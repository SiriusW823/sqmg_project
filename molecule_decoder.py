"""
==============================================================================
MoleculeDecoder — 將 CUDA-Q Bit-string 解碼為分子結構 (修正版 v2)
==============================================================================

本模組負責：
1. 解析 CUDA-Q cudaq.sample() 回傳的 bit-string
2. 將 3-bit 原子碼映射為化學元素
3. 將 2-bit 鍵碼映射為化學鍵類型
4. 使用 RDKit 建構分子物件並驗證化學有效性
5. 計算 QED (Quantitative Estimate of Drug-likeness) 分數

v2 修正重點
──────────
1. 【空白清理】parse_bitstring() 在解析前執行 .replace(" ", "")，
   處理 CUDA-Q 輸出可能帶有的空白字元。
2. 【長度防呆】bit-string 長度不等於 5N-2 時嚴格處理：
   過長→截斷，過短→補零，並記錄警告。
3. 【NONE 邊界】遇到 atom "000" 時立即 break，且明確記錄有效原子
   數量 n_valid_atoms，確保 bond 迴圈範圍為 [0, n_valid_atoms-2]，
   絕不會嘗試連接到不存在的原子 index。
4. 【RDKit 嚴格防呆】SanitizeMol 包裝在 try-except 中，
   任何化學無效的分子都會回傳 None，compute_fitness 據此給 0 分。

Bit-string 格式（與 sqmg_kernel.py v2 一致）：
  [bond_bits | atom_bits]
  bond_bits : 2(N-1) 位，每 2 位代表一組鍵結
  atom_bits : 3N 位，每 3 位代表一個原子

原子類型映射（8 States Mapping，來自 SQMG 規格）：
  |000⟩ → NONE (無原子 / 終止符)
  |001⟩ → C  (碳)
  |010⟩ → O  (氧)
  |011⟩ → N  (氮)
  |100⟩ → S  (硫)
  |101⟩ → P  (磷)
  |110⟩ → F  (氟)
  |111⟩ → Cl (氯)

鍵類型映射：
  |00⟩ → NONE (無鍵)
  |01⟩ → Single Bond (單鍵)
  |10⟩ → Double Bond (雙鍵)
  |11⟩ → Triple Bond (參鍵)
==============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdmolops


# ============================================================================
# 映射表 (Lookup Tables)
# ============================================================================

# 原子類型映射：3-bit string → (元素符號, 原子序)
# NONE 表示「分子建構到此為止」的終止訊號
ATOM_MAP: Dict[str, Optional[Tuple[str, int]]] = {
    "000": None,          # NONE / Terminator
    "001": ("C",  6),     # Carbon
    "010": ("O",  8),     # Oxygen
    "011": ("N",  7),     # Nitrogen
    "100": ("S", 16),     # Sulfur
    "101": ("P", 15),     # Phosphorus
    "110": ("F",  9),     # Fluorine
    "111": ("Cl", 17),    # Chlorine
}

# 鍵結類型映射：2-bit string → RDKit BondType
BOND_MAP: Dict[str, Optional[Chem.rdchem.BondType]] = {
    "00": None,                        # No bond
    "01": Chem.rdchem.BondType.SINGLE,  # Single bond
    "10": Chem.rdchem.BondType.DOUBLE,  # Double bond
    "11": Chem.rdchem.BondType.TRIPLE,  # Triple bond
}


class MoleculeDecoder:
    """
    將 CUDA-Q 量子線路的測量結果（bit-string 計數字典）
    解碼為化學分子結構。

    使用方式：
        decoder = MoleculeDecoder(max_atoms=4)
        results = decoder.decode_counts(counts_dict)
        # results: list of dicts with 'smiles', 'qed', 'valid', etc.
    """

    def __init__(self, max_atoms: int = 4):
        """
        Args:
            max_atoms: 最大重原子數量 N（需與 SQMGKernel 一致）
        """
        self.max_atoms = max_atoms
        self.n_bonds = max_atoms - 1
        self.expected_length = 5 * max_atoms - 2  # bit-string 預期長度

    # ────────────────────────────────────────────────────────────
    # Bit-string 解析
    # ────────────────────────────────────────────────────────────

    def parse_bitstring(self, bitstring: str) -> Optional[Tuple[List[str], List[str]]]:
        """
        將單一 bit-string 拆解為原子碼列表與鍵碼列表。

        CUDA-Q bit-string 格式（measurement order）：
          位置 0          .. 2(N-1)-1  → Bond 測量 (每 2 bit 一組)
          位置 2(N-1)     .. 5N-3      → Atom 測量 (每 3 bit 一組)

        v2 修正：
        • 【空白清理】CUDA-Q 某些後端的 bit-string 會夾帶空格
          （如 "01 10 001 010 ..."），解析前統一移除。
        • 【長度防呆】若清理後長度不等於 5N-2：
          - 過長 → 截斷為前 5N-2 位
          - 過短 → 右邊補零到 5N-2 位
          - 如果清理後全是非 0/1 字元 → 回傳 None

        Args:
            bitstring: CUDA-Q sample() 回傳的 bit-string（可能含空白）

        Returns:
            (atom_codes, bond_codes) 或 None（如果 bit-string 完全無效）
            atom_codes: ['001', '010', ...] — 每個原子的 3-bit 碼
            bond_codes: ['01', '10', ...]   — 每個鍵的 2-bit 碼
        """
        # ── 【修正 #2】空白清理 ──
        clean_bs = bitstring.replace(" ", "").strip()

        # 驗證：確保只包含 0 和 1
        if not clean_bs or not all(c in '01' for c in clean_bs):
            return None

        # ── 長度防呆 ──
        if len(clean_bs) != self.expected_length:
            if len(clean_bs) > self.expected_length:
                clean_bs = clean_bs[:self.expected_length]
            else:
                clean_bs = clean_bs.ljust(self.expected_length, '0')

        # ── 拆解 Bond 碼 ──
        bond_section_len = 2 * self.n_bonds
        bond_codes: List[str] = []
        for j in range(self.n_bonds):
            start = 2 * j
            bond_codes.append(clean_bs[start:start + 2])

        # ── 拆解 Atom 碼 ──
        atom_offset = bond_section_len
        atom_codes: List[str] = []
        for j in range(self.max_atoms):
            start = atom_offset + 3 * j
            atom_codes.append(clean_bs[start:start + 3])

        return atom_codes, bond_codes

    # ────────────────────────────────────────────────────────────
    # 分子建構
    # ────────────────────────────────────────────────────────────

    def build_molecule(
        self, atom_codes: List[str], bond_codes: List[str]
    ) -> Optional[Chem.RWMol]:
        """
        從原子碼與鍵碼建構 RDKit 可編輯分子物件 (RWMol)。

        邏輯流程：
        1. 逐一讀取 atom_codes，若遇到 NONE ("000") 立即停止。
        2. 記錄有效原子數量 n_valid_atoms。
        3. 根據 bond_codes 在「相鄰的有效原子間」添加化學鍵。
           【關鍵】Bond 迴圈的範圍為 [0, n_valid_atoms - 2]，
           確保不會嘗試連接到 NONE 原子的 index。
        4. 使用 RDKit SanitizeMol 嚴格驗證化學有效性。

        Args:
            atom_codes: 原子 3-bit 碼列表
            bond_codes: 鍵 2-bit 碼列表

        Returns:
            有效的 RDKit Mol 物件，或 None（如果分子無效）
        """
        mol = Chem.RWMol()
        atom_indices: List[int] = []

        # ── Step 1: 添加原子 ──
        for i, acode in enumerate(atom_codes):
            atom_info = ATOM_MAP.get(acode)

            if atom_info is None:
                break

            symbol, atomic_num = atom_info
            try:
                atom = Chem.Atom(atomic_num)
                idx = mol.AddAtom(atom)
                atom_indices.append(idx)
            except Exception:
                break

        n_valid_atoms = len(atom_indices)

        if n_valid_atoms < 1:
            return None

        # ── Step 2: 添加化學鍵 ──
        n_valid_bonds = min(len(bond_codes), n_valid_atoms - 1)

        for j in range(n_valid_bonds):
            bcode = bond_codes[j]
            bond_type = BOND_MAP.get(bcode)

            if bond_type is None:
                continue

            try:
                mol.AddBond(atom_indices[j], atom_indices[j + 1], bond_type)
            except Exception:
                continue

        # ── Step 3: Sanitize ──
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        return mol

    # ────────────────────────────────────────────────────────────
    # 後處理：取最大連通片段
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def get_largest_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        如果分子有多個不連通的片段，取最大的那一個。
        """
        try:
            frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                return None
            frags_sorted = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)
            return frags_sorted[0]
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────
    # v3: 結構救援 — 從 Sanitize 失敗的分子中提取最大有效子圖
    # ────────────────────────────────────────────────────────────

    def _try_salvage(
        self,
        atom_codes: List[str],
        bond_codes: List[str],
        n_decoded_atoms: int,
    ) -> Optional[Tuple[Chem.Mol, str, float, float]]:
        """
        當 build_molecule (含 SanitizeMol) 完全失敗時，
        嘗試從原始 bit-string 建構未 sanitize 的分子圖，
        再「逐片段」各自 sanitize，回傳最大有效子圖。

        Args:
            atom_codes:      原子 3-bit 碼列表
            bond_codes:      鍵 2-bit 碼列表
            n_decoded_atoms: 有效原子數量（遇到 NONE 前的原子數）

        Returns:
            (mol, smiles, qed, frag_coverage) 或 None
        """
        if n_decoded_atoms < 2:
            return None

        raw_mol = Chem.RWMol()
        atom_indices: List[int] = []

        for acode in atom_codes:
            atom_info = ATOM_MAP.get(acode)
            if atom_info is None:
                break
            try:
                idx = raw_mol.AddAtom(Chem.Atom(atom_info[1]))
                atom_indices.append(idx)
            except Exception:
                break

        n = len(atom_indices)
        if n < 2:
            return None

        for j in range(min(len(bond_codes), n - 1)):
            bt = BOND_MAP.get(bond_codes[j])
            if bt is None:
                continue
            try:
                raw_mol.AddBond(atom_indices[j], atom_indices[j + 1], bt)
            except Exception:
                continue

        try:
            frags = rdmolops.GetMolFrags(
                raw_mol, asMols=True, sanitizeFrags=False
            )
        except Exception:
            return None

        best_frag: Optional[Chem.Mol] = None
        best_size = 0

        for frag in (frags or []):
            try:
                Chem.SanitizeMol(frag)
                if frag.GetNumAtoms() > best_size:
                    best_frag = frag
                    best_size = frag.GetNumAtoms()
            except Exception:
                continue

        if best_frag is None or best_size == 0:
            return None

        smiles = self.to_smiles(best_frag)
        if smiles is None:
            return None

        qed = self.compute_qed(best_frag)
        coverage = best_size / n_decoded_atoms
        return best_frag, smiles, qed, coverage

    # ────────────────────────────────────────────────────────────
    # QED 計算
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_qed(mol: Chem.Mol) -> float:
        """
        計算分子的 QED (Quantitative Estimate of Drug-likeness) 分數。
        """
        try:
            mol_with_h = Chem.AddHs(mol)
            return QED.qed(mol_with_h)
        except Exception:
            return 0.0

    @staticmethod
    def to_smiles(mol: Chem.Mol) -> Optional[str]:
        """
        將 RDKit Mol 轉換為正規 SMILES 字串。
        """
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
            return smi if smi else None
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────
    # 主要 API：批次解碼
    # ────────────────────────────────────────────────────────────

    def decode_counts(
        self, counts: Dict[str, int], top_k: int = 0
    ) -> List[Dict]:
        """
        批次解碼 cudaq.sample() 回傳的計數字典。

        Args:
            counts:  {bitstring: count} 字典（來自 SQMGKernel.sample()）
            top_k:   僅處理前 k 個最常出現的 bit-string (0 = 全部)

        Returns:
            結果列表，每個元素為 dict
        """
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        if top_k > 0:
            sorted_items = sorted_items[:top_k]

        results: List[Dict] = []

        for bitstring, count in sorted_items:
            record = {
                'bitstring': bitstring,
                'count': count,
                'atom_codes': [],
                'bond_codes': [],
                'smiles': None,
                'qed': 0.0,
                'valid': False,
                'mol': None,
                'n_decoded_atoms': 0,
                'n_bonds_formed': 0,
                'partial_valid': False,
                'partial_smiles': None,
                'partial_qed': 0.0,
                'frag_coverage': 0.0,
                'partial_mol': None,
            }

            try:
                parse_result = self.parse_bitstring(bitstring)

                if parse_result is None:
                    results.append(record)
                    continue

                atom_codes, bond_codes = parse_result
                record['atom_codes'] = atom_codes
                record['bond_codes'] = bond_codes

                # ── v3: 計算結構元數據 ──
                n_decoded_atoms = 0
                for ac in atom_codes:
                    if ATOM_MAP.get(ac) is None:
                        break
                    n_decoded_atoms += 1
                record['n_decoded_atoms'] = n_decoded_atoms

                n_potential = min(
                    len(bond_codes), max(n_decoded_atoms - 1, 0)
                )
                n_bonds_formed = sum(
                    1 for j in range(n_potential)
                    if BOND_MAP.get(bond_codes[j]) is not None
                )
                record['n_bonds_formed'] = n_bonds_formed

                # ── 建構分子 ──
                mol = self.build_molecule(atom_codes, bond_codes)

                if mol is not None:
                    frag = self.get_largest_fragment(mol)
                    if frag is not None and frag.GetNumAtoms() > 0:
                        mol = frag

                    smiles = self.to_smiles(mol)
                    if smiles is not None:
                        qed_score = self.compute_qed(mol)
                        record['smiles'] = smiles
                        record['qed'] = qed_score
                        record['valid'] = True
                        record['mol'] = mol

                # ── v3: Sanitize 失敗 → 嘗試救援最大有效子圖 ──
                if not record['valid'] and n_decoded_atoms >= 2:
                    salvage = self._try_salvage(
                        atom_codes, bond_codes, n_decoded_atoms
                    )
                    if salvage is not None:
                        s_mol, s_smi, s_qed, s_cov = salvage
                        record['partial_valid'] = True
                        record['partial_smiles'] = s_smi
                        record['partial_qed'] = s_qed
                        record['frag_coverage'] = s_cov
                        record['partial_mol'] = s_mol

                # ── 防呆：嚴格確保 valid 與 partial_valid 互斥 ──
                if record.get('partial_valid'):
                    record['valid'] = False
                    record['smiles'] = None
                    record['qed'] = 0.0
                    record['mol'] = None

            except Exception:
                pass

            results.append(record)

        return results

    # ────────────────────────────────────────────────────────────
    # v5: Shaping Reward
    # ────────────────────────────────────────────────────────────

    def _score_molecule(self, record: Dict) -> float:
        """
        為單一 bit-string 解碼結果計算結構 Shaping Reward (v5)。
        """
        n_decoded = record.get('n_decoded_atoms', 0)
        n_bonds = record.get('n_bonds_formed', 0)

        if n_decoded == 0:
            return 0.0

        max_bonds = max(n_decoded - 1, 1)
        connectivity = min(n_bonds / max_bonds, 1.0)
        size_ratio = n_decoded / self.max_atoms

        if record.get('valid'):
            return 0.40 * size_ratio + 0.30 * connectivity + 0.30

        if record.get('partial_valid'):
            coverage = record.get('frag_coverage', 0.0)
            return 0.30 * size_ratio + 0.20 * connectivity + 0.20 * coverage

        return 0.20 * size_ratio + 0.10 * connectivity

    # ────────────────────────────────────────────────────────────
    # 適應度計算 (Shaping Reward Fitness)
    # ────────────────────────────────────────────────────────────

    def compute_fitness(
        self, counts: Dict[str, int],
    ) -> Tuple[float, List[Dict]]:
        """
        計算一組量子線路參數的單目標適應度 (v6 — SOQPSO)。

        回傳值為單一標量 Fitness，供 SOQPSOOptimizer 直接使用。

        公式：
          Fitness = Validity × Uniqueness × Length_Penalty

        目標定義：
        ──────────────────────────────────────────────────────────
        • Validity      = n_valid / n_total
        • Uniqueness    = n_unique_smiles / n_valid
        • Length_Penalty = 0.1 if avg_heavy_atoms < 3 else 1.0

        Args:
            counts: bit-string 計數字典（來自 cudaq.sample()）

        Returns:
            (fitness, decoded_results)
            fitness:         ∈ [0, 1] 標量
            decoded_results: 完整解碼結果列表
        """
        try:
            decoded = self.decode_counts(counts)
        except Exception:
            return 0.0, []

        if not decoded:
            return 0.0, decoded

        n_total = len(decoded)

        # ── 篩選完全有效分子（排除 partial_valid）──
        MIN_HEAVY_ATOMS = 3
        valid = []
        for r in decoded:
            if not r.get('valid') or r.get('partial_valid'):
                continue
            mol = r.get('mol')
            if mol is not None and mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
                r['valid'] = False
                continue
            valid.append(r)
        n_valid = len(valid)

        validity = n_valid / n_total if n_total > 0 else 0.0

        if n_valid > 0:
            unique_smiles = {
                r['smiles'] for r in valid if r.get('smiles')
            }
            uniqueness = len(unique_smiles) / n_valid

            # Length Penalty: 有效分子的平均重原子數
            heavy_counts = []
            for r in valid:
                mol = r.get('mol')
                if mol is not None:
                    heavy_counts.append(mol.GetNumHeavyAtoms())
            avg_heavy = sum(heavy_counts) / len(heavy_counts) if heavy_counts else 0
            length_penalty = 0.1 if avg_heavy < MIN_HEAVY_ATOMS else 1.0
        else:
            uniqueness = 0.0
            length_penalty = 0.1

        fitness = validity * uniqueness * length_penalty
        return fitness, decoded

    def summarize(self, decoded_results: List[Dict]) -> str:
        """
        產生解碼結果的文字摘要。
        """
        total = len(decoded_results)
        valid = [
            r for r in decoded_results
            if r.get('valid') and not r.get('partial_valid')
        ]
        partial = [
            r for r in decoded_results
            if r.get('partial_valid') and not r.get('valid')
        ]
        n_valid = len(valid)
        n_partial = len(partial)

        lines = [
            f"  Unique bitstrings   : {total}",
        ]
        if total > 0:
            lines.append(
                f"  Fully Valid         : {n_valid} / {total} "
                f"({100 * n_valid / total:.1f}%)"
            )
            lines.append(
                f"  Partial Valid       : {n_partial} / {total} "
                f"({100 * n_partial / total:.1f}%)"
            )

        if valid:
            qeds = [r['qed'] for r in valid]
            lines.append(f"  Mean QED (valid)    : {np.mean(qeds):.4f}")
            lines.append(f"  Max  QED (valid)    : {np.max(qeds):.4f}")
            lines.append(f"  Top SMILES (fully valid):")
            top5 = sorted(valid, key=lambda r: -r['qed'])[:5]
            for r in top5:
                lines.append(f"    {r['smiles']:30s}  QED={r['qed']:.4f}  "
                             f"(count={r['count']})")

        if partial:
            partial_qeds = [r.get('partial_qed', 0) for r in partial]
            lines.append(f"  Mean QED (partial)  : {np.mean(partial_qeds):.4f}")
            lines.append(f"  Top Partial SMILES (僅供參考，不計入有效性統計):")
            top3 = sorted(partial, key=lambda r: -r.get('partial_qed', 0))[:3]
            for r in top3:
                lines.append(
                    f"    {r.get('partial_smiles', '?'):30s}  "
                    f"QED={r.get('partial_qed', 0):.4f}  "
                    f"cov={r.get('frag_coverage', 0):.2f}"
                )

        return "\n".join(lines)
