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
        # cudaq.sample() 在某些後端（特別是 tensornet）輸出的
        # bit-string 會使用空格分隔不同暫存器的測量結果。
        # 必須在解析前統一清除，否則切片位置會全部錯亂。
        clean_bs = bitstring.replace(" ", "").strip()

        # 驗證：確保只包含 0 和 1
        if not clean_bs or not all(c in '01' for c in clean_bs):
            return None

        # ── 長度防呆 ──
        if len(clean_bs) != self.expected_length:
            if len(clean_bs) > self.expected_length:
                # 過長：截斷為預期長度
                clean_bs = clean_bs[:self.expected_length]
            else:
                # 過短：右邊補零
                # 原因：CUDA-Q 偶爾因測量截斷導致長度不足
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

        v2 修正：
        • 【NONE 邊界 #3】明確計算 n_valid_atoms，
          bond 迴圈以 min(j, n_valid_atoms - 2) 為上界。
        • 【RDKit 防呆 #4】SanitizeMol 包裝在 try-except 中，
          化學無效分子（如碳接 5 鍵）安全回傳 None。

        Args:
            atom_codes: 原子 3-bit 碼列表
            bond_codes: 鍵 2-bit 碼列表

        Returns:
            有效的 RDKit Mol 物件，或 None（如果分子無效）
        """
        mol = Chem.RWMol()
        atom_indices: List[int] = []  # RDKit 內部原子索引

        # ── Step 1: 添加原子 ──
        # 逐一處理 atom_codes，遇到 "000" (NONE) 立即終止。
        for i, acode in enumerate(atom_codes):
            atom_info = ATOM_MAP.get(acode)

            if atom_info is None:
                # 遇到 NONE (000) → 終止符
                # 【修正 #3】立即 break，不再處理後續原子。
                # 此原子位置 i 不會被加入 atom_indices，
                # 因此後續 bond 迴圈也不會嘗試連接到此 index。
                break

            symbol, atomic_num = atom_info
            try:
                atom = Chem.Atom(atomic_num)
                idx = mol.AddAtom(atom)
                atom_indices.append(idx)
            except Exception:
                # 如果 RDKit 拒絕此原子（極罕見），跳過
                break

        # 記錄有效原子數量
        n_valid_atoms = len(atom_indices)

        # 至少需要 1 個原子才能形成分子
        if n_valid_atoms < 1:
            return None

        # ── Step 2: 添加化學鍵 ──
        # 【修正 #3 核心】Bond 迴圈範圍嚴格限制在有效原子之間：
        #   - Bond j 連接 atom j 和 atom j+1
        #   - 有效範圍：j = 0 到 n_valid_atoms - 2
        #   - 如果 atom i 是 NONE，那 n_valid_atoms = i，
        #     最後一個 bond index = i-2，
        #     因此 bond i-1（連接 atom i-1 到 NONE 的 atom i）
        #     根本不會被執行。
        n_valid_bonds = min(len(bond_codes), n_valid_atoms - 1)

        for j in range(n_valid_bonds):
            bcode = bond_codes[j]
            bond_type = BOND_MAP.get(bcode)

            if bond_type is None:
                # 鍵碼 "00" → 不加鍵（原子 j 和 j+1 不直接相連）
                continue

            try:
                mol.AddBond(atom_indices[j], atom_indices[j + 1], bond_type)
            except Exception:
                # RDKit 在某些情況下會拒絕加鍵（如重複鍵、
                # 同一原子自環等），安全跳過。
                continue

        # ── Step 3: Sanitize（化學有效性嚴格驗證） ──
        # 【修正 #4】量子電路可能生成化學上不合理的結構，例如：
        #   - 碳原子接了 5 個鍵（超過 4 價）
        #   - 孤立的三鍵連接到鹵素原子
        #   - 非法的芳香結構
        # SanitizeMol 會檢查：價態、芳香性、環結構等。
        # 若任何檢查失敗，該分子視為無效，安全回傳 None。
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # 無法通過 sanitization → 分子化學上無效
            # 這在量子生成中是常見的，不是程式錯誤。
            return None

        return mol

    # ────────────────────────────────────────────────────────────
    # 後處理：取最大連通片段
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def get_largest_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        如果分子有多個不連通的片段，取最大的那一個。
        這是因為 bond "00" 可能產生斷裂的分子圖。

        Args:
            mol: RDKit Mol 物件

        Returns:
            最大片段的 Mol，或 None
        """
        try:
            frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                return None
            # 按重原子數排序，取最大片段
            frags_sorted = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)
            return frags_sorted[0]
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────
    # QED 計算
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_qed(mol: Chem.Mol) -> float:
        """
        計算分子的 QED (Quantitative Estimate of Drug-likeness) 分數。

        QED ∈ [0, 1]，越高表示越像藥物分子。
        QED 綜合考慮：分子量、LogP、TPSA、HBA、HBD、旋轉鍵數等。

        Args:
            mol: 經過 sanitize 的 RDKit Mol 物件

        Returns:
            QED 分數 (float)，若計算失敗回傳 0.0
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

        Args:
            mol: RDKit Mol 物件

        Returns:
            SMILES 字串，或 None（如果轉換失敗）
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

        流程：
        1. 按計數降序排列 bit-string
        2. 對每個 bit-string：清理 → 解析 → 建構分子 → 計算 QED
        3. 回傳所有結果（含有效與無效）

        Args:
            counts:  {bitstring: count} 字典（來自 SQMGKernel.sample()）
            top_k:   僅處理前 k 個最常出現的 bit-string (0 = 全部)

        Returns:
            結果列表，每個元素為 dict：
            {
                'bitstring': str,        # 原始 bit-string
                'count': int,            # 出現次數
                'atom_codes': list[str], # 原子 3-bit 碼列表
                'bond_codes': list[str], # 鍵 2-bit 碼列表
                'smiles': str or None,   # 正規 SMILES
                'qed': float,            # QED 分數
                'valid': bool,           # 是否為化學有效分子
                'mol': RDKit Mol or None,# RDKit 分子物件
            }
        """
        # 按計數降序排列
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
            }

            try:
                # ── 解析 bit-string ──
                # parse_bitstring 內部已處理空白清理與長度防呆
                parse_result = self.parse_bitstring(bitstring)

                # 如果 bit-string 完全無效（非 0/1 字元），跳過
                if parse_result is None:
                    results.append(record)
                    continue

                atom_codes, bond_codes = parse_result
                record['atom_codes'] = atom_codes
                record['bond_codes'] = bond_codes

                # ── 建構分子 ──
                mol = self.build_molecule(atom_codes, bond_codes)

                if mol is not None:
                    # 嘗試取最大連通片段（bond "00" 可能產生斷裂分子圖）
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

            except Exception:
                # 【修正 #4】任何未預期的錯誤都安全地跳過，
                # 不會導致主迴圈 crash。
                # 該 bit-string 的 record 維持 valid=False, qed=0.0。
                pass

            results.append(record)

        return results

    # ────────────────────────────────────────────────────────────
    # 統計彙整
    # ────────────────────────────────────────────────────────────

    def compute_fitness(self, counts: Dict[str, int],
                        alpha: float = 0.4) -> Tuple[float, List[Dict]]:
        """
        計算一組量子線路參數的適應度分數。

        適應度 = α × validity_ratio + (1 − α) × mean_qed

        【修正 #4 延伸】此函式呼叫 decode_counts()，後者內部的每個
        分子解碼都被 try-except 保護。即使所有分子都化學無效，
        此函式仍會回傳 fitness=0.0，保證主迴圈不會 crash。

        Args:
            counts: bit-string 計數字典
            alpha:  validity_ratio 的權重 (0~1)

        Returns:
            (fitness, decoded_results)
            fitness:         適應度分數 (float)，恆 ≥ 0
            decoded_results: 完整解碼結果列表
        """
        try:
            decoded = self.decode_counts(counts)
        except Exception:
            # 極端情況：整個 decode 過程失敗
            return 0.0, []

        if not decoded:
            return 0.0, decoded

        total_unique = len(decoded)
        valid_results = [r for r in decoded if r['valid']]
        n_valid = len(valid_results)

        # 有效性比率
        validity_ratio = n_valid / total_unique if total_unique > 0 else 0.0

        # 有效分子的平均 QED
        if n_valid > 0:
            mean_qed = np.mean([r['qed'] for r in valid_results])
        else:
            mean_qed = 0.0

        # 組合適應度
        fitness = alpha * validity_ratio + (1.0 - alpha) * float(mean_qed)

        return fitness, decoded

    def summarize(self, decoded_results: List[Dict]) -> str:
        """
        產生解碼結果的文字摘要。

        Args:
            decoded_results: decode_counts() 的回傳值

        Returns:
            格式化的文字摘要
        """
        total = len(decoded_results)
        valid = [r for r in decoded_results if r['valid']]
        n_valid = len(valid)

        lines = [
            f"  Unique bitstrings : {total}",
            f"  Valid molecules   : {n_valid} / {total} "
            f"({100 * n_valid / total:.1f}%)" if total > 0 else "",
        ]

        if valid:
            qeds = [r['qed'] for r in valid]
            lines.append(f"  Mean QED          : {np.mean(qeds):.4f}")
            lines.append(f"  Max  QED          : {np.max(qeds):.4f}")
            lines.append(f"  Top SMILES:")
            # 取 QED 最高的前 5 個
            top5 = sorted(valid, key=lambda r: -r['qed'])[:5]
            for r in top5:
                lines.append(f"    {r['smiles']:30s}  QED={r['qed']:.4f}  "
                             f"(count={r['count']})")

        return "\n".join(lines)
