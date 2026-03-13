from __future__ import annotations

"""
==============================================================================
MoleculeDecoder — 全上三角鍵結 (Full Upper-Triangular) bit-string 分子解碼器
==============================================================================

• 使用全上三角鄰接矩陣 N(N-1)/2 bonds，可表達分支與環狀結構
• bit-string 長度 = 3N + 2·N(N-1)/2 = N² + 2N
• 原子獨立解碼 (Independent Atom Decoding)：每個原子位置獨立判斷是否為 NONE，
  不再在第一個 NONE 處截斷，對應論文與 sqmg_circuit 的動態電路設計。
• BOND_MAP 支援 4 種鍵結類型：|00⟩=無鍵, |10⟩=單鍵, |11⟩=雙鍵, |01⟩=三鍵。
• compute_fitness 的 validity 與 uniqueness 均使用 shot-weighted 分母，
  提供連續且平滑的優化梯度訊號；論文定義的 bitstring-count 指標由 MoleculeEvaluator 負責。
• fitness = Validity_sw × Uniqueness_sw（無 QED / length_penalty）
==============================================================================
"""

import traceback
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    rdmolops = None
    RDKIT_AVAILABLE = False


ATOM_MAP: Dict[str, Optional[Tuple[str, int]]] = {
    '000': None,
    '001': ('C', 6),
    '010': ('O', 8),
    '011': ('N', 7),
    '100': ('S', 16),
    '101': ('P', 15),
    '110': ('F', 9),
    '111': ('Cl', 17),
}

if RDKIT_AVAILABLE:
    BondType = Chem.rdchem.BondType
    # ━━ BOND_MAP (v7: 4 reachable states) ━━
    # sqmg_circuit v7 的 Bond 子電路結構（3-gate）：
    #   Gate 1: RY(θ₁) on q_bond[0]              → |00⟩ ↔ |10⟩  (bond existence)
    #   Gate 2: CRY(θ₂) on q_bond[1], ctrl q[0]  → |10⟩ ↔ |11⟩  (single→double)
    #   Gate 3: CRY(θ₃) on q_bond[0], ctrl q[1]  → |11⟩ ↔ |01⟩  (double→triple)
    #
    # 從 |00⟩ 出發，4 種可達測量結果：
    #   |00⟩ (q0=0,q1=0) → 無鍵 (θ₁→0)
    #   |10⟩ (q0=1,q1=0) → 單鍵 SINGLE  (θ₁>0, θ₂→0)
    #   |11⟩ (q0=1,q1=1) → 雙鍵 DOUBLE  (θ₁>0, θ₂>0, θ₃→0)
    #   |01⟩ (q0=0,q1=1) → 三鍵 TRIPLE  (θ₁>0, θ₂>0, θ₃>0)  ← v7 新增
    BOND_MAP: Dict[str, Optional[BondType]] = {
        '00': None,
        '01': BondType.TRIPLE,    # v7: reachable via gate 3 (double→triple)
        '10': BondType.SINGLE,
        '11': BondType.DOUBLE,
    }
else:
    BondType = object
    BOND_MAP = {
        '00': None,
        '01': 'TRIPLE',
        '10': 'SINGLE',
        '11': 'DOUBLE',
    }


def generate_bond_pairs(max_atoms: int) -> List[Tuple[int, int]]:
    """回傳全上三角鍵對 (i, j) where i < j，對齊 SQMG 論文的鄰接矩陣設計。

    bond 數量 = N*(N-1)/2
    例：N=4 → (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)，共 6 個鍵
    """
    return [(i, j) for i in range(max_atoms) for j in range(i + 1, max_atoms)]


class MoleculeDecoder:
    """將 CUDA-Q sample 結果解碼為分子與品質指標。"""

    def __init__(self, max_atoms: int = 4, partial_validity_weight: float = 0.0):
        """初始化解碼器。

        expected_length = 3*N + 2*(N*(N-1)//2) = N² + 2N
        例：N=4 → 24 bits（12 bond bits + 12 atom bits）

        Args:
            max_atoms: 最大重原子數量 N。
            partial_validity_weight: Partial Validity 在 fitness 中的權重。
                論文定義的純 fitness = Validity × Uniqueness，對應 weight=0.0（預設）。
                設為小正數（如 0.05）可提供梯度引導，但偏離論文目標定義。
        """
        self.max_atoms = max_atoms
        self.partial_validity_weight = partial_validity_weight
        self.bond_pairs = generate_bond_pairs(max_atoms)
        self.expected_length = 3 * max_atoms + 2 * len(self.bond_pairs)

    def parse_bitstring(self, bitstring: str) -> Optional[Tuple[List[str], List[str]]]:
        clean_bs = bitstring.replace(' ', '').strip()
        if not clean_bs or not all(char in '01' for char in clean_bs):
            return None

        if len(clean_bs) != self.expected_length:
            if len(clean_bs) > self.expected_length:
                clean_bs = clean_bs[:self.expected_length]
            else:
                clean_bs = clean_bs.ljust(self.expected_length, '0')

        # 測量順序：原子位元 (3N) 先，鍵結位元 (N(N-1)) 後
        # 對應 v6 Dynamic Circuit 的 mz 呼叫順序
        atom_section_len = 3 * self.max_atoms
        atom_codes = [
            clean_bs[3 * atom_idx:3 * atom_idx + 3]
            for atom_idx in range(self.max_atoms)
        ]
        bond_offset = atom_section_len
        bond_codes = [
            clean_bs[bond_offset + idx:bond_offset + idx + 2]
            for idx in range(0, 2 * len(self.bond_pairs), 2)
        ]
        return atom_codes, bond_codes

    # 修改原因：新增 decode_bitstring，讓完整鄰接矩陣的單筆 bit-string 解析邏輯集中管理。
    def decode_bitstring(self, bitstring: str, count: int = 0) -> Dict:
        record = {
            'bitstring': bitstring,
            'count': count,
            'atom_codes': [],
            'bond_codes': [],
            'smiles': None,
            'valid': False,
            'mol': None,
            'n_decoded_atoms': 0,
            'n_bonds_formed': 0,
            'partial_valid': False,
            'partial_smiles': None,
            'frag_coverage': 0.0,
            'partial_mol': None,
        }

        parsed = self.parse_bitstring(bitstring)
        if parsed is None:
            return record

        atom_codes, bond_codes = parsed
        record['atom_codes'] = atom_codes
        record['bond_codes'] = bond_codes

        # ━━ 硬約束：第一個原子禁止為 NONE (|000⟩) ━━
        # 論文明確規定 "the first atom cannot be labeled as non-existent"，
        # 若第一個原子解碼為 NONE，直接標記為無效而非任由後續流程舊救。
        if ATOM_MAP.get(atom_codes[0]) is None:
            return record

        # ━━ 獨立原子解碼 (Independent Atom Decoding) ━━
        # 每個原子位置獨立判斷是否為 NONE，不在第一個 NONE 處截斷。
        # 對應 sqmg_circuit 的 atom_exists[i]：每個原子的量子態獨立測量與塌縮。
        # 例：['001','000','010'] → 原子 0 (C) 和原子 2 (O) 存在，原子 1 缺席。
        atom_present = [ATOM_MAP.get(code) is not None for code in atom_codes]
        n_decoded_atoms = sum(atom_present)
        record['n_decoded_atoms'] = n_decoded_atoms

        n_bonds_formed = 0
        for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
            if not atom_present[lhs] or not atom_present[rhs]:
                continue
            if pair_idx < len(bond_codes) and BOND_MAP.get(bond_codes[pair_idx]) is not None:
                n_bonds_formed += 1
        record['n_bonds_formed'] = n_bonds_formed

        # ━━ 硬體等價 CNOT 覆寫 (Hardware-equivalent Conditional CNOT) ━━
        # 模擬 QMG 論文中的電路層行為：當所有 bond bits 為零
        # （完全斷連）時，強制在第一對有效原子之間建立單鍵，
        # 確保基線連通性 (Connectivity)。
        if n_bonds_formed == 0 and n_decoded_atoms >= 2:
            for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
                if atom_present[lhs] and atom_present[rhs]:
                    bond_codes = list(bond_codes)  # 確保可寫
                    bond_codes[pair_idx] = '10'     # 強制 SINGLE bond
                    n_bonds_formed = 1
                    break

        mol = self.build_molecule(atom_codes, bond_codes)
        if mol is not None:
            smiles = self.to_smiles(mol)
            if smiles is not None:
                # 排除孤立單原子分子（原子數 < 2）
                if RDKIT_AVAILABLE and mol.GetNumHeavyAtoms() < 2:
                    pass
                # 排除斷連分子（disconnected graph）
                # 對應論文 Validity 定義：有效分子必須為連通結構。
                # 斷連分子（如 C.C）SMILES 包含 '.'，不符合化學意義的「一個分子」。
                elif RDKIT_AVAILABLE and len(rdmolops.GetMolFrags(mol)) > 1:
                    pass  # 不接受斷連分子；交由後續 salvage 擷取最大片段
                else:
                    record['smiles'] = smiles
                    record['valid'] = True
                    record['mol'] = mol
                    return record

        if n_decoded_atoms >= 2:
            salvage = self._try_salvage(atom_codes, bond_codes, atom_present)
            if salvage is not None:
                s_mol, s_smiles, coverage = salvage
                record['partial_valid'] = True
                record['partial_smiles'] = s_smiles
                record['frag_coverage'] = coverage
                record['partial_mol'] = s_mol

        return record

    def build_molecule(self, atom_codes: List[str], bond_codes: List[str]) -> Optional[Chem.RWMol]:
        if not RDKIT_AVAILABLE:
            return None
        mol = Chem.RWMol()
        # ━━ 獨立原子解碼：跳過 NONE 原子（不截斷），保留索引對映 ━━
        # atom_index_map[i] = rdkit atom idx，僅包含存在的原子。
        atom_index_map: Dict[int, int] = {}
        for i, atom_code in enumerate(atom_codes):
            atom_info = ATOM_MAP.get(atom_code)
            if atom_info is None:
                continue   # 跳過 NONE，不 break（支援 gap 原子）
            try:
                atom_index_map[i] = mol.AddAtom(Chem.Atom(atom_info[1]))
            except Exception:
                continue

        if len(atom_index_map) < 1:
            return None

        for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
            if pair_idx >= len(bond_codes):
                break
            if lhs not in atom_index_map or rhs not in atom_index_map:
                continue
            bond_type = BOND_MAP.get(bond_codes[pair_idx])
            if bond_type is None:
                continue
            try:
                mol.AddBond(atom_index_map[lhs], atom_index_map[rhs], bond_type)
            except Exception:
                continue

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return mol

    @staticmethod
    def get_largest_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
        # [DEPRECATED] 主流程已改為直接剔除斷連分子，此方法不再被呼叫。
        # 保留供外部工具或分析腳本使用。
        if not RDKIT_AVAILABLE:
            return None
        try:
            frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                return None
            return sorted(frags, key=lambda frag: frag.GetNumAtoms(), reverse=True)[0]
        except Exception:
            return None

    def _try_salvage(
        self,
        atom_codes: List[str],
        bond_codes: List[str],
        atom_present: List[bool],
    ) -> Optional[Tuple[Chem.Mol, str, float]]:
        if not RDKIT_AVAILABLE:
            return None
        raw_mol = Chem.RWMol()
        # 獨立原子解碼：使用 atom_present 跳過 NONE，保留索引對映
        atom_index_map: Dict[int, int] = {}
        for i, atom_code in enumerate(atom_codes):
            if not atom_present[i]:
                continue
            atom_info = ATOM_MAP.get(atom_code)
            if atom_info is None:
                continue
            try:
                atom_index_map[i] = raw_mol.AddAtom(Chem.Atom(atom_info[1]))
            except Exception:
                continue

        if len(atom_index_map) < 2:
            return None

        for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
            if pair_idx >= len(bond_codes):
                break
            if lhs not in atom_index_map or rhs not in atom_index_map:
                continue
            bond_type = BOND_MAP.get(bond_codes[pair_idx])
            if bond_type is None:
                continue
            try:
                raw_mol.AddBond(atom_index_map[lhs], atom_index_map[rhs], bond_type)
            except Exception:
                continue

        try:
            frags = rdmolops.GetMolFrags(raw_mol, asMols=True, sanitizeFrags=False)
        except Exception:
            return None

        best_frag = None
        best_size = 0
        for frag in frags or []:
            try:
                Chem.SanitizeMol(frag)
            except Exception:
                continue
            if frag.GetNumAtoms() > best_size:
                best_frag = frag
                best_size = frag.GetNumAtoms()

        if best_frag is None or best_size == 0:
            return None

        smiles = self.to_smiles(best_frag)
        if smiles is None:
            return None

        n_present = sum(atom_present)
        coverage = best_size / max(n_present, 1)
        return best_frag, smiles, coverage

    @staticmethod
    def to_smiles(mol: Chem.Mol) -> Optional[str]:
        if not RDKIT_AVAILABLE:
            return None
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            return smiles if smiles else None
        except Exception:
            return None

    def decode_counts(self, counts: Dict[str, int], top_k: int = 0) -> List[Dict]:
        sorted_items = sorted(counts.items(), key=lambda item: -item[1])
        if top_k > 0:
            sorted_items = sorted_items[:top_k]
        return [self.decode_bitstring(bitstring, count) for bitstring, count in sorted_items]

    def compute_fitness(self, counts: Dict[str, int]) -> Tuple[float, List[Dict]]:
        """計算單一標量適應度（shot-weighted Validity × Uniqueness）。

        compute_fitness 與 MoleculeEvaluator.evaluate() 的分母選擇：
        ─────────────────────────────────────────────────────────────
        • compute_fitness（此方法）：兩個指標均使用 shot-weighted 分母。
          - validity   = valid_shots  / total_shots
          - uniqueness = unique_SMILES / valid_shots
          優點：乘積 validity × uniqueness 量綱一致，梯度訊號平滑，
                高頻率出現的有效分子結構在 fitness 中權重更大，有利於優化收斂。

        • MoleculeEvaluator.evaluate()：使用 bitstring-count 分母（QMG Eq.4–5），
          每種 bitstring 等權重，可直接與論文報告數字比較。

        兩者僅在報告指標時有差異；優化目標語意相同（最大化有效率×多樣性）。
        """
        try:
            decoded = self.decode_counts(counts)
        except Exception as e:
            print(f"\n[MoleculeDecoder 嚴重錯誤] 無法解析 counts 字典:")
            print(f"錯誤訊息: {e}")
            traceback.print_exc()
            return 0.0, []

        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0, []

        # 1. Shot-weighted Validity
        valid = [
            r for r in decoded
            if r.get('valid') and not r.get('partial_valid')
        ]
        valid_shots = sum(counts.get(r['bitstring'], 0) for r in valid)
        validity = valid_shots / total_shots

        # 2. Partial Validity（梯度引導）
        partial = [
            r for r in decoded
            if r.get('partial_valid') and not r.get('valid')
        ]
        partial_shots = sum(counts.get(r['bitstring'], 0) for r in partial)
        partial_validity = partial_shots / total_shots

        # 3. Shot-weighted Uniqueness
        # 分母同樣使用 valid_shots（而非 valid bitstring 種數），
        # 使 validity × uniqueness 量綱一致：
        #   validity   = valid_shots / total_shots     → 機率意義的有效率
        #   uniqueness = unique_SMILES / valid_shots   → 每個有效 shot 帶來新分子的比例
        unique_smiles = {r['smiles'] for r in valid if r.get('smiles')}
        uniqueness = len(unique_smiles) / valid_shots if valid_shots > 0 else 0.0

        # 4. Fitness = validity_sw × uniqueness_sw + optional partial bonus
        fitness_score = (
            float(validity) * float(uniqueness)
            + self.partial_validity_weight * float(partial_validity)
        )
        return fitness_score, decoded

    def summarize(self, decoded_results: List[Dict]) -> str:
        total = len(decoded_results)
        valid = [record for record in decoded_results if record.get('valid') and not record.get('partial_valid')]
        partial = [record for record in decoded_results if record.get('partial_valid') and not record.get('valid')]

        lines = [f'  Unique bitstrings   : {total}']
        if total > 0:
            lines.append(f'  Fully Valid         : {len(valid)} / {total} ({100 * len(valid) / total:.1f}%)')
            lines.append(f'  Partial Valid       : {len(partial)} / {total} ({100 * len(partial) / total:.1f}%)')

        return '\n'.join(lines)