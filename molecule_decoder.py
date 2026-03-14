from __future__ import annotations

"""
==============================================================================
MoleculeDecoder — 全上三角鍵結 (Full Upper-Triangular) bit-string 分子解碼器
==============================================================================

v8 修改說明：

  ★ FIX-CONNECTIVITY ★
  原版「硬體等價 CNOT 覆寫」只在 n_bonds_formed == 0 時強制第一個有效鍵。
  新版新增 `_enforce_connectivity()` 方法：
    - 建立分子圖（Union-Find），偵測所有孤立 component。
    - 對每個孤立 component，在 bond_codes 中強制加入一個 SINGLE bond，
      使其與最大 component 相連。
    - 對應 QMG `build_removing_bond_disconnection_circuit` 的 decoder 層等價實作。
  效果：有效消除斷連分子（如 C.C.O），顯著提升 validity。

  ★ FIX-NONE-FIRST ★
  `decode_bitstring` 的第一個原子 NONE 早返邏輯不變，但新增 warning 追蹤計數，
  便於監控 X-gate bias 的實際效果。

  其他不變：
  - ATOM_MAP, BOND_MAP, bitstring 解析順序
  - compute_fitness（shot-weighted validity × uniqueness）
  - partial validity salvage 路徑

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
    '001': ('C',  6),
    '010': ('O',  8),
    '011': ('N',  7),
    '100': ('S',  16),
    '101': ('P',  15),
    '110': ('F',  9),
    '111': ('Cl', 17),
}

if RDKIT_AVAILABLE:
    BondType = Chem.rdchem.BondType
    BOND_MAP: Dict[str, Optional[BondType]] = {
        '00': None,
        '01': BondType.TRIPLE,
        '10': BondType.SINGLE,
        '11': BondType.DOUBLE,
    }
else:
    BondType = object
    BOND_MAP = {'00': None, '01': 'TRIPLE', '10': 'SINGLE', '11': 'DOUBLE'}


def generate_bond_pairs(max_atoms: int) -> List[Tuple[int, int]]:
    """全上三角鍵對 (i, j) where i < j。bond 數 = N*(N-1)/2。"""
    return [(i, j) for i in range(max_atoms) for j in range(i + 1, max_atoms)]


# ===========================================================================
# Union-Find：用於偵測分子圖連通 component
# ===========================================================================

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self, nodes: List[int]) -> Dict[int, List[int]]:
        """回傳各 root → [node, ...] 的 dict。"""
        comp: Dict[int, List[int]] = {}
        for n in nodes:
            r = self.find(n)
            comp.setdefault(r, []).append(n)
        return comp


# ===========================================================================
# MoleculeDecoder
# ===========================================================================

class MoleculeDecoder:
    """將 CUDA-Q sample 結果解碼為分子與品質指標。"""

    def __init__(self, max_atoms: int = 4, partial_validity_weight: float = 0.0):
        """
        Args:
            max_atoms: 最大重原子數 N。
                       expected_length = N² + 2N = 3N + 2·N(N-1)/2
            partial_validity_weight: Partial Validity 在 fitness 中的補充權重。
                0.0（預設）= 純 validity×uniqueness；小正數提供額外梯度訊號。
        """
        self.max_atoms  = max_atoms
        self.partial_validity_weight = partial_validity_weight
        self.bond_pairs = generate_bond_pairs(max_atoms)
        self.expected_length = 3 * max_atoms + 2 * len(self.bond_pairs)

        # 統計計數器（用於監控第一個原子 NONE 的頻率）
        self._stat_total_decoded  = 0
        self._stat_none_first     = 0
        self._stat_disconnected   = 0
        self._stat_connectivity_fixed = 0

    # ------------------------------------------------------------------
    # bitstring 解析
    # ------------------------------------------------------------------

    def parse_bitstring(
        self, bitstring: str
    ) -> Optional[Tuple[List[str], List[str]]]:
        clean_bs = bitstring.replace(' ', '').strip()
        if not clean_bs or not all(c in '01' for c in clean_bs):
            return None

        if len(clean_bs) != self.expected_length:
            if len(clean_bs) > self.expected_length:
                clean_bs = clean_bs[:self.expected_length]
            else:
                clean_bs = clean_bs.ljust(self.expected_length, '0')

        # 測量順序：原子位元 (3N) 先，鍵結位元 (2·n_bonds) 後
        atom_codes = [
            clean_bs[3 * i: 3 * i + 3]
            for i in range(self.max_atoms)
        ]
        bond_offset = 3 * self.max_atoms
        bond_codes = [
            clean_bs[bond_offset + 2 * k: bond_offset + 2 * k + 2]
            for k in range(len(self.bond_pairs))
        ]
        return atom_codes, bond_codes

    # ------------------------------------------------------------------
    # 連通性強制：Union-Find + 最小生成樹策略
    # ------------------------------------------------------------------

    def _enforce_connectivity(
        self,
        bond_codes: List[str],
        atom_present: List[bool],
    ) -> Tuple[List[str], bool]:
        """
        確保所有存在的原子形成一個連通分子圖。

        演算法：
          1. 以現有 bond_codes 建立 Union-Find 圖。
          2. 若只有 ≤ 1 個 component，無需修改。
          3. 若有多個 component，找出最大 component（主體），
             對其他每個 component 各強制加入一個 SINGLE bond（'10'），
             連接到主體中最近的原子，直到所有 component 合併。

        對應 QMG `build_removing_bond_disconnection_circuit` 的 decoder 層等價。

        Returns:
            (修正後的 bond_codes, 是否有修正)
        """
        present_atoms = [i for i, p in enumerate(atom_present) if p]
        if len(present_atoms) < 2:
            return bond_codes, False

        uf = _UnionFind(self.max_atoms)

        # 建立初始圖（只計有鍵的 bond）
        for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
            if (pair_idx < len(bond_codes)
                    and atom_present[lhs] and atom_present[rhs]
                    and BOND_MAP.get(bond_codes[pair_idx]) is not None):
                uf.union(lhs, rhs)

        comps = uf.components(present_atoms)
        if len(comps) <= 1:
            return bond_codes, False   # 已連通

        # 找最大 component 作為主體
        main_root = max(comps, key=lambda r: len(comps[r]))
        main_atoms = set(comps[main_root])

        bond_codes = list(bond_codes)   # 確保可寫
        fixed = False

        # 合併每個孤立 component 到主體
        for root, members in comps.items():
            if root == main_root:
                continue
            # 在 bond_pairs 中找一個可連接此 component 與主體的鍵
            connected = False
            for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
                if pair_idx >= len(bond_codes):
                    break
                one_in_main = (lhs in main_atoms) != (rhs in main_atoms)
                lhs_in_comp  = lhs in members
                rhs_in_comp  = rhs in members
                if one_in_main and (lhs_in_comp or rhs_in_comp):
                    bond_codes[pair_idx] = '10'   # 強制 SINGLE bond
                    # 將此 component 加入主體
                    for m in members:
                        main_atoms.add(m)
                    fixed = True
                    connected = True
                    break
            if not connected:
                # fallback：直接在 bond_pairs 中找第一個連接兩個 component 任意原子的鍵
                for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
                    if pair_idx >= len(bond_codes):
                        break
                    if atom_present[lhs] and atom_present[rhs]:
                        if uf.find(lhs) != uf.find(rhs):
                            bond_codes[pair_idx] = '10'
                            uf.union(lhs, rhs)
                            fixed = True
                            break

        return bond_codes, fixed

    # ------------------------------------------------------------------
    # 單筆 bitstring 解碼（主入口）
    # ------------------------------------------------------------------

    def decode_bitstring(self, bitstring: str, count: int = 0) -> Dict:
        self._stat_total_decoded += 1
        record = {
            'bitstring':       bitstring,
            'count':           count,
            'atom_codes':      [],
            'bond_codes':      [],
            'smiles':          None,
            'valid':           False,
            'mol':             None,
            'n_decoded_atoms': 0,
            'n_bonds_formed':  0,
            'partial_valid':   False,
            'partial_smiles':  None,
            'frag_coverage':   0.0,
            'partial_mol':     None,
        }

        parsed = self.parse_bitstring(bitstring)
        if parsed is None:
            return record

        atom_codes, bond_codes = parsed
        record['atom_codes'] = atom_codes
        record['bond_codes'] = bond_codes

        # ━━ 硬約束：第一個原子禁止為 NONE ━━
        if ATOM_MAP.get(atom_codes[0]) is None:
            self._stat_none_first += 1
            return record

        # ━━ 獨立原子解碼（Independent Atom Decoding）━━
        atom_present = [ATOM_MAP.get(code) is not None for code in atom_codes]
        n_decoded_atoms = sum(atom_present)
        record['n_decoded_atoms'] = n_decoded_atoms

        if n_decoded_atoms < 1:
            return record

        # ━━ FIX-CONNECTIVITY：強制連通（v8 新增）━━
        # 在 build_molecule 前執行，確保解碼結果對應連通分子圖，
        # 消除 C.C、C.N.O 等斷連情形（對應 QMG bond-disconnection circuit）。
        if n_decoded_atoms >= 2:
            bond_codes, was_fixed = self._enforce_connectivity(bond_codes, atom_present)
            if was_fixed:
                self._stat_connectivity_fixed += 1
                record['bond_codes'] = bond_codes

        # 統計有效 bond 數
        n_bonds_formed = sum(
            1 for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs)
            if (pair_idx < len(bond_codes)
                and atom_present[lhs] and atom_present[rhs]
                and BOND_MAP.get(bond_codes[pair_idx]) is not None)
        )
        record['n_bonds_formed'] = n_bonds_formed

        # ━━ 嘗試建構 RDKit 分子 ━━
        mol = self.build_molecule(atom_codes, bond_codes)
        if mol is not None:
            smiles = self.to_smiles(mol)
            if smiles is not None:
                if RDKIT_AVAILABLE and mol.GetNumHeavyAtoms() < 2:
                    pass  # 排除單原子分子
                elif RDKIT_AVAILABLE and len(rdmolops.GetMolFrags(mol)) > 1:
                    # 連通性修復後仍斷連（極少數情形），嘗試 salvage
                    self._stat_disconnected += 1
                else:
                    record['smiles'] = smiles
                    record['valid']  = True
                    record['mol']    = mol
                    return record

        # ━━ Partial Validity Salvage：提取最大連通片段 ━━
        if n_decoded_atoms >= 2:
            salvage = self._try_salvage(atom_codes, bond_codes, atom_present)
            if salvage is not None:
                s_mol, s_smiles, coverage = salvage
                record['partial_valid']   = True
                record['partial_smiles']  = s_smiles
                record['frag_coverage']   = coverage
                record['partial_mol']     = s_mol

        return record

    # ------------------------------------------------------------------
    # RDKit 分子建構
    # ------------------------------------------------------------------

    def build_molecule(
        self, atom_codes: List[str], bond_codes: List[str]
    ) -> Optional['Chem.RWMol']:
        if not RDKIT_AVAILABLE:
            return None
        mol = Chem.RWMol()
        atom_index_map: Dict[int, int] = {}

        for i, code in enumerate(atom_codes):
            info = ATOM_MAP.get(code)
            if info is None:
                continue
            try:
                atom_index_map[i] = mol.AddAtom(Chem.Atom(info[1]))
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

    # ------------------------------------------------------------------
    # 輔助方法
    # ------------------------------------------------------------------

    @staticmethod
    def get_largest_fragment(mol: 'Chem.Mol') -> Optional['Chem.Mol']:
        """提取最大連通片段（保留供外部分析用）。"""
        if not RDKIT_AVAILABLE:
            return None
        try:
            frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            return sorted(frags, key=lambda f: f.GetNumAtoms(), reverse=True)[0] if frags else None
        except Exception:
            return None

    def _try_salvage(
        self,
        atom_codes: List[str],
        bond_codes:  List[str],
        atom_present: List[bool],
    ) -> Optional[Tuple['Chem.Mol', str, float]]:
        """Partial Validity：提取最大有效片段作為 partial SMILES。"""
        if not RDKIT_AVAILABLE:
            return None
        raw_mol = Chem.RWMol()
        atom_index_map: Dict[int, int] = {}

        for i, code in enumerate(atom_codes):
            if not atom_present[i]:
                continue
            info = ATOM_MAP.get(code)
            if info is None:
                continue
            try:
                atom_index_map[i] = raw_mol.AddAtom(Chem.Atom(info[1]))
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

        best_frag, best_size = None, 0
        for frag in (frags or []):
            try:
                Chem.SanitizeMol(frag)
            except Exception:
                continue
            if frag.GetNumAtoms() > best_size:
                best_frag, best_size = frag, frag.GetNumAtoms()

        if best_frag is None or best_size == 0:
            return None
        smiles = self.to_smiles(best_frag)
        if smiles is None:
            return None
        coverage = best_size / max(sum(atom_present), 1)
        return best_frag, smiles, coverage

    @staticmethod
    def to_smiles(mol: 'Chem.Mol') -> Optional[str]:
        if not RDKIT_AVAILABLE:
            return None
        try:
            s = Chem.MolToSmiles(mol, canonical=True)
            return s if s else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 批次解碼
    # ------------------------------------------------------------------

    def decode_counts(
        self, counts: Dict[str, int], top_k: int = 0
    ) -> List[Dict]:
        sorted_items = sorted(counts.items(), key=lambda x: -x[1])
        if top_k > 0:
            sorted_items = sorted_items[:top_k]
        return [self.decode_bitstring(bs, cnt) for bs, cnt in sorted_items]

    # ------------------------------------------------------------------
    # 適應度計算（優化用，shot-weighted）
    # ------------------------------------------------------------------

    def compute_fitness(
        self, counts: Dict[str, int]
    ) -> Tuple[float, List[Dict]]:
        """
        計算單一標量適應度（shot-weighted Validity × Uniqueness）。

        定義：
          validity   = valid_shots  / total_shots     (shot-weighted)
          uniqueness = unique_SMILES / valid_shots    (shot-weighted 分母)
          fitness    = validity × uniqueness + partial_bonus

        與 MoleculeEvaluator.evaluate() 的差異：
          後者使用 bitstring-count 分母（QMG Eq.4–5），每種 bitstring 等權重，
          用於論文指標報告；此方法用於 optimizer 的連續梯度訊號。

        Returns:
            (fitness_score, decoded_results)
        """
        try:
            decoded = self.decode_counts(counts)
        except Exception as e:
            print(f"\n[MoleculeDecoder 錯誤] 無法解析 counts: {e}")
            traceback.print_exc()
            return 0.0, []

        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0, []

        # 1. Shot-weighted Validity
        valid = [r for r in decoded if r.get('valid') and not r.get('partial_valid')]
        valid_shots = sum(counts.get(r['bitstring'], 0) for r in valid)
        validity = valid_shots / total_shots

        # 2. Partial Validity（可選梯度引導）
        partial = [r for r in decoded if r.get('partial_valid') and not r.get('valid')]
        partial_shots = sum(counts.get(r['bitstring'], 0) for r in partial)
        partial_validity = partial_shots / total_shots

        # 3. Shot-weighted Uniqueness
        unique_smiles = {r['smiles'] for r in valid if r.get('smiles')}
        uniqueness = len(unique_smiles) / valid_shots if valid_shots > 0 else 0.0

        # 4. Fitness
        fitness_score = (
            float(validity) * float(uniqueness)
            + self.partial_validity_weight * float(partial_validity)
        )
        return fitness_score, decoded

    # ------------------------------------------------------------------
    # 摘要輸出
    # ------------------------------------------------------------------

    def summarize(self, decoded_results: List[Dict]) -> str:
        total   = len(decoded_results)
        valid   = [r for r in decoded_results if r.get('valid') and not r.get('partial_valid')]
        partial = [r for r in decoded_results if r.get('partial_valid') and not r.get('valid')]
        lines   = [f'  Unique bitstrings        : {total}']
        if total > 0:
            lines.append(f'  Fully Valid              : {len(valid)}/{total} ({100*len(valid)/total:.1f}%)')
            lines.append(f'  Partial Valid            : {len(partial)}/{total} ({100*len(partial)/total:.1f}%)')
        lines.append(f'  Connectivity fixes       : {self._stat_connectivity_fixed}')
        lines.append(f'  None-first rejections    : {self._stat_none_first}')
        return '\n'.join(lines)