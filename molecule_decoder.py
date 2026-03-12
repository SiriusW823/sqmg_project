from __future__ import annotations

"""
==============================================================================
MoleculeDecoder — 全上三角鍵結 (Full Upper-Triangular) bit-string 分子解碼器
==============================================================================

• 使用全上三角鄰接矩陣 N(N-1)/2 bonds，可表達分支與環狀結構
• bit-string 長度 = 3N + 2·N(N-1)/2 = N² + 2N
• validity 以總 shots 為分母，避免系統性高估
• fitness = Validity × Uniqueness（無 QED / length_penalty）
==============================================================================
"""

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
    BOND_MAP: Dict[str, Optional[BondType]] = {
        '00': None,
        '01': BondType.SINGLE,
        '10': BondType.DOUBLE,
        '11': BondType.TRIPLE,
    }
else:
    BondType = object
    BOND_MAP = {
        '00': None,
        '01': 'SINGLE',
        '10': 'DOUBLE',
        '11': 'TRIPLE',
    }


def generate_bond_pairs(max_atoms: int) -> List[Tuple[int, int]]:
    """回傳全上三角鍵對 (i, j) where i < j，對齊 SQMG 論文的鄰接矩陣設計。

    bond 數量 = N*(N-1)/2
    例：N=4 → (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)，共 6 個鍵
    """
    return [(i, j) for i in range(max_atoms) for j in range(i + 1, max_atoms)]


class MoleculeDecoder:
    """將 CUDA-Q sample 結果解碼為分子與品質指標。"""

    def __init__(self, max_atoms: int = 4):
        """初始化解碼器。

        expected_length = 3*N + 2*(N*(N-1)//2) = N² + 2N
        例：N=4 → 24 bits（12 bond bits + 12 atom bits）
        """
        self.max_atoms = max_atoms
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

        bond_section_len = 2 * len(self.bond_pairs)
        bond_codes = [
            clean_bs[idx:idx + 2]
            for idx in range(0, bond_section_len, 2)
        ]
        atom_offset = bond_section_len
        atom_codes = [
            clean_bs[atom_offset + 3 * atom_idx:atom_offset + 3 * atom_idx + 3]
            for atom_idx in range(self.max_atoms)
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

        n_decoded_atoms = 0
        for atom_code in atom_codes:
            if ATOM_MAP.get(atom_code) is None:
                break
            n_decoded_atoms += 1
        record['n_decoded_atoms'] = n_decoded_atoms

        n_bonds_formed = 0
        for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
            if lhs >= n_decoded_atoms or rhs >= n_decoded_atoms:
                continue
            if pair_idx < len(bond_codes) and BOND_MAP.get(bond_codes[pair_idx]) is not None:
                n_bonds_formed += 1
        record['n_bonds_formed'] = n_bonds_formed

        mol = self.build_molecule(atom_codes, bond_codes)
        if mol is not None:
            largest = self.get_largest_fragment(mol)
            if largest is not None and largest.GetNumAtoms() > 0:
                mol = largest
            smiles = self.to_smiles(mol)
            if smiles is not None:
                # 後處理過濾：排除孤立單原子分子（對應 Chen et al. JCTC 2025
                # Figure 1 的 "Remove bond disconnection" 步驟）。
                # 重原子數 < 2 的分子（如 C, O, N 單原子）不具化學意義，
                # 不標記為 valid，交由下方 salvage 流程處理。
                if RDKIT_AVAILABLE and mol.GetNumHeavyAtoms() < 2:
                    pass
                else:
                    record['smiles'] = smiles
                    record['valid'] = True
                    record['mol'] = mol
                    return record

        if n_decoded_atoms >= 2:
            salvage = self._try_salvage(atom_codes, bond_codes, n_decoded_atoms)
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
        atom_indices: List[int] = []

        for atom_code in atom_codes:
            atom_info = ATOM_MAP.get(atom_code)
            if atom_info is None:
                break
            try:
                atom_indices.append(mol.AddAtom(Chem.Atom(atom_info[1])))
            except Exception:
                break

        n_valid_atoms = len(atom_indices)
        if n_valid_atoms < 1:
            return None

        for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
            if pair_idx >= len(bond_codes):
                break
            if lhs >= n_valid_atoms or rhs >= n_valid_atoms:
                continue
            bond_type = BOND_MAP.get(bond_codes[pair_idx])
            if bond_type is None:
                continue
            try:
                mol.AddBond(atom_indices[lhs], atom_indices[rhs], bond_type)
            except Exception:
                continue

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return mol

    @staticmethod
    def get_largest_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
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
        n_decoded_atoms: int,
    ) -> Optional[Tuple[Chem.Mol, str, float]]:
        if not RDKIT_AVAILABLE:
            return None
        raw_mol = Chem.RWMol()
        atom_indices: List[int] = []
        for atom_code in atom_codes:
            atom_info = ATOM_MAP.get(atom_code)
            if atom_info is None:
                break
            try:
                atom_indices.append(raw_mol.AddAtom(Chem.Atom(atom_info[1])))
            except Exception:
                break

        if len(atom_indices) < 2:
            return None

        for pair_idx, (lhs, rhs) in enumerate(self.bond_pairs):
            if pair_idx >= len(bond_codes):
                break
            if lhs >= len(atom_indices) or rhs >= len(atom_indices):
                continue
            bond_type = BOND_MAP.get(bond_codes[pair_idx])
            if bond_type is None:
                continue
            try:
                raw_mol.AddBond(atom_indices[lhs], atom_indices[rhs], bond_type)
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

        coverage = best_size / max(n_decoded_atoms, 1)
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
        """計算單一標量適應度 = validity * uniqueness。"""
        try:
            decoded = self.decode_counts(counts)
        except Exception:
            return 0.0, []

        total_shots = sum(counts.values())
        valid = [
            r for r in decoded
            if r.get('valid') and not r.get('partial_valid')
        ]
        valid_shots = sum(
            counts.get(r['bitstring'], 0) for r in valid
        )
        validity = valid_shots / total_shots if total_shots > 0 else 0.0
        n_valid = len(valid)

        unique_smiles = {
            r['smiles']
            for r in valid
            if r.get('smiles')
        }
        uniqueness = len(unique_smiles) / n_valid if n_valid > 0 else 0.0

        fitness_score = float(validity) * float(uniqueness)
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