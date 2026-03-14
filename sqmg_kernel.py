"""
==============================================================================
SQMG Kernel — 3N+2 CUDA-Q 參數化量子線路 (v8 Static Circuit, sm_70 compatible)
==============================================================================

v8 變更：移除所有動態電路操作（mz mid-circuit, reset, if conditionals）。
V100 (sm_70) + CUDA-Q 0.12 不支援動態電路控制流，改採靜態全分配架構：

  架構：
    • Atom Register : 3N qubits，靜態 HEA
    • Bond Register : 2 × N(N-1)/2 qubits，靜態分配（不 reuse，不 reset）
    • 總 qubits     = 3N + N(N-1)  = N²+2N
    • 測量          : cudaq.sample() 自動測量全部 qubit，無需顯式 mz()

  適用範圍：N ≤ 4（N=4 時 24 qubits，StateVec = 256MB，V100 16GB 可用）
  N ≥ 5 的 StateVec 超出 V100 記憶體，請改用較少 atoms。

  Bitstring 順序（對應 MoleculeDecoder）：
    前 3N bits  = atom bits   （allocation order: q_atoms[0..3N-1]）
    後 N(N-1) bits = bond bits （allocation order: q_bonds[0..N(N-1)-1]）
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int):
    """
    SQMG 靜態量子線路 (v8, sm_70 compatible)。

    參數佈局：與 v7 完全相同
      Atom i : thetas[9i : 9i+9]
      Bond bp: thetas[9N + 3*bp : 9N + 3*bp + 3]
    總參數量 = 9N + 3·N(N-1)/2（不變）

    Qubit 佈局（v8 新增，靜態分配）：
      q_atoms[0..3N-1]           : atom qubits
      q_bonds[0..N(N-1)-1]       : bond qubits（每 bond 2 個，共 N(N-1)/2 個 bond）
    """
    n_bonds = n_atoms * (n_atoms - 1) // 2
    q_atoms = cudaq.qvector(3 * n_atoms)
    q_bonds = cudaq.qvector(2 * n_bonds)   # 靜態分配，無 reuse，無 reset

    # ================================================================
    # Atom 0 偏置
    # ================================================================
    x(q_atoms[0])

    # ================================================================
    # Atom Blocks: 1-Layer HEA (9 params/atom)
    # ================================================================
    param_idx = 0
    for i in range(n_atoms):
        ry(thetas[param_idx],     q_atoms[3 * i])
        ry(thetas[param_idx + 1], q_atoms[3 * i + 1])
        ry(thetas[param_idx + 2], q_atoms[3 * i + 2])
        rz(thetas[param_idx + 3], q_atoms[3 * i])
        rz(thetas[param_idx + 4], q_atoms[3 * i + 1])
        rz(thetas[param_idx + 5], q_atoms[3 * i + 2])
        x.ctrl(q_atoms[3 * i],     q_atoms[3 * i + 1])
        x.ctrl(q_atoms[3 * i + 1], q_atoms[3 * i + 2])
        x.ctrl(q_atoms[3 * i + 2], q_atoms[3 * i])
        ry(thetas[param_idx + 6], q_atoms[3 * i])
        ry(thetas[param_idx + 7], q_atoms[3 * i + 1])
        ry(thetas[param_idx + 8], q_atoms[3 * i + 2])
        param_idx += 9

    # ================================================================
    # Bond Blocks: 靜態全分配，每 bond 使用獨立 2 qubits
    # 無 mid-circuit measurement，無 reset，無條件判斷
    # ================================================================
    bp = param_idx
    bond_q = 0
    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):
            ry(thetas[bp],     q_bonds[bond_q])
            ry.ctrl(thetas[bp + 1], q_bonds[bond_q],     q_bonds[bond_q + 1])
            ry.ctrl(thetas[bp + 2], q_bonds[bond_q + 1], q_bonds[bond_q])
            bond_q += 2
            bp += 3
    # cudaq.sample() 自動測量所有 qubits（q_atoms 先，q_bonds 後）


class SQMGKernel:
    """
    SQMG 靜態量子線路封裝 (v8, sm_70 compatible)。

    總 qubits = N²+2N（靜態，無 reuse）
    總參數量  = 9N + 3·N(N-1)/2（與 v7 相同）
    Bitstring = N²+2N bits（與 v7 相同，格式不變）
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_bonds = max_atoms * (max_atoms - 1) // 2
        self.n_qubits = 3 * max_atoms + 2 * self.n_bonds  # 靜態全分配
        self.n_atom_params = 9 * max_atoms
        self.n_bond_params = 3 * self.n_bonds
        self.n_params = self.n_atom_params + self.n_bond_params

    def sample(self, params: np.ndarray) -> Dict[str, int]:
        theta_list = params.tolist()
        results = cudaq.sample(
            sqmg_circuit,
            theta_list,
            self.max_atoms,
            shots_count=self.shots,
        )
        counts: Dict[str, int] = {}
        for bitstring in results:
            clean_bs = bitstring.replace(" ", "")
            counts[clean_bs] = results.count(bitstring)
        return counts

    def get_expected_bitstring_length(self) -> int:
        return 3 * self.max_atoms + 2 * self.n_bonds  # N²+2N

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        bond_start_idx = self.n_atom_params
        lower[bond_start_idx::3] = 0.0
        upper[bond_start_idx::3] = np.pi / 2
        lower[bond_start_idx + 1::3] = 0.0
        upper[bond_start_idx + 1::3] = np.pi / 2
        lower[bond_start_idx + 2::3] = 0.0
        upper[bond_start_idx + 2::3] = np.pi / 2
        return lower, upper

    def describe(self) -> str:
        return (
            f"SQMG Static Circuit (v8, sm_70 compatible)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3N + N(N-1) = N²+2N, static)\n"
            f"  Total parameters  : {self.n_params}  (9N + 3·N(N-1)/2)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}  (N²+2N)\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Dynamic ops       : None (sm_70 compatible)"
        )
