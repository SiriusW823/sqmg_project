"""
==============================================================================
SQMG Kernel — 3N+2 CUDA-Q 參數化量子線路 (v12 arbitrary-N, qpp-cpu compatible)
==============================================================================

v12 相對原版的修改：
  1. 新增 n_atom_qubits 和 n_bonds 作為獨立 kernel 參數（Python 層預算）
     避免 kernel 內的 3 * n_atoms（runtime Mult，CUDA-Q JIT 不支援）
  2. 使用累加器 aq += 3 取代 q_atoms[3 * i]（加法取代乘法）
  3. Bond loop 直接用 range(n_bonds)，bond 參數透過 p 累加索引
  所有其他功能（bond reuse、測量順序、3-gate bond 電路、HEA 結構）完全不變。

支援任意 N（透過 max_atoms 參數），不限制 N=4。
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int, n_atom_qubits: int, n_bonds: int):
    """
    SQMG 3N+2 量子線路 (v12, 零 runtime 乘法, 任意 N)。

    Kernel 參數：
      thetas        : 電路參數（長度 = 9*n_atoms + 3*n_bonds，由 Python 預算）
      n_atoms       : 重原子數量 N
      n_atom_qubits : 原子量子位元數 = 3*N（Python 層預算，避免 kernel 內 3*N）
      n_bonds       : 鍵結數 = N*(N-1)/2（Python 層預算）

    測量順序：atom bits (3N) 先，bond bits (N(N-1)) 後。
    Bitstring 長度：3N + N(N-1) = N²+2N。
    """
    q_atoms = cudaq.qvector(n_atom_qubits)   # n_atom_qubits = 3*N，直接傳入，無乘法
    q_bond  = cudaq.qvector(2)

    # ── Atom 0 偏置 ──────────────────────────────────────────────
    x(q_atoms[0])

    # ── Atom HEA (9 params/atom) ──────────────────────────────────
    # aq: atom qubit index，每輪 +3（加法取代 3*i 乘法）
    # p:  parameter index，每輪 +9
    p  = 0
    aq = 0
    for i in range(n_atoms):
        ry(thetas[p],     q_atoms[aq])
        ry(thetas[p + 1], q_atoms[aq + 1])
        ry(thetas[p + 2], q_atoms[aq + 2])
        rz(thetas[p + 3], q_atoms[aq])
        rz(thetas[p + 4], q_atoms[aq + 1])
        rz(thetas[p + 5], q_atoms[aq + 2])
        x.ctrl(q_atoms[aq],     q_atoms[aq + 1])
        x.ctrl(q_atoms[aq + 1], q_atoms[aq + 2])
        x.ctrl(q_atoms[aq + 2], q_atoms[aq])
        ry(thetas[p + 6], q_atoms[aq])
        ry(thetas[p + 7], q_atoms[aq + 1])
        ry(thetas[p + 8], q_atoms[aq + 2])
        p  = p  + 9
        aq = aq + 3

    # ── 原子測量（atom bits 先進入 bitstring）──────────────────────
    aq = 0
    for i in range(n_atoms):
        mz(q_atoms[aq])
        mz(q_atoms[aq + 1])
        mz(q_atoms[aq + 2])
        aq = aq + 3

    # ── Bond Blocks: 無條件執行，bond reuse (mz + reset) ─────────
    # p 繼承 atom loop 結束後的值 = 9*n_atoms
    # 每個 bond 使用 3 個參數，累加索引，無乘法
    # MoleculeDecoder 負責過濾 NONE 原子的 bond
    for b in range(n_bonds):
        ry(thetas[p],     q_bond[0])                  # gate 1: bond existence
        ry.ctrl(thetas[p + 1], q_bond[0], q_bond[1])  # gate 2: single→double
        ry.ctrl(thetas[p + 2], q_bond[1], q_bond[0])  # gate 3: double→triple
        mz(q_bond[0])
        mz(q_bond[1])
        reset(q_bond[0])
        reset(q_bond[1])
        p = p + 3


class SQMGKernel:
    """
    SQMG v12 封裝，支援任意 N，qpp-cpu 相容。

    對外介面與原版完全相同：
      n_params, get_param_bounds(), sample(), describe()
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms     = max_atoms
        self.shots         = shots
        self.n_bonds       = max_atoms * (max_atoms - 1) // 2
        self.n_atom_qubits = 3 * max_atoms          # 預算，傳給 kernel 避免 3*N
        self.n_qubits      = self.n_atom_qubits + 2  # 3N+2
        self.n_atom_params = 9 * max_atoms
        self.n_bond_params = 3 * self.n_bonds
        self.n_params      = self.n_atom_params + self.n_bond_params

    def sample(self, params: np.ndarray) -> Dict[str, int]:
        results = cudaq.sample(
            sqmg_circuit,
            params.tolist(),
            self.max_atoms,
            self.n_atom_qubits,   # 3*N，Python 層預算
            self.n_bonds,          # N*(N-1)/2，Python 層預算
            shots_count=self.shots,
        )
        counts: Dict[str, int] = {}
        for bitstring in results:
            clean_bs = bitstring.replace(" ", "")
            counts[clean_bs] = results.count(bitstring)
        return counts

    def get_expected_bitstring_length(self) -> int:
        return self.n_atom_qubits + 2 * self.n_bonds   # 3N + N(N-1) = N²+2N

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        s = self.n_atom_params
        lower[s::3] = 0.0;    upper[s::3] = np.pi / 2    # bond_existence
        lower[s+1::3] = 0.0;  upper[s+1::3] = np.pi / 2  # bond_order
        lower[s+2::3] = 0.0;  upper[s+2::3] = np.pi / 2  # bond_triple_order
        return lower, upper

    def describe(self) -> str:
        return (
            f"SQMG 3N+2 Ansatz (v12, arbitrary N, qpp-cpu compatible)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3×{self.max_atoms} + 2)\n"
            f"  Total parameters  : {self.n_params}  (9×{self.max_atoms} + 3×{self.n_bonds})\n"
            f"  Atom params       : {self.n_atom_params}  (9 per atom, 1-layer HEA)\n"
            f"  Bond params       : {self.n_bond_params}  (3 per bond: RY + 2×Ctrl-RY)\n"
            f"  Bond pairs        : {self.n_bonds}  (upper-triangular N(N-1)/2)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}  (N²+2N)\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Atom ansatz       : [RY+RZ] -> Ring-CNOT -> [RY]\n"
            f"  Atom 0 bias       : X gate (non-NONE bias; hard filter in Decoder)\n"
            f"  Bond ansatz       : RY(exist) + Ctrl-RY(single→double) + Ctrl-RY(double→triple)\n"
            f"  Bond states       : |00⟩=無鍵, |10⟩=單鍵, |11⟩=雙鍵, |01⟩=三鍵  (4 reachable)\n"
            f"  Bond execution    : Unconditional + bond reuse (mz+reset)\n"
            f"  Runtime arithmetic: None (all Mult pre-computed in Python)"
        )
