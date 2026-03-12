"""
==============================================================================
SQMG Kernel — 3N+3 CUDA-Q 參數化量子線路 (v5 Full Upper-Triangular)
==============================================================================

架構概覽 (Scalable Quantum Molecular Generation, v5, N²+12N params)
─────────────────────────────────────────────────────────────────
• 原子暫存器 (Atom Register)：每個重原子靜態分配 3 顆量子位元，
  每個原子使用 3 層 Hardware-Efficient Ansatz，共 15 個參數。
    Layer 1: RY×3 + RZ×3 → Ring-CNOT
    Layer 2: RY×3 + RZ×3
    Layer 3: Linear-CNOT → RZ×3
• 鍵暫存器 (Bond Register)：2 顆量子位元，每個鍵使用 6 個參數。
    RY×2 + RZ×2 → CNOT → RZ×2
    全上三角矩陣：N(N-1)/2 bonds，透過 mid-circuit measurement + reset 重複使用
• 輔助位元 (Ancilla)：1 顆量子位元。
• 總量子位元數 = 3N + 3。
• 總參數量   = 15N + 6·N(N-1)/2 = 3N² + 12N（O(N²)）。
• Bit-string 長度 = N(N-1) + 3N = N² + 2N。
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int):
    """
    SQMG N²+12N 參數化量子線路 (v5 Full Upper-Triangular)。

    參數佈局 (Parameter Layout)
    ──────────────────────────
    Atom Register: 15 params/atom
        Atom i: thetas[15i : 15i+15]
            Layer 1 RY: thetas[15i+0..2], RZ: thetas[15i+3..5]
            Ring-CNOT
            Layer 2 RY: thetas[15i+6..8], RZ: thetas[15i+9..11]
            Linear-CNOT
            Layer 3 RZ: thetas[15i+12..14]

    Bond Register: 6 params/bond (全上三角，N(N-1)/2 bonds)
        Bond pair (i,j) 的 index bp_idx 按上三角順序排列
        thetas[15N + 6*bp_idx : 15N + 6*bp_idx + 6]
            RY×2 + RZ×2 → CNOT → RZ×2

    總參數量 = 15N + 6·N(N-1)/2 = 3N² + 12N
    Bit-string 長度 = N(N-1) + 3N = N² + 2N
    測量順序：先 bond bits（N(N-1) 個），後 atom bits（3N 個）
    """
    n_qubits = 3 * n_atoms + 3
    q = cudaq.qvector(n_qubits)

    bond_q0_idx = 3 * n_atoms
    bond_q1_idx = 3 * n_atoms + 1
    ancilla_idx = 3 * n_atoms + 2

    n_atom_params = 15 * n_atoms

    # ================================================================
    # Atom Blocks: 3-Layer HEA (15 params/atom)
    # ================================================================
    for i in range(n_atoms):
        ap = 15 * i
        base = 3 * i

        # Layer 1: RY + RZ
        ry(thetas[ap],     q[base])
        ry(thetas[ap + 1], q[base + 1])
        ry(thetas[ap + 2], q[base + 2])
        rz(thetas[ap + 3], q[base])
        rz(thetas[ap + 4], q[base + 1])
        rz(thetas[ap + 5], q[base + 2])
        # Ring-CNOT
        x.ctrl(q[base],     q[base + 1])
        x.ctrl(q[base + 1], q[base + 2])
        x.ctrl(q[base + 2], q[base])
        # Layer 2: RY + RZ
        ry(thetas[ap + 6],  q[base])
        ry(thetas[ap + 7],  q[base + 1])
        ry(thetas[ap + 8],  q[base + 2])
        rz(thetas[ap + 9],  q[base])
        rz(thetas[ap + 10], q[base + 1])
        rz(thetas[ap + 11], q[base + 2])
        # Layer 3: Linear-CNOT + RZ
        x.ctrl(q[base],     q[base + 1])
        x.ctrl(q[base + 1], q[base + 2])
        rz(thetas[ap + 12], q[base])
        rz(thetas[ap + 13], q[base + 1])
        rz(thetas[ap + 14], q[base + 2])

    # ================================================================
    # Bond Blocks: 全上三角 N(N-1)/2 bonds
    # 順序：(0,1),(0,2),...,(0,N-1),(1,2),...,(N-2,N-1)
    # ================================================================
    bp_idx = 0
    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):
            bp = n_atom_params + 6 * bp_idx

            # ---- Bond Block: RY+RZ → CNOT → RZ (6 params) ----
            ry(thetas[bp],     q[bond_q0_idx])
            ry(thetas[bp + 1], q[bond_q1_idx])
            rz(thetas[bp + 2], q[bond_q0_idx])
            rz(thetas[bp + 3], q[bond_q1_idx])
            x.ctrl(q[bond_q0_idx], q[bond_q1_idx])
            rz(thetas[bp + 4], q[bond_q0_idx])
            rz(thetas[bp + 5], q[bond_q1_idx])

            # ---- Ancilla OR for atom_i (非 NONE 偵測) ----
            base_i = 3 * atom_i
            x(q[base_i])
            x(q[base_i + 1])
            x(q[base_i + 2])
            x.ctrl(q[base_i], q[base_i + 1], q[base_i + 2], q[ancilla_idx])
            x(q[ancilla_idx])
            x(q[base_i])
            x(q[base_i + 1])
            x(q[base_i + 2])
            x.ctrl(q[ancilla_idx], q[bond_q0_idx])

            # Uncompute ancilla
            x(q[base_i])
            x(q[base_i + 1])
            x(q[base_i + 2])
            x(q[ancilla_idx])
            x.ctrl(q[base_i], q[base_i + 1], q[base_i + 2], q[ancilla_idx])
            x(q[base_i])
            x(q[base_i + 1])
            x(q[base_i + 2])

            # CZ phase: atom_i → bond
            z.ctrl(q[base_i],     q[bond_q0_idx])
            z.ctrl(q[base_i + 1], q[bond_q1_idx])
            z.ctrl(q[base_i + 2], q[bond_q0_idx])

            # ---- Ancilla OR for atom_j ----
            base_j = 3 * atom_j
            x(q[base_j])
            x(q[base_j + 1])
            x(q[base_j + 2])
            x.ctrl(q[base_j], q[base_j + 1], q[base_j + 2], q[ancilla_idx])
            x(q[ancilla_idx])
            x(q[base_j])
            x(q[base_j + 1])
            x(q[base_j + 2])
            x.ctrl(q[ancilla_idx], q[bond_q1_idx])

            # Uncompute ancilla
            x(q[base_j])
            x(q[base_j + 1])
            x(q[base_j + 2])
            x(q[ancilla_idx])
            x.ctrl(q[base_j], q[base_j + 1], q[base_j + 2], q[ancilla_idx])
            x(q[base_j])
            x(q[base_j + 1])
            x(q[base_j + 2])

            # CZ phase: atom_j → bond (cross)
            z.ctrl(q[base_j],     q[bond_q1_idx])
            z.ctrl(q[base_j + 1], q[bond_q0_idx])
            z.ctrl(q[base_j + 2], q[bond_q1_idx])

            # Bond internal CNOT
            x.ctrl(q[bond_q0_idx], q[bond_q1_idx])

            # Mid-circuit measurement
            mz(q[bond_q0_idx])
            mz(q[bond_q1_idx])

            # Reset for reuse
            reset(q[bond_q0_idx])
            reset(q[bond_q1_idx])

            bp_idx += 1

    # Final atom measurements
    for i in range(3 * n_atoms):
        mz(q[i])


class SQMGKernel:
    """
    SQMG N²+12N Full Upper-Triangular 量子線路的 Python 封裝 (v5)。

    全上三角矩陣 N(N-1)/2 bonds，可表達分支與環狀分子結構。
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_qubits = 3 * max_atoms + 3
        self.kernel_func = sqmg_circuit
        self.n_bonds = max_atoms * (max_atoms - 1) // 2
        self.n_atom_params = 15 * max_atoms
        self.n_bond_params = 6 * self.n_bonds
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
        return 2 * self.n_bonds + 3 * self.max_atoms  # N²+2N

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        return lower, upper

    def describe(self) -> str:
        return (
            f"SQMG 3N+3 Ansatz (v5 Full Upper-Triangular)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3×{self.max_atoms} + 3)\n"
            f"  Total parameters  : {self.n_params}  (15×{self.max_atoms} + 6×{self.n_bonds})\n"
            f"  Atom params       : {self.n_atom_params}  (15 per atom, 3-layer HEA)\n"
            f"  Bond params       : {self.n_bond_params}  (6 per bond, RY+RZ+CNOT+RZ)\n"
            f"  Bond pairs        : {self.n_bonds}  (upper-triangular N(N-1)/2)\n"
            f"  Ancilla           : 1  (OR non-NONE detection, uncomputed)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Ansatz per atom   : [RY+RZ] -> Ring-CNOT -> [RY+RZ] -> Linear-CNOT -> [RZ]\n"
            f"  Bond ansatz       : [RY+RZ] -> CNOT -> [RZ]"
        )
