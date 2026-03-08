"""
==============================================================================
SQMG Kernel — 3N+3 CUDA-Q 參數化量子線路 (v5 Enhanced Expressivity)
==============================================================================

架構概覽 (Scalable Quantum Molecular Generation, v5, 21N-6 params)
─────────────────────────────────────────────────────────────────
• 原子暫存器 (Atom Register)：每個重原子靜態分配 3 顆量子位元，
  每個原子使用 3 層 Hardware-Efficient Ansatz，共 15 個參數。
    Layer 1: RY×3 + RZ×3 → Ring-CNOT
    Layer 2: RY×3 + RZ×3
    Layer 3 (NEW): Linear-CNOT → RZ×3
• 鍵暫存器 (Bond Register)：2 顆量子位元，每個鍵使用 6 個參數。
    RY×2 + RZ×2 → CNOT → RZ×2
• 輔助位元 (Ancilla)：1 顆量子位元。
• 總量子位元數 = 3N + 3。
• 總參數量   = 15N + 6(N-1) = 21N - 6。
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int):
    """
    SQMG 21N-6 參數化量子線路 (v5 Enhanced Expressivity)。

    參數佈局 (Parameter Layout)
    ──────────────────────────
    Atom Register: 15 params/atom
        Atom i: thetas[15i : 15i+15]
            Layer 1 RY: thetas[15i+0..2], RZ: thetas[15i+3..5]
            Ring-CNOT
            Layer 2 RY: thetas[15i+6..8], RZ: thetas[15i+9..11]
            [NEW] Linear-CNOT
            [NEW] Layer 3 RZ: thetas[15i+12..14]

    Bond Register: 6 params/bond
        Bond j: thetas[15N + 6j : 15N + 6j + 6]
            RY: thetas[15N+6j..+1], RZ: thetas[15N+6j+2..+3]
            [NEW] CNOT(bond_q0 -> bond_q1)
            [NEW] RZ: thetas[15N+6j+4..+5]

    總參數量 = 21N - 6
    Bit-string 長度 = 5N - 2 (不變)
    """
    n_qubits = 3 * n_atoms + 3
    q = cudaq.qvector(n_qubits)

    bond_q0_idx = 3 * n_atoms
    bond_q1_idx = 3 * n_atoms + 1
    ancilla_idx = 3 * n_atoms + 2

    # ================================================================
    # Atom 0: 3-Layer HEA (15 params)
    # ================================================================
    # Layer 1: RY + RZ
    ry(thetas[0], q[0])
    ry(thetas[1], q[1])
    ry(thetas[2], q[2])
    rz(thetas[3], q[0])
    rz(thetas[4], q[1])
    rz(thetas[5], q[2])
    # Ring-CNOT
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    x.ctrl(q[2], q[0])
    # Layer 2: RY + RZ
    ry(thetas[6], q[0])
    ry(thetas[7], q[1])
    ry(thetas[8], q[2])
    rz(thetas[9], q[0])
    rz(thetas[10], q[1])
    rz(thetas[11], q[2])
    # [NEW] Layer 3: Linear-CNOT + RZ
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    rz(thetas[12], q[0])
    rz(thetas[13], q[1])
    rz(thetas[14], q[2])

    # ================================================================
    # Atom 1 ~ N-1 + Bond blocks
    # ================================================================
    for i in range(1, n_atoms):
        ap   = 15 * i
        base = 3 * i

        # ---- Atom Block i: 3-Layer HEA ----
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
        # [NEW] Layer 3: Linear-CNOT + RZ
        x.ctrl(q[base],     q[base + 1])
        x.ctrl(q[base + 1], q[base + 2])
        rz(thetas[ap + 12], q[base])
        rz(thetas[ap + 13], q[base + 1])
        rz(thetas[ap + 14], q[base + 2])

        # ---- Bond Block (atom i-1 <-> atom i): 6 params ----
        bp = 15 * n_atoms + 6 * (i - 1)
        # RY + RZ
        ry(thetas[bp],     q[bond_q0_idx])
        ry(thetas[bp + 1], q[bond_q1_idx])
        rz(thetas[bp + 2], q[bond_q0_idx])
        rz(thetas[bp + 3], q[bond_q1_idx])
        # [NEW] Entanglement Layer: CNOT + RZ
        x.ctrl(q[bond_q0_idx], q[bond_q1_idx])
        rz(thetas[bp + 4], q[bond_q0_idx])
        rz(thetas[bp + 5], q[bond_q1_idx])

        # ---- Ancilla OR for atom i-1 ----
        prev_base = 3 * (i - 1)
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])
        x.ctrl(q[prev_base], q[prev_base + 1], q[prev_base + 2],
               q[ancilla_idx])
        x(q[ancilla_idx])
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])
        x.ctrl(q[ancilla_idx], q[bond_q0_idx])
        # Uncompute ancilla
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])
        x(q[ancilla_idx])
        x.ctrl(q[prev_base], q[prev_base + 1], q[prev_base + 2],
               q[ancilla_idx])
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])

        # CZ phase: atom i-1 -> bond
        z.ctrl(q[prev_base],     q[bond_q0_idx])
        z.ctrl(q[prev_base + 1], q[bond_q1_idx])
        z.ctrl(q[prev_base + 2], q[bond_q0_idx])

        # ---- Ancilla OR for atom i ----
        curr_base = 3 * i
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])
        x.ctrl(q[curr_base], q[curr_base + 1], q[curr_base + 2],
               q[ancilla_idx])
        x(q[ancilla_idx])
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])
        x.ctrl(q[ancilla_idx], q[bond_q1_idx])
        # Uncompute ancilla
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])
        x(q[ancilla_idx])
        x.ctrl(q[curr_base], q[curr_base + 1], q[curr_base + 2],
               q[ancilla_idx])
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])

        # CZ phase: atom i -> bond (cross)
        z.ctrl(q[curr_base],     q[bond_q1_idx])
        z.ctrl(q[curr_base + 1], q[bond_q0_idx])
        z.ctrl(q[curr_base + 2], q[bond_q1_idx])

        # Bond internal CNOT
        x.ctrl(q[bond_q0_idx], q[bond_q1_idx])

        # Mid-circuit measurement
        mz(q[bond_q0_idx])
        mz(q[bond_q1_idx])

        # Reset for reuse
        reset(q[bond_q0_idx])
        reset(q[bond_q1_idx])

    # Final atom measurements
    for i in range(3 * n_atoms):
        mz(q[i])


class SQMGKernel:
    """
    SQMG 21N-6 Enhanced Expressivity 量子線路的 Python 封裝 (v5)。
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_qubits = 3 * max_atoms + 3
        # [v5] 15 per atom + 6 per bond = 21N - 6
        self.n_atom_params = 15 * max_atoms
        self.n_bond_params = 6 * (max_atoms - 1)
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
        return 5 * self.max_atoms - 2

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        return lower, upper

    def describe(self) -> str:
        return (
            f"SQMG 3N+3 Ansatz (v5 Enhanced Expressivity)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3x{self.max_atoms} + 3)\n"
            f"  Total parameters  : {self.n_params}  (21x{self.max_atoms} - 6)\n"
            f"  Atom params       : {self.n_atom_params}  (15 per atom, 3-layer HEA)\n"
            f"  Bond params       : {self.n_bond_params}  (6 per bond, RY+RZ+CNOT+RZ)\n"
            f"  Ancilla           : 1  (OR non-NONE detection, uncomputed)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Ansatz per atom   : [RY+RZ] -> Ring-CNOT -> [RY+RZ] -> Linear-CNOT -> [RZ]\n"
            f"  Bond ansatz       : [RY+RZ] -> CNOT -> [RZ]"
        )
