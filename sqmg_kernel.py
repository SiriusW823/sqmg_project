"""
SQMG Kernel — v11 Fully Unrolled, V100/sm_70 compatible
Zero loops, zero arithmetic, zero variable kernel params.
Hardcoded for N=4 (24 qubits, 54 optimizer params).
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit_n4(thetas: list[float]):
    """
    Fully unrolled SQMG circuit for N=4.
    qvector: 24 qubits (q[0..11]=atoms, q[12..23]=bonds)
    thetas: 66 values (pre-expanded by _expand_params)
      [0..35] = atom params (9 per atom)
      [36..65] = bond params (5 per bond, pre-scaled)
    """
    q = cudaq.qvector(24)

    # ── Atom 0 bias ──
    x(q[0])

    # ── Atom HEA blocks (9 params/atom, fully unrolled) ──
    # Atom 0 (qubits 0,1,2)
    ry(thetas[0],   q[0])
    ry(thetas[1], q[1])
    ry(thetas[2], q[2])
    rz(thetas[3], q[0])
    rz(thetas[4], q[1])
    rz(thetas[5], q[2])
    x.ctrl(q[0],   q[1])
    x.ctrl(q[1], q[2])
    x.ctrl(q[2], q[0])
    ry(thetas[6], q[0])
    ry(thetas[7], q[1])
    ry(thetas[8], q[2])

    # Atom 1 (qubits 3,4,5)
    ry(thetas[9],   q[3])
    ry(thetas[10], q[4])
    ry(thetas[11], q[5])
    rz(thetas[12], q[3])
    rz(thetas[13], q[4])
    rz(thetas[14], q[5])
    x.ctrl(q[3],   q[4])
    x.ctrl(q[4], q[5])
    x.ctrl(q[5], q[3])
    ry(thetas[15], q[3])
    ry(thetas[16], q[4])
    ry(thetas[17], q[5])

    # Atom 2 (qubits 6,7,8)
    ry(thetas[18],   q[6])
    ry(thetas[19], q[7])
    ry(thetas[20], q[8])
    rz(thetas[21], q[6])
    rz(thetas[22], q[7])
    rz(thetas[23], q[8])
    x.ctrl(q[6],   q[7])
    x.ctrl(q[7], q[8])
    x.ctrl(q[8], q[6])
    ry(thetas[24], q[6])
    ry(thetas[25], q[7])
    ry(thetas[26], q[8])

    # Atom 3 (qubits 9,10,11)
    ry(thetas[27],   q[9])
    ry(thetas[28], q[10])
    ry(thetas[29], q[11])
    rz(thetas[30], q[9])
    rz(thetas[31], q[10])
    rz(thetas[32], q[11])
    x.ctrl(q[9],   q[10])
    x.ctrl(q[10], q[11])
    x.ctrl(q[11], q[9])
    ry(thetas[33], q[9])
    ry(thetas[34], q[10])
    ry(thetas[35], q[11])

    # ── Bond blocks (5 pre-scaled params/bond, fully unrolled) ──
    # CRY(θ, ctrl, tgt) = ry(θ/2,tgt) cx(ctrl,tgt) ry(-θ/2,tgt) cx(ctrl,tgt)
    # Bond (0,1) → qubits 12,13, thetas[36..40]
    # Gate 1: RY(θ₁) — bond existence
    ry(thetas[36], q[12])
    # Gate 2: CRY(θ₂, ctrl=12, tgt=13) — single→double
    ry(thetas[37], q[13])
    x.ctrl(q[12], q[13])
    ry(thetas[38], q[13])
    x.ctrl(q[12], q[13])
    # Gate 3: CRY(θ₃, ctrl=13, tgt=12) — double→triple
    ry(thetas[39], q[12])
    x.ctrl(q[13], q[12])
    ry(thetas[40], q[12])
    x.ctrl(q[13], q[12])

    # Bond (0,2) → qubits 14,15, thetas[41..45]
    # Gate 1: RY(θ₁) — bond existence
    ry(thetas[41], q[14])
    # Gate 2: CRY(θ₂, ctrl=14, tgt=15) — single→double
    ry(thetas[42], q[15])
    x.ctrl(q[14], q[15])
    ry(thetas[43], q[15])
    x.ctrl(q[14], q[15])
    # Gate 3: CRY(θ₃, ctrl=15, tgt=14) — double→triple
    ry(thetas[44], q[14])
    x.ctrl(q[15], q[14])
    ry(thetas[45], q[14])
    x.ctrl(q[15], q[14])

    # Bond (0,3) → qubits 16,17, thetas[46..50]
    # Gate 1: RY(θ₁) — bond existence
    ry(thetas[46], q[16])
    # Gate 2: CRY(θ₂, ctrl=16, tgt=17) — single→double
    ry(thetas[47], q[17])
    x.ctrl(q[16], q[17])
    ry(thetas[48], q[17])
    x.ctrl(q[16], q[17])
    # Gate 3: CRY(θ₃, ctrl=17, tgt=16) — double→triple
    ry(thetas[49], q[16])
    x.ctrl(q[17], q[16])
    ry(thetas[50], q[16])
    x.ctrl(q[17], q[16])

    # Bond (1,2) → qubits 18,19, thetas[51..55]
    # Gate 1: RY(θ₁) — bond existence
    ry(thetas[51], q[18])
    # Gate 2: CRY(θ₂, ctrl=18, tgt=19) — single→double
    ry(thetas[52], q[19])
    x.ctrl(q[18], q[19])
    ry(thetas[53], q[19])
    x.ctrl(q[18], q[19])
    # Gate 3: CRY(θ₃, ctrl=19, tgt=18) — double→triple
    ry(thetas[54], q[18])
    x.ctrl(q[19], q[18])
    ry(thetas[55], q[18])
    x.ctrl(q[19], q[18])

    # Bond (1,3) → qubits 20,21, thetas[56..60]
    # Gate 1: RY(θ₁) — bond existence
    ry(thetas[56], q[20])
    # Gate 2: CRY(θ₂, ctrl=20, tgt=21) — single→double
    ry(thetas[57], q[21])
    x.ctrl(q[20], q[21])
    ry(thetas[58], q[21])
    x.ctrl(q[20], q[21])
    # Gate 3: CRY(θ₃, ctrl=21, tgt=20) — double→triple
    ry(thetas[59], q[20])
    x.ctrl(q[21], q[20])
    ry(thetas[60], q[20])
    x.ctrl(q[21], q[20])

    # Bond (2,3) → qubits 22,23, thetas[61..65]
    # Gate 1: RY(θ₁) — bond existence
    ry(thetas[61], q[22])
    # Gate 2: CRY(θ₂, ctrl=22, tgt=23) — single→double
    ry(thetas[62], q[23])
    x.ctrl(q[22], q[23])
    ry(thetas[63], q[23])
    x.ctrl(q[22], q[23])
    # Gate 3: CRY(θ₃, ctrl=23, tgt=22) — double→triple
    ry(thetas[64], q[22])
    x.ctrl(q[23], q[22])
    ry(thetas[65], q[22])
    x.ctrl(q[23], q[22])


class SQMGKernel:
    """
    SQMG v11 — fully unrolled for N=4, V100/sm_70 compatible.
    Optimizer interface identical to v7 (3-param bond format).
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        if max_atoms != 4:
            raise ValueError(f"v11 kernel only supports N=4, got N={max_atoms}")
        self.max_atoms = 4
        self.shots = shots
        self.n_bonds = 6
        self.n_qubits = 24
        self.n_atom_params = 36
        self.n_bond_params = 18
        self.n_params = 54  # optimizer interface
        self._n_expanded = 66  # actual kernel input size

    def _expand_params(self, params: np.ndarray) -> list:
        """3-param bond → 5-param expanded (θ₂/2, -θ₂/2, θ₃/2, -θ₃/2 pre-computed)."""
        out = params[:self.n_atom_params].tolist()
        b = params[36:]
        for i in range(6):
            t1 = float(b[3*i])
            t2 = float(b[3*i+1])
            t3 = float(b[3*i+2])
            out += [t1, t2/2, -t2/2, t3/2, -t3/2]
        return out

    def sample(self, params: np.ndarray) -> Dict[str, int]:
        expanded = self._expand_params(params)
        results = cudaq.sample(sqmg_circuit_n4, expanded, shots_count=self.shots)
        counts: Dict[str, int] = {}
        for bs in results:
            clean = bs.replace(" ", "")
            counts[clean] = results.count(bs)
        return counts

    def get_expected_bitstring_length(self) -> int:
        return 24

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(54, -np.pi)
        upper = np.full(54,  np.pi)
        s = 36
        lower[s::3] = 0.0;   upper[s::3] = np.pi / 2
        lower[s+1::3] = 0.0; upper[s+1::3] = np.pi / 2
        lower[s+2::3] = 0.0; upper[s+2::3] = np.pi / 2
        return lower, upper

    def describe(self) -> str:
        return (
            "SQMG v11 (fully unrolled, V100/sm_70 zero-arithmetic)\n"
            "  Max atoms (N)    : 4 (hardcoded)\n"
            "  Total qubits     : 24\n"
            "  Optimizer params : 54\n"
            "  Kernel inputs    : 66 (pre-expanded)\n"
            f"  Shots            : {self.shots}"
        )
