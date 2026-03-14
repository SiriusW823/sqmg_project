"""
==============================================================================
SQMG Kernel — v9 Static Circuit, V100/sm_70 + nvidia backend compatible
==============================================================================

v9 修正項目（相對 v8）：
  1. 分解 ry.ctrl() → 使用 rx/ry/x.ctrl() 基礎閘組合
     原因：CUDA-Q 0.12 的 ry.ctrl() 參數化受控閘在 sm_70 PTX 編譯時產生
           architecture mismatch，分解為基礎閘可迴避此問題。
  2. n_bonds 改為獨立 kernel 參數（避免 kernel 內做整數除法）
  3. 無 mid-circuit mz，無 reset，無 if 條件（與 v8 相同）

CRY 分解公式（標準 CNOT + RY 等效）：
  CRY(θ, ctrl, target) =
      ry( θ/2, target)
      x.ctrl(ctrl, target)
      ry(-θ/2, target)
      x.ctrl(ctrl, target)
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int, n_bonds: int):
    """
    SQMG v9 靜態量子線路（V100 / sm_70 相容）。

    參數佈局（與 v7/v8 完全相同）：
      Atom i : thetas[9i : 9i+9]
      Bond bp: thetas[9*n_atoms + 3*bp : +3]
    總參數量 = 9*n_atoms + 3*n_bonds

    Qubit 佈局：
      q_atoms[0 .. 3*n_atoms-1]  : atom qubits
      q_bonds[0 .. 2*n_bonds-1]  : bond qubits（靜態分配，每 bond 2 qubits）
    """
    q_atoms = cudaq.qvector(3 * n_atoms)
    q_bonds = cudaq.qvector(2 * n_bonds)

    # ── Atom 0 偏置 ──────────────────────────────────────────────────
    x(q_atoms[0])

    # ── Atom HEA Blocks (9 params/atom) ──────────────────────────────
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

    # ── Bond Blocks (3 params/bond, CRY 分解為基礎閘) ────────────────
    # CRY(θ, ctrl, tgt) = ry(θ/2,tgt) cx(ctrl,tgt) ry(-θ/2,tgt) cx(ctrl,tgt)
    #
    # Gate 1: RY(θ₁) on q_bonds[bq]       → |00⟩↔|10⟩  bond existence
    # Gate 2: CRY(θ₂, q_bonds[bq], q_bonds[bq+1]) → |10⟩↔|11⟩ single→double
    # Gate 3: CRY(θ₃, q_bonds[bq+1], q_bonds[bq]) → |11⟩↔|01⟩ double→triple
    bp = param_idx
    bq = 0
    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):
            # Gate 1: RY(θ₁) — bond existence
            ry(thetas[bp], q_bonds[bq])

            # Gate 2: CRY(θ₂, ctrl=q_bonds[bq], tgt=q_bonds[bq+1])
            ry( thetas[bp + 1] * 0.5,  q_bonds[bq + 1])
            x.ctrl(q_bonds[bq],        q_bonds[bq + 1])
            ry(-thetas[bp + 1] * 0.5,  q_bonds[bq + 1])
            x.ctrl(q_bonds[bq],        q_bonds[bq + 1])

            # Gate 3: CRY(θ₃, ctrl=q_bonds[bq+1], tgt=q_bonds[bq])
            ry( thetas[bp + 2] * 0.5,  q_bonds[bq])
            x.ctrl(q_bonds[bq + 1],    q_bonds[bq])
            ry(-thetas[bp + 2] * 0.5,  q_bonds[bq])
            x.ctrl(q_bonds[bq + 1],    q_bonds[bq])

            bq += 2
            bp += 3
    # cudaq.sample() 自動測量所有 qubits（q_atoms 先，q_bonds 後）


class SQMGKernel:
    """SQMG v9 封裝，V100/sm_70 + nvidia backend 相容。"""

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_bonds = max_atoms * (max_atoms - 1) // 2
        self.n_qubits = 3 * max_atoms + 2 * self.n_bonds
        self.n_atom_params = 9 * max_atoms
        self.n_bond_params = 3 * self.n_bonds
        self.n_params = self.n_atom_params + self.n_bond_params

    def sample(self, params: np.ndarray) -> Dict[str, int]:
        results = cudaq.sample(
            sqmg_circuit,
            params.tolist(),
            self.max_atoms,
            self.n_bonds,          # v9: 新增 n_bonds 參數
            shots_count=self.shots,
        )
        counts: Dict[str, int] = {}
        for bitstring in results:
            clean_bs = bitstring.replace(" ", "")
            counts[clean_bs] = results.count(bitstring)
        return counts

    def get_expected_bitstring_length(self) -> int:
        return 3 * self.max_atoms + 2 * self.n_bonds

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        s = self.n_atom_params
        lower[s::3] = 0.0;      upper[s::3] = np.pi / 2
        lower[s+1::3] = 0.0;    upper[s+1::3] = np.pi / 2
        lower[s+2::3] = 0.0;    upper[s+2::3] = np.pi / 2
        return lower, upper

    def describe(self) -> str:
        return (
            f"SQMG Static Circuit (v9, V100/sm_70 + nvidia compatible)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3N + N(N-1), static alloc)\n"
            f"  Total parameters  : {self.n_params}  (9N + 3·N(N-1)/2)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Dynamic ops       : None\n"
            f"  CRY decomposition : ry + cx + ry + cx (basic gates only)"
        )
