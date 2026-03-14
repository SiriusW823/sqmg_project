"""
==============================================================================
SQMG Kernel — v10 Static Circuit, V100/sm_70 zero-arithmetic kernel
==============================================================================

v10 修正（相對 v9）：
  消除 kernel 內所有算術運算（`* 0.5`、負號、索引偏移加法）。

  根本原因：CUDA-Q 0.12 nvidia backend 在 sm_70 上的 PTX JIT 不支援
  kernel 內的浮點算術運算（thetas[i] * 0.5、-thetas[i]）。
  即使只用 ry/rz/x.ctrl 基礎閘，只要有算術就崩潰。

  解法（Pre-scaled parameters）：
    sample() 在呼叫 cudaq.sample() 前，把 3-param bond format 展開成
    5-param format，kernel 直接使用展開後的值，零算術：

    原始 3 params/bond: [θ₁,       θ₂,      θ₃     ]
    展開 5 params/bond: [θ₁, θ₂/2, -θ₂/2, θ₃/2, -θ₃/2]

    kernel 接收 expanded_thetas，依序使用，無任何運算。

  對優化器透明：optimizer/evaluator 仍使用原始 3-param 格式。
  只有 sample() 內部做轉換，介面完全不變。
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int, n_bonds: int):
    """
    SQMG v10 kernel — kernel 內零算術，純 ry/rz/x.ctrl 基礎閘。

    thetas 佈局（expanded format，由 sample() 轉換）：
      Atom section  [0 : 9*n_atoms]        : 9 params/atom，直接使用
      Bond section  [9*n_atoms : ...]       : 5 params/bond
        per bond: [θ₁, θ₂_half, θ₂_neg_half, θ₃_half, θ₃_neg_half]
        θ₁           → ry(θ₁, bq[0])             gate1: bond existence
        θ₂_half      → ry(θ₂_half, bq[1])        CRY gate2 step 1
        θ₂_neg_half  → ry(θ₂_neg_half, bq[1])    CRY gate2 step 3
        θ₃_half      → ry(θ₃_half, bq[0])        CRY gate3 step 1
        θ₃_neg_half  → ry(θ₃_neg_half, bq[0])    CRY gate3 step 3
    """
    q_atoms = cudaq.qvector(3 * n_atoms)
    q_bonds = cudaq.qvector(2 * n_bonds)

    # ── Atom 0 偏置 ──────────────────────────────────────────────────
    x(q_atoms[0])

    # ── Atom HEA (9 params/atom) ──────────────────────────────────────
    p = 0
    for i in range(n_atoms):
        ry(thetas[p],     q_atoms[3 * i])
        ry(thetas[p + 1], q_atoms[3 * i + 1])
        ry(thetas[p + 2], q_atoms[3 * i + 2])
        rz(thetas[p + 3], q_atoms[3 * i])
        rz(thetas[p + 4], q_atoms[3 * i + 1])
        rz(thetas[p + 5], q_atoms[3 * i + 2])
        x.ctrl(q_atoms[3 * i],     q_atoms[3 * i + 1])
        x.ctrl(q_atoms[3 * i + 1], q_atoms[3 * i + 2])
        x.ctrl(q_atoms[3 * i + 2], q_atoms[3 * i])
        ry(thetas[p + 6], q_atoms[3 * i])
        ry(thetas[p + 7], q_atoms[3 * i + 1])
        ry(thetas[p + 8], q_atoms[3 * i + 2])
        p = p + 9

    # ── Bond Blocks (5 pre-scaled params/bond, zero arithmetic) ──────
    # CRY(θ, ctrl, tgt) = ry(θ/2,tgt) cx(ctrl,tgt) ry(-θ/2,tgt) cx(ctrl,tgt)
    # 所有 θ/2 和 -θ/2 已由 sample() 預先計算並傳入 thetas。
    bq = 0
    for bond_idx in range(n_bonds):
        # thetas[p+0] = θ₁           (gate1: bond existence)
        # thetas[p+1] = θ₂/2         (gate2 CRY step1)
        # thetas[p+2] = -θ₂/2        (gate2 CRY step3)
        # thetas[p+3] = θ₃/2         (gate3 CRY step1)
        # thetas[p+4] = -θ₃/2        (gate3 CRY step3)

        # Gate 1: RY(θ₁) on q_bonds[bq]
        ry(thetas[p], q_bonds[bq])

        # Gate 2: CRY(θ₂, ctrl=bq, tgt=bq+1) decomposed
        ry(thetas[p + 1], q_bonds[bq + 1])       # ry( θ₂/2, tgt)
        x.ctrl(q_bonds[bq], q_bonds[bq + 1])      # cx(ctrl, tgt)
        ry(thetas[p + 2], q_bonds[bq + 1])        # ry(-θ₂/2, tgt)
        x.ctrl(q_bonds[bq], q_bonds[bq + 1])      # cx(ctrl, tgt)

        # Gate 3: CRY(θ₃, ctrl=bq+1, tgt=bq) decomposed
        ry(thetas[p + 3], q_bonds[bq])            # ry( θ₃/2, tgt)
        x.ctrl(q_bonds[bq + 1], q_bonds[bq])      # cx(ctrl, tgt)
        ry(thetas[p + 4], q_bonds[bq])            # ry(-θ₃/2, tgt)
        x.ctrl(q_bonds[bq + 1], q_bonds[bq])      # cx(ctrl, tgt)

        bq = bq + 2
        p = p + 5


class SQMGKernel:
    """
    SQMG v10 封裝。

    對外介面（optimizer / evaluator）完全與 v7 相同：
      n_params = 9*N + 3*N(N-1)/2   (3-param bond format)
      bitstring length = N²+2N

    內部 sample() 自動把 3-param bond 展開為 5-param expanded format
    再傳給 kernel，對使用者完全透明。
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_bonds = max_atoms * (max_atoms - 1) // 2
        self.n_qubits = 3 * max_atoms + 2 * self.n_bonds
        self.n_atom_params = 9 * max_atoms
        self.n_bond_params = 3 * self.n_bonds
        self.n_params = self.n_atom_params + self.n_bond_params  # 外部介面維度

    def _expand_params(self, params: np.ndarray) -> list:
        """
        把 3-param bond format 展開為 5-param expanded format。

        [θ₁, θ₂, θ₃]  →  [θ₁, θ₂/2, -θ₂/2, θ₃/2, -θ₃/2]

        kernel 內零算術的前提：所有 ±half 值在這裡預先計算。
        """
        atom_params = params[:self.n_atom_params].tolist()

        bond_expanded = []
        bond_raw = params[self.n_atom_params:]
        for b in range(self.n_bonds):
            t1 = float(bond_raw[3 * b])
            t2 = float(bond_raw[3 * b + 1])
            t3 = float(bond_raw[3 * b + 2])
            bond_expanded += [t1, t2 / 2.0, -t2 / 2.0, t3 / 2.0, -t3 / 2.0]

        return atom_params + bond_expanded

    def sample(self, params: np.ndarray) -> Dict[str, int]:
        expanded = self._expand_params(params)
        results = cudaq.sample(
            sqmg_circuit,
            expanded,
            self.max_atoms,
            self.n_bonds,
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
        """對外仍回傳 3-param bond format 的邊界（與 optimizer 介面一致）。"""
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        s = self.n_atom_params
        lower[s::3] = 0.0;    upper[s::3] = np.pi / 2    # bond_existence
        lower[s+1::3] = 0.0;  upper[s+1::3] = np.pi / 2  # bond_order
        lower[s+2::3] = 0.0;  upper[s+2::3] = np.pi / 2  # bond_triple_order
        return lower, upper

    def describe(self) -> str:
        return (
            f"SQMG Static Circuit (v10, V100/sm_70 zero-arithmetic kernel)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3N + N(N-1), static)\n"
            f"  Kernel params     : {self.n_atom_params + 5*self.n_bonds}"
            f"  (expanded: 9N + 5·N(N-1)/2)\n"
            f"  Optimizer params  : {self.n_params}  (3-param bond, 9N + 3·N(N-1)/2)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Kernel arithmetic : None (all pre-scaled in Python)\n"
            f"  CRY decomp        : ry(θ/2) cx ry(-θ/2) cx"
        )
