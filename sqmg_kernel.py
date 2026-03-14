"""
==============================================================================
SQMG Kernel — 3N+2 CUDA-Q 參數化量子線路 (v7 Dynamic Circuit)
==============================================================================

架構概覽 (Scalable Quantum Molecular Generation, v7, 3-gate bond subcircuit)
──────────────────────────────────────────────────────────────────────────────
• 原子暫存器 (Atom Register)：每個重原子靜態分配 3 顆量子位元，
  1 層 HEA（RY×3+RZ×3+Ring-CNOT+RY×3），共 9 個參數。
• 無 Cross-atom coupling（暫未包含）。
• 鍵暫存器 (Bond Register)：2 顆量子位元，每個鍵使用 3 個參數
  （RY + Ctrl-RY + Ctrl-RY），全上三角 N(N-1)/2 bonds。
• 動態電路 (Dynamic Circuit)：原子先測量，Bond 條件執行。
• Atom 0 硬約束：X 閘強制非 NONE。
• 無 Ancilla。
• 總量子位元數 = 3N + 2。
• 總參數量   = 9N + 3·N(N-1)/2。
• Bit-string 長度 = 3N + N(N-1) = N² + 2N（不變）。
• 測量順序：原子位元 (3N) 先，鍵結位元 (N(N-1)) 後。

Bond 子電路設計 (3-gate, 4 reachable states)
─────────────────────────────────────────────
  Gate 1 (綠): RY(θ₁) on q_bond[0]              → |00⟩ ↔ |10⟩  (bond existence)
  Gate 2 (藍): CRY(θ₂) on q_bond[1], ctrl q_bond[0]  → |10⟩ ↔ |11⟩  (single→double)
  Gate 3 (橘): CRY(θ₃) on q_bond[0], ctrl q_bond[1]  → |11⟩ ↔ |01⟩  (double→triple)

  可達測量結果：
    |00⟩ = 無鍵 (NONE)
    |10⟩ = 單鍵 (SINGLE)
    |11⟩ = 雙鍵 (DOUBLE)
    |01⟩ = 三鍵 (TRIPLE)  ← v7 新增，v6 不可達

  參數邊界 (Hierarchical Bounds, 對應 QMG Eq.2-3)：
    θ₁ (bond_existence)    : [0, π/2]  → P(bond) ≤ 50%
    θ₂ (bond_order)        : [0, π/2]  → P(double|bond) ≤ 50%
    θ₃ (bond_triple_order) : [0, π/2]  → P(triple|double) ≤ 50%
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int):
    """
    SQMG 3N+2 參數化量子線路 (v7 Dynamic Circuit, 3-gate bond subcircuit)。

    參數佈局 (Parameter Layout)
    ──────────────────────────
    Atom Register: 9 params/atom
        Atom i: thetas[9i : 9i+9]
            Layer 1 RY: thetas[9i+0..2], RZ: thetas[9i+3..5]
            Ring-CNOT
            Layer 2 RY: thetas[9i+6..8]

    Bond Register: 3 params/bond (全上三角，N(N-1)/2 bonds)
        Bond pair bp_idx 按上三角順序排列
        thetas[9N + 3*bp_idx : +3]
            Param 1 (RY on q_bond[0])            : 控制鍵結是否存在 (|00⟩→|10⟩)
            Param 2 (CRY on q_bond[1], ctrl q[0]): 控制鍵結階數 single→double (|10⟩→|11⟩)
            Param 3 (CRY on q_bond[0], ctrl q[1]): 控制鍵結階數 double→triple (|11⟩→|01⟩)

    總參數量 = 9N + 3·N(N-1)/2
    Bit-string 長度 = 3N + N(N-1) = N² + 2N
    測量順序：先 atom bits（3N 個），後 bond bits（N(N-1) 個）
    """
    q_atoms = cudaq.qvector(3 * n_atoms)
    q_bond = cudaq.qvector(2)

    # ================================================================
    # Atom 0 統計偏置：X 閘降低 NONE 的輸出機率
    # ================================================================
    # 論文："the first atom cannot be labeled as non-existent"
    # 實作說明：在電路前端加 X(q_atoms[0]) 讓 q_atoms[0] 從 |0⟩ 偏移至 |1⟩，
    # 使後續 RY 旋轉從 |1⟩ 出發，統計上降低 atom_codes[0] 輸出 |000⟩ 的機率。
    # 注意：這是統計偏置而非電路層的硬性排除 —— 若 thetas[0] 接近 π，
    #       q_atoms[0] 仍可旋轉回 |0⟩。真正的硬性過濾由 MoleculeDecoder 完成。
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
        # Ring-CNOT
        x.ctrl(q_atoms[3 * i],     q_atoms[3 * i + 1])
        x.ctrl(q_atoms[3 * i + 1], q_atoms[3 * i + 2])
        x.ctrl(q_atoms[3 * i + 2], q_atoms[3 * i])
        # Layer 2: RY×3
        ry(thetas[param_idx + 6], q_atoms[3 * i])
        ry(thetas[param_idx + 7], q_atoms[3 * i + 1])
        ry(thetas[param_idx + 8], q_atoms[3 * i + 2])
        param_idx += 9

    # (已移除無論文對應的 Cross-atom CRY)

    # ================================================================
    # 測量所有原子，準備條件執行 (Dynamic Circuit)
    # ================================================================
    # CUDA-Q @kernel 限制說明：
    #   ✗ [False] * n_atoms   — n_atoms 為 runtime 參數，JIT 無法靜態分配
    #   ✗ atom_exists[i] = v  — @kernel 內不支援 list 的 index 賦值
    #   ✓ cudaq.quake_value    — 可累積 mz() 的 bool 結果
    #
    # 解決策略：直接測量所有原子 bits，僅保留各原子的「是否存在」bool，
    # 並儲存至 cudaq 可接受的 list[bool] 初始化形式（用 append 而非 index 賦值）。
    atom_exists: list[bool] = []
    for i in range(n_atoms):
        m0 = mz(q_atoms[3 * i])
        m1 = mz(q_atoms[3 * i + 1])
        m2 = mz(q_atoms[3 * i + 2])
        atom_exists.append(m0 or m1 or m2)

    # ================================================================
    # Bond Blocks: 全上三角 N(N-1)/2 bonds (Dynamic Circuit)
    # 順序：(0,1),(0,2),...,(0,N-1),(1,2),...,(N-2,N-1)
    #
    # 條件執行 (Conditional Bond Execution)：
    # 只有當兩端原子都存在時，才激活 Bond 模組，淨化機率景觀。
    #
    # Bond 子電路物理語義 (3-gate, 4 reachable states)：
    #   Gate 1 (RY on q_bond[0])              : |00⟩ ↔ |10⟩  bond existence
    #   Gate 2 (CRY on q_bond[1], ctrl q[0])  : |10⟩ ↔ |11⟩  single→double
    #   Gate 3 (CRY on q_bond[0], ctrl q[1])  : |11⟩ ↔ |01⟩  double→triple
    #
    #   可達態：|00⟩=無鍵, |10⟩=單鍵, |11⟩=雙鍵, |01⟩=三鍵
    # ================================================================
    bp = param_idx
    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):

            # 條件執行：兩端原子均存在時才激活 Bond
            if atom_exists[atom_i] and atom_exists[atom_j]:
                ry(thetas[bp],     q_bond[0])                      # gate 1: bond existence
                ry.ctrl(thetas[bp + 1], q_bond[0], q_bond[1])      # gate 2: single→double
                ry.ctrl(thetas[bp + 2], q_bond[1], q_bond[0])      # gate 3: double→triple

            # 測量與重置 (Bond reuse)
            mz(q_bond[0])
            mz(q_bond[1])
            reset(q_bond[0])
            reset(q_bond[1])
            bp += 3


class SQMGKernel:
    """
    SQMG 3N+2 Full Upper-Triangular 量子線路的 Python 封裝 (v7)。

    參數量公式：9N + 3·N(N-1)/2
    Bitstring 長度：N² + 2N（不變，bond bits 仍為 2 per bond）
    全上三角矩陣 N(N-1)/2 bonds，可表達分支與環狀分子結構。
    可達鍵結類型：|00⟩=無鍵, |10⟩=單鍵, |11⟩=雙鍵, |01⟩=三鍵 (3-gate bond subcircuit)。
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_qubits = 3 * max_atoms + 2
        self.kernel_func = sqmg_circuit
        self.n_bonds = max_atoms * (max_atoms - 1) // 2
        self.n_atom_params = 9 * max_atoms
        self.n_bond_params = 3 * self.n_bonds          # v7: 3 params/bond (was 2)
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
        """
        獲取量子閘參數的搜尋邊界。
        【實作化學先驗限制 + 階層邊界 (Hierarchical Bounds)】

        Bond 參數每 3 個一組，對應 QMG Eq.2-3：
          Param 0 (bond_existence)    : [0, π/2]  → P(bond exists) ≤ 50%
          Param 1 (bond_order)        : [0, π/2]  → P(double|bond) ≤ 50%
          Param 2 (bond_triple_order) : [0, π/2]  → P(triple|double) ≤ 50%
        """
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)

        bond_start_idx = self.n_atom_params

        # Param 0 per bond: bond_existence (|00⟩→|10⟩), [0, π/2]
        lower[bond_start_idx::3] = 0.0
        upper[bond_start_idx::3] = np.pi / 2

        # Param 1 per bond: bond_order single→double (|10⟩→|11⟩), [0, π/2]
        # 對應 QMG Eq.2：使雙鍵機率 ≤ 50%
        lower[bond_start_idx + 1::3] = 0.0
        upper[bond_start_idx + 1::3] = np.pi / 2

        # Param 2 per bond: bond_triple_order double→triple (|11⟩→|01⟩), [0, π/2]
        # 對應 QMG Eq.3 精神：使三鍵機率 ≤ 50%（比雙鍵更少見）
        lower[bond_start_idx + 2::3] = 0.0
        upper[bond_start_idx + 2::3] = np.pi / 2

        return lower, upper

    def describe(self) -> str:
        return (
            f"SQMG 3N+2 Ansatz (v7 Dynamic Circuit, 3-gate bond subcircuit)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3×{self.max_atoms} + 2)\n"
            f"  Total parameters  : {self.n_params}  (9×{self.max_atoms} + 3×{self.n_bonds})\n"
            f"  Atom params       : {self.n_atom_params}  (9 per atom, 1-layer HEA)\n"
            f"  Bond params       : {self.n_bond_params}  (3 per bond: RY + 2×Ctrl-RY)\n"
            f"  Bond pairs        : {self.n_bonds}  (upper-triangular N(N-1)/2)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}  (N²+2N, unchanged)\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Atom ansatz       : [RY+RZ] -> Ring-CNOT -> [RY]\n"
            f"  Atom 0 bias       : X gate (statistical bias toward non-NONE; hard filter in Decoder)\n"
            f"  Bond ansatz       : RY(exist) + Ctrl-RY(single→double) + Ctrl-RY(double→triple)\n"
            f"  Bond states       : |00⟩=無鍵, |10⟩=單鍵, |11⟩=雙鍵, |01⟩=三鍵  (4 reachable)\n"
            f"  Bond execution    : Conditional (dynamic circuit)"
        )
