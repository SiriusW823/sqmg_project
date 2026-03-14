"""
==============================================================================
SQMG Kernel — 3N+2 CUDA-Q 參數化量子線路 (v13 closure-fixed, qpp-cpu compatible)
==============================================================================

v13 相對 v12 的關鍵修改：
  ★ 根本原因修復 ★
  v12 將 n_atom_qubits / n_bonds 作為 kernel 函式「參數」傳入，導致 CUDA-Q JIT
  在 MLIR 層試圖以 runtime i64 值計算 cc.stdvec<i1> 的大小，觸發：
    RuntimeError: Invalid type for Binary Op <class 'ast.Mult'>
    (cc.stdvec_init(..., i64), cc.load(i64))

  修正策略：Closure Pattern
  - 使用工廠函式 `build_sqmg_kernel(max_atoms)` 讓所有尺寸常數
    在 Python 層（JIT 定義時）固化為字面值（compile-time constants）。
  - @cudaq.kernel 函式只保留唯一 runtime 參數：`thetas: list[float]`。
  - cudaq.qvector(n_atom_qubits) 中的 n_atom_qubits 此時是 closure 常數
    （Python int），MLIR JIT 可靜態配置量子暫存器，測量計數也因此確定。

  其他細節修改：
  - Bond section 改用獨立計數器 pb = n_atom_params（closure 常數），
    避免 p 在 atom loop 後的累積值被誤判為 runtime 變數。
  - range(max_atoms) 與 range(n_bonds) 均為 closure 常數，loop 可被 JIT 展開。
  - SQMGKernel.sample() 只傳入 thetas.tolist()，不再傳 n_atom_qubits / n_bonds。

所有其他功能（bond reuse、測量順序、3-gate bond 電路、HEA 結構）完全不變。
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple


# ===========================================================================
# 工廠函式：以 closure 固化所有尺寸常數，回傳 @cudaq.kernel 函式物件
# ===========================================================================

def build_sqmg_kernel(max_atoms: int):
    """
    產生針對特定 max_atoms 編譯的 CUDA-Q kernel 函式。

    所有尺寸常數（n_atom_qubits, n_bonds, n_atom_params）
    在此函式呼叫時即固化為 Python int，對 @cudaq.kernel JIT 而言是
    compile-time 字面值，不再從函式參數讀取。

    Args:
        max_atoms: 最大重原子數量 N（編譯期常數）。

    Returns:
        @cudaq.kernel 修飾的 sqmg_circuit 函式，只接受 thetas: list[float]。
    """
    # ── 編譯期常數（全部在 Python 層算好） ──────────────────────────────
    n_atom_qubits: int = 3 * max_atoms           # = 3N，cudaq.qvector 的靜態大小
    n_bonds:       int = max_atoms * (max_atoms - 1) // 2   # N*(N-1)/2
    n_atom_params: int = 9 * max_atoms           # HEA 參數量 = 9N
    # ──────────────────────────────────────────────────────────────────────

    @cudaq.kernel
    def sqmg_circuit(thetas: list[float]):
        """
        SQMG 3N+2 量子線路 (v13, closure 固化所有尺寸常數)。

        線路拓樸：
          Atom Register  : n_atom_qubits (= 3*N) qubits，靜態分配，不重複使用
          Bond Register  : 2 qubits，動態 bond reuse (mz + reset)

        測量順序：atom bits (3N) 先，bond bits (2 * n_bonds) 後。
        Bitstring 長度  : 3N + 2*N*(N-1)/2 = N² + 2N。

        closure 變數（均為 Python int，JIT 可見為 compile-time constants）：
          n_atom_qubits, max_atoms, n_bonds, n_atom_params
        """
        # ── 量子暫存器（大小為 closure 常數，JIT 靜態配置） ────────────
        q_atoms = cudaq.qvector(n_atom_qubits)   # 3N qubits，compile-time
        q_bond  = cudaq.qvector(2)               # Bond Reuse 暫存器

        # ── Atom 0 非 NONE 偏置：X gate 讓 q_atoms[0] 起點為 |1⟩ ───────
        # 配合 lb_vec[0] = 0 的 optimizer 約束，降低第一個原子為 NONE 的機率。
        x(q_atoms[0])

        # ── Atom HEA：9 params/atom，1 層 HEA（RY×3 + RZ×3 + Ring-CNOT + RY×3）
        # range(max_atoms)：max_atoms 是 closure 常數，JIT 可展開此迴圈。
        # p 和 aq 為純加法累加（無乘法），JIT unroll 後所有索引均為常數。
        p  = 0
        aq = 0
        for i in range(max_atoms):
            ry(thetas[p],     q_atoms[aq])
            ry(thetas[p + 1], q_atoms[aq + 1])
            ry(thetas[p + 2], q_atoms[aq + 2])
            rz(thetas[p + 3], q_atoms[aq])
            rz(thetas[p + 4], q_atoms[aq + 1])
            rz(thetas[p + 5], q_atoms[aq + 2])
            # Ring-CNOT 糾纏層
            x.ctrl(q_atoms[aq],     q_atoms[aq + 1])
            x.ctrl(q_atoms[aq + 1], q_atoms[aq + 2])
            x.ctrl(q_atoms[aq + 2], q_atoms[aq])
            # 後置 RY 旋轉
            ry(thetas[p + 6], q_atoms[aq])
            ry(thetas[p + 7], q_atoms[aq + 1])
            ry(thetas[p + 8], q_atoms[aq + 2])
            p  = p  + 9
            aq = aq + 3

        # ── 原子測量（atom bits 先進入 bitstring）──────────────────────
        # 測量後 q_atoms 塌縮；後續 Bond 電路不再讀取 q_atoms。
        aq = 0
        for i in range(max_atoms):
            mz(q_atoms[aq])
            mz(q_atoms[aq + 1])
            mz(q_atoms[aq + 2])
            aq = aq + 3

        # ── Bond Blocks：無條件執行，bond reuse（mz + reset）────────────
        # pb 以 n_atom_params（closure 常數 = 9N）為起點，只做加法。
        # range(n_bonds)：n_bonds 是 closure 常數，JIT 可展開此迴圈。
        # NONE 原子的 bond 由 MoleculeDecoder 負責過濾，電路層無條件執行。
        pb = n_atom_params
        for b in range(n_bonds):
            # Gate 1: RY(θ₁)                     → |00⟩ ↔ |10⟩  bond existence
            ry(thetas[pb],     q_bond[0])
            # Gate 2: CRY(θ₂, ctrl=q[0])          → |10⟩ ↔ |11⟩  single → double
            ry.ctrl(thetas[pb + 1], q_bond[0], q_bond[1])
            # Gate 3: CRY(θ₃, ctrl=q[1])          → |11⟩ ↔ |01⟩  double → triple
            ry.ctrl(thetas[pb + 2], q_bond[1], q_bond[0])
            mz(q_bond[0])
            mz(q_bond[1])
            reset(q_bond[0])
            reset(q_bond[1])
            pb = pb + 3

    return sqmg_circuit


# ===========================================================================
# SQMGKernel：封裝類別，對外介面與 v12 完全相同
# ===========================================================================

class SQMGKernel:
    """
    SQMG v13 封裝。

    使用 build_sqmg_kernel(max_atoms) 建立 closure-fixed kernel。
    sample() 只傳入 thetas（單一 runtime 參數），不再傳尺寸常數。

    對外介面：
      n_params, get_param_bounds(), sample(), describe()
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        self.max_atoms     = max_atoms
        self.shots         = shots
        self.n_bonds       = max_atoms * (max_atoms - 1) // 2
        self.n_atom_qubits = 3 * max_atoms
        self.n_qubits      = self.n_atom_qubits + 2          # 3N+2
        self.n_atom_params = 9 * max_atoms
        self.n_bond_params = 3 * self.n_bonds
        self.n_params      = self.n_atom_params + self.n_bond_params

        # ★ 核心修改：以 closure pattern 建立 kernel，消除 runtime 乘法 ★
        self._kernel = build_sqmg_kernel(max_atoms)

    def sample(self, params: np.ndarray) -> Dict[str, int]:
        """
        執行量子線路取樣，回傳 {bitstring: count}。

        v13 變更：只傳 thetas（list[float]），不再傳 n_atom_qubits / n_bonds，
        因為所有尺寸已由 closure 固化在 self._kernel 中。
        """
        results = cudaq.sample(
            self._kernel,
            params.tolist(),      # 唯一 runtime 參數
            shots_count=self.shots,
        )
        counts: Dict[str, int] = {}
        for bitstring in results:
            clean_bs = bitstring.replace(" ", "")
            counts[clean_bs] = results.count(bitstring)
        return counts

    def get_expected_bitstring_length(self) -> int:
        """Bitstring 長度 = 3N + 2*N*(N-1)/2 = N² + 2N"""
        return self.n_atom_qubits + 2 * self.n_bonds

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        回傳參數邊界向量（帶化學先驗約束）。

        Atom params  : [-π, π]（HEA 自由旋轉）
        Bond params  : [0, π/2]（Hierarchical bounds，P(bond) ≤ 50%）
          - bond_existence    : [0, π/2]  → P(|10⟩) ≤ 50%
          - bond_order        : [0, π/2]  → P(double|bond) ≤ 50%
          - bond_triple_order : [0, π/2]  → P(triple|double) ≤ 50%
        """
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        s = self.n_atom_params
        lower[s::3] = 0.0;    upper[s::3] = np.pi / 2    # bond_existence
        lower[s+1::3] = 0.0;  upper[s+1::3] = np.pi / 2  # bond_order
        lower[s+2::3] = 0.0;  upper[s+2::3] = np.pi / 2  # bond_triple_order
        return lower, upper

    def describe(self) -> str:
        bs_len = self.get_expected_bitstring_length()
        return (
            f"SQMG 3N+2 Ansatz (v13, closure-fixed, qpp-cpu compatible)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3×{self.max_atoms} + 2)\n"
            f"  Total parameters  : {self.n_params}  (9×{self.max_atoms} + 3×{self.n_bonds})\n"
            f"  Atom params       : {self.n_atom_params}  (9 per atom, 1-layer HEA)\n"
            f"  Bond params       : {self.n_bond_params}  (3 per bond: RY + 2×Ctrl-RY)\n"
            f"  Bond pairs        : {self.n_bonds}  (upper-triangular N(N-1)/2)\n"
            f"  Bitstring length  : {bs_len}  (N²+2N = {bs_len})\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Atom ansatz       : [RY+RZ] -> Ring-CNOT -> [RY]\n"
            f"  Atom 0 bias       : X gate (non-NONE bias; hard filter in Decoder)\n"
            f"  Bond ansatz       : RY(exist) + Ctrl-RY(single→double) + Ctrl-RY(double→triple)\n"
            f"  Bond states       : |00⟩=無鍵, |10⟩=單鍵, |11⟩=雙鍵, |01⟩=三鍵  (4 reachable)\n"
            f"  Bond execution    : Unconditional + bond reuse (mz+reset)\n"
            f"  JIT fix (v13)     : Closure pattern — all size constants are Python ints\n"
            f"                      at kernel definition time, no runtime Mult ops\n"
            f"  Runtime arithmetic: None (qvector sizes, loop bounds, pb start are constants)"
        )