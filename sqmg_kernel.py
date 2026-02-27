"""
==============================================================================
SQMG Kernel — 3N+2 CUDA-Q 參數化量子線路
==============================================================================

架構概覽 (Scalable Quantum Molecular Generation, 3N+2 Ansatz)
─────────────────────────────────────────────────────────────
• 原子暫存器 (Atom Register)：每個重原子靜態分配 3 顆量子位元，
  可表達 2^3 = 8 種狀態（含 NONE 終止符）。
• 鍵暫存器 (Bond Register)：僅使用 2 顆量子位元，透過「Mid-circuit
  Measurement + Reset」在每個原子建構步驟間動態重複使用。
  →  2 顆量子位元可表達 4 種鍵結類型 (None / Single / Double / Triple)。
• 總量子位元數 = 3N + 2（N = 最大重原子數量）。

與 QMG 原始論文（PDF）的差異
──────────────────────────
1. 框架從 Qiskit / Cirq 全面改為 NVIDIA CUDA-Q，
   以利用 GPU 加速的 Tensor Network 模擬。
2. 採用「Atom no reuse, Bond reuse」策略，維持線性量子位元擴展。
3. 以 QPSO（量子粒子群演算法）取代傳統貝葉斯優化。

CUDA-Q 防範資訊遺失重點
───────────────────────
CUDA-Q 的編譯器會積極優化圖形（尤其在 Tensor Network 後端），
可能「吞掉」未顯式追蹤的中途測量結果。因此本 Kernel 務必：
  ✓ 每個 bond 的中途測量都使用顯式 mz()
  ✓ 所有測量結果都留在量子程式的 classical register 中
  ✓ 使用 cudaq.sample() 收集完整 bit-string（絕不使用 observe()）
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple

# ============================================================================
# CUDA-Q Kernel（Module-level 定義）
# ============================================================================
# CUDA-Q 的 @cudaq.kernel 裝飾器會將 Python 函式 JIT 編譯為
# 量子程式（MLIR → QIR → 後端原生碼）。
# 限制：Kernel 內部不能使用任意 Python 物件 / Numpy / Class；
#       只能使用基本型別（int, float, bool, list[float]）和 CUDA-Q 內建量子閘。
# ============================================================================


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int):
    """
    SQMG 3N+2 參數化量子線路。

    參數佈局 (Parameter Layout)
    ──────────────────────────
    thetas 的前 6·N 個元素歸屬於 Atom Register：
        Atom i → thetas[6i : 6i+6]  (i = 0 .. N-1)
            ┌ qubit 3i+0 ← RY(thetas[6i+0]), RZ(thetas[6i+1])
            │ qubit 3i+1 ← RY(thetas[6i+2]), RZ(thetas[6i+3])
            └ qubit 3i+2 ← RY(thetas[6i+4]), RZ(thetas[6i+5])

    thetas 的後 4·(N-1) 個元素歸屬於 Bond Register：
        Bond j → thetas[6N + 4j : 6N + 4j + 4]  (j = 0 .. N-2)
            ┌ bond_q0 ← RY(thetas[6N+4j+0]), RZ(thetas[6N+4j+1])
            └ bond_q1 ← RY(thetas[6N+4j+2]), RZ(thetas[6N+4j+3])

    總參數量 = 6N + 4(N-1) = 10N - 4

    Bit-string 輸出格式（measurement order, 左 → 右）
    ───────────────────────────────────────────────────
    [bond_0_b0, bond_0_b1,  ← 第 0 組鍵結測量 (2 bits)
     bond_1_b0, bond_1_b1,  ← 第 1 組鍵結測量 (2 bits)
     ...
     bond_{N-2}_b0, bond_{N-2}_b1,
     atom_0_b0, atom_0_b1, atom_0_b2,  ← 第 0 個原子 (3 bits)
     atom_1_b0, atom_1_b1, atom_1_b2,
     ...
     atom_{N-1}_b0, atom_{N-1}_b1, atom_{N-1}_b2]

    總 bit 數 = 2(N-1) + 3N = 5N - 2
    """

    # ── 量子位元分配 ──────────────────────────────────────────
    n_qubits = 3 * n_atoms + 2
    q = cudaq.qvector(n_qubits)

    # Bond Register 的全域量子位元索引（位於暫存器最後 2 顆）
    bond_q0_idx = 3 * n_atoms       # 鍵暫存器 qubit 0
    bond_q1_idx = 3 * n_atoms + 1   # 鍵暫存器 qubit 1

    # ================================================================
    # Atom 0 — 初始原子參數化
    # ================================================================
    # 使用 RY + RZ 的組合提供完整的 SU(2) 單量子位元旋轉自由度。
    # RY 控制 |0⟩ ↔ |1⟩ 的振幅分配（Population）
    # RZ 控制相位（Phase），驅動量子干涉效果
    # ================================================================
    ry(thetas[0], q[0])
    rz(thetas[1], q[0])
    ry(thetas[2], q[1])
    rz(thetas[3], q[1])
    ry(thetas[4], q[2])
    rz(thetas[5], q[2])

    # 原子內部糾纏 (Intra-atom Entanglement)
    # CNOT 鏈：q[0]→q[1]→q[2]
    # 使得 3 顆量子位元不僅是獨立旋轉，而是能表達具有
    # 量子關聯的 8 態疊加，增加表達力。
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])

    # ================================================================
    # 迭代建構 Atom 1 ~ Atom N-1（含 Bond Block）
    # ================================================================
    for i in range(1, n_atoms):

        # ────────────────────────────────────────
        # Step A: Bond Block (原子 i-1 ↔ 原子 i)
        # ────────────────────────────────────────
        # 計算此 bond 在 thetas 中的起始索引
        bp = 6 * n_atoms + 4 * (i - 1)

        # 對 Bond Register 施加參數化旋轉
        ry(thetas[bp],     q[bond_q0_idx])
        rz(thetas[bp + 1], q[bond_q0_idx])
        ry(thetas[bp + 2], q[bond_q1_idx])
        rz(thetas[bp + 3], q[bond_q1_idx])

        # 交叉糾纏 (Cross-Entanglement)：
        # 將前一個原子最後一顆量子位元 (q[3(i-1)+2]) 作為控制位元，
        # 與 bond_q0 做 CNOT。
        # → 確保鍵結選擇與前一個原子類型之間存在量子關聯。
        x.ctrl(q[3 * (i - 1) + 2], q[bond_q0_idx])

        # 額外的 bond 內部糾纏，增加鍵結的表達能力
        x.ctrl(q[bond_q0_idx], q[bond_q1_idx])

        # ── CUDA-Q: 顯式中途測量 (Mid-Circuit Measurement) ──────
        # 關鍵！mz() 會將測量結果寫入 classical register。
        # cudaq.sample() 執行時會自動收集這些中途測量結果，
        # 並將其放入回傳 bit-string 的對應位置。
        # 測量順序：先 bond_q0，再 bond_q1（與 bit-string 格式一致）
        mz(q[bond_q0_idx])
        mz(q[bond_q1_idx])

        # ── CUDA-Q: 量子位元重置 (Qubit Reset) ──────────────────
        # 將 bond register 重置為 |0⟩ 態，實現 "Bond Reuse" 策略。
        # 這是 3N+2 架構的核心——用有限的量子位元表達 N-1 個鍵結。
        # CUDA-Q 0.6+ 的 reset() 在 Tensor Network 後端完全支持。
        reset(q[bond_q0_idx])
        reset(q[bond_q1_idx])

        # ────────────────────────────────────────
        # Step B: Atom Block i
        # ────────────────────────────────────────
        ap   = 6 * i         # Atom i 的參數起始索引
        base = 3 * i         # Atom i 的量子位元起始索引

        ry(thetas[ap],     q[base])
        rz(thetas[ap + 1], q[base])
        ry(thetas[ap + 2], q[base + 1])
        rz(thetas[ap + 3], q[base + 1])
        ry(thetas[ap + 4], q[base + 2])
        rz(thetas[ap + 5], q[base + 2])

        # 原子內部糾纏（同 Atom 0）
        x.ctrl(q[base],     q[base + 1])
        x.ctrl(q[base + 1], q[base + 2])

    # ================================================================
    # 最終測量 — 測量所有 Atom Register 的量子位元
    # ================================================================
    # 注意：Bond Register 已在中途測量完畢，此處不再重複測量。
    # 這些 atom 測量結果會 **接在** 中途 bond 測量之後，
    # 形成完整的 bit-string：[bonds | atoms]
    # ================================================================
    for i in range(3 * n_atoms):
        mz(q[i])


# ============================================================================
# SQMGKernel 封裝類別
# ============================================================================

class SQMGKernel:
    """
    SQMG 3N+2 量子線路的 Python 封裝。

    使用方式：
        kernel = SQMGKernel(max_atoms=4, shots=1024)
        params = np.random.uniform(-np.pi, np.pi, kernel.n_params)
        counts = kernel.sample(params)
        # counts: {'010110010100001110': 15, ...}
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        """
        Args:
            max_atoms: 最大重原子數量 N（決定量子位元數 = 3N+2）
            shots:     每次取樣的重複次數
        """
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_qubits = 3 * max_atoms + 2

        # 參數數量計算（見 sqmg_circuit docstring）
        self.n_atom_params = 6 * max_atoms        # 每個原子 6 參數
        self.n_bond_params = 4 * (max_atoms - 1)  # 每個鍵 4 參數
        self.n_params = self.n_atom_params + self.n_bond_params  # = 10N - 4

    def sample(self, params: np.ndarray) -> Dict[str, int]:
        """
        使用給定參數對量子線路進行取樣。

        Args:
            params: 旋轉角度參數 (shape: [n_params,])

        Returns:
            bit-string 計數字典 {bitstring: count}

        CUDA-Q 注意：
        • 使用 cudaq.sample() 而非 cudaq.observe()。
          observe() 僅回傳期望值，會遺失個別 bit-string 的資訊。
        • sample() 回傳完整的 SampleResult，保留所有中途 + 最終測量。
        """
        # CUDA-Q kernel 要求 Python list[float]，不接受 numpy array
        theta_list = params.tolist()

        # ── 關鍵呼叫：cudaq.sample() ──
        # shots_count 指定取樣次數；每次取樣會獨立執行整個量子線路
        results = cudaq.sample(
            sqmg_circuit,
            theta_list,
            self.max_atoms,
            shots_count=self.shots
        )

        # 將 CUDA-Q 的 SampleResult 轉換為 Python dict
        # SampleResult 可直接迭代取得所有不同的 bit-string
        counts: Dict[str, int] = {}
        for bitstring in results:
            counts[bitstring] = results.count(bitstring)

        return counts

    def get_expected_bitstring_length(self) -> int:
        """
        回傳 bit-string 的預期長度 = 5N - 2
        其中前 2(N-1) 位為 bond 測量，後 3N 位為 atom 測量。
        """
        return 5 * self.max_atoms - 2

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        回傳參數的上下界（用於 QPSO 初始化與邊界約束）。
        所有旋轉角度的有效範圍為 [-π, π]。
        """
        lower = np.full(self.n_params, -np.pi)
        upper = np.full(self.n_params,  np.pi)
        return lower, upper

    def describe(self) -> str:
        """回傳線路架構的文字描述。"""
        return (
            f"SQMG 3N+2 Ansatz\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3×{self.max_atoms} + 2)\n"
            f"  Total parameters  : {self.n_params}  (10×{self.max_atoms} − 4)\n"
            f"  Atom params       : {self.n_atom_params}\n"
            f"  Bond params       : {self.n_bond_params}\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}\n"
            f"  Shots per sample  : {self.shots}"
        )
