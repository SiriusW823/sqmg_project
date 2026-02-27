"""
==============================================================================
SQMG Kernel — 3N+2 CUDA-Q 參數化量子線路 (修正版 v2)
==============================================================================

架構概覽 (Scalable Quantum Molecular Generation, 3N+2 Ansatz)
─────────────────────────────────────────────────────────────
• 原子暫存器 (Atom Register)：每個重原子靜態分配 3 顆量子位元，
  可表達 2^3 = 8 種狀態（含 NONE 終止符）。
  每顆量子位元使用 1 個 RY 旋轉參數。
• 鍵暫存器 (Bond Register)：僅使用 2 顆量子位元，透過「Mid-circuit
  Measurement + Reset」在每個原子建構步驟間動態重複使用。
  每顆量子位元使用 1 個 RY 旋轉參數。
  → 2 顆量子位元可表達 4 種鍵結類型 (None / Single / Double / Triple)。
• 總量子位元數 = 3N + 2（N = 最大重原子數量）。
• 總參數量   = 3N + 2(N-1) = 5N - 2。

v2 修正重點
──────────
1. 【參數對齊】每個 qubit 僅使用 1 個 RY 參數（移除冗餘 RZ），
   使總參數量精準等於 3N + 2(N−1) = 5N − 2。
2. 【線路順序】改為「先建 Atom i → 再建 Bond(i−1, i)」，
   使 Bond 的 CNOT 糾纏能同時連接 atom i−1 與 atom i。
3. 【雙向交叉糾纏】Bond block 的 CNOT 同時從 atom i−1 和
   atom i 引出控制位元，確保鍵結選擇與兩端原子類型都有量子關聯。
4. 【空白防護】sample() 在建構計數字典時清除 CUDA-Q 可能插入的空格。

CUDA-Q 防範資訊遺失重點
───────────────────────
✓ 每個 bond 的中途測量都使用顯式 mz()
✓ 所有測量結果保留在 classical register
✓ 使用 cudaq.sample() 收集完整 bit-string（絕不使用 observe()）
==============================================================================
"""

import cudaq
import numpy as np
from typing import Dict, Tuple

# ============================================================================
# CUDA-Q Kernel（Module-level 定義）
# ============================================================================
# @cudaq.kernel 裝飾器將 Python 函式 JIT 編譯為量子程式
# （MLIR → QIR → 後端原生碼）。
# 限制：Kernel 內部只能使用基本型別 (int, float, bool, list[float])
#       和 CUDA-Q 內建量子閘，不能使用 Numpy / Class / 任意 Python 物件。
# ============================================================================


@cudaq.kernel
def sqmg_circuit(thetas: list[float], n_atoms: int):
    """
    SQMG 3N+2 參數化量子線路（修正版 v2）。

    參數佈局 (Parameter Layout)
    ──────────────────────────
    每個量子位元分配 1 個 RY 旋轉參數（控制 |0⟩ ↔ |1⟩ 的機率振幅）。
    CNOT 糾纏閘負責在量子位元間建立關聯，不需要額外參數。

    Atom Register: 3 params/atom × N atoms = 3N params
        Atom i → thetas[3i : 3i+3]  (i = 0 .. N-1)
            qubit 3i+0 ← RY(thetas[3i+0])
            qubit 3i+1 ← RY(thetas[3i+1])
            qubit 3i+2 ← RY(thetas[3i+2])

    Bond Register: 2 params/bond × (N-1) bonds = 2(N-1) params
        Bond j → thetas[3N + 2j : 3N + 2j + 2]  (j = 0 .. N-2)
            bond_q0 ← RY(thetas[3N + 2j])
            bond_q1 ← RY(thetas[3N + 2j + 1])

    總參數量 = 3N + 2(N-1) = 5N - 2

    參數索引越界驗證（以 N=4 為例）
    ────────────────────────────────
    • Atom 0: thetas[0..2],  Atom 1: thetas[3..5]
      Atom 2: thetas[6..8],  Atom 3: thetas[9..11]
    • Bond 0: thetas[12..13], Bond 1: thetas[14..15], Bond 2: thetas[16..17]
    • 最大索引 = 17 = 5×4 − 3  →  陣列大小 = 18 = 5×4 − 2  ✓ 無越界

    線路執行順序（修正為 Atom-first）
    ─────────────────────────────────
    Atom 0  →  [ Atom 1  →  Bond(0,1) ]  →  [ Atom 2  →  Bond(1,2) ]  → ...
    先建構 Atom i 的量子態，再建構 Bond(i-1, i)。
    這使得 Bond 的 CNOT 可以同時連接到 atom i-1 與 atom i 的量子位元。

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
    # 使用 RY 控制 |0⟩ ↔ |1⟩ 的振幅分配（Population）。
    # 每個 qubit 僅需 1 個參數即可控制其在計算基底上的機率分佈。
    # 後續的 CNOT 糾纏閘會在 qubit 間引入量子關聯。
    # ================================================================
    # param_idx: 0, 1, 2
    ry(thetas[0], q[0])
    ry(thetas[1], q[1])
    ry(thetas[2], q[2])

    # 原子內部糾纏 (Intra-atom Entanglement)
    # CNOT 鏈：q[0]→q[1]→q[2]
    # 使 3 顆量子位元能表達具有量子關聯的 8 態疊加，增加表達力。
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])

    # ================================================================
    # 迭代建構 Atom 1 ~ Atom N-1
    # 順序：先 Atom i → 再 Bond(i-1, i)
    # ================================================================
    for i in range(1, n_atoms):

        # ────────────────────────────────────────
        # Step A: Atom Block i（先建構原子）
        # ────────────────────────────────────────
        # Atom i 的 3 個 RY 參數索引：thetas[3i], thetas[3i+1], thetas[3i+2]
        ap   = 3 * i         # Atom i 的參數起始索引
        base = 3 * i         # Atom i 的量子位元起始索引

        ry(thetas[ap],     q[base])
        ry(thetas[ap + 1], q[base + 1])
        ry(thetas[ap + 2], q[base + 2])

        # 原子內部糾纏（同 Atom 0）
        x.ctrl(q[base],     q[base + 1])
        x.ctrl(q[base + 1], q[base + 2])

        # ────────────────────────────────────────
        # Step B: Bond Block (原子 i-1 ↔ 原子 i)
        # ────────────────────────────────────────
        # 現在 Atom i 已經建構完成，Bond 的 CNOT 可以同時
        # 連接到 atom i-1 和 atom i，實現雙向交叉糾纏。
        #
        # Bond j 的 2 個 RY 參數索引：
        #   thetas[3N + 2j], thetas[3N + 2j + 1]
        #   其中 j = i - 1
        bp = 3 * n_atoms + 2 * (i - 1)

        # 對 Bond Register 施加參數化旋轉
        ry(thetas[bp],     q[bond_q0_idx])
        ry(thetas[bp + 1], q[bond_q1_idx])

        # ── 雙向交叉糾纏 (Bidirectional Cross-Entanglement) ──
        #
        # CNOT 1: atom i-1 的最後一顆量子位元 (q[3(i-1)+2])
        #         作為控制位元 → bond_q0 作為目標位元。
        #         → 確保鍵結選擇受「前一個原子類型」影響。
        x.ctrl(q[3 * (i - 1) + 2], q[bond_q0_idx])

        # CNOT 2: atom i 的第一顆量子位元 (q[3i])
        #         作為控制位元 → bond_q1 作為目標位元。
        #         → 確保鍵結選擇同時受「當前原子類型」影響。
        #         （這是 v2 修正新增的糾纏，修復了原始版本
        #           Bond block 僅連接 atom i-1 的缺陷。）
        x.ctrl(q[3 * i], q[bond_q1_idx])

        # CNOT 3: bond 內部糾纏
        #         bond_q0 → bond_q1，使 2-bit 鍵碼具有關聯性。
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
        reset(q[bond_q0_idx])
        reset(q[bond_q1_idx])

    # ================================================================
    # 最終測量 — 測量所有 Atom Register 的量子位元
    # ================================================================
    # Bond Register 已在中途測量完畢，此處不再重複測量。
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

        # 參數數量計算（修正版）：
        # 每個 qubit 僅 1 個 RY 參數 → 3 params/atom, 2 params/bond
        self.n_atom_params = 3 * max_atoms        # 每個原子 3 參數
        self.n_bond_params = 2 * (max_atoms - 1)  # 每個鍵 2 參數
        self.n_params = self.n_atom_params + self.n_bond_params  # = 5N - 2

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
        • 回傳的 bit-string 可能含有空格（視後端而定），
          此處統一清除空白，避免下游解碼錯誤。
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
        # 【修正】清除 bit-string 中可能的空白字元 (CUDA-Q 已知問題)
        counts: Dict[str, int] = {}
        for bitstring in results:
            clean_bs = bitstring.replace(" ", "")
            counts[clean_bs] = results.count(bitstring)

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
            f"SQMG 3N+2 Ansatz (v2)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3×{self.max_atoms} + 2)\n"
            f"  Total parameters  : {self.n_params}  (5×{self.max_atoms} − 2)\n"
            f"  Atom params       : {self.n_atom_params}  (3 per atom)\n"
            f"  Bond params       : {self.n_bond_params}  (2 per bond)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Circuit order     : Atom-first (Atom i → Bond(i-1,i))\n"
            f"  Entanglement      : Bidirectional (atom i-1 & atom i → bond)"
        )
