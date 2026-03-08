"""
==============================================================================
SQMG Kernel — 3N+3 CUDA-Q 參數化量子線路 (v4 Deep HEA)
==============================================================================

架構概覽 (Scalable Quantum Molecular Generation, v4 Deep HEA, 16N-4 params)
───────────────────────────────────────────────────────────────────────────────
• 原子暫存器 (Atom Register)：每個重原子靜態分配 3 顆量子位元，
  可表達 2^3 = 8 種狀態（含 NONE 終止符）。
  每個原子使用 2 層 Hardware-Efficient Ansatz（RY+RZ+Ring-CNOT），共 12 個參數。
• 鍵暫存器 (Bond Register)：僅使用 2 顆量子位元，透過「Mid-circuit
  Measurement + Reset」在每個原子建構步驟間動態重複使用。
  每顆鍵位元使用 RY + RZ 旋轉參數，共 4 個參數。
  → 2 顆量子位元可表達 4 種鍵結類型 (None / Single / Double / Triple)。
• 輔助位元 (Ancilla)：1 顆量子位元，用於 De Morgan OR 特徵提取。
  判斷原子是否 ≠ 000 (NONE)，門控 Bond 糾纏。
  不測量、不攜帶參數，每次使用後 uncompute 恢復 |0⟩。
• 總量子位元數 = 3N + 3（N = 最大重原子數量）。
• 總參數量   = 12N + 4(N-1) = 16N - 4。

v4 升級重點 (Deep Hardware-Efficient Ansatz + Ancilla-OR)
──────────────────────────────────────────────────────────
0. 【2-Layer HEA】每個 Atom 使用 2 層 RY+RZ 旋轉 + Ring-CNOT 糾纏，
   總參數量由 5N-2 提升至 16N-4，大幅增加化學空間探索自由度。
   Bond 亦加入 RZ 旋轉（每鍵 4 參數）。
1. 【Ancilla OR 閘】使用 De Morgan 定理實作 OR：
     OR(a,b,c) = NOT(AND(NOT a, NOT b, NOT c))
   只要原子 ≠ 000，ancilla = |1⟩。
   步驟：X 翻轉 3 顆原子位元 → 3-Controlled X → X ancilla → 恢復原子。
2. 【Ancilla-Gated CNOT】以 ancilla 為控制位元，對 Bond 施加 CNOT。
   所有合法元素（C/O/N/S/P/F/Cl）均觸發 Bond 糾纏。
   NONE 原子不觸發 → Bond 完全由 RY 參數決定。
3. 【CZ 相位糾纏】每顆原子位元透過 CZ 門將「類型相位足跡」
   傳遞至 Bond 位元。交叉分配目標（atom i-1 → bq0/bq1 交替，
   atom i → bq1/bq0 互補），透過量子干涉使不同元素產生不同鍵結傾向。
4. 【Ancilla Uncompute】每次使用後嚴格反運算恢復 ancilla 為 |0⟩，
   確保跨 Bond Block 重複使用時無殘留態。

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
    SQMG 16N-4 參數化量子線路（v4 Deep HEA）。

    參數佈局 (Parameter Layout)
    ──────────────────────────
    每個原子使用 2 層 Hardware-Efficient Ansatz:
      Layer 1: RY×3 + RZ×3 → Ring-CNOT
      Layer 2: RY×3 + RZ×3
    共 12 個參數控制原子態疊加與內部糾纏。
    每個鍵使用 RY×2 + RZ×2 = 4 個參數。

    Atom Register: 12 params/atom × N atoms = 12N params
        Atom i → thetas[12i : 12i+12]  (i = 0 .. N-1)
            Layer 1 RY: thetas[12i+0..2] → q[3i+0..2]
            Layer 1 RZ: thetas[12i+3..5] → q[3i+0..2]
            Ring-CNOT:  q[3i]→q[3i+1]→q[3i+2]→q[3i]
            Layer 2 RY: thetas[12i+6..8] → q[3i+0..2]
            Layer 2 RZ: thetas[12i+9..11] → q[3i+0..2]

    Bond Register: 4 params/bond × (N-1) bonds = 4(N-1) params
        Bond j → thetas[12N + 4j : 12N + 4j + 4]  (j = 0 .. N-2)
            bond_q0 ← RY(thetas[12N + 4j]),   RZ(thetas[12N + 4j + 2])
            bond_q1 ← RY(thetas[12N + 4j + 1]), RZ(thetas[12N + 4j + 3])

    總參數量 = 12N + 4(N-1) = 16N - 4

    參數索引越界驗證（以 N=4 為例）
    ────────────────────────────────
    • Atom 0: thetas[0..11],  Atom 1: thetas[12..23]
      Atom 2: thetas[24..35], Atom 3: thetas[36..47]
    • Bond 0: thetas[48..51], Bond 1: thetas[52..55], Bond 2: thetas[56..59]
    • 最大索引 = 59 = 16×4 − 5  →  陣列大小 = 60 = 16×4 − 4  ✓ 無越界

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
    # 3N (atom) + 2 (bond) + 1 (ancilla) = 3N + 3
    n_qubits = 3 * n_atoms + 3
    q = cudaq.qvector(n_qubits)

    # Bond Register 的全域量子位元索引
    bond_q0_idx = 3 * n_atoms       # 鍵暫存器 qubit 0
    bond_q1_idx = 3 * n_atoms + 1   # 鍵暫存器 qubit 1
    # Ancilla 位元（OR 特徵提取用，不測量）
    ancilla_idx = 3 * n_atoms + 2

    # ================================================================
    # Atom 0 — 初始原子參數化 (2-Layer HEA)
    # ================================================================
    # Layer 1: RY 控制振幅 + RZ 控制相位 → Ring-CNOT 循環糾纏
    # Layer 2: RY + RZ 二次參數化
    # 12 個參數 (thetas[0..11]) 充分探索 3-qubit Hilbert 空間。
    # ================================================================

    # ── Layer 1: RY + RZ ──
    ry(thetas[0], q[0])
    ry(thetas[1], q[1])
    ry(thetas[2], q[2])
    rz(thetas[3], q[0])
    rz(thetas[4], q[1])
    rz(thetas[5], q[2])

    # Ring-CNOT 糾纏：q[0]→q[1]→q[2]→q[0]
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    x.ctrl(q[2], q[0])

    # ── Layer 2: RY + RZ ──
    ry(thetas[6], q[0])
    ry(thetas[7], q[1])
    ry(thetas[8], q[2])
    rz(thetas[9], q[0])
    rz(thetas[10], q[1])
    rz(thetas[11], q[2])

    # ================================================================
    # 迭代建構 Atom 1 ~ Atom N-1
    # 順序：先 Atom i → 再 Bond(i-1, i)
    # ================================================================
    for i in range(1, n_atoms):

        # ────────────────────────────────────────
        # Step A: Atom Block i（2-Layer HEA）
        # ────────────────────────────────────────
        # Atom i 的 12 個參數索引：thetas[12i .. 12i+11]
        ap   = 12 * i        # Atom i 的參數起始索引 (12 params/atom)
        base = 3 * i         # Atom i 的量子位元起始索引

        # ── Layer 1: RY + RZ ──
        ry(thetas[ap],     q[base])
        ry(thetas[ap + 1], q[base + 1])
        ry(thetas[ap + 2], q[base + 2])
        rz(thetas[ap + 3], q[base])
        rz(thetas[ap + 4], q[base + 1])
        rz(thetas[ap + 5], q[base + 2])

        # Ring-CNOT 糾纏
        x.ctrl(q[base],     q[base + 1])
        x.ctrl(q[base + 1], q[base + 2])
        x.ctrl(q[base + 2], q[base])

        # ── Layer 2: RY + RZ ──
        ry(thetas[ap + 6],  q[base])
        ry(thetas[ap + 7],  q[base + 1])
        ry(thetas[ap + 8],  q[base + 2])
        rz(thetas[ap + 9],  q[base])
        rz(thetas[ap + 10], q[base + 1])
        rz(thetas[ap + 11], q[base + 2])

        # ────────────────────────────────────────
        # Step B: Bond Block (原子 i-1 ↔ 原子 i)
        # ────────────────────────────────────────
        # v4: RY+RZ 參數化 + Ancilla OR 特徵提取 + CZ 相位糾纏
        #
        # Bond j 的 4 個參數索引：
        #   thetas[12N + 4j .. 12N + 4j + 3]
        bp = 12 * n_atoms + 4 * (i - 1)

        # 對 Bond Register 施加參數化旋轉 (RY + RZ)
        ry(thetas[bp],     q[bond_q0_idx])
        ry(thetas[bp + 1], q[bond_q1_idx])
        rz(thetas[bp + 2], q[bond_q0_idx])
        rz(thetas[bp + 3], q[bond_q1_idx])

        # ══════════════════════════════════════════════
        # Atom i-1: Ancilla OR 特徵提取 + Bond 糾纏
        # ══════════════════════════════════════════════
        prev_base = 3 * (i - 1)

        # [Compute] ancilla = OR(atom_{i-1} q0, q1, q2)
        # De Morgan: OR(a,b,c) = NOT( AND(NOT a, NOT b, NOT c) )
        # Step 1: X-flip → 000 (NONE) 變為 111
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])
        # Step 2: 3-Controlled X → ancilla=1 iff 翻轉後全 1 (原始=000)
        x.ctrl(q[prev_base], q[prev_base + 1], q[prev_base + 2],
               q[ancilla_idx])
        # Step 3: X ancilla → ancilla=1 iff 原始 ≠ 000 (NOT NONE)
        x(q[ancilla_idx])
        # Step 4: 恢復原子位元
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])

        # [Apply] Ancilla-gated CNOT → bond_q0
        # 只要 atom i-1 ≠ NONE，就翻轉 bond_q0
        x.ctrl(q[ancilla_idx], q[bond_q0_idx])

        # [Uncompute] 恢復 ancilla = |0⟩（反向 compute 步驟）
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])
        x(q[ancilla_idx])
        x.ctrl(q[prev_base], q[prev_base + 1], q[prev_base + 2],
               q[ancilla_idx])
        x(q[prev_base])
        x(q[prev_base + 1])
        x(q[prev_base + 2])

        # [Phase] CZ 相位糾纏：原子類型→Bond 交叉分配
        # CZ 僅在 |11⟩ 態添加 -1 相位，不翻轉位元。
        # 不同元素產生不同相位足跡，透過量子干涉影響鍵結類型。
        z.ctrl(q[prev_base],     q[bond_q0_idx])  # atom bit 0 → bq0
        z.ctrl(q[prev_base + 1], q[bond_q1_idx])  # atom bit 1 → bq1
        z.ctrl(q[prev_base + 2], q[bond_q0_idx])  # atom bit 2 → bq0

        # ══════════════════════════════════════════════
        # Atom i: Ancilla OR 特徵提取 + Bond 糾纏
        # ══════════════════════════════════════════════
        curr_base = 3 * i

        # [Compute] ancilla = OR(atom_i q0, q1, q2)
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])
        x.ctrl(q[curr_base], q[curr_base + 1], q[curr_base + 2],
               q[ancilla_idx])
        x(q[ancilla_idx])
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])

        # [Apply] Ancilla-gated CNOT → bond_q1
        x.ctrl(q[ancilla_idx], q[bond_q1_idx])

        # [Uncompute] 恢復 ancilla = |0⟩
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])
        x(q[ancilla_idx])
        x.ctrl(q[curr_base], q[curr_base + 1], q[curr_base + 2],
               q[ancilla_idx])
        x(q[curr_base])
        x(q[curr_base + 1])
        x(q[curr_base + 2])

        # [Phase] CZ 相位糾纏：互補分配（與 atom i-1 交叉）
        z.ctrl(q[curr_base],     q[bond_q1_idx])  # atom bit 0 → bq1
        z.ctrl(q[curr_base + 1], q[bond_q0_idx])  # atom bit 1 → bq0
        z.ctrl(q[curr_base + 2], q[bond_q1_idx])  # atom bit 2 → bq1

        # ── Bond 內部糾纏 ──
        # bond_q0 → bond_q1，使 2-bit 鍵碼具有關聯性。
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
    SQMG 16N-4 Deep HEA 量子線路的 Python 封裝。

    使用方式：
        kernel = SQMGKernel(max_atoms=4, shots=1024)
        params = np.random.uniform(-np.pi, np.pi, kernel.n_params)
        counts = kernel.sample(params)
        # counts: {'010110010100001110': 15, ...}
    """

    def __init__(self, max_atoms: int = 4, shots: int = 1024):
        """
        Args:
            max_atoms: 最大重原子數量 N（決定量子位元數 = 3N+3）
            shots:     每次取樣的重複次數
        """
        self.max_atoms = max_atoms
        self.shots = shots
        self.n_qubits = 3 * max_atoms + 3  # 3N atom + 2 bond + 1 ancilla

        # 參數數量計算：
        # 每個原子 12 參數 (2 層 HEA: RY×3 + RZ×3 每層)
        # 每個鍵   4 參數 (RY×2 + RZ×2)
        # Ancilla 不攜帶參數（僅做 OR 計算 + uncompute）
        self.n_atom_params = 12 * max_atoms        # 每個原子 12 參數
        self.n_bond_params = 4 * (max_atoms - 1)   # 每個鍵 4 參數
        self.n_params = self.n_atom_params + self.n_bond_params  # = 16N - 4

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
            f"SQMG 3N+3 Ansatz (v4 Deep HEA + Ancilla-OR)\n"
            f"  Max atoms (N)     : {self.max_atoms}\n"
            f"  Total qubits      : {self.n_qubits}  (3×{self.max_atoms} + 3)\n"
            f"  Total parameters  : {self.n_params}  (16×{self.max_atoms} − 4)\n"
            f"  Atom params       : {self.n_atom_params}  (12 per atom, 2-layer HEA)\n"
            f"  Bond params       : {self.n_bond_params}  (4 per bond, RY+RZ)\n"
            f"  Ancilla           : 1  (OR non-NONE detection, uncomputed)\n"
            f"  Bitstring length  : {self.get_expected_bitstring_length()}\n"
            f"  Shots per sample  : {self.shots}\n"
            f"  Circuit order     : Atom-first (Atom i → Bond(i-1,i))\n"
            f"  Ansatz per atom   : [RY+RZ] → Ring-CNOT → [RY+RZ]\n"
            f"  Entanglement      : Ring-CNOT + Ancilla-OR gated CNOT + CZ phase"
        )
