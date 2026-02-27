"""
==============================================================================
QuantumOptimizer — 量子粒子群優化 (QPSO) 演算法
==============================================================================

本模組實作 Quantum Particle Swarm Optimization (QPSO)，
用來最大化分子生成的 Validity 與 QED 分數。

■ 為什麼用 QPSO 而非傳統 BO？
  ─ 原始 QMG 論文使用 GPEI / SAASBO 等貝葉斯優化 (BO)。
  ─ 本專案（SQMG）的核心創新之一是採用「全量子優化器」。
  ─ QPSO 基於量子力學的 Delta 勢阱模型與薛丁格方程式的
    機率分佈來更新粒子位置，無須計算梯度，適合處理
    「非連續 / 高維 / 多模態」的量子線路參數搜尋空間。

■ QPSO 核心數學
  ─────────────
  1. mbest（Mean Best Position，平均最佳位置）：
     所有粒子個人最佳位置 (pbest) 的平均值。

         mbest_d = (1/M) × Σᵢ pbest_i,d

     mbest 扮演「全域吸引子」的角色，引導粒子群往整體
     有前景的區域移動。

  2. 局部吸引子 (Local Attractor)：
     每個粒子在每個維度上的吸引點，是 pbest 與 gbest 的
     隨機線性組合：

         p_i,d = φ × pbest_i,d + (1 − φ) × gbest_d
         φ ~ Uniform(0, 1)

  3. 位置更新（Delta 勢阱模型）：
     根據 QPSO 的量子行為（非牛頓軌跡），粒子的新位置
     遵循 Delta 勢阱的機率分佈：

         x_i,d = p_i,d ± α × |mbest_d − x_i,d| × ln(1/u)
         u ~ Uniform(0, 1)

     ± 號以 50% 的機率隨機選取。

  4. 收縮-擴張係數 (Contraction-Expansion Coefficient, α)：
     α 控制搜尋範圍：
       α 大 → 探索 (Exploration)：粒子跳得遠，避免陷入局部最優
       α 小 → 利用 (Exploitation)：粒子收斂到良好區域
     通常線性遞減：α(t) = α_max − (α_max − α_min) × (t / T)

■ QPSO 與 CUDA-Q kernel 參數的互動
  ──────────────────────────────────
  • 每個「粒子」代表一組量子線路旋轉角度 θ = [θ₀, θ₁, ..., θ_{D-1}]
    其中 D = 10N − 4（N 為重原子數量）。
  • 在每一輪迭代中：
    1. 將粒子位置（參數陣列）傳入 CUDA-Q kernel
    2. kernel 使用這些角度執行參數化量子線路
    3. cudaq.sample() 回傳 bit-string 計數
    4. MoleculeDecoder 將 bit-strings 解碼為分子
    5. 計算適應度（Validity + QED）
    6. QPSO 根據適應度更新粒子位置（不需要梯度！）
==============================================================================
"""

import numpy as np
from typing import Callable, List, Optional, Tuple


class QuantumOptimizer:
    """
    QPSO (Quantum Particle Swarm Optimization) 優化器。

    用於搜尋 SQMG 量子線路的最佳旋轉角度參數，
    最大化分子生成的 Validity 與 QED 分數。

    使用方式：
        optimizer = QuantumOptimizer(
            n_params=36,          # 10N-4, N=4
            n_particles=20,
            max_iterations=50,
            fitness_fn=my_fitness_function,
        )
        best_params, best_fitness, history = optimizer.optimize()
    """

    def __init__(
        self,
        n_params: int,
        n_particles: int = 20,
        max_iterations: int = 50,
        fitness_fn: Optional[Callable[[np.ndarray], float]] = None,
        alpha_max: float = 1.0,
        alpha_min: float = 0.5,
        param_lower: float = -np.pi,
        param_upper: float = np.pi,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        初始化 QPSO 優化器。

        Args:
            n_params:        參數空間維度 D（= 10N − 4）
            n_particles:     粒子數量 M（建議 15~30）
            max_iterations:  最大迭代次數 T
            fitness_fn:      適應度函式 f(params) → float
                             其中 params 是 shape (D,) 的 numpy array
            alpha_max:       收縮-擴張係數的最大值（初期，鼓勵探索）
            alpha_min:       收縮-擴張係數的最小值（末期，促進收斂）
            param_lower:     參數下界（所有維度共用）
            param_upper:     參數上界（所有維度共用）
            seed:            隨機數種子（可重現性）
            verbose:         是否印出每輪進度
        """
        self.D = n_params
        self.M = n_particles
        self.T = max_iterations
        self.fitness_fn = fitness_fn
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.lb = param_lower
        self.ub = param_upper
        self.verbose = verbose

        # 隨機數生成器（支援可重現性）
        self.rng = np.random.default_rng(seed)

        # ── 初始化粒子群 ──
        # 每個粒子的位置 x[i] ∈ ℝ^D，在 [lb, ub] 內均勻隨機分佈
        self.positions = self.rng.uniform(
            self.lb, self.ub, size=(self.M, self.D)
        )

        # 個人最佳位置 (pbest) 和對應適應度
        self.pbest = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf)

        # 全域最佳位置 (gbest) 和對應適應度
        self.gbest = self.positions[0].copy()
        self.gbest_fitness = -np.inf

        # 優化歷史紀錄
        self.history: List[dict] = []

    # ────────────────────────────────────────────────────────────
    # α 排程 (Contraction-Expansion Coefficient Scheduling)
    # ────────────────────────────────────────────────────────────

    def _get_alpha(self, t: int) -> float:
        """
        計算第 t 輪的收縮-擴張係數 α。

        線性遞減排程：α(t) = α_max − (α_max − α_min) × (t / T)

        • 初期 (t ≈ 0)：α ≈ α_max → 大步搜索，全域探索
        • 末期 (t ≈ T)：α ≈ α_min → 小步微調，收斂到最優解

        Args:
            t: 目前迭代輪次 (0-indexed)

        Returns:
            當前 α 值
        """
        return self.alpha_max - (self.alpha_max - self.alpha_min) * (t / max(self.T - 1, 1))

    # ────────────────────────────────────────────────────────────
    # mbest 計算 (Mean Best Position)
    # ────────────────────────────────────────────────────────────

    def _compute_mbest(self) -> np.ndarray:
        """
        計算 mbest（所有粒子個人最佳位置的平均值）。

            mbest_d = (1/M) × Σᵢ pbest_i,d

        mbest 作為「量子勢阱的中心」，代表粒子群的集體知識。
        在 Delta 勢阱模型中，mbest 決定了每個粒子振盪的中心位置，
        使整個粒子群朝向有前景的參數空間區域移動。

        Returns:
            mbest: shape (D,) 的平均最佳位置向量
        """
        return np.mean(self.pbest, axis=0)

    # ────────────────────────────────────────────────────────────
    # QPSO 位置更新核心
    # ────────────────────────────────────────────────────────────

    def _update_position(
        self, x: np.ndarray, pbest_i: np.ndarray,
        gbest: np.ndarray, mbest: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        使用 QPSO Delta 勢阱模型更新單一粒子的位置。

        數學推導（簡述）：
        ──────────────────
        在量子力學中，被束縛在 Delta 勢阱中的粒子不具有確定的
        軌跡（不同於經典 PSO 的速度更新）。其位置遵循機率分佈：

            |ψ(x)|² = (1 / L) × exp(−2|x − p| / L)

        其中 L = 2α|mbest − x|（勢阱寬度）。

        透過逆變換取樣法（Inverse Transform Sampling），
        從上述分佈中採樣新位置：

            x_new = p ± α × |mbest − x| × ln(1/u)

        Args:
            x:       當前位置 (D,)
            pbest_i: 此粒子的個人最佳位置 (D,)
            gbest:   全域最佳位置 (D,)
            mbest:   平均最佳位置 (D,)
            alpha:   收縮-擴張係數

        Returns:
            x_new: 更新後的位置 (D,)
        """
        D = self.D

        # ── Step 1: 計算局部吸引子 (Local Attractor) ──
        # φ 對每個維度獨立隨機，使搜索方向多樣化
        phi = self.rng.uniform(0, 1, size=D)
        p = phi * pbest_i + (1.0 - phi) * gbest

        # ── Step 2: Delta 勢阱採樣 ──
        # u ~ Uniform(0, 1)，取 max 以避免 ln(0) → -inf
        u = np.maximum(self.rng.uniform(0, 1, size=D), 1e-10)

        # |mbest − x| × ln(1/u) 構成量子漲落的振幅
        # 直覺：距離 mbest 越遠的維度，搜索步幅越大
        quantum_step = alpha * np.abs(mbest - x) * np.log(1.0 / u)

        # ── Step 3: 隨機選擇 ± 方向 ──
        # 以 50% 的機率選擇正或負方向，模擬量子態的對稱性
        sign = np.where(self.rng.uniform(0, 1, size=D) < 0.5, 1.0, -1.0)

        x_new = p + sign * quantum_step

        # ── Step 4: 邊界約束 ──
        # 將位置夾到 [lb, ub] 範圍內（物理上對應旋轉角度 [-π, π]）
        x_new = np.clip(x_new, self.lb, self.ub)

        return x_new

    # ────────────────────────────────────────────────────────────
    # 主要優化迴圈
    # ────────────────────────────────────────────────────────────

    def optimize(self) -> Tuple[np.ndarray, float, List[dict]]:
        """
        執行 QPSO 優化迭代。

        完整流程：
        ─────────
        1. 對每個粒子評估適應度
        2. 更新個人最佳 (pbest) 與全域最佳 (gbest)
        3. 計算 mbest 與 α
        4. 使用 Delta 勢阱模型更新粒子位置
        5. 記錄歷史並印出進度
        6. 重複直到達到最大迭代次數

        Returns:
            (gbest, gbest_fitness, history)
            gbest:          最佳參數向量 (D,)
            gbest_fitness:  最佳適應度分數
            history:        每輪的統計紀錄
        """
        if self.fitness_fn is None:
            raise ValueError("fitness_fn 未設定！請在初始化時提供適應度函式。")

        print("=" * 70)
        print("QPSO 量子粒子群優化 啟動")
        print(f"  粒子數 (M)     : {self.M}")
        print(f"  參數維度 (D)   : {self.D}")
        print(f"  最大迭代 (T)   : {self.T}")
        print(f"  α 範圍         : {self.alpha_max} → {self.alpha_min}")
        print(f"  參數範圍       : [{self.lb:.4f}, {self.ub:.4f}]")
        print("=" * 70)

        # ── 初始適應度評估 ──
        print("\n[初始化] 評估所有粒子的初始適應度...")
        for i in range(self.M):
            fitness = self.fitness_fn(self.positions[i])
            self.pbest_fitness[i] = fitness
            self.pbest[i] = self.positions[i].copy()

            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest = self.positions[i].copy()

        print(f"[初始化] 完成。初始全域最佳適應度: {self.gbest_fitness:.6f}\n")

        # ── 主迭代迴圈 ──
        for t in range(self.T):
            # 計算當前 α
            alpha = self._get_alpha(t)

            # 計算 mbest（平均最佳位置）
            mbest = self._compute_mbest()

            iteration_fitnesses: List[float] = []

            # ── 更新每個粒子 ──
            for i in range(self.M):
                # Step 1: QPSO 位置更新
                self.positions[i] = self._update_position(
                    x=self.positions[i],
                    pbest_i=self.pbest[i],
                    gbest=self.gbest,
                    mbest=mbest,
                    alpha=alpha,
                )

                # Step 2: 評估新位置的適應度
                # 這裡會呼叫 CUDA-Q kernel → sample → decode → compute QED
                fitness = self.fitness_fn(self.positions[i])
                iteration_fitnesses.append(fitness)

                # Step 3: 更新個人最佳 (pbest)
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.positions[i].copy()

                # Step 4: 更新全域最佳 (gbest)
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest = self.positions[i].copy()

            # ── 記錄歷史 ──
            iter_record = {
                'iteration': t,
                'alpha': alpha,
                'gbest_fitness': self.gbest_fitness,
                'mean_fitness': float(np.mean(iteration_fitnesses)),
                'max_fitness': float(np.max(iteration_fitnesses)),
                'min_fitness': float(np.min(iteration_fitnesses)),
                'std_fitness': float(np.std(iteration_fitnesses)),
            }
            self.history.append(iter_record)

            # ── 進度輸出 ──
            if self.verbose:
                print(
                    f"[Iter {t + 1:3d}/{self.T}]  "
                    f"α={alpha:.4f}  "
                    f"gbest={self.gbest_fitness:.6f}  "
                    f"mean={iter_record['mean_fitness']:.6f}  "
                    f"max={iter_record['max_fitness']:.6f}  "
                    f"std={iter_record['std_fitness']:.6f}"
                )

        print("\n" + "=" * 70)
        print("QPSO 優化完成")
        print(f"  最佳適應度 : {self.gbest_fitness:.6f}")
        print(f"  最佳參數   : (前 6 維) {self.gbest[:6].round(4)}")
        print("=" * 70)

        return self.gbest.copy(), self.gbest_fitness, self.history

    # ────────────────────────────────────────────────────────────
    # 工具函式
    # ────────────────────────────────────────────────────────────

    def get_convergence_curve(self) -> Tuple[List[int], List[float]]:
        """
        取得收斂曲線資料（用於繪圖）。

        Returns:
            (iterations, gbest_fitnesses)
        """
        iterations = [h['iteration'] for h in self.history]
        fitnesses = [h['gbest_fitness'] for h in self.history]
        return iterations, fitnesses

    def reset(self):
        """
        重置優化器狀態，使用新的隨機粒子位置。
        保留相同的超參數設定。
        """
        self.positions = self.rng.uniform(
            self.lb, self.ub, size=(self.M, self.D)
        )
        self.pbest = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf)
        self.gbest = self.positions[0].copy()
        self.gbest_fitness = -np.inf
        self.history = []
