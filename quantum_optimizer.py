"""
==============================================================================
QuantumOptimizer — 量子粒子群優化 (QPSO) 演算法  v3
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
         mbest_d = (1/M) × Σᵢ pbest_i,d

  2. 局部吸引子 (Local Attractor)：
         p_i,d = φ × pbest_i,d + (1 − φ) × gbest_d
         φ ~ Uniform(0, 1)

  3. 位置更新（Delta 勢阱模型）：
         x_i,d = p_i,d ± α × |mbest_d − x_i,d| × ln(1/u)
         u ~ Uniform(0, 1)

  4. 收縮-擴張係數 α：控制探索 vs 利用的平衡。

■ v3 新增三大抗停滯機制
  ────────────────────
  1. 非線性 α 排程：Cosine Annealing + 隨機擾動 + 停滯提升
     α(t) = α_min + ½(α_max − α_min)(1 + cos(πt/T)) + perturbation

  2. Cauchy 變異 (Mutation)：以機率 p_mut 對粒子施加
     Cauchy 分佈的跳躍變異，提供比 Gaussian 更重尾的探索。
     對 gbest 也定期施加變異以探索鄰近盆地。

  3. 停滯偵測 & 部分重初始化：
     連續 N_stag 代 gbest 未改善 → 對最差 reinit_frac 粒子
     進行隨機重初始化，並暫時提升 α 以脫離局部最優。

■ QPSO 與 CUDA-Q kernel 參數的互動
  ──────────────────────────────────
  • 每個「粒子」代表一組量子線路旋轉角度 θ = [θ₀, θ₁, ..., θ_{D-1}]
    其中 D = 5N − 2（N 為重原子數量）。
  • 在每一輪迭代中：
    1. 將粒子位置（參數陣列）傳入 CUDA-Q kernel
    2. kernel 使用這些角度執行參數化量子線路
    3. cudaq.sample() 回傳 bit-string 計數
    4. MoleculeDecoder 將 bit-strings 解碼為分子
    5. 計算適應度（Shaping Reward）
    6. QPSO 根據適應度更新粒子位置（不需要梯度！）
==============================================================================
"""

import math
import numpy as np
from typing import Callable, List, Optional, Tuple


class QuantumOptimizer:
    """
    QPSO (Quantum Particle Swarm Optimization) 優化器 — v3 抗停滯版。

    相對於 v1/v2 的改進：
      • Cosine Annealing α 排程（替代線性遞減）
      • Cauchy 變異機制（重尾探索）
      • 停滯偵測 + 部分粒子重初始化
      • 多樣性監控指標

    使用方式：
        optimizer = QuantumOptimizer(
            n_params=18,            # 5N-2, N=4
            n_particles=20,
            max_iterations=50,
            fitness_fn=my_fitness_function,
            stagnation_limit=5,     # 連續 5 代未改善就觸發重初始化
            mutation_prob=0.15,     # 15% 機率施加 Cauchy 變異
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
        iteration_callback: Optional[Callable[[int, dict], None]] = None,
        # ── v3: 抗停滯超參數 ──
        stagnation_limit: int = 5,
        reinit_fraction: float = 0.3,
        mutation_prob: float = 0.15,
        mutation_scale: float = 0.3,
        alpha_perturb_std: float = 0.05,
        alpha_stag_boost: float = 0.3,
    ):
        """
        初始化 QPSO 優化器。

        Args:
            n_params:        參數空間維度 D（= 5N − 2）
            n_particles:     粒子數量 M（建議 15~30）
            max_iterations:  最大迭代次數 T
            fitness_fn:      適應度函式 f(params) → float
            alpha_max:       α 的最大值（初期，鼓勵探索）
            alpha_min:       α 的最小值（末期，促進收斂）
            param_lower:     參數下界
            param_upper:     參數上界
            seed:            隨機數種子
            verbose:         是否印出每輪進度
            iteration_callback: 每輪迭代結束後回呼

            stagnation_limit:  連續幾代 gbest 未改善就觸發重初始化
            reinit_fraction:   停滯時重初始化的粒子比例 (0~1)
            mutation_prob:     每個粒子在每輪被 Cauchy 變異的機率
            mutation_scale:    Cauchy 變異的尺度因子 (相對於 param range)
            alpha_perturb_std: α 隨機擾動的標準差
            alpha_stag_boost:  停滯觸發時 α 的額外提升量
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
        self.iteration_callback = iteration_callback

        # v3 新參數
        self.stagnation_limit = stagnation_limit
        self.reinit_fraction = reinit_fraction
        self.mutation_prob = mutation_prob
        self.mutation_scale = mutation_scale * (param_upper - param_lower)
        self.alpha_perturb_std = alpha_perturb_std
        self.alpha_stag_boost = alpha_stag_boost

        # 隨機數生成器
        self.rng = np.random.default_rng(seed)

        # ── 初始化粒子群 ──
        self.positions = self.rng.uniform(
            self.lb, self.ub, size=(self.M, self.D)
        )
        self.pbest = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf)
        self.gbest = self.positions[0].copy()
        self.gbest_fitness = -np.inf

        # v3: 停滯追蹤
        self._stagnation_counter = 0
        self._total_reinits = 0
        self._total_mutations = 0

        # 優化歷史紀錄
        self.history: List[dict] = []

    # ────────────────────────────────────────────────────────────
    # v3: Cosine Annealing α 排程
    # ────────────────────────────────────────────────────────────

    def _get_alpha(self, t: int) -> float:
        """
        計算第 t 輪的收縮-擴張係數 α（Cosine Annealing + 隨機擾動）。

        Cosine Annealing：
            α_base(t) = α_min + ½(α_max − α_min)(1 + cos(πt / T))

        相比線性遞減的優勢：
        ┌──────────────────────────────────────────────────────┐
        │ • 初期衰減慢 → 維持更久的高 α 探索期               │
        │ • 中期衰減快 → 避免浪費算力在中間地帶               │
        │ • 末期衰減緩 → 保留微幅探索能力，不完全鎖死         │
        │ • 加上隨機擾動 → 打破確定性衰減軌跡                 │
        └──────────────────────────────────────────────────────┘

        Extra：若處於停滯狀態，額外加上 alpha_stag_boost 以擴大搜索。

        Args:
            t: 目前迭代輪次 (0-indexed)

        Returns:
            當前 α 值（已 clip 到合理範圍）
        """
        progress = t / max(self.T - 1, 1)

        # Cosine annealing 基線
        alpha_base = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (
            1.0 + math.cos(math.pi * progress)
        )

        # 隨機擾動（高斯）
        perturbation = self.rng.normal(0, self.alpha_perturb_std)

        # 停滯提升
        stag_boost = 0.0
        if self._stagnation_counter >= self.stagnation_limit:
            stag_boost = self.alpha_stag_boost

        alpha = alpha_base + perturbation + stag_boost

        # 上下界保護
        alpha_upper = self.alpha_max + self.alpha_stag_boost
        return float(np.clip(alpha, self.alpha_min * 0.8, alpha_upper))

    # ────────────────────────────────────────────────────────────
    # mbest 計算
    # ────────────────────────────────────────────────────────────

    def _compute_mbest(self) -> np.ndarray:
        """
        計算 mbest（所有粒子個人最佳位置的平均值）。

            mbest_d = (1/M) × Σᵢ pbest_i,d

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

            x_new = p ± α × |mbest − x| × ln(1/u)

        Args:
            x, pbest_i, gbest, mbest: 位置向量 (D,)
            alpha: 收縮-擴張係數

        Returns:
            x_new: 更新後的位置 (D,)
        """
        D = self.D

        # Step 1: 局部吸引子
        phi = self.rng.uniform(0, 1, size=D)
        p = phi * pbest_i + (1.0 - phi) * gbest

        # Step 2: Delta 勢阱採樣
        u = np.maximum(self.rng.uniform(0, 1, size=D), 1e-10)
        quantum_step = alpha * np.abs(mbest - x) * np.log(1.0 / u)

        # Step 3: 隨機 ± 方向
        sign = np.where(self.rng.uniform(0, 1, size=D) < 0.5, 1.0, -1.0)

        x_new = p + sign * quantum_step

        # Step 4: 邊界約束
        x_new = np.clip(x_new, self.lb, self.ub)

        return x_new

    # ────────────────────────────────────────────────────────────
    # v3: Cauchy 變異 (Mutation)
    # ────────────────────────────────────────────────────────────

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        """
        對位置向量施加 Cauchy 分佈變異。

        為何用 Cauchy 而非 Gaussian？
        ────────────────────────────
        Cauchy 分佈具有「重尾」(Heavy tail) 特性，
        產生的跳躍距離分佈更廣：
          • Gaussian：99.7% 的跳躍在 ±3σ 以內
          • Cauchy  ：經常產生 >3σ 甚至 >10σ 的大跳躍

        這使粒子能偶爾「跳出」當前盆地，探索遠處的搜索空間。
        在分子生成中，這對應於「嘗試完全不同的原子/鍵組合」。

        實作：
          1. 隨機選取 D 維度的一個子集（proportion ~ 0.3~0.5）
          2. 對選取的維度施加 Cauchy 擾動
          3. 未選取的維度保持不變（局部結構保持）

        Args:
            x: 原始位置向量 (D,)

        Returns:
            x_mut: 變異後的位置向量 (D,)
        """
        x_mut = x.copy()

        # 隨機選取 30%~50% 的維度進行變異
        n_mutate = max(1, int(self.D * self.rng.uniform(0.3, 0.5)))
        dims = self.rng.choice(self.D, size=n_mutate, replace=False)

        # Cauchy 擾動 = standard_cauchy × scale
        cauchy_noise = self.rng.standard_cauchy(size=n_mutate) * self.mutation_scale

        x_mut[dims] += cauchy_noise
        x_mut = np.clip(x_mut, self.lb, self.ub)

        return x_mut

    def _mutate_gbest(self) -> np.ndarray:
        """
        對 gbest 施加小幅 Cauchy 變異，探索最優解的鄰近盆地。

        與一般粒子變異不同：
          • 只擾動 10%~20% 的維度（保守探索）
          • 擾動幅度為一般變異的 50%

        Returns:
            gbest 的變異版本 (D,)
        """
        x_mut = self.gbest.copy()

        n_mutate = max(1, int(self.D * self.rng.uniform(0.1, 0.2)))
        dims = self.rng.choice(self.D, size=n_mutate, replace=False)

        cauchy_noise = self.rng.standard_cauchy(size=n_mutate) * (self.mutation_scale * 0.5)
        x_mut[dims] += cauchy_noise

        return np.clip(x_mut, self.lb, self.ub)

    # ────────────────────────────────────────────────────────────
    # v3: 停滯偵測 & 部分粒子重初始化
    # ────────────────────────────────────────────────────────────

    def _check_and_reinit(self, prev_gbest_fitness: float) -> bool:
        """
        檢查是否處於停滯狀態，若是則對最差粒子重初始化。

        停滯判定：
          連續 stagnation_limit 輪 gbest_fitness 完全未改善
          （改善定義：提升 > 1e-8）

        重初始化策略：
        ┌──────────────────────────────────────────────────────┐
        │ 1. 按 pbest_fitness 排序，找出最差的                 │
        │    reinit_fraction × M 個粒子                        │
        │ 2. 將這些粒子的位置替換為：                          │
        │    • 前半數：完全隨機（全域探索）                     │
        │    • 後半數：在 gbest 附近的高斯擾動（局部探索）      │
        │ 3. 重置這些粒子的 pbest 與 pbest_fitness             │
        │ 4. 重置停滯計數器                                    │
        └──────────────────────────────────────────────────────┘

        Args:
            prev_gbest_fitness: 上一輪的 gbest_fitness

        Returns:
            是否觸發了重初始化
        """
        # 判斷是否有改善
        improved = (self.gbest_fitness - prev_gbest_fitness) > 1e-8

        if improved:
            self._stagnation_counter = 0
            return False

        self._stagnation_counter += 1

        if self._stagnation_counter < self.stagnation_limit:
            return False

        # ── 觸發重初始化 ──
        n_reinit = max(1, int(self.M * self.reinit_fraction))

        # 找出 pbest_fitness 最差的粒子
        worst_indices = np.argsort(self.pbest_fitness)[:n_reinit]

        for k, idx in enumerate(worst_indices):
            if k < n_reinit // 2:
                # 策略 A：完全隨機（全域探索）
                self.positions[idx] = self.rng.uniform(
                    self.lb, self.ub, size=self.D
                )
            else:
                # 策略 B：gbest 附近的高斯擾動（局部探索）
                noise_std = (self.ub - self.lb) * 0.25
                noise = self.rng.normal(0, noise_std, size=self.D)
                self.positions[idx] = np.clip(
                    self.gbest + noise, self.lb, self.ub
                )

            # 重置該粒子的 pbest
            self.pbest[idx] = self.positions[idx].copy()
            self.pbest_fitness[idx] = -np.inf

        self._stagnation_counter = 0
        self._total_reinits += 1

        if self.verbose:
            print(
                f"  ⚡ [停滯偵測] 連續 {self.stagnation_limit} 代無進步，"
                f"已重初始化 {n_reinit} 個粒子（共 {self._total_reinits} 次）"
            )

        return True

    # ────────────────────────────────────────────────────────────
    # v3: 多樣性度量
    # ────────────────────────────────────────────────────────────

    def _compute_diversity(self) -> float:
        """
        計算粒子群的多樣性指標。

        定義：所有粒子位置在各維度上的平均標準差。

            diversity = (1/D) × Σ_d std(positions[:, d])

        diversity 高 → 粒子分散 → 探索狀態
        diversity 低 → 粒子聚集 → 可能已收斂或停滯

        Returns:
            diversity ∈ [0, +∞)
        """
        return float(np.mean(np.std(self.positions, axis=0)))

    # ────────────────────────────────────────────────────────────
    # 主要優化迴圈
    # ────────────────────────────────────────────────────────────

    def optimize(self) -> Tuple[np.ndarray, float, List[dict]]:
        """
        執行 QPSO 優化迭代（v3：含 Cauchy 變異 + 停滯偵測）。

        完整流程（每輪迭代）：
        ─────────────────────
        1. 計算 α（Cosine Annealing + 擾動 + 停滯提升）
        2. 計算 mbest
        3. 對每個粒子：
           a. QPSO Delta 勢阱位置更新
           b. 以機率 p_mut 施加 Cauchy 變異
           c. 評估適應度
           d. 更新 pbest / gbest
        4. 對 gbest 施加 Cauchy 探索（評估但不取代除非更好）
        5. 停滯偵測：若連續 N 代無改善 → 重初始化最差粒子
        6. 記錄歷史 + 回呼 + 進度輸出

        Returns:
            (gbest, gbest_fitness, history)
        """
        if self.fitness_fn is None:
            raise ValueError("fitness_fn 未設定！請在初始化時提供適應度函式。")

        print("=" * 70)
        print("QPSO 量子粒子群優化 v3（抗停滯版）啟動")
        print(f"  粒子數 (M)       : {self.M}")
        print(f"  參數維度 (D)     : {self.D}")
        print(f"  最大迭代 (T)     : {self.T}")
        print(f"  α 範圍           : {self.alpha_max} → {self.alpha_min} (cosine)")
        print(f"  參數範圍         : [{self.lb:.4f}, {self.ub:.4f}]")
        print(f"  停滯門檻         : {self.stagnation_limit} 代")
        print(f"  重初始化比例     : {self.reinit_fraction:.0%}")
        print(f"  Cauchy 變異機率  : {self.mutation_prob:.0%}")
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
            prev_gbest_fitness = self.gbest_fitness

            # 計算當前 α（Cosine Annealing + 擾動 + 停滯提升）
            alpha = self._get_alpha(t)

            # 計算 mbest（平均最佳位置）
            mbest = self._compute_mbest()

            iteration_fitnesses: List[float] = []
            n_mutated_this_iter = 0

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

                # Step 2 (v3): Cauchy 變異
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    n_mutated_this_iter += 1

                # Step 3: 評估新位置的適應度
                fitness = self.fitness_fn(self.positions[i])
                iteration_fitnesses.append(fitness)

                # Step 4: 更新個人最佳 (pbest)
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.positions[i].copy()

                # Step 5: 更新全域最佳 (gbest)
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest = self.positions[i].copy()

            # ── v3: 對 gbest 進行鄰域探索 ──
            gbest_candidate = self._mutate_gbest()
            gbest_cand_fitness = self.fitness_fn(gbest_candidate)
            iteration_fitnesses.append(gbest_cand_fitness)

            if gbest_cand_fitness > self.gbest_fitness:
                self.gbest_fitness = gbest_cand_fitness
                self.gbest = gbest_candidate.copy()
                if self.verbose:
                    print(
                        f"  🔬 [Gbest 變異] 在鄰域發現更優解！"
                        f" fitness: {gbest_cand_fitness:.6f}"
                    )

            self._total_mutations += n_mutated_this_iter

            # ── v3: 停滯偵測 & 重初始化 ──
            self._check_and_reinit(prev_gbest_fitness)

            # ── 多樣性度量 ──
            diversity = self._compute_diversity()

            # ── 記錄歷史 ──
            iter_record = {
                'iteration': t,
                'alpha': alpha,
                'gbest_fitness': self.gbest_fitness,
                'gbest_params': self.gbest.copy(),
                'mean_fitness': float(np.mean(iteration_fitnesses)),
                'max_fitness': float(np.max(iteration_fitnesses)),
                'min_fitness': float(np.min(iteration_fitnesses)),
                'std_fitness': float(np.std(iteration_fitnesses)),
                # v3 附加指標
                'diversity': diversity,
                'stagnation_counter': self._stagnation_counter,
                'n_mutated': n_mutated_this_iter,
            }
            self.history.append(iter_record)

            # ── 回呼 ──
            if self.iteration_callback is not None:
                try:
                    self.iteration_callback(t, iter_record)
                except Exception as e:
                    if self.verbose:
                        print(f"  [Callback 警告] {e}")

            # ── 進度輸出 ──
            if self.verbose:
                stag_marker = (
                    f"  stag={self._stagnation_counter}"
                    if self._stagnation_counter > 0 else ""
                )
                print(
                    f"[Iter {t + 1:3d}/{self.T}]  "
                    f"α={alpha:.4f}  "
                    f"gbest={self.gbest_fitness:.6f}  "
                    f"mean={iter_record['mean_fitness']:.6f}  "
                    f"div={diversity:.4f}  "
                    f"mut={n_mutated_this_iter}"
                    f"{stag_marker}"
                )

        # ── 最終報告 ──
        print("\n" + "=" * 70)
        print("QPSO v3 優化完成")
        print(f"  最佳適應度       : {self.gbest_fitness:.6f}")
        print(f"  最佳參數 (前6維) : {self.gbest[:6].round(4)}")
        print(f"  總重初始化次數   : {self._total_reinits}")
        print(f"  總變異粒子次數   : {self._total_mutations}")
        print(f"  最終多樣性       : {self._compute_diversity():.4f}")
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
        self._stagnation_counter = 0
        self._total_reinits = 0
        self._total_mutations = 0
        self.history = []
