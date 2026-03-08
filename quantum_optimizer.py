"""
==============================================================================
SOQPSOOptimizer — 単目標量子粒子群優化 (Single-Objective QPSO) v6
==============================================================================

本模組實作 Single-Objective Quantum Particle Swarm Optimization (SOQPSO)，
集中所有算力最大化單一的 Fitness 分數。

■ 設計理念：單目標重火力打擊
  - 移除 Pareto Archive，改用傳統 gbest / pbest 追蹤
  - 所有粒子圍繞 gbest 與 mbest 收斂
  - 保留 Cosine α、Cauchy 變異、停滯偵測等抗停滯機制

■ SOQPSO 核心數學
  1. mbest = (1/M) × Σᵢ pbest_i
  2. 局部吸引子: p_i = φ × pbest_i + (1 − φ) × gbest,  φ ~ U(0,1)
  3. 位置更新 (Delta 勢阱):
       x_i = p_i ± α × |mbest − x_i| × ln(1/u),  u ~ U(0,1)

■ fitness_fn 回傳值
  fitness_fn(params) → float  (單一標量)
==============================================================================
"""

import math
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


class SOQPSOOptimizer:
    """
    SOQPSO (Single-Objective Quantum Particle Swarm Optimization) — v6。

    使用方式：
        optimizer = SOQPSOOptimizer(
            n_params=78,
            n_particles=40,
            max_iterations=30,
            fitness_fn=my_fn,     # returns float
        )
        best_params, best_fitness, history = optimizer.optimize()
    """

    def __init__(
        self,
        n_params: int,
        n_particles: int = 40,
        max_iterations: int = 30,
        fitness_fn: Optional[Callable[[np.ndarray], float]] = None,
        alpha_max: float = 1.0,
        alpha_min: float = 0.5,
        param_lower: float = -np.pi,
        param_upper: float = np.pi,
        seed: Optional[int] = None,
        verbose: bool = True,
        iteration_callback: Optional[Callable[[int, dict], None]] = None,
        stagnation_limit: int = 5,
        reinit_fraction: float = 0.3,
        mutation_prob: float = 0.15,
        mutation_scale: float = 0.3,
        alpha_perturb_std: float = 0.05,
        alpha_stag_boost: float = 0.3,
    ):
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

        # 抗停滯參數
        self.stagnation_limit = stagnation_limit
        self.reinit_fraction = reinit_fraction
        self.mutation_prob = mutation_prob
        self.mutation_scale = mutation_scale * (param_upper - param_lower)
        self.alpha_perturb_std = alpha_perturb_std
        self.alpha_stag_boost = alpha_stag_boost

        self.rng = np.random.default_rng(seed)

        # 初始化粒子群
        self.positions = self.rng.uniform(
            self.lb, self.ub, size=(self.M, self.D)
        )
        self.pbest = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf)

        # 全域最佳
        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_fitness: float = -np.inf

        # 停滯追蹤
        self._stagnation_counter = 0
        self._prev_gbest_fitness = -np.inf
        self._total_reinits = 0
        self._total_mutations = 0

        self.history: List[dict] = []

    # ────────────────────────────────────────────────────────────
    # pbest / gbest 更新
    # ────────────────────────────────────────────────────────────

    def _update_pbest(self, i: int, fitness: float):
        """若新 fitness 超越 pbest，則更新。"""
        if fitness > self.pbest_fitness[i]:
            self.pbest[i] = self.positions[i].copy()
            self.pbest_fitness[i] = fitness

    def _update_gbest(self, i: int, fitness: float):
        """若新 fitness 超越 gbest，則更新。"""
        if fitness > self.gbest_fitness:
            self.gbest_position = self.positions[i].copy()
            self.gbest_fitness = fitness

    # ────────────────────────────────────────────────────────────
    # Cosine Annealing α
    # ────────────────────────────────────────────────────────────

    def _get_alpha(self, t: int) -> float:
        progress = t / max(self.T - 1, 1)
        alpha_base = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (
            1.0 + math.cos(math.pi * progress)
        )
        perturbation = self.rng.normal(0, self.alpha_perturb_std)
        stag_boost = self.alpha_stag_boost if self._stagnation_counter >= self.stagnation_limit else 0.0
        alpha = alpha_base + perturbation + stag_boost
        return float(np.clip(alpha, self.alpha_min * 0.8,
                             self.alpha_max + self.alpha_stag_boost))

    # ────────────────────────────────────────────────────────────
    # mbest
    # ────────────────────────────────────────────────────────────

    def _compute_mbest(self) -> np.ndarray:
        return np.mean(self.pbest, axis=0)

    # ────────────────────────────────────────────────────────────
    # QPSO 位置更新 (Delta 勢阱)
    # ────────────────────────────────────────────────────────────

    def _update_position(
        self, x: np.ndarray, pbest_i: np.ndarray,
        gbest: np.ndarray, mbest: np.ndarray, alpha: float,
    ) -> np.ndarray:
        """
        Delta 勢阱位置更新，圍繞 gbest 與 pbest 的線性組合。

            p = φ × pbest_i + (1 − φ) × gbest
            x_new = p ± α × |mbest − x| × ln(1/u)
        """
        D = self.D
        phi = self.rng.uniform(0, 1, size=D)
        p = phi * pbest_i + (1.0 - phi) * gbest
        u = np.maximum(self.rng.uniform(0, 1, size=D), 1e-10)
        quantum_step = alpha * np.abs(mbest - x) * np.log(1.0 / u)
        sign = np.where(self.rng.uniform(0, 1, size=D) < 0.5, 1.0, -1.0)
        x_new = p + sign * quantum_step
        return np.clip(x_new, self.lb, self.ub)

    # ────────────────────────────────────────────────────────────
    # Cauchy 變異
    # ────────────────────────────────────────────────────────────

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        """重尾 Cauchy 變異，促進跳出局部最優。"""
        x_mut = x.copy()
        n_mutate = max(1, int(self.D * self.rng.uniform(0.3, 0.5)))
        dims = self.rng.choice(self.D, size=n_mutate, replace=False)
        cauchy_noise = self.rng.standard_cauchy(size=n_mutate) * self.mutation_scale
        x_mut[dims] += cauchy_noise
        return np.clip(x_mut, self.lb, self.ub)

    # ────────────────────────────────────────────────────────────
    # 多樣性度量
    # ────────────────────────────────────────────────────────────

    def _compute_diversity(self) -> float:
        return float(np.mean(np.std(self.positions, axis=0)))

    # ────────────────────────────────────────────────────────────
    # 停滯偵測 & 重初始化
    # ────────────────────────────────────────────────────────────

    def _check_and_reinit(self) -> bool:
        """
        基於 gbest 是否改善的停滯偵測。
        連續 N 代 gbest 未改善 → 重初始化最差粒子。
        """
        if self.gbest_fitness > self._prev_gbest_fitness + 1e-8:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1
        self._prev_gbest_fitness = self.gbest_fitness

        if self._stagnation_counter < self.stagnation_limit:
            return False

        n_reinit = max(1, int(self.M * self.reinit_fraction))
        worst_indices = np.argsort(self.pbest_fitness)[:n_reinit]

        for k, idx in enumerate(worst_indices):
            if k < n_reinit // 2:
                # 完全隨機 (全域探索)
                self.positions[idx] = self.rng.uniform(
                    self.lb, self.ub, size=self.D
                )
            else:
                # gbest 附近高斯擾動 (局部探索)
                if self.gbest_position is not None:
                    noise = self.rng.normal(
                        0, (self.ub - self.lb) * 0.25, size=self.D
                    )
                    self.positions[idx] = np.clip(
                        self.gbest_position + noise, self.lb, self.ub
                    )
                else:
                    self.positions[idx] = self.rng.uniform(
                        self.lb, self.ub, size=self.D
                    )
            self.pbest[idx] = self.positions[idx].copy()
            self.pbest_fitness[idx] = -np.inf

        self._stagnation_counter = 0
        self._total_reinits += 1

        if self.verbose:
            print(
                f"  ⚡ [停滯偵測] 連續 {self.stagnation_limit} 代 "
                f"gbest 未改善，已重初始化 {n_reinit} 個粒子"
                f"（共 {self._total_reinits} 次）"
            )
        return True

    # ────────────────────────────────────────────────────────────
    # 主要優化迴圈
    # ────────────────────────────────────────────────────────────

    def optimize(self) -> Tuple[np.ndarray, float, List[dict]]:
        """
        執行 SOQPSO 單目標優化。

        Returns:
            (best_params, best_fitness, history)
        """
        if self.fitness_fn is None:
            raise ValueError("fitness_fn 未設定！")

        print("=" * 70)
        print("SOQPSO 單目標量子粒子群優化 v6 啟動")
        print(f"  粒子數 (M)       : {self.M}")
        print(f"  參數維度 (D)     : {self.D}")
        print(f"  最大迭代 (T)     : {self.T}")
        print(f"  α 範圍           : {self.alpha_max} → {self.alpha_min} (cosine)")
        print(f"  參數範圍         : [{self.lb:.4f}, {self.ub:.4f}]")
        print(f"  目標             : 最大化 Fitness (單目標)")
        print(f"  停滯門檻         : {self.stagnation_limit} 代")
        print(f"  重初始化比例     : {self.reinit_fraction:.0%}")
        print(f"  Cauchy 變異機率  : {self.mutation_prob:.0%}")
        print("=" * 70)

        # 初始適應度評估
        print("\n[初始化] 評估所有粒子的初始適應度...")
        for i in range(self.M):
            fit = self.fitness_fn(self.positions[i])
            self.pbest_fitness[i] = fit
            self.pbest[i] = self.positions[i].copy()
            self._update_gbest(i, fit)

        self._prev_gbest_fitness = self.gbest_fitness
        print(
            f"[初始化] 完成。Gbest Fitness: {self.gbest_fitness:.6f}\n"
        )

        # 主迭代迴圈
        for t in range(self.T):
            alpha = self._get_alpha(t)
            mbest = self._compute_mbest()

            iter_fitnesses: List[float] = []
            n_mutated_this_iter = 0

            for i in range(self.M):
                # QPSO 位置更新
                self.positions[i] = self._update_position(
                    x=self.positions[i],
                    pbest_i=self.pbest[i],
                    gbest=self.gbest_position,
                    mbest=mbest,
                    alpha=alpha,
                )

                # Cauchy 變異
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    n_mutated_this_iter += 1

                # 評估
                fit = self.fitness_fn(self.positions[i])
                iter_fitnesses.append(fit)

                # 更新 pbest & gbest
                self._update_pbest(i, fit)
                self._update_gbest(i, fit)

            self._total_mutations += n_mutated_this_iter

            # 停滯偵測
            self._check_and_reinit()

            diversity = self._compute_diversity()

            # 歷史紀錄
            iter_record = {
                'iteration': t,
                'gbest_fitness': self.gbest_fitness,
                'gbest_params': self.gbest_position.copy() if self.gbest_position is not None else None,
                'mean_fitness': float(np.mean(iter_fitnesses)),
                'max_fitness': float(np.max(iter_fitnesses)),
                'min_fitness': float(np.min(iter_fitnesses)),
                'alpha': alpha,
                'diversity': diversity,
                'stagnation_counter': self._stagnation_counter,
                'n_mutated': n_mutated_this_iter,
            }
            self.history.append(iter_record)

            # 回呼
            if self.iteration_callback is not None:
                try:
                    self.iteration_callback(t, iter_record)
                except Exception as e:
                    if self.verbose:
                        print(f"  [Callback 警告] {e}")

            # 進度輸出
            if self.verbose:
                stag_marker = (
                    f"  stag={self._stagnation_counter}"
                    if self._stagnation_counter > 0 else ""
                )
                print(
                    f"[Iter {t + 1:3d}/{self.T}]  "
                    f"α={alpha:.4f}  "
                    f"gbest={self.gbest_fitness:.6f}  "
                    f"mean={float(np.mean(iter_fitnesses)):.4f}  "
                    f"div={diversity:.4f}  "
                    f"mut={n_mutated_this_iter}"
                    f"{stag_marker}"
                )

        # 最終報告
        print("\n" + "=" * 70)
        print("SOQPSO v6 優化完成")
        print(f"  Gbest Fitness    : {self.gbest_fitness:.6f}")
        print(f"  總重初始化次數   : {self._total_reinits}")
        print(f"  總變異粒子次數   : {self._total_mutations}")
        print(f"  最終多樣性       : {self._compute_diversity():.4f}")
        print("=" * 70)

        return self.gbest_position.copy(), self.gbest_fitness, self.history

    def reset(self):
        """重置優化器狀態。"""
        self.positions = self.rng.uniform(
            self.lb, self.ub, size=(self.M, self.D)
        )
        self.pbest = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf)
        self.gbest_position = None
        self.gbest_fitness = -np.inf
        self._stagnation_counter = 0
        self._prev_gbest_fitness = -np.inf
        self._total_reinits = 0
        self._total_mutations = 0
        self.history = []
