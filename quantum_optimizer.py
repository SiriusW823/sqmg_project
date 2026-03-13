"""
==============================================================================
SOQPSOOptimizer — 單目標量子粒子群優化 (SOQPSO)
==============================================================================

極大化單一標量適應度 fitness_score = validity * uniqueness。

  • 支援化學先驗限制（bond 角度範圍與角度和約束）
  • Delta 勢阱模型圍繞 mbest (平均最佳位置) 與全域 gbest 進行收斂
==============================================================================
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class SOQPSOOptimizer:
    """單目標 Quantum Particle Swarm Optimization (Delta 勢阱模型)。"""

    def __init__(
        self,
        n_params: int,
        n_particles: int = 30,
        max_iterations: int = 150,
        fitness_fn: Optional[Callable[[np.ndarray], Tuple[float, List[str], List[dict]]]] = None,
        kernel=None,
        decoder=None,
        shots: int = 1024,
        alpha_max: float = 1.2,
        alpha_min: float = 0.4,
        param_lower: Union[float, np.ndarray] = -np.pi,
        param_upper: Union[float, np.ndarray] = np.pi,
        seed: Optional[int] = None,
        verbose: bool = True,
        iteration_callback: Optional[Callable[[int, dict], None]] = None,
        stagnation_limit: int = 10,
        reinit_fraction: float = 0.2,
        mutation_prob: float = 0.10,
        mutation_scale: float = 0.3,
        alpha_perturb_std: float = 0.05,
        alpha_stag_boost: float = 0.25,
        use_chem_constraints: bool = True,
        atom_param_indices: Optional[List[int]] = None,
        bond_param_indices: Optional[List[Tuple[int, str]]] = None,
    ):
        self.D = n_params
        self.M = n_particles
        self.T = max_iterations
        self.fitness_fn = fitness_fn
        self.kernel = kernel
        self.decoder = decoder
        self.shots = shots if shots is not None else getattr(kernel, 'shots', 1024)
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.verbose = verbose
        self.iteration_callback = iteration_callback
        self.stagnation_limit = stagnation_limit
        self.reinit_fraction = reinit_fraction
        self.mutation_prob = mutation_prob
        self.alpha_perturb_std = alpha_perturb_std
        self.alpha_stag_boost = alpha_stag_boost
        self.use_chem_constraints = use_chem_constraints

        self.atom_param_indices = atom_param_indices or list(range(self.D))
        self.bond_param_indices = bond_param_indices or []
        self.bond_angle_pairs: List[Tuple[int, int]] = []
        for pair_start in range(0, len(self.bond_param_indices), 3):
            if pair_start + 1 < len(self.bond_param_indices):
                lhs_idx = self.bond_param_indices[pair_start][0]
                rhs_idx = self.bond_param_indices[pair_start + 1][0]
                self.bond_angle_pairs.append((lhs_idx, rhs_idx))

        # 【支援陣列型別的邊界】實作化學先驗限制
        if isinstance(param_lower, (float, int)):
            self.param_lower = np.full(self.D, float(param_lower), dtype=np.float64)
        else:
            self.param_lower = np.asarray(param_lower, dtype=np.float64)

        if isinstance(param_upper, (float, int)):
            self.param_upper = np.full(self.D, float(param_upper), dtype=np.float64)
        else:
            self.param_upper = np.asarray(param_upper, dtype=np.float64)

        assert self.param_lower.shape == (self.D,), "param_lower 維度必須等於 n_params"
        assert self.param_upper.shape == (self.D,), "param_upper 維度必須等於 n_params"

        # lb_vec / ub_vec 為實際使用的邊界向量（在 param_lower/upper 基礎上疊加化學限制）
        self.lb_vec = self.param_lower.copy()
        self.ub_vec = self.param_upper.copy()

        if use_chem_constraints and bond_param_indices:
            for idx, bond_type in bond_param_indices:
                if idx < self.D:
                    if bond_type == 'bond_existence':
                        # 控制鍵結是否存在（|00⟩→|10⟩）：範圍 [0, π/2]
                        # 由 sqmg_kernel.get_param_bounds() 已設定；此處取交集確保一致。
                        self.lb_vec[idx] = max(self.lb_vec[idx], 0.0)
                        self.ub_vec[idx] = min(self.ub_vec[idx], np.pi / 2)
                    elif bond_type == 'bond_order':
                        # 控制鍵結階數升級（|10⟩→|11⟩ 單鍵→雙鍵）：範圍 [0, π/2]
                        # 對應 QMG Eq.2，使雙鍵機率 ≤ 50%。
                        self.lb_vec[idx] = max(self.lb_vec[idx], 0.0)
                        self.ub_vec[idx] = min(self.ub_vec[idx], np.pi / 2)
                    elif bond_type == 'bond_triple_order':
                        # 控制鍵結階數升級（|11⟩→|01⟩ 雙鍵→三鍵）：範圍 [0, π/2]
                        # 對應 QMG Eq.3 精神，使三鍵機率 ≤ 50%（比雙鍵更少見）。
                        self.lb_vec[idx] = max(self.lb_vec[idx], 0.0)
                        self.ub_vec[idx] = min(self.ub_vec[idx], np.pi / 2)
                    # 向後相容：舊標籤仍接受，避免已存在的呼叫端報錯
                    elif bond_type in ('single_double', 'double_triple'):
                        self.lb_vec[idx] = max(self.lb_vec[idx], 0.0)
                        self.ub_vec[idx] = min(self.ub_vec[idx], np.pi / 2)

        if use_chem_constraints:
            # 第一個 atom 的第一個 RY 參數限制在 [0, π]，
            # 確保 q[0] 偏向 |1⟩，降低第一個原子輸出 NONE(|000⟩) 的機率
            self.lb_vec[0] = max(self.lb_vec[0], 0.0)
            self.ub_vec[0] = min(self.ub_vec[0], np.pi)

        # mutation_scale 使用向量化範圍
        self.mutation_scale = mutation_scale * (self.ub_vec - self.lb_vec)

        self.rng = np.random.default_rng(seed)
        self.positions = self._random_positions(self.M)
        self.pbest = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf, dtype=np.float64)

        # 全域最佳
        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_fitness: float = -np.inf
        self.gbest_decoded: List[dict] = []

        self.history: List[dict] = []
        self._stagnation_counter = 0
        self._prev_best_score = -np.inf
        self._total_reinits = 0
        self._total_mutations = 0

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------
    def _random_positions(self, n_samples: int) -> np.ndarray:
        """生成隨機初始位置，使用廣播機制適應 Array 邊界。"""
        positions = self.param_lower + self.rng.random((n_samples, self.D)) * (self.param_upper - self.param_lower)
        return np.array([self._apply_chem_constraints(pos) for pos in positions])

    def _get_alpha(self, t: int) -> float:
        progress = t / max(self.T - 1, 1)
        alpha_base = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1.0 + math.cos(math.pi * progress))
        perturbation = self.rng.normal(0, self.alpha_perturb_std)
        stag_boost = self.alpha_stag_boost if self._stagnation_counter >= self.stagnation_limit else 0.0
        alpha = alpha_base + perturbation + stag_boost
        return float(np.clip(alpha, self.alpha_min, self.alpha_max + self.alpha_stag_boost))

    def _compute_mbest(self) -> np.ndarray:
        return np.mean(self.pbest, axis=0)

    def _apply_chem_constraints(self, x: np.ndarray) -> np.ndarray:
        """Bond Existence 參數的化學先驗約束（全上三角拓撲適用版）。

        ── 問題背景 ──
        QMG 論文 Eq.1 原本針對「循序動態電路」設計：atom i 只與之前的 atoms {1,...,i-1}
        形成鍵結，因此每個新原子的所有 bond existence 角之和可以獨立設為 π（等式約束）。

        SQMG 使用全上三角矩陣：每個原子與其餘所有原子均有潛在鍵結，因此同一個
        bond existence 參數（如 bond(0,1)）同時出現在 atom 0 和 atom 1 的約束集合中。
        對兩個原子分別套用「等式約束 sum=π」會形成相互覆寫的矛盾系統，無法同時滿足。

        ── 本實作策略 ──
        改為對每個 bond existence 參數套用「per-bond 上界約束」：
          每個 bond existence 角 θ_exist ≤ π / (N-1)
        語意：每個 bond 最多佔用一個原子總成鍵預算 π 的 1/(N-1)，
        確保任何單一原子不會因某一鍵而耗盡所有成鍵預算。

        這是從 Eq.1 精神導出的可行近似，避免全上三角拓撲下的約束衝突，
        同時保留化學先驗（防止超價）。
        """
        x_new = np.clip(np.array(x, dtype=np.float64).copy(), self.lb_vec, self.ub_vec)
        if not self.use_chem_constraints:
            return x_new

        n_atoms = self.kernel.max_atoms if hasattr(self, 'kernel') and self.kernel else 0
        if n_atoms < 2:
            return x_new

        bond_start_idx = n_atoms * 9

        # per-bond 上界：每個 bond existence 角 ≤ π / (N-1)
        # 直覺：若每個原子最多參與 N-1 個鍵，且每個鍵的貢獻均等，
        # 則單鍵的存在角上限為 π/(N-1)，確保每個原子的成鍵預算不超過 π。
        per_bond_max = np.pi / max(n_atoms - 1, 1)

        n_bonds = n_atoms * (n_atoms - 1) // 2
        for bond_idx in range(n_bonds):
            theta_exist_idx = bond_start_idx + bond_idx * 3  # v7: 3 params/bond (was 2)
            if theta_exist_idx < len(x_new):
                x_new[theta_exist_idx] = np.clip(
                    x_new[theta_exist_idx],
                    self.lb_vec[theta_exist_idx],
                    min(self.ub_vec[theta_exist_idx], per_bond_max),
                )

        return np.clip(x_new, self.lb_vec, self.ub_vec)

    # ------------------------------------------------------------------
    # 粒子位置更新 (Delta 勢阱模型)
    # ------------------------------------------------------------------
    def _update_position(
        self,
        x: np.ndarray,
        pbest_i: np.ndarray,
        gbest: np.ndarray,
        mbest: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        phi = self.rng.uniform(0, 1, size=self.D)
        attractor = phi * pbest_i + (1.0 - phi) * gbest
        u = np.maximum(self.rng.uniform(0, 1, size=self.D), 1e-10)
        quantum_step = alpha * np.abs(mbest - x) * np.log(1.0 / u)
        sign = np.where(self.rng.uniform(0, 1, size=self.D) < 0.5, 1.0, -1.0)
        x_new = attractor + sign * quantum_step
        x_new = np.clip(x_new, self.lb_vec, self.ub_vec)
        return self._apply_chem_constraints(x_new)

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        x_mut = x.copy()
        n_mutate = max(1, int(self.D * self.rng.uniform(0.2, 0.4)))
        dims = self.rng.choice(self.D, size=n_mutate, replace=False)
        noise = self.rng.standard_cauchy(size=n_mutate) * self.mutation_scale[dims]
        x_mut[dims] += noise
        x_mut = np.clip(x_mut, self.lb_vec, self.ub_vec)
        return self._apply_chem_constraints(x_mut)

    def _compute_diversity(self) -> float:
        return float(np.mean(np.std(self.positions, axis=0)))

    # ------------------------------------------------------------------
    # pbest / gbest 更新
    # ------------------------------------------------------------------
    def _update_pbest(self, i: int, fitness: float):
        if fitness > self.pbest_fitness[i]:
            self.pbest[i] = self.positions[i].copy()
            self.pbest_fitness[i] = fitness

    def _update_gbest(self, i: int, fitness: float, decoded: Optional[List[dict]] = None):
        if fitness > self.gbest_fitness:
            self.gbest_position = self.positions[i].copy()
            self.gbest_fitness = fitness
            self.gbest_decoded = decoded if decoded is not None else []

    # ------------------------------------------------------------------
    # 停滯偵測與重初始化
    # ------------------------------------------------------------------
    def _update_stagnation(self, best_score: float):
        if best_score > self._prev_best_score + 1e-8:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1
        self._prev_best_score = best_score

    def _check_and_reinit(self):
        if self._stagnation_counter < self.stagnation_limit:
            return

        n_reinit = max(1, int(self.M * self.reinit_fraction))
        worst_indices = np.argsort(self.pbest_fitness)[:n_reinit]

        for offset, idx in enumerate(worst_indices):
            if offset < n_reinit // 2 or self.gbest_position is None:
                self.positions[idx] = self.rng.uniform(self.lb_vec, self.ub_vec)
                self.positions[idx] = self._apply_chem_constraints(self.positions[idx])
            else:
                noise = self.rng.normal(0.0, 0.15 * (self.ub_vec - self.lb_vec), size=self.D)
                self.positions[idx] = np.clip(self.gbest_position + noise, self.lb_vec, self.ub_vec)
                self.positions[idx] = self._apply_chem_constraints(self.positions[idx])
            self.pbest[idx] = self.positions[idx].copy()
            self.pbest_fitness[idx] = -np.inf

        self._stagnation_counter = 0
        self._total_reinits += 1
        if self.verbose:
            print(
                f"  [停滯偵測] 已重初始化 {n_reinit} 個粒子 "
                f"(累計 {self._total_reinits} 次)"
            )

    # ------------------------------------------------------------------
    # 群體評估
    # ------------------------------------------------------------------
    def _evaluate_swarm(self) -> List[Tuple[int, float, List[str], List[dict]]]:
        if self.fitness_fn is None:
            raise ValueError('fitness_fn 未設定。')

        results: List[Tuple[int, float, List[str], List[dict]]] = []
        for i in range(self.M):
            fitness_score, smiles_list, decoded = self.fitness_fn(self.positions[i])
            results.append((i, float(fitness_score), list(smiles_list or []), list(decoded)))
        return results

    # ------------------------------------------------------------------
    # 主優化迴圈
    # ------------------------------------------------------------------
    def optimize(self) -> Tuple[np.ndarray, float, List[dict]]:
        if self.fitness_fn is None and (self.kernel is None or self.decoder is None):
            raise ValueError('至少需要 fitness_fn，或同時提供 kernel 與 decoder。')

        if self.verbose:
            print('=' * 70)
            print('SOQPSO 單目標量子粒子群優化啟動')
            print(f'  粒子數 (M)       : {self.M}')
            print(f'  參數維度 (D)     : {self.D}')
            print(f'  最大迭代 (T)     : {self.T}')
            print(f'  α 範圍           : {self.alpha_max} → {self.alpha_min}')
            print(f'  有效化學限制     : {self.use_chem_constraints}')
            print(f'  停滯門檻         : {self.stagnation_limit} 代')
            print(f'  重初始化比例     : {self.reinit_fraction:.0%}')
            print(f'  變異機率         : {self.mutation_prob:.0%}')
            print('=' * 70)

        # 初始評估
        initial_results = self._evaluate_swarm()
        for i, fitness, _smiles, decoded in initial_results:
            self._update_pbest(i, fitness)
            self._update_gbest(i, fitness, decoded)

        if self.gbest_position is not None:
            self._prev_best_score = self.gbest_fitness

        # 迭代主迴圈
        for t in range(self.T):
            alpha = self._get_alpha(t)
            mbest = self._compute_mbest()
            gbest = self.gbest_position if self.gbest_position is not None else self.positions[0]
            n_mutated_this_iter = 0

            for i in range(self.M):
                self.positions[i] = self._update_position(
                    x=self.positions[i],
                    pbest_i=self.pbest[i],
                    gbest=gbest,
                    mbest=mbest,
                    alpha=alpha,
                )

                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    n_mutated_this_iter += 1

            iter_results = self._evaluate_swarm()
            iter_fitnesses: List[float] = []

            for i, fitness, _smiles, decoded in iter_results:
                iter_fitnesses.append(fitness)
                self._update_pbest(i, fitness)
                self._update_gbest(i, fitness, decoded)

            self._total_mutations += n_mutated_this_iter
            self._update_stagnation(self.gbest_fitness)
            self._check_and_reinit()
            diversity = self._compute_diversity()

            iter_record = {
                'iteration': t,
                'alpha': alpha,
                'gbest_fitness': self.gbest_fitness,
                'gbest_params': self.gbest_position.copy() if self.gbest_position is not None else np.zeros(self.D),
                'gbest_decoded': list(self.gbest_decoded),
                'mean_fitness': float(np.mean(iter_fitnesses)) if iter_fitnesses else 0.0,
                'max_fitness': float(np.max(iter_fitnesses)) if iter_fitnesses else 0.0,
                'diversity': diversity,
                'stagnation_counter': self._stagnation_counter,
                'n_mutated': n_mutated_this_iter,
            }
            self.history.append(iter_record)

            if self.iteration_callback is not None:
                try:
                    self.iteration_callback(t, iter_record)
                except Exception as exc:
                    if self.verbose:
                        print(f'  [Callback 警告] {exc}')

            if self.verbose:
                print(
                    f"[Iter {t + 1:3d}/{self.T}] "
                    f"α={alpha:.4f} "
                    f"gbest={self.gbest_fitness:.6f} "
                    f"mean={iter_record['mean_fitness']:.4f} "
                    f"max={iter_record['max_fitness']:.4f} "
                    f"div={diversity:.4f} "
                    f"mut={n_mutated_this_iter}"
                )

        # 回傳最終結果
        if self.gbest_position is None:
            return np.zeros(self.D, dtype=np.float64), 0.0, self.history

        if self.verbose:
            print('\n' + '=' * 70)
            print('SOQPSO 優化完成')
            print(f'  Best fitness      : {self.gbest_fitness:.6f}')
            print(f'  總重初始化次數   : {self._total_reinits}')
            print(f'  總變異粒子次數   : {self._total_mutations}')
            print('=' * 70)

        return self.gbest_position.copy(), self.gbest_fitness, self.history

    def reset(self):
        self.positions = self._random_positions(self.M)
        self.pbest = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf, dtype=np.float64)
        self.gbest_position = None
        self.gbest_fitness = -np.inf
        self.gbest_decoded = []
        self.history = []
        self._stagnation_counter = 0
        self._prev_best_score = -np.inf
        self._total_reinits = 0
        self._total_mutations = 0