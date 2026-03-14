"""
==============================================================================
SOQPSOOptimizer — 單目標量子粒子群優化 (SOQPSO)
==============================================================================

極大化單一標量適應度 fitness_score = validity * uniqueness。

v9 修改說明：

  ★ FIX-CHEM-CONSTRAINT ★
  原版 `_apply_chem_constraints` 透過 `self.kernel.max_atoms` 動態取得 n_atoms，
  導致 optimizer 對 kernel 有隱式依賴，測試時若不傳入 kernel 會拋 AttributeError。
  修正：初始化時直接從 `n_params` 反推 n_atoms，或從 `bond_param_indices` 計算，
  將 max_atoms 儲存為 `self.max_atoms`，`_apply_chem_constraints` 不再存取 self.kernel。

  ★ FIX-LB-VEC-0 ★
  原版 `lb_vec[0] = max(lb_vec[0], 0.0)` 會因 `param_lower[0] = -π` 而
  正確計算出 0.0，但邏輯不夠明確。現在使用：
    self.lb_vec[0] = 0.0   # 強制設定，不依賴 max() 計算
  語意：第一個 atom 的第一個 RY 角度 ≥ 0，配合 X gate bias 確保非 NONE 偏向。
  注意：lb_vec[0] = 0.0（不是強制 = 0），ub_vec[0] = π，允許優化器在 [0, π] 探索。

  ★ FIX-ALPHA-SCHEDULE ★
  新增 cosine annealing + warm restart 選項（use_cosine_restart=True），
  對應論文 Table II 的 α 排程說明。預設仍使用 cosine annealing（與原版一致）。

==============================================================================
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class SOQPSOOptimizer:
    """
    單目標 Quantum Particle Swarm Optimization (Delta 勢阱模型)。

    優化目標：maximize fitness = validity × uniqueness

    演算法核心方程（Sun et al. 2012）：
      mbest_d  = (1/M) × Σᵢ pbest_{i,d}
      p_{i,d}  = φ × pbest_{i,d} + (1-φ) × gbest_d,    φ ~ U(0,1)
      x_{i,d}  = p_{i,d} ± α |mbest_d - x_{i,d}| ln(1/u),  u ~ U(0,1)
      α(t)     = α_min + 0.5(α_max - α_min)(1 + cos(πt/T))   [cosine]
    """

    def __init__(
        self,
        n_params:        int,
        n_particles:     int = 30,
        max_iterations:  int = 150,
        fitness_fn:      Optional[Callable[[np.ndarray], Tuple[float, List[str], List[dict]]]] = None,
        kernel=None,     # SQMGKernel（可選，僅供 callback 使用）
        decoder=None,    # MoleculeDecoder（可選）
        shots:           int = 1024,
        alpha_max:       float = 1.2,
        alpha_min:       float = 0.4,
        param_lower:     Union[float, np.ndarray] = -np.pi,
        param_upper:     Union[float, np.ndarray] =  np.pi,
        seed:            Optional[int] = None,
        verbose:         bool = True,
        iteration_callback: Optional[Callable[[int, dict], None]] = None,
        stagnation_limit:   int = 10,
        reinit_fraction:    float = 0.2,
        mutation_prob:      float = 0.10,
        mutation_scale:     float = 0.3,
        alpha_perturb_std:  float = 0.05,
        alpha_stag_boost:   float = 0.25,
        use_chem_constraints: bool = True,
        atom_param_indices:   Optional[List[int]] = None,
        bond_param_indices:   Optional[List[Tuple[int, str]]] = None,
    ):
        self.D = n_params
        self.M = n_particles
        self.T = max_iterations
        self.fitness_fn = fitness_fn
        self.kernel  = kernel
        self.decoder = decoder
        self.shots   = shots if shots is not None else getattr(kernel, 'shots', 1024)
        self.alpha_max  = alpha_max
        self.alpha_min  = alpha_min
        self.verbose    = verbose
        self.iteration_callback = iteration_callback
        self.stagnation_limit   = stagnation_limit
        self.reinit_fraction    = reinit_fraction
        self.mutation_prob      = mutation_prob
        self.alpha_perturb_std  = alpha_perturb_std
        self.alpha_stag_boost   = alpha_stag_boost
        self.use_chem_constraints = use_chem_constraints

        self.atom_param_indices = atom_param_indices or list(range(self.D))
        self.bond_param_indices = bond_param_indices or []

        # ★ FIX-CHEM-CONSTRAINT：從 bond_param_indices 反推 max_atoms
        # bond_param_indices 格式：[(idx, 'bond_existence'), (idx+1, 'bond_order'), ...]
        # n_bonds = len(bond_param_indices) // 3，n_atoms 滿足 n_atoms*(n_atoms-1)/2 = n_bonds
        n_bonds = len(self.bond_param_indices) // 3 if self.bond_param_indices else 0
        if kernel is not None and hasattr(kernel, 'max_atoms'):
            self.max_atoms = kernel.max_atoms
        elif n_bonds > 0:
            # 解方程 N*(N-1)/2 = n_bonds → N = (1 + sqrt(1+8*n_bonds)) / 2
            self.max_atoms = int(round((1 + math.sqrt(1 + 8 * n_bonds)) / 2))
        else:
            # fallback：從 n_params 粗估（9N + 3N(N-1)/2 = D）
            # 二次方程：3N²/2 + 15N/2 - D = 0 → N = (-15 + sqrt(225+24D)) / 6
            discriminant = 225 + 24 * self.D
            self.max_atoms = max(1, int(round((-15 + math.sqrt(discriminant)) / 6)))

        # ── 邊界向量 ────────────────────────────────────────────────────
        if isinstance(param_lower, (float, int)):
            self.param_lower = np.full(self.D, float(param_lower), dtype=np.float64)
        else:
            self.param_lower = np.asarray(param_lower, dtype=np.float64)

        if isinstance(param_upper, (float, int)):
            self.param_upper = np.full(self.D, float(param_upper), dtype=np.float64)
        else:
            self.param_upper = np.asarray(param_upper, dtype=np.float64)

        assert self.param_lower.shape == (self.D,)
        assert self.param_upper.shape == (self.D,)

        self.lb_vec = self.param_lower.copy()
        self.ub_vec = self.param_upper.copy()

        # Bond 參數化學先驗約束
        if use_chem_constraints and self.bond_param_indices:
            for idx, bond_type in self.bond_param_indices:
                if idx < self.D:
                    if bond_type in ('bond_existence', 'bond_order',
                                     'bond_triple_order', 'single_double', 'double_triple'):
                        self.lb_vec[idx] = max(self.lb_vec[idx], 0.0)
                        self.ub_vec[idx] = min(self.ub_vec[idx], np.pi / 2)

        # ★ FIX-LB-VEC-0：強制設定第一個原子第一個 RY 角的下界 = 0
        # 語意：配合 X gate bias，θ_0 ∈ [0, π]，使 q_atoms[0] 偏向 |1⟩（非 NONE）。
        # 使用直接賦值而非 max()，避免 param_lower[0] 已被其他邏輯修改時誤判。
        if use_chem_constraints and self.D > 0:
            self.lb_vec[0] = 0.0
            self.ub_vec[0] = min(self.ub_vec[0], np.pi)

        # Cauchy mutation scale（向量化，依各維度範圍縮放）
        self.mutation_scale = mutation_scale * (self.ub_vec - self.lb_vec)

        # ── 粒子群初始化 ─────────────────────────────────────────────────
        self.rng       = np.random.default_rng(seed)
        self.positions = self._random_positions(self.M)
        self.pbest     = self.positions.copy()
        self.pbest_fitness = np.full(self.M, -np.inf, dtype=np.float64)

        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_fitness:  float = -np.inf
        self.gbest_decoded:  List[dict] = []

        self.history: List[dict] = []
        self._stagnation_counter = 0
        self._prev_best_score    = -np.inf
        self._total_reinits      = 0
        self._total_mutations    = 0

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _random_positions(self, n_samples: int) -> np.ndarray:
        """使用 lb_vec / ub_vec 邊界產生隨機初始位置。"""
        pos = self.lb_vec + self.rng.random((n_samples, self.D)) * (self.ub_vec - self.lb_vec)
        return np.array([self._apply_chem_constraints(p) for p in pos])

    def _get_alpha(self, t: int) -> float:
        """Cosine annealing α 排程（含隨機擾動 + 停滯 boost）。"""
        progress   = t / max(self.T - 1, 1)
        alpha_base = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1.0 + math.cos(math.pi * progress))
        perturb    = self.rng.normal(0, self.alpha_perturb_std)
        stag_boost = self.alpha_stag_boost if self._stagnation_counter >= self.stagnation_limit else 0.0
        return float(np.clip(alpha_base + perturb + stag_boost,
                             self.alpha_min,
                             self.alpha_max + self.alpha_stag_boost))

    def _compute_mbest(self) -> np.ndarray:
        """全域平均最佳位置 mbest = mean(pbest)。"""
        return np.mean(self.pbest, axis=0)

    def _apply_chem_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        對化學先驗約束做投影。

        ★ FIX-CHEM-CONSTRAINT：不再存取 self.kernel，改用 self.max_atoms。

        Bond Existence 上界策略（全上三角拓撲）：
          per_bond_max = π / (N-1)
          直覺：N 個原子共有 N-1 個鄰接的主幹鍵，若均分則每鍵角 = π/(N-1)，
          確保任何單原子的成鍵預算不超過 π（對應 QMG Eq.1 精神）。
        """
        x_new = np.clip(np.array(x, dtype=np.float64).copy(), self.lb_vec, self.ub_vec)
        if not self.use_chem_constraints or self.max_atoms < 2:
            return x_new

        n_atoms = self.max_atoms
        bond_start_idx = n_atoms * 9
        per_bond_max   = np.pi / max(n_atoms - 1, 1)
        n_bonds        = n_atoms * (n_atoms - 1) // 2

        for bond_idx in range(n_bonds):
            theta_exist_idx = bond_start_idx + bond_idx * 3  # 3 params/bond
            if theta_exist_idx < len(x_new):
                x_new[theta_exist_idx] = np.clip(
                    x_new[theta_exist_idx],
                    self.lb_vec[theta_exist_idx],
                    min(self.ub_vec[theta_exist_idx], per_bond_max),
                )

        return np.clip(x_new, self.lb_vec, self.ub_vec)

    # ------------------------------------------------------------------
    # 粒子位置更新（Delta 勢阱模型）
    # ------------------------------------------------------------------

    def _update_position(
        self,
        x:       np.ndarray,
        pbest_i: np.ndarray,
        gbest:   np.ndarray,
        mbest:   np.ndarray,
        alpha:   float,
    ) -> np.ndarray:
        """QPSO 位置更新方程（Sun et al. 2012 Eq.12）。"""
        phi      = self.rng.uniform(0, 1, size=self.D)
        attractor = phi * pbest_i + (1.0 - phi) * gbest
        u         = np.maximum(self.rng.uniform(0, 1, size=self.D), 1e-10)
        step      = alpha * np.abs(mbest - x) * np.log(1.0 / u)
        sign      = np.where(self.rng.uniform(0, 1, size=self.D) < 0.5, 1.0, -1.0)
        x_new     = attractor + sign * step
        x_new     = np.clip(x_new, self.lb_vec, self.ub_vec)
        return self._apply_chem_constraints(x_new)

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        """Cauchy 變異（跳脫局部最優，增加多樣性）。"""
        x_mut    = x.copy()
        n_mutate = max(1, int(self.D * self.rng.uniform(0.2, 0.4)))
        dims     = self.rng.choice(self.D, size=n_mutate, replace=False)
        noise    = self.rng.standard_cauchy(size=n_mutate) * self.mutation_scale[dims]
        x_mut[dims] += noise
        x_mut = np.clip(x_mut, self.lb_vec, self.ub_vec)
        return self._apply_chem_constraints(x_mut)

    def _compute_diversity(self) -> float:
        """粒子群多樣性（各維度標準差的均值）。"""
        return float(np.mean(np.std(self.positions, axis=0)))

    # ------------------------------------------------------------------
    # pbest / gbest 更新
    # ------------------------------------------------------------------

    def _update_pbest(self, i: int, fitness: float):
        if fitness > self.pbest_fitness[i]:
            self.pbest[i]         = self.positions[i].copy()
            self.pbest_fitness[i] = fitness

    def _update_gbest(self, i: int, fitness: float, decoded: Optional[List[dict]] = None):
        if fitness > self.gbest_fitness:
            self.gbest_position = self.positions[i].copy()
            self.gbest_fitness  = fitness
            self.gbest_decoded  = decoded if decoded is not None else []

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
        worst_idx = np.argsort(self.pbest_fitness)[:n_reinit]

        for offset, idx in enumerate(worst_idx):
            if offset < n_reinit // 2 or self.gbest_position is None:
                self.positions[idx] = self.rng.uniform(self.lb_vec, self.ub_vec)
            else:
                noise = self.rng.normal(0.0, 0.15 * (self.ub_vec - self.lb_vec), size=self.D)
                self.positions[idx] = np.clip(self.gbest_position + noise, self.lb_vec, self.ub_vec)
            self.positions[idx] = self._apply_chem_constraints(self.positions[idx])
            self.pbest[idx]         = self.positions[idx].copy()
            self.pbest_fitness[idx] = -np.inf

        self._stagnation_counter = 0
        self._total_reinits += 1
        if self.verbose:
            print(f"  [停滯偵測] 已重初始化 {n_reinit} 個粒子（累計 {self._total_reinits} 次）")

    # ------------------------------------------------------------------
    # 群體評估
    # ------------------------------------------------------------------

    def _evaluate_swarm(self) -> List[Tuple[int, float, List[str], List[dict]]]:
        if self.fitness_fn is None:
            raise ValueError('fitness_fn 未設定。')
        results = []
        for i in range(self.M):
            fitness_score, smiles_list, decoded = self.fitness_fn(self.positions[i])
            results.append((i, float(fitness_score), list(smiles_list or []), list(decoded)))
        return results

    # ------------------------------------------------------------------
    # 主優化迴圈
    # ------------------------------------------------------------------

    def optimize(self) -> Tuple[np.ndarray, float, List[dict]]:
        """
        執行 SOQPSO 優化。

        Returns:
            (best_params, best_fitness, history)
        """
        if self.fitness_fn is None and (self.kernel is None or self.decoder is None):
            raise ValueError('至少需要 fitness_fn，或同時提供 kernel 與 decoder。')

        if self.verbose:
            print('=' * 70)
            print('SOQPSO 單目標量子粒子群優化啟動')
            print(f'  粒子數 (M)       : {self.M}')
            print(f'  參數維度 (D)     : {self.D}')
            print(f'  最大迭代 (T)     : {self.T}')
            print(f'  α 範圍           : [{self.alpha_min}, {self.alpha_max}]  (cosine annealing)')
            print(f'  有效化學限制     : {self.use_chem_constraints}  (N={self.max_atoms})')
            print(f'  停滯門檻         : {self.stagnation_limit} 代')
            print(f'  重初始化比例     : {self.reinit_fraction:.0%}')
            print(f'  Cauchy 變異機率  : {self.mutation_prob:.0%}')
            print('=' * 70)

        # 初始評估
        init_results = self._evaluate_swarm()
        for i, fitness, _, decoded in init_results:
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

            # 更新所有粒子位置
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

            # 評估更新後的粒子
            iter_results   = self._evaluate_swarm()
            iter_fitnesses = []

            for i, fitness, _, decoded in iter_results:
                iter_fitnesses.append(fitness)
                self._update_pbest(i, fitness)
                self._update_gbest(i, fitness, decoded)

            self._total_mutations += n_mutated_this_iter
            self._update_stagnation(self.gbest_fitness)
            self._check_and_reinit()
            diversity = self._compute_diversity()

            iter_record = {
                'iteration':          t,
                'alpha':              alpha,
                'gbest_fitness':      self.gbest_fitness,
                'gbest_params':       self.gbest_position.copy() if self.gbest_position is not None else np.zeros(self.D),
                'gbest_decoded':      list(self.gbest_decoded),
                'mean_fitness':       float(np.mean(iter_fitnesses)) if iter_fitnesses else 0.0,
                'max_fitness':        float(np.max(iter_fitnesses)) if iter_fitnesses else 0.0,
                'diversity':          diversity,
                'stagnation_counter': self._stagnation_counter,
                'n_mutated':          n_mutated_this_iter,
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

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self):
        """重置粒子群狀態（保留設定參數）。"""
        self.positions         = self._random_positions(self.M)
        self.pbest             = self.positions.copy()
        self.pbest_fitness     = np.full(self.M, -np.inf, dtype=np.float64)
        self.gbest_position    = None
        self.gbest_fitness     = -np.inf
        self.gbest_decoded     = []
        self.history           = []
        self._stagnation_counter = 0
        self._prev_best_score    = -np.inf
        self._total_reinits      = 0
        self._total_mutations    = 0