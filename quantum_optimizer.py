"""
==============================================================================
SOQPSOOptimizer — 單目標量子粒子群優化 (SOQPSO)
==============================================================================

極大化單一標量適應度 fitness_score = validity * uniqueness。
  • 支援 CUDA-Q sample_async + mqpu 多 GPU 非同步評估
  • 支援化學先驗限制（bond 角度範圍與角度和約束）
  • Delta 勢阱模型圍繞 mbest (平均最佳位置) 與全域 gbest 進行收斂
==============================================================================
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cudaq
except ImportError:
    cudaq = None


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
        param_lower: float = -np.pi,
        param_upper: float = np.pi,
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
        use_async_sampling: bool = True,
        rank: int = 0,
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
        self.lb = param_lower
        self.ub = param_upper
        self.verbose = verbose
        self.iteration_callback = iteration_callback
        self.stagnation_limit = stagnation_limit
        self.reinit_fraction = reinit_fraction
        self.mutation_prob = mutation_prob
        self.mutation_scale = mutation_scale * (param_upper - param_lower)
        self.alpha_perturb_std = alpha_perturb_std
        self.alpha_stag_boost = alpha_stag_boost
        self.use_chem_constraints = use_chem_constraints
        self.use_async_sampling = use_async_sampling
        self.rank = rank

        self.atom_param_indices = atom_param_indices or list(range(self.D))
        self.bond_param_indices = bond_param_indices or []
        self.bond_angle_pairs: List[Tuple[int, int]] = []
        for pair_start in range(0, len(self.bond_param_indices), 2):
            if pair_start + 1 < len(self.bond_param_indices):
                lhs_idx = self.bond_param_indices[pair_start][0]
                rhs_idx = self.bond_param_indices[pair_start + 1][0]
                self.bond_angle_pairs.append((lhs_idx, rhs_idx))

        self.lb_vec = np.full(self.D, self.lb, dtype=np.float64)
        self.ub_vec = np.full(self.D, self.ub, dtype=np.float64)

        if use_chem_constraints and bond_param_indices:
            for idx, bond_type in bond_param_indices:
                if idx < self.D:
                    if bond_type == 'single_double':
                        self.lb_vec[idx] = 0.0
                        self.ub_vec[idx] = np.pi
                    elif bond_type == 'double_triple':
                        self.lb_vec[idx] = 0.0
                        self.ub_vec[idx] = np.pi / 2.0

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
        positions = self.rng.uniform(self.lb_vec, self.ub_vec, size=(n_samples, self.D))
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
        x_new = np.clip(np.array(x, dtype=np.float64).copy(), self.lb_vec, self.ub_vec)
        if not self.use_chem_constraints:
            return x_new

        for idx0, idx1 in self.bond_angle_pairs:
            total = float(x_new[idx0] + x_new[idx1])
            target_total = float(np.clip(total, np.pi / 2.0, 3.0 * np.pi / 2.0))
            if abs(target_total - total) < 1e-12:
                continue

            ratio = float(x_new[idx0] / total) if abs(total) > 1e-12 else 2.0 / 3.0
            theta0 = float(np.clip(target_total * ratio, self.lb_vec[idx0], self.ub_vec[idx0]))
            theta1 = float(np.clip(target_total - theta0, self.lb_vec[idx1], self.ub_vec[idx1]))

            repaired_total = theta0 + theta1
            if repaired_total < np.pi / 2.0:
                deficit = np.pi / 2.0 - repaired_total
                add1 = min(deficit, self.ub_vec[idx1] - theta1)
                theta1 += add1
                deficit -= add1
                if deficit > 0.0:
                    theta0 = min(self.ub_vec[idx0], theta0 + deficit)
            elif repaired_total > 3.0 * np.pi / 2.0:
                excess = repaired_total - 3.0 * np.pi / 2.0
                cut1 = min(excess, theta1 - self.lb_vec[idx1])
                theta1 -= cut1
                excess -= cut1
                if excess > 0.0:
                    theta0 = max(self.lb_vec[idx0], theta0 - excess)

            x_new[idx0] = theta0
            x_new[idx1] = theta1

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
        noise = self.rng.standard_cauchy(size=n_mutate) * self.mutation_scale
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
        if self.verbose and self.rank == 0:
            print(
                f"  [停滯偵測] 已重初始化 {n_reinit} 個粒子 "
                f"(累計 {self._total_reinits} 次)"
            )

    # ------------------------------------------------------------------
    # 群體評估
    # ------------------------------------------------------------------
    def _get_num_qpus(self) -> int:
        if cudaq is None:
            return 1
        try:
            target = cudaq.get_target()
            value = getattr(target, 'num_qpus', None)
            if callable(value):
                return max(1, int(value()))
            if value is not None:
                return max(1, int(value))
        except Exception:
            pass
        try:
            return max(1, int(cudaq.num_available_gpus()))
        except Exception:
            return 1

    def _evaluate_with_async_sampling(self) -> List[Tuple[int, float, List[str], List[dict]]]:
        if cudaq is None or self.kernel is None or self.decoder is None:
            raise RuntimeError("cudaq / kernel / decoder 未正確設定，無法進行非同步評估")
        if not hasattr(cudaq, 'sample_async'):
            raise RuntimeError("cudaq.sample_async 不可用，請 fallback 至同步評估")

        n_qpus = self._get_num_qpus()
        futures = []
        for i in range(self.M):
            qpu_id = i % n_qpus
            future = cudaq.sample_async(
                self.kernel.kernel_func,
                self.positions[i].tolist(),
                self.kernel.max_atoms,
                shots_count=self.shots,
                qpu_id=qpu_id,
            )
            futures.append((i, future))

        results: List[Tuple[int, float, List[str], List[dict]]] = []
        for i, future in futures:
            counts_result = future.get()
            counts_dict: Dict[str, int] = {}
            for bitstring, count in counts_result.items():
                counts_dict[bitstring.replace(' ', '')] = int(count)

            fitness_score, decoded = self.decoder.compute_fitness(counts_dict)
            smiles_list = [
                record['smiles']
                for record in decoded
                if record.get('valid') and record.get('smiles') and not record.get('partial_valid')
            ]
            results.append((i, float(fitness_score), smiles_list, decoded))

        return results

    def _evaluate_with_fitness_fn(self) -> List[Tuple[int, float, List[str], List[dict]]]:
        if self.fitness_fn is None:
            raise ValueError('fitness_fn 未設定，且無法使用 CUDA-Q 非同步評估。')

        results: List[Tuple[int, float, List[str], List[dict]]] = []
        for i in range(self.M):
            fitness_score, smiles_list, decoded = self.fitness_fn(self.positions[i])
            results.append((i, float(fitness_score), list(smiles_list or []), list(decoded)))
        return results

    def _evaluate_swarm(self) -> List[Tuple[int, float, List[str], List[dict]]]:
        if self.use_async_sampling and self.kernel is not None and self.decoder is not None:
            try:
                return self._evaluate_with_async_sampling()
            except Exception as exc:
                if self.verbose and self.rank == 0:
                    print(f"  [Async 評估警告] 退回同步 fitness_fn：{exc}")
        return self._evaluate_with_fitness_fn()

    # ------------------------------------------------------------------
    # 主優化迴圈
    # ------------------------------------------------------------------
    def optimize(self) -> Tuple[np.ndarray, float, List[dict]]:
        if self.fitness_fn is None and (self.kernel is None or self.decoder is None):
            raise ValueError('至少需要 fitness_fn，或同時提供 kernel 與 decoder。')

        if self.rank == 0:
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
                    if self.verbose and self.rank == 0:
                        print(f'  [Callback 警告] {exc}')

            if self.verbose and self.rank == 0:
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

        if self.rank == 0:
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