"""
==============================================================================
MOQPSOOptimizer — 多目標量子粒子群優化 (MOQPSO)
==============================================================================
# MODIFIED: FIX-P0-1, FIX-P0-2, FIX-P1-1, FIX-QED, FIX-HYPER, FIX-CHEM, FIX-GPU

本模組提供兩個核心元件：
  1. ParetoArchive    — 維護 validity / uniqueness 的非支配解集合
  2. MOQPSOOptimizer  — 僅使用量子取樣回饋的多目標 QPSO 優化器

設計重點：
  • 目標值固定為 (validity, uniqueness)
  • 支援 CUDA-Q sample_async + mqpu 多 GPU 非同步評估
  • 支援化學先驗限制（bond 角度範圍與角度和約束）
==============================================================================
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cudaq
except ImportError:
    cudaq = None


ObjectiveTuple = Tuple[float, float]


class ParetoArchive:
    """維護 validity / uniqueness 的 Pareto Archive。"""

    def __init__(self):
        self.entries: List[Dict] = []

    @staticmethod
    def dominates(lhs: Dict[str, float], rhs: Dict[str, float]) -> bool:
        return (
            lhs['validity'] >= rhs['validity']
            and lhs['uniqueness'] >= rhs['uniqueness']
            and (
                lhs['validity'] > rhs['validity']
                or lhs['uniqueness'] > rhs['uniqueness']
            )
        )

    @staticmethod
    def compromise_score(objectives: Dict[str, float]) -> float:
        validity = float(objectives.get('validity', 0.0))
        uniqueness = float(objectives.get('uniqueness', 0.0))
        return validity * uniqueness + 1e-3 * (validity + uniqueness)

    def try_add(
        self,
        params: np.ndarray,
        objectives: Dict[str, float],
        smiles: Optional[List[str]] = None,
    ) -> bool:
        """嘗試加入一個新候選解，並保留其有效 SMILES。"""
        candidate = {
            'params': np.array(params, dtype=np.float64).copy(),
            'objectives': {
                'validity': float(objectives.get('validity', 0.0)),
                'uniqueness': float(objectives.get('uniqueness', 0.0)),
            },
            'smiles': sorted({s for s in (smiles or []) if s}),
        }

        merged_entries: List[Dict] = []
        for entry in self.entries:
            same_params = np.allclose(entry['params'], candidate['params'])
            same_objectives = (
                entry['objectives']['validity'] == candidate['objectives']['validity']
                and entry['objectives']['uniqueness'] == candidate['objectives']['uniqueness']
            )
            if same_params and same_objectives:
                entry['smiles'] = sorted(set(entry.get('smiles', [])) | set(candidate['smiles']))
                return True
            merged_entries.append(entry)

        if any(self.dominates(entry['objectives'], candidate['objectives']) for entry in merged_entries):
            return False

        survivors: List[Dict] = []
        for entry in merged_entries:
            if self.dominates(candidate['objectives'], entry['objectives']):
                continue
            survivors.append(entry)

        survivors.append(candidate)
        self.entries = survivors
        return True

    def best_compromise(self) -> Optional[Dict]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda item: self.compromise_score(item['objectives']))

    def best_metric(self, metric: str) -> float:
        if not self.entries:
            return 0.0
        return float(max(entry['objectives'].get(metric, 0.0) for entry in self.entries))

    def as_sorted_list(self) -> List[Dict]:
        return sorted(
            self.entries,
            key=lambda item: self.compromise_score(item['objectives']),
            reverse=True,
        )

    def __len__(self) -> int:
        return len(self.entries)


class MOQPSOOptimizer:
    """多目標 Quantum Particle Swarm Optimization。"""

    def __init__(
        self,
        n_params: int,
        n_particles: int = 30,
        max_iterations: int = 150,
        fitness_fn: Optional[Callable[[np.ndarray], Tuple[float, float, List[str]]]] = None,
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
        # [FIX-HYPER] 調整停滯與重初始化超參數
        stagnation_limit: int = 10,
        reinit_fraction: float = 0.2,
        mutation_prob: float = 0.10,
        mutation_scale: float = 0.3,
        alpha_perturb_std: float = 0.05,
        alpha_stag_boost: float = 0.25,
        # [FIX-CHEM] Chemistry Constraints 參數
        use_chem_constraints: bool = True,
        atom_param_indices: Optional[List[int]] = None,
        bond_param_indices: Optional[List[Tuple[int, str]]] = None,
        use_async_sampling: bool = True,
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

        self.atom_param_indices = atom_param_indices or list(range(self.D))
        self.bond_param_indices = bond_param_indices or []
        self.bond_angle_pairs: List[Tuple[int, int]] = []
        for pair_start in range(0, len(self.bond_param_indices), 2):
            if pair_start + 1 < len(self.bond_param_indices):
                lhs_idx = self.bond_param_indices[pair_start][0]
                rhs_idx = self.bond_param_indices[pair_start + 1][0]
                self.bond_angle_pairs.append((lhs_idx, rhs_idx))

        # [FIX-CHEM] 建立非均勻邊界向量（Chemistry Constraints）
        self.lb_vec = np.full(self.D, self.lb, dtype=np.float64)
        self.ub_vec = np.full(self.D, self.ub, dtype=np.float64)

        if use_chem_constraints and bond_param_indices:
            for idx, bond_type in bond_param_indices:
                if idx < self.D:
                    if bond_type == 'single_double':
                        # Single/Double 切換角：限制在 [0, π]
                        self.lb_vec[idx] = 0.0
                        self.ub_vec[idx] = np.pi
                    elif bond_type == 'double_triple':
                        # Double/Triple 切換角：限制在 [0, π/2]
                        self.lb_vec[idx] = 0.0
                        self.ub_vec[idx] = np.pi / 2.0

        self.rng = np.random.default_rng(seed)
        self.positions = self._random_positions(self.M)
        self.pbest = self.positions.copy()
        self.pbest_objectives: List[Optional[ObjectiveTuple]] = [None for _ in range(self.M)]

        self.archive = ParetoArchive()
        self.history: List[dict] = []
        self._stagnation_counter = 0
        self._prev_best_score = -np.inf
        self._total_reinits = 0
        self._total_mutations = 0

    @staticmethod
    def _dominates(lhs: ObjectiveTuple, rhs: ObjectiveTuple) -> bool:
        return lhs[0] >= rhs[0] and lhs[1] >= rhs[1] and (lhs[0] > rhs[0] or lhs[1] > rhs[1])

    @staticmethod
    def _score_objective(obj: ObjectiveTuple) -> float:
        return obj[0] * obj[1] + 1e-3 * (obj[0] + obj[1])

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
        source = []
        for i in range(self.M):
            if self.pbest_objectives[i] is None:
                source.append(self.positions[i])
            else:
                source.append(self.pbest[i])
        return np.mean(np.asarray(source), axis=0)

    def _select_leader(self) -> np.ndarray:
        if len(self.archive) == 0:
            return self.positions[0].copy()
        ranked = self.archive.as_sorted_list()
        k = min(len(ranked), max(1, len(ranked) // 3))
        leader = ranked[self.rng.integers(0, k)]
        return leader['params'].copy()

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

    def _update_position(
        self,
        x: np.ndarray,
        pbest_i: np.ndarray,
        leader: np.ndarray,
        mbest: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        phi = self.rng.uniform(0, 1, size=self.D)
        attractor = phi * pbest_i + (1.0 - phi) * leader
        u = np.maximum(self.rng.uniform(0, 1, size=self.D), 1e-10)
        quantum_step = alpha * np.abs(mbest - x) * np.log(1.0 / u)
        sign = np.where(self.rng.uniform(0, 1, size=self.D) < 0.5, 1.0, -1.0)
        x_new = attractor + sign * quantum_step
        # [FIX-CHEM] 使用非均勻邊界向量（支援 Chemistry Constraints）
        x_new = np.clip(x_new, self.lb_vec, self.ub_vec)
        return self._apply_chem_constraints(x_new)

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        x_mut = x.copy()
        n_mutate = max(1, int(self.D * self.rng.uniform(0.2, 0.4)))
        dims = self.rng.choice(self.D, size=n_mutate, replace=False)
        noise = self.rng.standard_cauchy(size=n_mutate) * self.mutation_scale
        x_mut[dims] += noise
        # [FIX-CHEM] 使用非均勻邊界向量
        x_mut = np.clip(x_mut, self.lb_vec, self.ub_vec)
        return self._apply_chem_constraints(x_mut)

    def _compute_diversity(self) -> float:
        return float(np.mean(np.std(self.positions, axis=0)))

    def _update_pbest(self, i: int, obj: ObjectiveTuple):
        current = self.pbest_objectives[i]
        if current is None or self._dominates(obj, current):
            self.pbest[i] = self.positions[i].copy()
            self.pbest_objectives[i] = obj
            return

        if not self._dominates(current, obj) and self._score_objective(obj) > self._score_objective(current):
            self.pbest[i] = self.positions[i].copy()
            self.pbest_objectives[i] = obj

    def _best_entry(self) -> Optional[Dict]:
        return self.archive.best_compromise()

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
        scores = []
        for obj in self.pbest_objectives:
            scores.append(self._score_objective(obj) if obj is not None else -np.inf)
        worst_indices = np.argsort(np.asarray(scores))[:n_reinit]
        best_entry = self._best_entry()

        for offset, idx in enumerate(worst_indices):
            if offset < n_reinit // 2 or best_entry is None:
                # [FIX-CHEM] 使用非均勻邊界初始化
                self.positions[idx] = self.rng.uniform(self.lb_vec, self.ub_vec)
            else:
                noise = self.rng.normal(0.0, 0.15 * (self.ub_vec - self.lb_vec), size=self.D)
                ref = best_entry['params']
                # [FIX-CHEM] 使用非均勻邊界裁剪
                self.positions[idx] = np.clip(ref + noise, self.lb_vec, self.ub_vec)
                self.positions[idx] = self._apply_chem_constraints(self.positions[idx])
            self.pbest[idx] = self.positions[idx].copy()
            self.pbest_objectives[idx] = None

        self._stagnation_counter = 0
        self._total_reinits += 1
        if self.verbose:
            print(
                f"  [停滯偵測] 已重初始化 {n_reinit} 個粒子 "
                f"(累計 {self._total_reinits} 次)"
            )

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

    def _evaluate_with_async_sampling(self) -> List[Tuple[int, ObjectiveTuple, List[str]]]:
        if cudaq is None or self.kernel is None or self.decoder is None or not hasattr(cudaq, 'sample_async'):
            return []

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

        results: List[Tuple[int, ObjectiveTuple, List[str]]] = []
        for i, future in futures:
            counts_result = future.get()
            counts_dict: Dict[str, int] = {}
            for bitstring, count in counts_result.items():
                counts_dict[bitstring.replace(' ', '')] = int(count)

            (validity, uniqueness), decoded = self.decoder.compute_fitness(counts_dict)
            smiles_list = [
                record['smiles']
                for record in decoded
                if record.get('valid') and record.get('smiles') and not record.get('partial_valid')
            ]
            results.append((i, (float(validity), float(uniqueness)), smiles_list))

        return results

    def _evaluate_with_fitness_fn(self) -> List[Tuple[int, ObjectiveTuple, List[str]]]:
        if self.fitness_fn is None:
            raise ValueError('fitness_fn 未設定，且無法使用 CUDA-Q 非同步評估。')

        results: List[Tuple[int, ObjectiveTuple, List[str]]] = []
        for i in range(self.M):
            validity, uniqueness, smiles_list = self.fitness_fn(self.positions[i])
            results.append((i, (float(validity), float(uniqueness)), list(smiles_list or [])))
        return results

    def _evaluate_swarm(self) -> List[Tuple[int, ObjectiveTuple, List[str]]]:
        if self.use_async_sampling and self.kernel is not None and self.decoder is not None:
            try:
                return self._evaluate_with_async_sampling()
            except Exception as exc:
                if self.verbose:
                    print(f"  [Async 評估警告] 退回同步 fitness_fn：{exc}")
        return self._evaluate_with_fitness_fn()

    def optimize(self) -> Tuple[np.ndarray, Dict[str, float], List[dict]]:
        if self.fitness_fn is None and (self.kernel is None or self.decoder is None):
            raise ValueError('至少需要 fitness_fn，或同時提供 kernel 與 decoder。')

        print('=' * 70)
        print('MOQPSO 多目標量子粒子群優化啟動')
        print(f'  粒子數 (M)       : {self.M}')
        print(f'  參數維度 (D)     : {self.D}')
        print(f'  最大迭代 (T)     : {self.T}')
        print(f'  α 範圍           : {self.alpha_max} → {self.alpha_min}')
        print(f'  有效化學限制     : {self.use_chem_constraints}')
        print(f'  停滯門檻         : {self.stagnation_limit} 代')
        print(f'  重初始化比例     : {self.reinit_fraction:.0%}')
        print(f'  變異機率         : {self.mutation_prob:.0%}')
        print('=' * 70)

        initial_results = self._evaluate_swarm()
        for i, obj, smiles_list in initial_results:
            self._update_pbest(i, obj)
            # [FIX-P0-2] 解構三個回傳值，並將 smiles 傳入 archive
            init_smiles = smiles_list
            self.archive.try_add(
                params=self.positions[i],
                objectives={'validity': obj[0], 'uniqueness': obj[1]},
                smiles=init_smiles,
            )

        best_entry = self._best_entry()
        if best_entry is not None:
            self._prev_best_score = self._score_objective(
                (best_entry['objectives']['validity'], best_entry['objectives']['uniqueness'])
            )

        for t in range(self.T):
            alpha = self._get_alpha(t)
            mbest = self._compute_mbest()
            leader = self._select_leader()
            n_mutated_this_iter = 0

            for i in range(self.M):
                anchor = self.pbest[i] if self.pbest_objectives[i] is not None else self.positions[i]
                self.positions[i] = self._update_position(
                    x=self.positions[i],
                    pbest_i=anchor,
                    leader=leader,
                    mbest=mbest,
                    alpha=alpha,
                )

                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    n_mutated_this_iter += 1

            iter_results = self._evaluate_swarm()
            iter_validities: List[float] = []
            iter_uniqueness: List[float] = []

            for i, obj, smiles_list in iter_results:
                iter_validities.append(obj[0])
                iter_uniqueness.append(obj[1])
                self._update_pbest(i, obj)
                # [FIX-P0-2] 解構三個回傳值，並將 smiles 傳入 archive
                loop_smiles = smiles_list
                self.archive.try_add(
                    params=self.positions[i],
                    objectives={'validity': obj[0], 'uniqueness': obj[1]},
                    smiles=loop_smiles,
                )

            self._total_mutations += n_mutated_this_iter
            best_entry = self._best_entry()
            best_validity = self.archive.best_metric('validity')
            best_uniqueness = self.archive.best_metric('uniqueness')
            best_params = best_entry['params'].copy() if best_entry is not None else np.zeros(self.D, dtype=np.float64)
            best_obj = (
                float(best_entry['objectives']['validity']) if best_entry is not None else 0.0,
                float(best_entry['objectives']['uniqueness']) if best_entry is not None else 0.0,
            )
            best_score = self._score_objective(best_obj)

            self._update_stagnation(best_score)
            self._check_and_reinit()
            diversity = self._compute_diversity()

            iter_record = {
                'iteration': t,
                'alpha': alpha,
                'archive_size': len(self.archive),
                'best_validity': best_validity,
                'best_uniqueness': best_uniqueness,
                'best_compromise_params': best_params,
                'mean_validity': float(np.mean(iter_validities)) if iter_validities else 0.0,
                'mean_uniqueness': float(np.mean(iter_uniqueness)) if iter_uniqueness else 0.0,
                'max_validity': float(np.max(iter_validities)) if iter_validities else 0.0,
                'max_uniqueness': float(np.max(iter_uniqueness)) if iter_uniqueness else 0.0,
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
                    f"best_v={best_validity:.4f} "
                    f"best_u={best_uniqueness:.4f} "
                    f"mean_v={iter_record['mean_validity']:.4f} "
                    f"mean_u={iter_record['mean_uniqueness']:.4f} "
                    f"archive={len(self.archive)} "
                    f"mut={n_mutated_this_iter}"
                )

        best_entry = self._best_entry()
        if best_entry is None:
            return np.zeros(self.D, dtype=np.float64), {'validity': 0.0, 'uniqueness': 0.0}, self.history

        print('\n' + '=' * 70)
        print('MOQPSO 優化完成')
        print(f"  Best validity     : {best_entry['objectives']['validity']:.6f}")
        print(f"  Best uniqueness   : {best_entry['objectives']['uniqueness']:.6f}")
        print(f"  Pareto size       : {len(self.archive)}")
        print(f"  總重初始化次數   : {self._total_reinits}")
        print(f"  總變異粒子次數   : {self._total_mutations}")
        print('=' * 70)

        return best_entry['params'].copy(), best_entry['objectives'].copy(), self.history

    def reset(self):
        self.positions = self._random_positions(self.M)
        self.pbest = self.positions.copy()
        self.pbest_objectives = [None for _ in range(self.M)]
        self.archive = ParetoArchive()
        self.history = []
        self._stagnation_counter = 0
        self._prev_best_score = -np.inf
        self._total_reinits = 0
        self._total_mutations = 0