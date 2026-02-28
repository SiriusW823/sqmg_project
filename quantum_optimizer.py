"""
==============================================================================
MOQPSOOptimizer — 多目標量子粒子群優化 (MOQPSO) v5
==============================================================================

本模組實作 Multi-Objective Quantum Particle Swarm Optimization (MOQPSO)，
用來「同時最大化」分子生成的 Validity（有效性）與 Uniqueness（唯一性）。

■ 為什麼升級為 MOQPSO？
  ─ v3 QPSO 使用單一 gbest 追蹤標量適應度，
    無法在 Validity 與 Uniqueness 之間自動權衡。
  ─ MOQPSO 使用 **Pareto Archive** 取代固定 gbest：
    • 非支配解外部存檔（支配關係判定）
    • 擁擠距離 (Crowding Distance) 輪盤選擇 Archive Leader
    • pbest 更新遵循 Pareto 支配規則
  ─ 保留 v3 的抗停滯機制（Cosine α、Cauchy 變異、停滯偵測）。

■ MOQPSO 核心數學
  ─────────────────
  1. mbest（平均最佳位置）：
         mbest_d = (1/M) × Σᵢ pbest_i,d

  2. 局部吸引子（Archive-guided）：
         guide  ← ParetoArchive.select_leader（擁擠距離輪盤）
         p_i,d  = φ × pbest_i,d + (1 − φ) × guide_d
         φ ~ Uniform(0, 1)

  3. 位置更新（Delta 勢阱模型）：
         x_i,d  = p_i,d ± α × |mbest_d − x_i,d| × ln(1/u)
         u ~ Uniform(0, 1)

  4. pbest 更新（Pareto 支配規則）：
         ─ 新解支配舊 pbest → 更新
         ─ 舊 pbest 支配新解 → 保留
         ─ 互不支配 → 50% 機率替換

  5. Archive 更新：
         每次評估都嘗試 try_add，移除被支配解，
         超出容量時按擁擠距離截斷。

■ 目標軸定義
  ───────────
  1. validity   — RDKit SanitizeMol 成功的 bit-string 比例
  2. uniqueness — 有效分子中不重複 SMILES 的比例

■ fitness_fn 回傳值
  ──────────────────
  fitness_fn(params) → Tuple[float, float]  即 (validity, uniqueness)
  （不再是單一標量！）
==============================================================================
"""

import math
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


# ============================================================================
# Pareto Archive — 多目標非支配解外部存檔（含擁擠距離）
# ============================================================================

class ParetoArchive:
    """
    外部存檔 (External Archive) — 記錄搜尋過程中的非支配解。

    在多目標最佳化中，不同解可能在 Validity 高但 Uniqueness 低，
    或 Validity 低但 Uniqueness 高之間權衡。Pareto Archive 保存
    所有「不被其他解支配」的解，形成 Pareto 前緣 (Pareto Frontier)。

    支配關係定義：
      解 A 支配 (dominates) 解 B ⟺
        ∀i: A_i ≥ B_i  且  ∃j: A_j > B_j
      （A 在所有目標上不差，且至少一個目標嚴格更好）

    目標軸：
      1. validity   — RDKit 解析成功比例（越高越好）
      2. uniqueness — 有效分子唯一性比例（越高越好）

    擁擠距離 (Crowding Distance)：
      衡量每個解在 Pareto 前緣上的「稀疏程度」。
      擁擠距離越大 → 解越孤立 → 被選為 Leader 的機率越高
      → 促進粒子群的多樣性探索。

    使用方式：
        archive = ParetoArchive(max_size=100)
        archive.try_add(
            params=params_vector,
            objectives={'validity': 0.72, 'uniqueness': 0.85},
            smiles=['CCO', 'CC=O'],
        )
        guide = archive.select_leader(rng)
        print(archive.summary())
    """

    def __init__(
        self,
        max_size: int = 100,
        objectives: Tuple[str, ...] = ('validity', 'uniqueness'),
    ):
        """
        Args:
            max_size:   存檔最大容量（超出後按擁擠距離截斷）
            objectives: 目標維度名稱（皆為越大越好）
        """
        self.max_size = max_size
        self.obj_keys = objectives
        self.archive: List[Dict] = []

    # ────────────────────────────────────────────────────────────

    def try_add(
        self,
        params: np.ndarray,
        objectives: Dict[str, float],
        smiles: Optional[List[str]] = None,
    ) -> bool:
        """
        嘗試將一組解加入存檔。

        流程：
        1. 計算新解的目標向量
        2. 跳過平凡解（所有目標 ≤ 0）
        3. 檢查是否被任何現有解支配 → 若是則拒絕
        4. 移除被新解支配的現有解
        5. 加入新解
        6. 若超出容量 → 按擁擠距離截斷

        Args:
            params:     量子線路參數向量
            objectives: 目標字典，須包含 self.obj_keys 中的所有 key
            smiles:     本次生成的有效 SMILES 列表（可選）

        Returns:
            True 表示成功加入，False 表示被支配而拒絕
        """
        obj_vec = tuple(objectives.get(k, 0.0) for k in self.obj_keys)

        # 跳過全零/全負解（沒有任何有效分子）
        if all(v <= 0.0 for v in obj_vec):
            return False

        # 檢查支配關係 & 移除被新解支配的舊解
        new_archive: List[Dict] = []
        dominated = False

        for entry in self.archive:
            exist_vec = entry['obj_vec']
            if self._dominates(exist_vec, obj_vec):
                dominated = True
                break
            if not self._dominates(obj_vec, exist_vec):
                new_archive.append(entry)
            # else: obj_vec dominates exist_vec → 移除舊解

        if dominated:
            return False

        # 加入新解
        new_archive.append({
            'params': params.copy() if hasattr(params, 'copy') else np.array(params),
            'objectives': dict(objectives),
            'obj_vec': obj_vec,
            'smiles': list(smiles) if smiles else [],
        })

        # 容量截斷：按擁擠距離降序保留 max_size 個
        if len(new_archive) > self.max_size:
            distances = self._compute_crowding_distances(new_archive)
            # 擁擠距離大的保留（邊界解 inf 永遠保留）
            indexed = sorted(
                range(len(new_archive)),
                key=lambda i: -distances[i],
            )
            new_archive = [new_archive[i] for i in indexed[:self.max_size]]

        self.archive = new_archive
        return True

    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _dominates(a: tuple, b: tuple) -> bool:
        """
        判斷目標向量 a 是否支配 b。

        支配條件：a 在所有維度 ≥ b，且至少一個維度 > b。
        """
        all_ge = all(ai >= bi for ai, bi in zip(a, b))
        any_gt = any(ai > bi for ai, bi in zip(a, b))
        return all_ge and any_gt

    # ────────────────────────────────────────────────────────────
    # 擁擠距離 (Crowding Distance)
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_crowding_distances(entries: List[Dict]) -> List[float]:
        """
        計算每個解在 Pareto 前緣上的擁擠距離。

        對每個目標維度 m：
          1. 按目標值排序
          2. 兩端邊界解設為 ∞
          3. 中間解 = (左鄰 − 右鄰) / 目標值域

        最終擁擠距離 = 各維度之和。

        擁擠距離大 → 解位於 Pareto 前緣的稀疏區域
        擁擠距離小 → 解被其他解包圍

        Returns:
            各解的擁擠距離列表
        """
        n = len(entries)
        if n == 0:
            return []
        if n <= 2:
            return [float('inf')] * n

        n_obj = len(entries[0]['obj_vec'])
        distances = [0.0] * n

        for m in range(n_obj):
            # 按第 m 個目標值排序
            sorted_idx = sorted(
                range(n), key=lambda i: entries[i]['obj_vec'][m]
            )

            # 邊界解設為 ∞
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')

            # 目標值域
            f_max = entries[sorted_idx[-1]]['obj_vec'][m]
            f_min = entries[sorted_idx[0]]['obj_vec'][m]
            obj_range = f_max - f_min
            if obj_range < 1e-12:
                continue

            # 中間解的擁擠距離
            for k in range(1, n - 1):
                prev_val = entries[sorted_idx[k - 1]]['obj_vec'][m]
                next_val = entries[sorted_idx[k + 1]]['obj_vec'][m]
                distances[sorted_idx[k]] += (next_val - prev_val) / obj_range

        return distances

    # ────────────────────────────────────────────────────────────
    # Archive Leader 選擇（擁擠距離輪盤）
    # ────────────────────────────────────────────────────────────

    def select_leader(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        """
        從 Archive 中以擁擠距離為權重進行輪盤選擇，
        取出一組參數作為粒子的吸引子 (guide)。

        擁擠距離越大 → 被選中機率越高
        → 促使粒子群探索 Pareto 前緣的不同區域。

        Args:
            rng: NumPy 隨機數生成器

        Returns:
            選中解的參數向量 (D,)，若 Archive 為空則回傳 None
        """
        if not self.archive:
            return None
        if len(self.archive) == 1:
            return self.archive[0]['params'].copy()

        distances = self._compute_crowding_distances(self.archive)

        # ∞ 替換為有限最大值的 2 倍（確保邊界解高機率被選）
        max_finite = max(
            (d for d in distances if d != float('inf')),
            default=1.0,
        )
        weights = [
            d if d != float('inf') else max_finite * 2.0
            for d in distances
        ]

        total = sum(weights)
        if total < 1e-12:
            idx = rng.integers(0, len(self.archive))
        else:
            probs = np.array(weights) / total
            idx = rng.choice(len(self.archive), p=probs)

        return self.archive[idx]['params'].copy()

    # ────────────────────────────────────────────────────────────
    # 查詢工具
    # ────────────────────────────────────────────────────────────

    def get_pareto_front(self) -> List[Dict]:
        """回傳目前 Pareto 前緣的所有解（副本）。"""
        return [dict(e) for e in self.archive]

    def __len__(self) -> int:
        return len(self.archive)

    def best_compromise(self) -> Optional[Dict]:
        """
        回傳「折中最優解」— 目標向量元素和最大的解。

        在 (validity, uniqueness) 空間中，折中最優 =
        validity + uniqueness 最大的非支配解。
        """
        if not self.archive:
            return None
        return max(self.archive, key=lambda e: sum(e['obj_vec']))

    def summary(self) -> str:
        """產生 Pareto Archive 的文字摘要。"""
        if not self.archive:
            return "  Pareto Archive: empty (0 non-dominated solutions)"

        lines = [
            f"  Pareto Archive: {len(self.archive)} non-dominated solutions",
            f"  Objectives: {', '.join(self.obj_keys)}",
        ]

        # 按折中分數（目標和）降序列出前 5 名
        top = sorted(self.archive, key=lambda x: -sum(x['obj_vec']))[:5]
        for i, entry in enumerate(top, 1):
            obj_str = ", ".join(
                f"{k}={entry['objectives'].get(k, 0):.4f}"
                for k in self.obj_keys
            )
            n_smi = len(entry.get('smiles', []))
            best_smi = entry['smiles'][0] if entry.get('smiles') else '?'
            lines.append(
                f"    #{i}: [{obj_str}]  "
                f"({n_smi} mols, e.g. {best_smi})"
            )

        return "\n".join(lines)


# ============================================================================
# MOQPSOOptimizer — 多目標量子粒子群優化
# ============================================================================

class MOQPSOOptimizer:
    """
    MOQPSO (Multi-Objective Quantum Particle Swarm Optimization) — v5。

    相對於 v3 單目標 QPSO 的核心改動：
      • fitness_fn 回傳 (validity, uniqueness) 元組（非標量）
      • 移除固定 gbest：改用 ParetoArchive 的擁擠距離輪盤選擇 Leader
      • pbest 更新：Pareto 支配規則 + 50% 隨機替換
      • 保留 v3 的 Cosine Annealing α、Cauchy 變異、停滯偵測

    使用方式：
        archive = ParetoArchive(max_size=100)
        optimizer = MOQPSOOptimizer(
            n_params=18,
            n_particles=20,
            max_iterations=50,
            fitness_fn=my_fn,     # returns (validity, uniqueness)
            archive=archive,
        )
        best_params, best_obj, history = optimizer.optimize()
    """

    def __init__(
        self,
        n_params: int,
        n_particles: int = 20,
        max_iterations: int = 50,
        fitness_fn: Optional[Callable[[np.ndarray], Tuple[float, float]]] = None,
        archive: Optional[ParetoArchive] = None,
        alpha_max: float = 1.0,
        alpha_min: float = 0.5,
        param_lower: float = -np.pi,
        param_upper: float = np.pi,
        seed: Optional[int] = None,
        verbose: bool = True,
        iteration_callback: Optional[Callable[[int, dict], None]] = None,
        # ── 抗停滯超參數（繼承自 v3）──
        stagnation_limit: int = 5,
        reinit_fraction: float = 0.3,
        mutation_prob: float = 0.15,
        mutation_scale: float = 0.3,
        alpha_perturb_std: float = 0.05,
        alpha_stag_boost: float = 0.3,
    ):
        """
        初始化 MOQPSO 優化器。

        Args:
            n_params:          參數空間維度 D（= 5N − 2）
            n_particles:       粒子數量 M（建議 15~30）
            max_iterations:    最大迭代次數 T
            fitness_fn:        目標函式 f(params) → (validity, uniqueness)
            archive:           外部 Pareto Archive（若 None 則自動建立）
            alpha_max:         α 的最大值（初期，鼓勵探索）
            alpha_min:         α 的最小值（末期，促進收斂）
            param_lower:       參數下界
            param_upper:       參數上界
            seed:              隨機數種子
            verbose:           是否印出每輪進度
            iteration_callback: 每輪迭代結束後回呼

            stagnation_limit:  連續幾代 Archive 未變化就觸發重初始化
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
        self.archive = archive if archive is not None else ParetoArchive()
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

        # 隨機數生成器
        self.rng = np.random.default_rng(seed)

        # ── 初始化粒子群 ──
        self.positions = self.rng.uniform(
            self.lb, self.ub, size=(self.M, self.D)
        )
        self.pbest = self.positions.copy()
        # pbest_objectives[i] = (validity, uniqueness) or None
        self.pbest_objectives: List[Optional[Tuple[float, float]]] = \
            [None] * self.M

        # 停滯追蹤（基於 Archive 大小變化）
        self._stagnation_counter = 0
        self._prev_archive_size = 0
        self._total_reinits = 0
        self._total_mutations = 0

        # 優化歷史紀錄
        self.history: List[dict] = []

    # ────────────────────────────────────────────────────────────
    # Pareto 支配判斷
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        """
        判斷目標向量 a 是否支配 b。

        支配條件：
          ∀i: a_i ≥ b_i  且  ∃j: a_j > b_j
        """
        all_ge = all(ai >= bi for ai, bi in zip(a, b))
        any_gt = any(ai > bi for ai, bi in zip(a, b))
        return all_ge and any_gt

    # ────────────────────────────────────────────────────────────
    # pbest 更新（Pareto 支配規則）
    # ────────────────────────────────────────────────────────────

    def _update_pbest(self, i: int, new_obj: Tuple[float, float]):
        """
        使用 Pareto 支配規則更新粒子 i 的個人最佳 (pbest)。

        規則：
          • 首次評估 → 直接設定
          • 新解支配舊 pbest → 更新
          • 舊 pbest 支配新解 → 保留
          • 互不支配 → 50% 機率替換（維持多樣性）

        Args:
            i:       粒子索引
            new_obj: 新位置的目標向量 (validity, uniqueness)
        """
        old_obj = self.pbest_objectives[i]

        if old_obj is None:
            # 首次評估
            self.pbest[i] = self.positions[i].copy()
            self.pbest_objectives[i] = new_obj
            return

        if self._dominates(new_obj, old_obj):
            # 新解支配舊 pbest → 更新
            self.pbest[i] = self.positions[i].copy()
            self.pbest_objectives[i] = new_obj
        elif self._dominates(old_obj, new_obj):
            # 舊 pbest 支配新解 → 保留
            pass
        else:
            # 互不支配 → 50% 機率替換
            if self.rng.random() < 0.5:
                self.pbest[i] = self.positions[i].copy()
                self.pbest_objectives[i] = new_obj

    # ────────────────────────────────────────────────────────────
    # Cosine Annealing α 排程
    # ────────────────────────────────────────────────────────────

    def _get_alpha(self, t: int) -> float:
        """
        計算第 t 輪的收縮-擴張係數 α（Cosine Annealing + 擾動 + 停滯提升）。

            α_base(t) = α_min + ½(α_max − α_min)(1 + cos(πt/T))

        Args:
            t: 目前迭代輪次 (0-indexed)

        Returns:
            當前 α 值（已 clip 到合理範圍）
        """
        progress = t / max(self.T - 1, 1)

        alpha_base = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (
            1.0 + math.cos(math.pi * progress)
        )

        perturbation = self.rng.normal(0, self.alpha_perturb_std)

        stag_boost = 0.0
        if self._stagnation_counter >= self.stagnation_limit:
            stag_boost = self.alpha_stag_boost

        alpha = alpha_base + perturbation + stag_boost

        alpha_upper = self.alpha_max + self.alpha_stag_boost
        return float(np.clip(alpha, self.alpha_min * 0.8, alpha_upper))

    # ────────────────────────────────────────────────────────────
    # mbest 計算
    # ────────────────────────────────────────────────────────────

    def _compute_mbest(self) -> np.ndarray:
        """
        計算 mbest（所有粒子個人最佳位置的平均值）。

            mbest_d = (1/M) × Σᵢ pbest_i,d
        """
        return np.mean(self.pbest, axis=0)

    # ────────────────────────────────────────────────────────────
    # MOQPSO 位置更新核心
    # ────────────────────────────────────────────────────────────

    def _update_position(
        self, x: np.ndarray, pbest_i: np.ndarray,
        guide: np.ndarray, mbest: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        使用 QPSO Delta 勢阱模型更新單一粒子的位置。

        與 v3 的差異：gbest 被替換為 guide（從 Archive 選出）。

            p = φ × pbest_i + (1 − φ) × guide
            x_new = p ± α × |mbest − x| × ln(1/u)

        Args:
            x:       目前位置 (D,)
            pbest_i: 粒子個人最佳位置 (D,)
            guide:   從 Pareto Archive 選出的吸引子 (D,)
            mbest:   全體 pbest 平均 (D,)
            alpha:   收縮-擴張係數

        Returns:
            x_new: 更新後的位置 (D,)
        """
        D = self.D

        # Step 1: 局部吸引子（Archive-guided）
        phi = self.rng.uniform(0, 1, size=D)
        p = phi * pbest_i + (1.0 - phi) * guide

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
    # Cauchy 變異 (Mutation)
    # ────────────────────────────────────────────────────────────

    def _cauchy_mutation(self, x: np.ndarray) -> np.ndarray:
        """
        對位置向量施加 Cauchy 分佈變異（重尾探索）。

        為何用 Cauchy 而非 Gaussian？
        ─ Cauchy 的重尾 (heavy tail) 讓粒子偶爾產生大幅跳躍，
          跳出當前盆地，探索遠處的搜索空間。

        實作：
          1. 隨機選取 30%~50% 的維度
          2. 對選取維度施加 Cauchy 擾動
          3. 未選取維度保持不變

        Args:
            x: 原始位置向量 (D,)

        Returns:
            x_mut: 變異後的位置向量 (D,)
        """
        x_mut = x.copy()

        n_mutate = max(1, int(self.D * self.rng.uniform(0.3, 0.5)))
        dims = self.rng.choice(self.D, size=n_mutate, replace=False)

        cauchy_noise = self.rng.standard_cauchy(size=n_mutate) * self.mutation_scale

        x_mut[dims] += cauchy_noise
        x_mut = np.clip(x_mut, self.lb, self.ub)

        return x_mut

    # ────────────────────────────────────────────────────────────
    # 多樣性度量
    # ────────────────────────────────────────────────────────────

    def _compute_diversity(self) -> float:
        """
        計算粒子群的多樣性指標（位置標準差的平均值）。

            diversity = (1/D) × Σ_d std(positions[:, d])
        """
        return float(np.mean(np.std(self.positions, axis=0)))

    # ────────────────────────────────────────────────────────────
    # 停滯偵測 & 部分粒子重初始化
    # ────────────────────────────────────────────────────────────

    def _check_and_reinit(self) -> bool:
        """
        基於 Archive 大小穩定度的停滯偵測。

        在 MOQPSO 中，「停滯」定義為 Archive 連續 N 代
        未新增任何非支配解（大小不增）。此時重初始化最差粒子。

        重初始化策略：
          • 前半數：完全隨機（全域探索）
          • 後半數：在 Archive Leader 附近的高斯擾動（局部探索）

        Returns:
            是否觸發了重初始化
        """
        current_size = len(self.archive)

        if current_size > self._prev_archive_size:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        self._prev_archive_size = current_size

        if self._stagnation_counter < self.stagnation_limit:
            return False

        # ── 觸發重初始化 ──
        n_reinit = max(1, int(self.M * self.reinit_fraction))

        # 按 pbest 目標和排序，找出最差粒子
        obj_sums = []
        for i in range(self.M):
            if self.pbest_objectives[i] is not None:
                obj_sums.append(sum(self.pbest_objectives[i]))
            else:
                obj_sums.append(-np.inf)

        worst_indices = np.argsort(obj_sums)[:n_reinit]

        for k, idx in enumerate(worst_indices):
            if k < n_reinit // 2:
                # 策略 A：完全隨機（全域探索）
                self.positions[idx] = self.rng.uniform(
                    self.lb, self.ub, size=self.D
                )
            else:
                # 策略 B：Archive Leader 附近的高斯擾動
                ref = self.archive.select_leader(self.rng)
                if ref is not None:
                    noise = self.rng.normal(
                        0, (self.ub - self.lb) * 0.25, size=self.D
                    )
                    self.positions[idx] = np.clip(
                        ref + noise, self.lb, self.ub
                    )
                else:
                    self.positions[idx] = self.rng.uniform(
                        self.lb, self.ub, size=self.D
                    )

            # 重置該粒子的 pbest
            self.pbest[idx] = self.positions[idx].copy()
            self.pbest_objectives[idx] = None

        self._stagnation_counter = 0
        self._total_reinits += 1

        if self.verbose:
            print(
                f"  ⚡ [停滯偵測] 連續 {self.stagnation_limit} 代 "
                f"Archive 無新增非支配解，"
                f"已重初始化 {n_reinit} 個粒子"
                f"（共 {self._total_reinits} 次）"
            )

        return True

    # ────────────────────────────────────────────────────────────
    # 主要優化迴圈
    # ────────────────────────────────────────────────────────────

    def optimize(self) -> Tuple[np.ndarray, Tuple[float, float], List[dict]]:
        """
        執行 MOQPSO 多目標優化迭代。

        完整流程（每輪迭代）：
        ─────────────────────
        1. 計算 α（Cosine Annealing + 擾動 + 停滯提升）
        2. 計算 mbest
        3. 對每個粒子：
           a. 從 Archive 選擇 guide（擁擠距離輪盤）
           b. QPSO Delta 勢阱位置更新（guide 取代固定 gbest）
           c. 以機率 p_mut 施加 Cauchy 變異
           d. 評估目標 (validity, uniqueness)
           e. 更新 pbest（Pareto 支配規則）
           f. 嘗試加入 Archive
        4. 停滯偵測：若 Archive 連續 N 代未變化 → 重初始化
        5. 記錄歷史 + 回呼 + 進度輸出

        Returns:
            (best_params, best_obj, history)
            best_params: 折中最優解的參數向量
            best_obj:    折中最優解的目標 (validity, uniqueness)
            history:     每輪迭代的紀錄列表
        """
        if self.fitness_fn is None:
            raise ValueError("fitness_fn 未設定！請在初始化時提供目標函式。")

        print("=" * 70)
        print("MOQPSO 多目標量子粒子群優化 v5 啟動")
        print(f"  粒子數 (M)       : {self.M}")
        print(f"  參數維度 (D)     : {self.D}")
        print(f"  最大迭代 (T)     : {self.T}")
        print(f"  α 範圍           : {self.alpha_max} → {self.alpha_min} (cosine)")
        print(f"  參數範圍         : [{self.lb:.4f}, {self.ub:.4f}]")
        print(f"  目標             : Validity × Uniqueness（雙目標最大化）")
        print(f"  吸引子選擇       : Pareto Archive 擁擠距離輪盤")
        print(f"  pbest 更新       : Pareto 支配規則 + 50% 隨機替換")
        print(f"  停滯門檻         : {self.stagnation_limit} 代")
        print(f"  重初始化比例     : {self.reinit_fraction:.0%}")
        print(f"  Cauchy 變異機率  : {self.mutation_prob:.0%}")
        print("=" * 70)

        # ── 初始適應度評估 ──
        print("\n[初始化] 評估所有粒子的初始目標值...")
        for i in range(self.M):
            obj = self.fitness_fn(self.positions[i])
            self.pbest_objectives[i] = obj
            self.pbest[i] = self.positions[i].copy()
            self.archive.try_add(
                params=self.positions[i],
                objectives={
                    'validity': obj[0],
                    'uniqueness': obj[1],
                },
            )

        compromise = self.archive.best_compromise()
        init_obj = compromise['obj_vec'] if compromise else (0, 0)
        self._prev_archive_size = len(self.archive)
        print(
            f"[初始化] 完成。Archive 大小: {len(self.archive)}  "
            f"最優折中: val={init_obj[0]:.4f}, uniq={init_obj[1]:.4f}\n"
        )

        # ── 主迭代迴圈 ──
        for t in range(self.T):
            # 計算當前 α
            alpha = self._get_alpha(t)

            # 計算 mbest
            mbest = self._compute_mbest()

            iter_validities: List[float] = []
            iter_uniquenesses: List[float] = []
            n_mutated_this_iter = 0

            # ── 更新每個粒子 ──
            for i in range(self.M):
                # Step 1: 從 Archive 選擇 guide
                guide = self.archive.select_leader(self.rng)
                if guide is None:
                    # Archive 為空時 fallback 到 pbest
                    guide = self.pbest[i]

                # Step 2: QPSO 位置更新
                self.positions[i] = self._update_position(
                    x=self.positions[i],
                    pbest_i=self.pbest[i],
                    guide=guide,
                    mbest=mbest,
                    alpha=alpha,
                )

                # Step 3: Cauchy 變異
                if self.rng.random() < self.mutation_prob:
                    self.positions[i] = self._cauchy_mutation(self.positions[i])
                    n_mutated_this_iter += 1

                # Step 4: 評估新位置的目標
                obj = self.fitness_fn(self.positions[i])
                iter_validities.append(obj[0])
                iter_uniquenesses.append(obj[1])

                # Step 5: 更新 pbest（Pareto 支配規則）
                self._update_pbest(i, obj)

                # Step 6: 嘗試加入 Archive
                self.archive.try_add(
                    params=self.positions[i],
                    objectives={
                        'validity': obj[0],
                        'uniqueness': obj[1],
                    },
                )

            self._total_mutations += n_mutated_this_iter

            # ── 停滯偵測 & 重初始化 ──
            self._check_and_reinit()

            # ── 多樣性度量 ──
            diversity = self._compute_diversity()

            # ── 本輪最優 ──
            compromise = self.archive.best_compromise()
            best_obj = compromise['obj_vec'] if compromise else (0, 0)
            best_params_this = (
                compromise['params'].copy() if compromise else None
            )

            # ── 記錄歷史 ──
            iter_record = {
                'iteration': t,
                'alpha': alpha,
                'archive_size': len(self.archive),
                'best_validity': best_obj[0],
                'best_uniqueness': best_obj[1],
                'best_compromise_params': best_params_this,
                'mean_validity': float(np.mean(iter_validities)),
                'mean_uniqueness': float(np.mean(iter_uniquenesses)),
                'max_validity': float(np.max(iter_validities)),
                'max_uniqueness': float(np.max(iter_uniquenesses)),
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
                    f"arch={len(self.archive):3d}  "
                    f"best_val={best_obj[0]:.4f}  "
                    f"best_uniq={best_obj[1]:.4f}  "
                    f"div={diversity:.4f}  "
                    f"mut={n_mutated_this_iter}"
                    f"{stag_marker}"
                )

        # ── 最終報告 ──
        compromise = self.archive.best_compromise()
        best_params = compromise['params'] if compromise else self.pbest[0]
        best_obj = compromise['obj_vec'] if compromise else (0, 0)

        print("\n" + "=" * 70)
        print("MOQPSO v5 優化完成")
        print(f"  Archive 大小     : {len(self.archive)}")
        print(f"  最優折中解       : validity={best_obj[0]:.4f}, "
              f"uniqueness={best_obj[1]:.4f}")
        print(f"  總重初始化次數   : {self._total_reinits}")
        print(f"  總變異粒子次數   : {self._total_mutations}")
        print(f"  最終多樣性       : {self._compute_diversity():.4f}")
        print("=" * 70)

        return best_params.copy(), best_obj, self.history

    # ────────────────────────────────────────────────────────────
    # 工具函式
    # ────────────────────────────────────────────────────────────

    def get_convergence_curve(
        self,
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        取得收斂曲線資料（用於繪圖）。

        Returns:
            (iterations, best_validities, best_uniquenesses)
        """
        iterations = [h['iteration'] for h in self.history]
        best_vals = [h['best_validity'] for h in self.history]
        best_uniqs = [h['best_uniqueness'] for h in self.history]
        return iterations, best_vals, best_uniqs

    def reset(self):
        """
        重置優化器狀態，使用新的隨機粒子位置。
        保留相同的超參數設定。Archive 不重置。
        """
        self.positions = self.rng.uniform(
            self.lb, self.ub, size=(self.M, self.D)
        )
        self.pbest = self.positions.copy()
        self.pbest_objectives = [None] * self.M
        self._stagnation_counter = 0
        self._prev_archive_size = len(self.archive)
        self._total_reinits = 0
        self._total_mutations = 0
        self.history = []
