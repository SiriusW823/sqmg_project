"""
==============================================================================
Evaluator — 三大化學生成指標計算模組
==============================================================================

計算論文標準的分子生成評估指標（QMG Chen et al. JCTC 2025 定義）：
  1. Validity   (有效性)  — 通過 RDKit Sanitize 的比例
  2. Uniqueness (唯一性)  — 有效分子中不重複 SMILES 的比例
  3. Novelty    (新穎性)  — 唯一有效分子中不在參考資料集內的比例

v4 修改說明：
  - 新增 evaluate_shot_weighted()：提供 shot-weighted 指標，與 compute_fitness 一致，
    方便論文中比較 bitstring-count 指標與 shot-weighted 指標的差異。
  - format_metrics() 加入 shot-weighted 欄位（可選輸出）。
  - 兩種指標並存，由呼叫端選擇使用哪一種報告給論文。

==============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# 預設參考分子資料集（Mock Data）
# ============================================================================

DEFAULT_REFERENCE_SMILES: Set[str] = {
    # ── 常見藥物分子 ──
    "CC(=O)Oc1ccccc1C(=O)O",                    # Aspirin
    "CC(=O)Nc1ccc(O)cc1",                        # Acetaminophen
    "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",       # Testosterone
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",             # Caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",               # Ibuprofen
    # ── 簡單有機分子 ──
    "C", "CC", "C=C", "C#C",
    "CO", "C=O", "CC=O", "CC(=O)C",
    "CCO", "CC(O)=O", "C(=O)O",
    "CCN", "CN", "CS",
    "ClC", "FC",
    "N", "O", "S",
    "c1ccccc1", "c1ccncc1", "c1ccoc1",
    "c1ccsc1", "c1cc[nH]c1",
    "CC#N", "C(Cl)(Cl)Cl",
}


class MoleculeEvaluator:
    """
    分子生成品質評估器。

    提供兩種計算模式：

    1. evaluate()：QMG Eq.4–5 定義，bitstring-count 分母。
       每種 bitstring 等權重，可直接與論文報告數字比較。
         Validity   = N_valid_bitstrings / N_total_bitstrings
         Uniqueness = N_unique_smiles    / N_valid_bitstrings
         Novelty    = N_novel_smiles     / N_unique_smiles

    2. evaluate_shot_weighted()：shot-weighted 分母。
       高頻 bitstring 佔更大權重，與 compute_fitness 語意一致。
         Validity   = Σ(valid_count)    / Σ(total_count)
         Uniqueness = N_unique_smiles   / Σ(valid_count)

    使用方式：
        evaluator = MoleculeEvaluator()
        metrics   = evaluator.evaluate(decoded_results)
    """

    def __init__(self, reference_smiles: Optional[Set[str]] = None):
        self.reference_smiles = reference_smiles if reference_smiles is not None \
                                else DEFAULT_REFERENCE_SMILES.copy()

    # ------------------------------------------------------------------
    # 主要評估方法（QMG 論文定義，bitstring-count）
    # ------------------------------------------------------------------

    def evaluate(self, decoded_results: List[Dict]) -> Dict[str, float]:
        """
        計算一組解碼結果的完整評估指標（QMG 論文定義，bitstring-count 分母）。

          validity   = N_valid   / N_total       (QMG Eq.4)
          uniqueness = N_unique  / N_valid        (QMG Eq.5)
          novelty    = N_novel   / N_unique

        每種 bitstring 等權重，可對照論文報告數字。

        Args:
            decoded_results: MoleculeDecoder.decode_counts() 的回傳值。

        Returns:
            包含 'validity', 'uniqueness', 'novelty' 等欄位的指標字典。
        """
        n_total = len(decoded_results)
        if n_total == 0:
            return self._empty_metrics()

        # 1. Validity（每種 bitstring 等權重）
        valid_results = [r for r in decoded_results if r.get('valid', False)]
        n_valid   = len(valid_results)
        validity  = n_valid / n_total

        if n_valid == 0:
            m = self._empty_metrics()
            m['n_total'] = n_total
            m['validity'] = validity
            return m

        # 收集有效 SMILES
        valid_smiles_list = [r['smiles'] for r in valid_results if r.get('smiles')]

        # 2. Uniqueness（QMG Eq.5）
        unique_smiles_set = set(valid_smiles_list)
        n_unique   = len(unique_smiles_set)
        uniqueness = n_unique / n_valid

        # 3. Novelty
        novel_smiles = [s for s in unique_smiles_set if s not in self.reference_smiles]
        n_novel  = len(novel_smiles)
        novelty  = n_novel / n_unique if n_unique > 0 else 0.0

        return {
            'n_total':       n_total,
            'n_valid':       n_valid,
            'n_unique':      n_unique,
            'n_novel':       n_novel,
            'validity':      validity,
            'uniqueness':    uniqueness,
            'novelty':       novelty,
            'valid_smiles':  valid_smiles_list,
            'unique_smiles': sorted(unique_smiles_set),
            'novel_smiles':  sorted(novel_smiles),
        }

    # ------------------------------------------------------------------
    # Shot-weighted 指標（與 compute_fitness 語意一致）
    # ------------------------------------------------------------------

    def evaluate_shot_weighted(self, decoded_results: List[Dict]) -> Dict[str, float]:
        """
        計算 shot-weighted 指標（分母為 total_shots，非 bitstring 種數）。

        與 evaluate() 的差異：
          - 高頻出現的 bitstring 在 validity 中佔更大比重。
          - uniqueness 分母為 valid_shots（而非 valid bitstring 種數）。

        適用於：優化收斂監控、與 compute_fitness 的數值對比。
        論文指標報告應使用 evaluate()（bitstring-count）。
        """
        if not decoded_results:
            return self._empty_metrics()

        total_shots = sum(r.get('count', 0) for r in decoded_results)
        if total_shots == 0:
            # fallback：每種 bitstring count=1
            total_shots = len(decoded_results)
            for r in decoded_results:
                if r.get('count', 0) == 0:
                    r['count'] = 1

        valid   = [r for r in decoded_results if r.get('valid')]
        valid_shots = sum(r.get('count', 1) for r in valid)
        validity_sw = valid_shots / total_shots

        unique_smiles = set(r['smiles'] for r in valid if r.get('smiles'))
        n_unique      = len(unique_smiles)
        uniqueness_sw = n_unique / valid_shots if valid_shots > 0 else 0.0

        novel = [s for s in unique_smiles if s not in self.reference_smiles]
        novelty_sw = len(novel) / n_unique if n_unique > 0 else 0.0

        return {
            'n_total':        len(decoded_results),
            'n_valid_shots':  valid_shots,
            'total_shots':    total_shots,
            'n_unique':       n_unique,
            'n_novel':        len(novel),
            'validity':       validity_sw,
            'uniqueness':     uniqueness_sw,
            'novelty':        novelty_sw,
            'valid_smiles':   [r['smiles'] for r in valid if r.get('smiles')],
            'unique_smiles':  sorted(unique_smiles),
            'novel_smiles':   sorted(novel),
        }

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------

    def evaluate_from_counts(
        self,
        counts:  Dict[str, int],
        decoder,
    ) -> Tuple[float, Dict[str, float], List[Dict]]:
        """
        從 bit-string 計數字典計算目標值與完整指標。

        Returns:
            (fitness_score, metrics_bitstring_count, decoded_results)
        """
        fitness_score, decoded = decoder.compute_fitness(counts)
        metrics = self.evaluate(decoded)
        return fitness_score, metrics, decoded

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            'n_total': 0, 'n_valid': 0, 'n_unique': 0, 'n_novel': 0,
            'validity': 0.0, 'uniqueness': 0.0, 'novelty': 0.0,
            'valid_smiles': [], 'unique_smiles': [], 'novel_smiles': [],
        }

    def format_metrics(
        self,
        metrics: Dict[str, float],
        show_shot_weighted: bool = False,
    ) -> str:
        """
        格式化指標為可讀文字。

        Args:
            metrics:             evaluate() 或 evaluate_shot_weighted() 的輸出。
            show_shot_weighted:  若 True，在標題標注「Shot-Weighted」。
        """
        mode = " [Shot-Weighted]" if show_shot_weighted else " [Bitstring-Count, QMG Eq.4-5]"
        n_v = metrics.get('n_valid', metrics.get('n_valid_shots', 0))
        return (
            f"  Metrics{mode}\n"
            f"  Validity   : {metrics['validity']:.4f}  ({n_v}/{metrics['n_total']})\n"
            f"  Uniqueness : {metrics['uniqueness']:.4f}  ({metrics['n_unique']}/{n_v})\n"
            f"  Novelty    : {metrics['novelty']:.4f}  ({metrics['n_novel']}/{metrics['n_unique']})"
        )