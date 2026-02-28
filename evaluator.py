"""
==============================================================================
Evaluator — 三大化學生成指標計算模組
==============================================================================

計算論文標準的分子生成評估指標：
  1. Validity (有效性)  — 通過 RDKit Sanitize 的比例
  2. Uniqueness (唯一性) — 有效分子中不重複 SMILES 的比例
  3. Novelty (新穎性)   — 唯一有效分子中不在參考資料集內的比例

此模組獨立於 CUDA-Q 量子線路和優化器，僅依賴 MoleculeDecoder 的
解碼結果 (decoded_results: List[Dict])。
==============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# 預設參考分子資料集 (Mock Data)
# ============================================================================
# 當使用者未提供自訂參考資料集時，使用以下常見藥物 / 有機小分子作為
# Novelty 計算的比較基準。可透過 MoleculeEvaluator 的 reference_smiles
# 參數覆蓋。
# ============================================================================

DEFAULT_REFERENCE_SMILES: Set[str] = {
    # ── 常見藥物分子 ──
    "CC(=O)Oc1ccccc1C(=O)O",       # Aspirin (阿斯匹靈)
    "CC(=O)Nc1ccc(O)cc1",           # Acetaminophen (乙醯胺酚)
    "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",  # Testosterone (睪固酮)
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",        # Caffeine (咖啡因)
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",           # Ibuprofen (布洛芬)
    # ── 簡單有機分子 ──
    "C",           # Methane (甲烷)
    "CC",          # Ethane (乙烷)
    "C=C",         # Ethylene (乙烯)
    "C#C",         # Acetylene (乙炔)
    "CO",          # Methanol (甲醇)
    "C=O",         # Formaldehyde (甲醛)
    "CC=O",        # Acetaldehyde (乙醛)
    "CC(=O)C",     # Acetone (丙酮)
    "CCO",         # Ethanol (乙醇)
    "CC(O)=O",     # Acetic acid (醋酸)
    "C(=O)O",      # Formic acid (甲酸)
    "CCN",         # Ethylamine (乙胺)
    "CN",          # Methylamine (甲胺)
    "CS",          # Methanethiol (甲硫醇)
    "ClC",         # Chloromethane (氯甲烷)
    "FC",          # Fluoromethane (氟甲烷)
    "N",           # Ammonia (氨)
    "O",           # Water (水)
    "S",           # Hydrogen sulfide (硫化氫)
    "c1ccccc1",    # Benzene (苯)
    "c1ccncc1",    # Pyridine (吡啶)
    "c1ccoc1",     # Furan (呋喃)
    "c1ccsc1",     # Thiophene (噻吩)
    "c1cc[nH]c1",  # Pyrrole (吡咯)
    "CC#N",        # Acetonitrile (乙腈)
    "C(Cl)(Cl)Cl", # Chloroform (氯仿)
}


class MoleculeEvaluator:
    """
    分子生成品質評估器。

    計算三大生成指標 (Validity, Uniqueness, Novelty) 以及
    彙整統計量 (Mean QED, Max QED 等)。

    使用方式：
        evaluator = MoleculeEvaluator(reference_smiles=my_set)
        metrics = evaluator.evaluate(decoded_results)
        print(metrics)
        # {'validity': 0.65, 'uniqueness': 0.85, 'novelty': 0.92,
        #  'mean_qed': 0.42, 'max_qed': 0.71, ...}
    """

    def __init__(self, reference_smiles: Optional[Set[str]] = None):
        """
        Args:
            reference_smiles: 參考分子的 canonical SMILES 集合。
                              用於計算 Novelty（不在此集合中的比例）。
                              若為 None，使用預設的 Mock 資料集。
        """
        if reference_smiles is not None:
            self.reference_smiles = reference_smiles
        else:
            self.reference_smiles = DEFAULT_REFERENCE_SMILES.copy()

    def evaluate(self, decoded_results: List[Dict]) -> Dict[str, float]:
        """
        計算一組解碼結果的完整評估指標。

        定義：
        ─────
        • N_total     = len(decoded_results)
        • N_valid     = 有效分子數（valid == True）
        • N_unique    = 有效分子中不重複的 SMILES 數量
        • N_novel     = 唯一有效分子中不在 reference_smiles 內的數量

        指標公式：
        ──────────
        • Validity   = N_valid / N_total        （若 N_total == 0 → 0）
        • Uniqueness = N_unique / N_valid        （若 N_valid == 0 → 0）
        • Novelty    = N_novel / N_unique        （若 N_unique == 0 → 0）

        Args:
            decoded_results: MoleculeDecoder.decode_counts() 的回傳值

        Returns:
            指標字典：
            {
                'n_total': int,
                'n_valid': int,
                'n_unique': int,
                'n_novel': int,
                'validity': float,       # [0, 1]
                'uniqueness': float,     # [0, 1]
                'novelty': float,        # [0, 1]
                'mean_qed': float,
                'max_qed': float,
                'valid_smiles': list[str],     # 所有有效 SMILES（含重複）
                'unique_smiles': list[str],    # 不重複的有效 SMILES
                'novel_smiles': list[str],     # 不在參考集中的 SMILES
            }
        """
        n_total = len(decoded_results)

        if n_total == 0:
            return self._empty_metrics()

        # ── 1. Validity ──
        valid_results = [r for r in decoded_results if r.get('valid', False)]
        n_valid = len(valid_results)
        validity = n_valid / n_total

        if n_valid == 0:
            metrics = self._empty_metrics()
            metrics['n_total'] = n_total
            metrics['validity'] = 0.0
            return metrics

        # 收集所有有效 SMILES（含重複）
        valid_smiles_list = [r['smiles'] for r in valid_results
                            if r.get('smiles') is not None]

        # ── 2. Uniqueness ──
        unique_smiles_set = set(valid_smiles_list)
        n_unique = len(unique_smiles_set)
        uniqueness = n_unique / n_valid if n_valid > 0 else 0.0

        # ── 3. Novelty ──
        novel_smiles = [s for s in unique_smiles_set
                        if s not in self.reference_smiles]
        n_novel = len(novel_smiles)
        novelty = n_novel / n_unique if n_unique > 0 else 0.0

        # ── QED 統計 ──
        qeds = [r['qed'] for r in valid_results]
        mean_qed = float(np.mean(qeds)) if qeds else 0.0
        max_qed = float(np.max(qeds)) if qeds else 0.0

        return {
            'n_total': n_total,
            'n_valid': n_valid,
            'n_unique': n_unique,
            'n_novel': n_novel,
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'mean_qed': mean_qed,
            'max_qed': max_qed,
            'valid_smiles': valid_smiles_list,
            'unique_smiles': sorted(unique_smiles_set),
            'novel_smiles': sorted(novel_smiles),
        }

    def evaluate_from_counts(
        self, counts: Dict[str, int], decoder,
    ) -> Tuple[Tuple[float, float], Dict[str, float], List[Dict]]:
        """
        直接從 bit-string 計數字典計算目標值與完整指標。

        便利方法，整合 decoder.compute_fitness() + evaluate()。

        v5 變更：compute_fitness 回傳 (validity, uniqueness) 元組。

        Args:
            counts:  {bitstring: count} 字典
            decoder: MoleculeDecoder 實例

        Returns:
            ((validity, uniqueness), metrics, decoded_results)
        """
        (validity, uniqueness), decoded = decoder.compute_fitness(counts)
        metrics = self.evaluate(decoded)
        return (validity, uniqueness), metrics, decoded

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        """回傳所有指標為零的預設字典。"""
        return {
            'n_total': 0,
            'n_valid': 0,
            'n_unique': 0,
            'n_novel': 0,
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': 0.0,
            'mean_qed': 0.0,
            'max_qed': 0.0,
            'valid_smiles': [],
            'unique_smiles': [],
            'novel_smiles': [],
        }

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """格式化指標為可讀文字。"""
        return (
            f"  Validity   : {metrics['validity']:.4f}  "
            f"({metrics['n_valid']}/{metrics['n_total']})\n"
            f"  Uniqueness : {metrics['uniqueness']:.4f}  "
            f"({metrics['n_unique']}/{metrics['n_valid']})\n"
            f"  Novelty    : {metrics['novelty']:.4f}  "
            f"({metrics['n_novel']}/{metrics['n_unique']})\n"
            f"  Mean QED   : {metrics['mean_qed']:.4f}\n"
            f"  Max QED    : {metrics['max_qed']:.4f}"
        )
