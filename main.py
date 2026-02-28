"""
==============================================================================
SQMG Main Loop — 可擴展量子分子生成系統 主流程
==============================================================================

本模組整合六大核心元件：
  1. SQMGKernel        — CUDA-Q 3N+2 參數化量子線路
  2. MoleculeDecoder   — Bit-string 到分子結構的解碼器
  3. QuantumOptimizer  — QPSO 量子粒子群優化器
  4. MoleculeEvaluator — Validity / Uniqueness / Novelty 評估指標
  5. plot_utils        — 視覺化模組（收斂軌跡、Pareto 前緣、化學空間）
  6. Main_Loop         — 初始化、優化迭代、指標收集、CSV 匯出、結果輸出

執行方式：
    python main.py [--max_atoms N] [--particles M] [--iterations T] [--shots S]

環境需求：
    ┌──────────────────────────────────────────────────────┐
    │ ⚠ 重要：numpy 必須 < 2.0（Numpy 2.0+ 的 C++ ABI     │
    │   變更會導致 RDKit 崩潰！）                           │
    │                                                      │
    │   pip install "numpy>=1.24,<2.0"                     │
    │   pip install rdkit                                  │
    │   pip install cuda-quantum-cu12  (or cu11)           │
    └──────────────────────────────────────────────────────┘

CUDA-Q 後端設定：
    • 'qpp-cpu'    — CPU 模擬（預設，不需 GPU）
    • 'nvidia'     — GPU 態向量模擬（需要 NVIDIA GPU + CUDA）
    • 'tensornet'  — GPU 張量網路模擬（大 N 時推薦）

    可透過 CUDAQ_DEFAULT_SIMULATOR 環境變數或 cudaq.set_target() 設定。
==============================================================================
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

# ── CUDA-Q Import ──
try:
    import cudaq
except ImportError:
    print("=" * 70)
    print("錯誤：無法匯入 CUDA-Q (cudaq)。")
    print("請先安裝 CUDA-Q：")
    print("  pip install cuda-quantum-cu12")
    print("或參考 https://nvidia.github.io/cuda-quantum/latest/install.html")
    print("=" * 70)
    sys.exit(1)

# ── RDKit Import ──
try:
    from rdkit import Chem
    from rdkit import RDLogger
    # 抑制 RDKit 的冗長警告訊息（無效分子會產生大量 WARNING）
    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    print("=" * 70)
    print("錯誤：無法匯入 RDKit。")
    print("請先安裝 RDKit：")
    print("  pip install rdkit")
    print("⚠ 並確保 numpy < 2.0：pip install 'numpy>=1.24,<2.0'")
    print("=" * 70)
    sys.exit(1)

# ── 本地模組 Import ──
from sqmg_kernel import SQMGKernel
from molecule_decoder import MoleculeDecoder
from quantum_optimizer import QuantumOptimizer, ParetoArchive
from evaluator import MoleculeEvaluator
from plot_utils import plot_all


# ============================================================================
# CUDA-Q 後端組態
# ============================================================================

def configure_cudaq_backend(target: str = "qpp-cpu"):
    """
    設定 CUDA-Q 模擬後端。

    Args:
        target: 後端名稱
            'qpp-cpu'   — C++ 態向量模擬器（CPU，適合小 N）
            'nvidia'    — GPU 態向量模擬器（需 NVIDIA GPU）
            'tensornet' — GPU 張量網路模擬器（大 N 推薦，需 GPU）

    CUDA-Q 注意：
        set_target() 必須在任何 kernel 呼叫前執行。
        切換後端後，已編譯的 kernel 會自動重新編譯。
    """
    try:
        cudaq.set_target(target)
        print(f"[CUDA-Q] 後端已設定為: {target}")
    except Exception as e:
        print(f"[CUDA-Q] 警告：無法設定後端 '{target}'，使用預設後端。")
        print(f"         錯誤訊息: {e}")
        try:
            cudaq.set_target("qpp-cpu")
            print("[CUDA-Q] 已降級為 qpp-cpu 後端。")
        except Exception:
            print("[CUDA-Q] 使用系統預設後端。")


# ============================================================================
# 適應度函式 (Fitness Function)
# ============================================================================

def create_fitness_function(
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
    alpha: float = 0.4,
    verbose_eval: bool = False,
    pareto_archive: 'ParetoArchive | None' = None,
):
    """
    建立連接 CUDA-Q kernel ↔ MoleculeDecoder ↔ QPSO 的適應度函式。

    閉包 (Closure) 內封裝了 kernel 與 decoder 的參考，
    使 QPSO 只需要傳入參數向量即可得到適應度分數。

    v4 新增：
      • Fitness = w1×QED + w2×SA + w3×Size_Reward + w4×Connectivity − Penalty
        （實際由 decoder._score_molecule 內部計算）
      • 若提供 pareto_archive，每次評估自動更新非支配解存檔

    Args:
        kernel:         SQMGKernel 實例
        decoder:        MoleculeDecoder 實例
        alpha:          shaping vs QED 的權重（0~1）
        verbose_eval:   若 True，每次評估都印出詳細結果
        pareto_archive: Pareto 非支配解存檔（可選）

    Returns:
        fitness_fn: Callable[[np.ndarray], float]
    """
    eval_count = [0]

    def fitness_fn(params: np.ndarray) -> float:
        """
        實際的適應度評估函式。

        流程：
        1. params → CUDA-Q kernel (cudaq.sample)
        2. bit-strings → MoleculeDecoder
        3. 分子 → RDKit → Validity + QED + SA + Size + Penalty
        4. → fitness score
        5. (可選) 更新 Pareto Archive
        """
        eval_count[0] += 1

        try:
            # ── Step 1: 量子取樣 ──
            counts = kernel.sample(params)

            # ── Step 2: 解碼 & 計算適應度 ──
            fitness, decoded = decoder.compute_fitness(counts, alpha=alpha)

            if verbose_eval:
                valid_count = sum(1 for r in decoded if r['valid'])
                print(
                    f"    [Eval #{eval_count[0]}] "
                    f"Unique BS={len(decoded)} "
                    f"Valid={valid_count} "
                    f"Fitness={fitness:.6f}"
                )

            # ── Step 3 (v4): 更新 Pareto Archive ──
            if pareto_archive is not None and decoded:
                valid = [
                    r for r in decoded
                    if r.get('valid') and not r.get('partial_valid')
                ]
                n_total = len(decoded)
                if valid:
                    qeds = [r['qed'] for r in valid]
                    validity = len(valid) / n_total
                    unique_smi = list({r['smiles'] for r in valid
                                       if r.get('smiles')})
                    uniqueness = (
                        len(unique_smi) / len(valid) if len(valid) > 0 else 0
                    )
                    pareto_archive.try_add(
                        params=params,
                        objectives={
                            'mean_qed': float(np.mean(qeds)),
                            'val_x_uniq': validity * uniqueness,
                        },
                        fitness=fitness,
                        smiles=unique_smi,
                    )

            return fitness

        except Exception as e:
            if verbose_eval:
                print(f"    [Eval #{eval_count[0]}] 錯誤: {e}")
            return 0.0

    return fitness_fn


# ============================================================================
# 迭代回呼 — 每輪收集 Validity / Uniqueness / Novelty / Mean QED
# ============================================================================

def create_iteration_callback(
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
    evaluator: MoleculeEvaluator,
    extended_history: List[Dict],
    all_molecules: List[Dict],
):
    """
    建立 QPSO iteration_callback，用於每輪結束後評估 gbest 並記錄指標。

    每一輪迭代結束時，optimizer 會把 gbest_params 放在 record 中傳進來，
    callback 用這組參數進行 sample → decode → evaluate，將真實的
    Validity / Uniqueness / Novelty / Mean_QED 寫入 extended_history。

    Args:
        kernel:           SQMGKernel 實例
        decoder:          MoleculeDecoder 實例
        evaluator:        MoleculeEvaluator 實例
        extended_history:  存放每輪指標的外部列表（會被 mutation 填充）
        all_molecules:     累積所有有效分子的外部列表

    Returns:
        callback: Callable[[int, dict], None]
    """
    # 用 set 做跨輪去重，避免每輪都重建
    _seen_smiles: set = set()
    for m in all_molecules:
        _seen_smiles.add(m['smiles'])

    def callback(iteration: int, record: dict):
        """
        在每輪 QPSO 迭代結束後被呼叫。
        使用 record['gbest_params'] 重新取樣並計算評估指標。

        重要：此處的 evaluator.evaluate() 只計算 valid=True
        （完全通過 SanitizeMol）的分子，不包含 partial_valid。
        """
        # ── 預設值（若評估失敗仍有合理紀錄）──
        validity = 0.0
        uniqueness = 0.0
        novelty = 0.0
        mean_qed = 0.0

        gbest_params = record.get('gbest_params')
        if gbest_params is None:
            print(f"  [Callback Iter {iteration + 1}] 警告：gbest_params 為 None")
        else:
            try:
                counts = kernel.sample(gbest_params)
                decoded = decoder.decode_counts(counts)

                # evaluator.evaluate 只計算 valid=True 的分子
                # （partial_valid 不會被納入 Validity/Uniqueness/Novelty）
                metrics = evaluator.evaluate(decoded)

                validity = metrics['validity']
                uniqueness = metrics['uniqueness']
                novelty = metrics['novelty']
                mean_qed = metrics['mean_qed']

                # 累積有效分子（跨輪去重，只取 valid=True）
                for r in decoded:
                    if r.get('valid') and not r.get('partial_valid'):
                        smi = r.get('smiles')
                        if smi and smi not in _seen_smiles:
                            all_molecules.append({
                                'smiles': smi,
                                'qed': r['qed'],
                                'mol': r.get('mol'),
                            })
                            _seen_smiles.add(smi)
            except Exception as e:
                print(f"  [Callback Iter {iteration + 1}] 評估失敗: {e}")

        ext_record = {
            'iteration': record['iteration'],
            'gbest_fitness': record['gbest_fitness'],
            'mean_fitness': record.get('mean_fitness', 0),
            'max_fitness': record.get('max_fitness', 0),
            'min_fitness': record.get('min_fitness', 0),
            'alpha': record.get('alpha', 0),
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'mean_qed': mean_qed,
            # v3: QPSO 抗停滯診斷指標
            'diversity': record.get('diversity', 0),
            'stagnation_counter': record.get('stagnation_counter', 0),
            'n_mutated': record.get('n_mutated', 0),
        }
        extended_history.append(ext_record)

    return callback


# ============================================================================
# CSV 匯出
# ============================================================================

def export_history_csv(history: List[Dict], filepath: str):
    """匯出迭代歷史指標至 CSV 檔案（含 v3 抗停滯診斷欄位）。"""
    if not history:
        return
    fieldnames = [
        'Iteration', 'Gbest_Fitness', 'Mean_Fitness', 'Alpha',
        'Validity', 'Uniqueness', 'Novelty', 'Mean_QED',
        'Diversity', 'Stagnation', 'N_Mutated',
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in history:
            writer.writerow({
                'Iteration': h['iteration'] + 1,
                'Gbest_Fitness': f"{h['gbest_fitness']:.6f}",
                'Mean_Fitness': f"{h.get('mean_fitness', 0):.6f}",
                'Alpha': f"{h.get('alpha', 0):.4f}",
                'Validity': f"{h.get('validity', 0):.4f}",
                'Uniqueness': f"{h.get('uniqueness', 0):.4f}",
                'Novelty': f"{h.get('novelty', 0):.4f}",
                'Mean_QED': f"{h.get('mean_qed', 0):.6f}",
                'Diversity': f"{h.get('diversity', 0):.4f}",
                'Stagnation': h.get('stagnation_counter', 0),
                'N_Mutated': h.get('n_mutated', 0),
            })
    print(f"  歷史指標已匯出至: {filepath}")


def export_molecules_csv(molecules: List[Dict], filepath: str):
    """匯出所有生成的有效分子至 CSV 檔案。"""
    if not molecules:
        return
    fieldnames = ['SMILES', 'QED']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        sorted_mols = sorted(molecules, key=lambda m: -m.get('qed', 0))
        for m in sorted_mols:
            writer.writerow({
                'SMILES': m['smiles'],
                'QED': f"{m.get('qed', 0):.6f}",
            })
    print(f"  分子列表已匯出至: {filepath}")


def export_archive_csv(archive: ParetoArchive, filepath: str):
    """匯出 Pareto Archive 非支配解至 CSV 檔案。"""
    front = archive.get_pareto_front()
    if not front:
        return
    fieldnames = ['Rank', 'Fitness', 'Mean_QED', 'Val_x_Uniq',
                  'N_Molecules', 'Top_SMILES']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        sorted_front = sorted(front, key=lambda x: -x['fitness'])
        for rank, entry in enumerate(sorted_front, 1):
            obj = entry.get('objectives', {})
            smiles_list = entry.get('smiles', [])
            writer.writerow({
                'Rank': rank,
                'Fitness': f"{entry['fitness']:.6f}",
                'Mean_QED': f"{obj.get('mean_qed', 0):.6f}",
                'Val_x_Uniq': f"{obj.get('val_x_uniq', 0):.6f}",
                'N_Molecules': len(smiles_list),
                'Top_SMILES': '; '.join(smiles_list[:5]),
            })
    print(f"  Pareto Archive 已匯出至: {filepath}")


# ============================================================================
# 結果分析與輸出
# ============================================================================

def analyze_best_result(
    best_params: np.ndarray,
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
):
    """
    對最佳參數進行詳細分析並輸出結果。

    Args:
        best_params: QPSO 找到的最佳參數
        kernel:      SQMGKernel 實例
        decoder:     MoleculeDecoder 實例
    """
    print("\n" + "=" * 70)
    print("最終結果分析")
    print("=" * 70)

    # 使用更多 shots 進行最終取樣以獲得更精確的統計
    original_shots = kernel.shots
    kernel.shots = max(original_shots * 4, 4096)

    counts = kernel.sample(best_params)
    decoded = decoder.decode_counts(counts)

    # 恢復原始 shots
    kernel.shots = original_shots

    # ── 統計彙整 ──
    total_unique = len(decoded)
    valid_results = [r for r in decoded if r['valid']]
    n_valid = len(valid_results)

    print(f"\n解碼統計（{kernel.shots * 4} shots 取樣）：")
    print(f"  不同 bit-string 數 : {total_unique}")
    print(f"  有效分子數         : {n_valid}")
    if total_unique > 0:
        print(f"  有效性比率 (Validity): {100 * n_valid / total_unique:.1f}%")

    if valid_results:
        qeds = [r['qed'] for r in valid_results]
        print(f"  平均 QED           : {np.mean(qeds):.4f}")
        print(f"  最高 QED           : {np.max(qeds):.4f}")

        # ── 所有有效分子排序輸出 ──
        print(f"\n所有有效分子（按 QED 降序）：")
        print(f"{'排名':>4}  {'SMILES':30s}  {'QED':>8}  {'計數':>6}  {'原子碼':20s}  {'鍵碼':15s}")
        print("-" * 90)

        sorted_valid = sorted(valid_results, key=lambda r: -r['qed'])
        for rank, r in enumerate(sorted_valid, 1):
            atom_str = " ".join(r['atom_codes'])
            bond_str = " ".join(r['bond_codes'])
            print(
                f"{rank:4d}  {r['smiles']:30s}  "
                f"{r['qed']:8.4f}  {r['count']:6d}  "
                f"{atom_str:20s}  {bond_str:15s}"
            )

        # ── 最佳分子詳細資訊 ──
        best_mol_record = sorted_valid[0]
        print(f"\n★ 最佳分子：")
        print(f"  SMILES   : {best_mol_record['smiles']}")
        print(f"  QED      : {best_mol_record['qed']:.6f}")
        print(f"  Bit-string: {best_mol_record['bitstring']}")
        print(f"  原子碼   : {best_mol_record['atom_codes']}")
        print(f"  鍵碼     : {best_mol_record['bond_codes']}")
    else:
        print("\n⚠ 未找到有效分子。建議：")
        print("  1. 增加粒子數（--particles）或迭代次數（--iterations）")
        print("  2. 增加取樣次數（--shots）")
        print("  3. 調整 α 權重以更重視有效性")

    # ── 輸出最佳參數（供後續使用）──
    print(f"\n最佳參數向量（可用於重現結果）：")
    print(f"  np.array({best_params.tolist()})")

    return valid_results


# ============================================================================
# 主流程 (Main Loop)
# ============================================================================

def main():
    """
    SQMG 主流程：初始化 → QPSO 優化迭代 → 結果輸出。

    這是整個「可擴展量子分子生成系統」的進入點。
    流程概覽：
    ┌────────────────────────────────────────────────────────┐
    │ 1. 設定 CUDA-Q 後端                                    │
    │ 2. 初始化 SQMGKernel (3N+2 量子線路)                   │
    │ 3. 初始化 MoleculeDecoder (bit-string → 分子)          │
    │ 4. 建立適應度函式 (kernel + decoder → fitness)          │
    │ 5. 初始化 QuantumOptimizer (QPSO)                      │
    │ 6. 執行 QPSO 優化迭代                                  │
    │    ┌──────────────────────────────────────────────────┐ │
    │    │ for each iteration:                             │ │
    │    │   for each particle:                            │ │
    │    │     params → CUDA-Q sample → decode → fitness   │ │
    │    │   update pbest, gbest, positions (QPSO)         │ │
    │    └──────────────────────────────────────────────────┘ │
    │ 7. 分析最佳結果，輸出高分分子                            │
    └────────────────────────────────────────────────────────┘
    """

    # ── 命令列參數 ──
    parser = argparse.ArgumentParser(
        description="SQMG — 可擴展量子分子生成系統 (Scalable Quantum Molecular Generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python main.py                         # 使用預設參數
  python main.py --max_atoms 6           # 支持最多 6 個重原子
  python main.py --particles 30 --iterations 100  # 更多粒子和迭代
  python main.py --backend tensornet     # 使用 GPU 張量網路後端
        """
    )
    parser.add_argument("--max_atoms", type=int, default=4,
                        help="最大重原子數量 N (default: 4)")
    parser.add_argument("--particles", type=int, default=20,
                        help="QPSO 粒子數量 M (default: 20)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="QPSO 最大迭代次數 T (default: 30)")
    parser.add_argument("--shots", type=int, default=512,
                        help="每次量子取樣的 shots 數 (default: 512)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="適應度權重: α×validity + (1-α)×QED (default: 0.4)")
    parser.add_argument("--alpha_max", type=float, default=1.0,
                        help="QPSO 收縮-擴張係數最大值 (default: 1.0)")
    parser.add_argument("--alpha_min", type=float, default=0.5,
                        help="QPSO 收縮-擴張係數最小值 (default: 0.5)")
    parser.add_argument("--backend", type=str, default="qpp-cpu",
                        choices=["qpp-cpu", "nvidia", "tensornet"],
                        help="CUDA-Q 模擬後端 (default: qpp-cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機數種子 (default: 42)")
    parser.add_argument("--verbose_eval", action="store_true",
                        help="每次適應度評估都印出詳細資訊")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="結果輸出目錄 (default: ./output)")
    parser.add_argument("--dimred", type=str, default="pca",
                        choices=["pca", "tsne"],
                        help="化學空間降維方法 (default: pca)")

    args = parser.parse_args()

    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 系統資訊 ──
    print("╔" + "═" * 68 + "╗")
    print("║  SQMG — 可擴展量子分子生成系統                                     ║")
    print("║  Scalable Quantum Molecular Generation with CUDA-Q & QPSO         ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CUDA-Q 版本: {cudaq.__version__ if hasattr(cudaq, '__version__') else '未知'}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"NumPy 版本 : {np.__version__}")

    # ── 檢查 NumPy 版本 ──
    np_major = int(np.__version__.split('.')[0])
    if np_major >= 2:
        print("\n⚠  警告：偵測到 NumPy 2.0+！")
        print("   Numpy 2.0 的 C++ ABI 變更可能導致 RDKit 崩潰。")
        print("   強烈建議降級：pip install 'numpy>=1.24,<2.0'")

    # ================================================================
    # Step 1: 設定 CUDA-Q 後端
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 1: 設定 CUDA-Q 模擬後端")
    print(f"{'─' * 70}")
    configure_cudaq_backend(args.backend)

    # ================================================================
    # Step 2: 初始化 SQMGKernel
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 2: 初始化 SQMG 3N+2 量子線路")
    print(f"{'─' * 70}")

    kernel = SQMGKernel(max_atoms=args.max_atoms, shots=args.shots)
    print(kernel.describe())

    # ================================================================
    # Step 3: 初始化 MoleculeDecoder
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 3: 初始化分子解碼器")
    print(f"{'─' * 70}")

    decoder = MoleculeDecoder(max_atoms=args.max_atoms)
    print(f"  Max atoms         : {decoder.max_atoms}")
    print(f"  Expected BS length: {decoder.expected_length}")
    print(f"  Atom mapping      : 000→NONE, 001→C, 010→O, 011→N, "
          f"100→S, 101→P, 110→F, 111→Cl")
    print(f"  Bond mapping      : 00→None, 01→Single, 10→Double, 11→Triple")

    # ================================================================
    # Step 4: 建立適應度函式
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 4: 建立適應度函式")
    print(f"{'─' * 70}")

    pareto_archive = ParetoArchive(
        max_size=100,
        objectives=('mean_qed', 'val_x_uniq'),
    )

    fitness_fn = create_fitness_function(
        kernel=kernel,
        decoder=decoder,
        alpha=args.alpha,
        verbose_eval=args.verbose_eval,
        pareto_archive=pareto_archive,
    )
    print(f"  Fitness = w1×QED + w2×SA + w3×Size_Reward + w4×Conn − Penalty")
    print(f"  Alpha (探索 vs 利用): {args.alpha}")
    print(f"  Pareto Archive 已初始化 (max_size=100, obj=mean_qed × val×uniq)")

    # ================================================================
    # Step 5: 初始化 Evaluator & 回呼
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 5: 初始化 MoleculeEvaluator & 迭代回呼")
    print(f"{'─' * 70}")

    evaluator = MoleculeEvaluator()
    extended_history: List[Dict] = []
    all_molecules: List[Dict] = []

    iteration_callback = create_iteration_callback(
        kernel=kernel,
        decoder=decoder,
        evaluator=evaluator,
        extended_history=extended_history,
        all_molecules=all_molecules,
    )
    print("  MoleculeEvaluator 已初始化")
    print(f"  參考分子集大小: {len(evaluator.reference_smiles)}")
    print(f"  輸出目錄: {args.output_dir}")

    # ================================================================
    # Step 6: 初始化 QPSO 優化器
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 6: 初始化 QPSO 量子粒子群優化器")
    print(f"{'─' * 70}")

    optimizer = QuantumOptimizer(
        n_params=kernel.n_params,
        n_particles=args.particles,
        max_iterations=args.iterations,
        fitness_fn=fitness_fn,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        seed=args.seed,
        verbose=True,
        iteration_callback=iteration_callback,
    )

    # ================================================================
    # Step 7: 執行 QPSO 優化（含迭代指標收集）
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 7: 執行 QPSO 優化迭代")
    print(f"{'─' * 70}")

    start_time = time.time()
    best_params, best_fitness, history = optimizer.optimize()
    elapsed = time.time() - start_time

    print(f"\n總耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")
    print(f"  extended_history 已收集 {len(extended_history)} 輪真實指標")

    # ================================================================
    # Step 8: 分析最佳結果
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 8: 分析最佳結果")
    print(f"{'─' * 70}")

    valid_results = analyze_best_result(best_params, kernel, decoder)

    # 將 analyze 結果中的有效分子也加入 all_molecules
    # 只取 valid=True 且 partial_valid 非 True 的分子
    if valid_results:
        existing_smiles = {m['smiles'] for m in all_molecules}
        for r in valid_results:
            smi = r.get('smiles')
            if smi and not r.get('partial_valid') and smi not in existing_smiles:
                all_molecules.append({
                    'smiles': smi,
                    'qed': r['qed'],
                    'mol': r.get('mol'),
                })
                existing_smiles.add(smi)

    # ── 最終評估指標 ──
    print(f"\n{'─' * 70}")
    print("評估指標 (Validity / Uniqueness / Novelty)：")
    print(f"{'─' * 70}")

    if valid_results:
        # 用最終取樣結果做完整評估
        final_decoded = []
        counts_final = kernel.sample(best_params)
        final_decoded = decoder.decode_counts(counts_final)
        final_metrics = evaluator.evaluate(final_decoded)
        print(evaluator.format_metrics(final_metrics))

    # ── 收斂曲線資料（ASCII）──
    print(f"\n{'─' * 70}")
    print("收斂曲線（gbest fitness vs. iteration）：")
    print(f"{'─' * 70}")
    iters, fitnesses = optimizer.get_convergence_curve()
    for it, fit in zip(iters, fitnesses):
        bar_len = int(fit * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  Iter {it + 1:3d}: {bar} {fit:.4f}")

    # ================================================================
    # Step 9: CSV 匯出 & 視覺化
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 9: CSV 匯出 & 視覺化")
    print(f"{'─' * 70}")

    # CSV 匯出
    export_history_csv(
        extended_history,
        os.path.join(args.output_dir, "history_metrics.csv"),
    )
    export_molecules_csv(
        all_molecules,
        os.path.join(args.output_dir, "generated_molecules.csv"),
    )
    export_archive_csv(
        pareto_archive,
        os.path.join(args.output_dir, "pareto_archive.csv"),
    )

    # Pareto Archive 摘要
    print(f"\n{'\u2500' * 70}")
    print("Pareto Archive (非支配解)\uff1a")
    print(f"{'\u2500' * 70}")
    print(pareto_archive.summary())

    # 視覺化圖表
    try:
        plot_paths = plot_all(
            history=extended_history,
            molecules=all_molecules,
            output_dir=args.output_dir,
            show=False,
            dimred_method=args.dimred,
        )
        print(f"\n  已生成 {len(plot_paths)} 張圖表。")
    except Exception as e:
        print(f"\n  [視覺化警告] 圖表生成失敗: {e}")
        print("  （可能缺少 matplotlib / seaborn / scikit-learn）")

    print(f"\n{'═' * 70}")
    print("SQMG 執行完成。所有結果已輸出至:")
    print(f"  {os.path.abspath(args.output_dir)}")
    print(f"{'═' * 70}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
