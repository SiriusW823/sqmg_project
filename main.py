"""
==============================================================================
SQMG Main Loop — 可擴展量子分子生成系統 主流程 (Multi-GPU Ready)
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

CUDA-Q 後端設定：
    • 'qpp-cpu'    — CPU 模擬（預設，不需 GPU）
    • 'nvidia'     — GPU 態向量模擬（需要 NVIDIA GPU + CUDA）
    • 'tensornet'  — GPU 張量網路模擬（大 N 時推薦）
    • 'nvidia-mgq' — 多 GPU 分散式模擬 (需透過 mpirun 啟動)
==============================================================================
"""

import argparse
import builtins
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

# ============================================================================
# MPI 多 GPU 環境設定 (避免輸出混亂與檔案寫入衝突)
# ============================================================================
def get_mpi_rank():
    """獲取當前 MPI Process 的 Rank，若無 MPI 則回傳 0"""
    for env_var in ['OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'MV2_COMM_WORLD_RANK']:
        if env_var in os.environ:
            return int(os.environ[env_var])
    return 0

MPI_RANK = get_mpi_rank()
IS_MAIN_PROCESS = (MPI_RANK == 0)

# 覆寫內建 print，只允許 Rank 0 輸出，保持終端機乾淨
_original_print = builtins.print
def rank0_print(*args, **kwargs):
    if IS_MAIN_PROCESS:
        _original_print(*args, **kwargs)
builtins.print = rank0_print


# ── CUDA-Q Import ──
try:
    import cudaq
except ImportError:
    print("=" * 70)
    print("錯誤：無法匯入 CUDA-Q (cudaq)。")
    print("請先安裝 CUDA-Q：")
    print("  pip install cuda-quantum-cu12")
    print("=" * 70)
    sys.exit(1)

# ── RDKit Import ──
try:
    from rdkit import Chem
    from rdkit import RDLogger
    # 抑制 RDKit 的冗長警告訊息
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
from quantum_optimizer import MOQPSOOptimizer, ParetoArchive
from evaluator import MoleculeEvaluator
from plot_utils import plot_all


# ============================================================================
# CUDA-Q 後端組態
# ============================================================================

def configure_cudaq_backend(target: str = "qpp-cpu"):
    """
    設定 CUDA-Q 模擬後端。
    """
    try:
        cudaq.set_target(target)
        print(f"[CUDA-Q] 後端已設定為: {target}")
    except Exception as e:
        print(f"[CUDA-Q] 警告：無法設定後端 '{target}'，使用預設後端。")
        print(f"        錯誤訊息: {e}")
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
    verbose_eval: bool = False,
):
    eval_count = [0]

    def fitness_fn(params: np.ndarray):
        eval_count[0] += 1
        try:
            counts = kernel.sample(params)
            (validity, uniqueness), decoded = decoder.compute_fitness(counts)

            if verbose_eval:
                valid_count = sum(
                    1 for r in decoded
                    if r.get('valid') and not r.get('partial_valid')
                )
                print(
                    f"    [Eval #{eval_count[0]}] "
                    f"BS={len(decoded)} "
                    f"Valid={valid_count} "
                    f"val={validity:.4f} "
                    f"uniq={uniqueness:.4f}"
                )
            return (validity, uniqueness)

        except Exception as e:
            if verbose_eval:
                print(f"    [Eval #{eval_count[0]}] 錯誤: {e}")
            return (0.0, 0.0)

    return fitness_fn


# ============================================================================
# 迭代回呼
# ============================================================================

def create_iteration_callback(
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
    evaluator: MoleculeEvaluator,
    extended_history: List[Dict],
    all_molecules: List[Dict],
):
    _seen_smiles: set = set()
    for m in all_molecules:
        _seen_smiles.add(m['smiles'])

    def callback(iteration: int, record: dict):
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
                metrics = evaluator.evaluate(decoded)

                validity = metrics['validity']
                uniqueness = metrics['uniqueness']
                novelty = metrics['novelty']
                mean_qed = metrics['mean_qed']

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
    front = archive.get_pareto_front()
    if not front:
        return
    fieldnames = ['Rank', 'Validity', 'Uniqueness', 'N_Molecules', 'Top_SMILES']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        sorted_front = sorted(front, key=lambda x: -sum(x['obj_vec']))
        for rank, entry in enumerate(sorted_front, 1):
            obj = entry.get('objectives', {})
            smiles_list = entry.get('smiles', [])
            writer.writerow({
                'Rank': rank,
                'Validity': f"{obj.get('validity', 0):.6f}",
                'Uniqueness': f"{obj.get('uniqueness', 0):.6f}",
                'N_Molecules': len(smiles_list),
                'Top_SMILES': '; '.join(smiles_list[:5]),
            })
    print(f"  Pareto Archive 已匯出至: {filepath}")


# ============================================================================
# 結果分析與輸出
# ============================================================================

def analyze_best_result(best_params: np.ndarray, kernel: SQMGKernel, decoder: MoleculeDecoder):
    print("\n" + "=" * 70)
    print("最終結果分析")
    print("=" * 70)

    original_shots = kernel.shots
    kernel.shots = max(original_shots * 4, 4096)
    counts = kernel.sample(best_params)
    decoded = decoder.decode_counts(counts)
    kernel.shots = original_shots

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

        best_mol_record = sorted_valid[0]
        print(f"\n★ 最佳分子：")
        print(f"  SMILES   : {best_mol_record['smiles']}")
        print(f"  QED      : {best_mol_record['qed']:.6f}")
        print(f"  Bit-string: {best_mol_record['bitstring']}")
        print(f"  原子碼   : {best_mol_record['atom_codes']}")
        print(f"  鍵碼     : {best_mol_record['bond_codes']}")
    else:
        print("\n⚠ 未找到有效分子。")

    print(f"\n最佳參數向量（可用於重現結果）：")
    print(f"  np.array({best_params.tolist()})")

    return valid_results


# ============================================================================
# 主流程 (Main Loop)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SQMG — 可擴展量子分子生成系統 (Multi-GPU Ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max_atoms", type=int, default=4,
                        help="最大重原子數量 N (default: 4)")
    parser.add_argument("--particles", type=int, default=20,
                        help="QPSO 粒子數量 M (default: 20)")
    parser.add_argument("--iterations", type=int, default=30,
                        help="QPSO 最大迭代次數 T (default: 30)")
    parser.add_argument("--shots", type=int, default=512,
                        help="每次量子取樣的 shots 數 (default: 512)")
    parser.add_argument("--alpha_max", type=float, default=1.0,
                        help="QPSO 收縮-擴張係數最大值 (default: 1.0)")
    parser.add_argument("--alpha_min", type=float, default=0.5,
                        help="QPSO 收縮-擴張係數最小值 (default: 0.5)")
    
    # 【已修正】這裡將 nvidia-mgq 加入白名單
    parser.add_argument("--backend", type=str, default="qpp-cpu",
                        choices=["qpp-cpu", "nvidia", "tensornet", "nvidia-mgq"],
                        help="CUDA-Q 模擬後端 (新增 nvidia-mgq 支援多 GPU 分散運算)")
    
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

    # 建立輸出目錄 (所有 Rank 都可以安全執行，exist_ok 保證不報錯)
    os.makedirs(args.output_dir, exist_ok=True)

    print("╔" + "═" * 68 + "╗")
    print("║  SQMG — 可擴展量子分子生成系統                                     ║")
    print("║  Scalable Quantum Molecular Generation with CUDA-Q & QPSO         ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MPI Rank: {MPI_RANK} (Main Process: {IS_MAIN_PROCESS})")

    # ================================================================
    # Step 1~6: 初始化元件
    # ================================================================
    print(f"\n{'─' * 70}\nStep 1: 設定 CUDA-Q 模擬後端\n{'─' * 70}")
    configure_cudaq_backend(args.backend)

    print(f"\n{'─' * 70}\nStep 2: 初始化 SQMG 3N+2 量子線路\n{'─' * 70}")
    kernel = SQMGKernel(max_atoms=args.max_atoms, shots=args.shots)
    print(kernel.describe())

    print(f"\n{'─' * 70}\nStep 3: 初始化分子解碼器\n{'─' * 70}")
    decoder = MoleculeDecoder(max_atoms=args.max_atoms)

    print(f"\n{'─' * 70}\nStep 4: 建立適應度函式\n{'─' * 70}")
    pareto_archive = ParetoArchive(max_size=100, objectives=('validity', 'uniqueness'))
    fitness_fn = create_fitness_function(kernel, decoder, args.verbose_eval)

    print(f"\n{'─' * 70}\nStep 5: 初始化 MoleculeEvaluator & 迭代回呼\n{'─' * 70}")
    evaluator = MoleculeEvaluator()
    extended_history: List[Dict] = []
    all_molecules: List[Dict] = []
    iteration_callback = create_iteration_callback(
        kernel, decoder, evaluator, extended_history, all_molecules
    )

    print(f"\n{'─' * 70}\nStep 6: 初始化 MOQPSO 多目標量子粒子群優化器\n{'─' * 70}")
    optimizer = MOQPSOOptimizer(
        n_params=kernel.n_params,
        n_particles=args.particles,
        max_iterations=args.iterations,
        fitness_fn=fitness_fn,
        archive=pareto_archive,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        seed=args.seed,
        verbose=True,
        iteration_callback=iteration_callback,
    )

    # ================================================================
    # Step 7: 執行 MOQPSO 多目標優化
    # ================================================================
    print(f"\n{'─' * 70}\nStep 7: 執行 MOQPSO 多目標優化迭代\n{'─' * 70}")
    start_time = time.time()
    best_params, best_obj, history = optimizer.optimize()
    elapsed = time.time() - start_time

    print(f"\n總耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")
    print(f"  最優折中解: validity={best_obj[0]:.4f}, uniqueness={best_obj[1]:.4f}")

    # ================================================================
    # Step 8: 分析最佳結果
    # ================================================================
    print(f"\n{'─' * 70}\nStep 8: 分析最佳結果\n{'─' * 70}")
    valid_results = analyze_best_result(best_params, kernel, decoder)

    if valid_results:
        existing_smiles = {m['smiles'] for m in all_molecules}
        for r in valid_results:
            smi = r.get('smiles')
            if smi and not r.get('partial_valid') and smi not in existing_smiles:
                all_molecules.append({
                    'smiles': smi, 'qed': r['qed'], 'mol': r.get('mol'),
                })
                existing_smiles.add(smi)

    # ================================================================
    # Step 9: CSV 匯出 & 視覺化 (僅限主程序執行)
    # ================================================================
    if IS_MAIN_PROCESS:
        print(f"\n{'─' * 70}\nStep 9: CSV 匯出 & 視覺化\n{'─' * 70}")

        export_history_csv(extended_history, os.path.join(args.output_dir, "history_metrics.csv"))
        export_molecules_csv(all_molecules, os.path.join(args.output_dir, "generated_molecules.csv"))
        export_archive_csv(pareto_archive, os.path.join(args.output_dir, "pareto_archive.csv"))

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

        print(f"\n{'═' * 70}")
        print("SQMG 執行完成。所有結果已輸出至:")
        print(f"  {os.path.abspath(args.output_dir)}")
        print(f"{'═' * 70}")

if __name__ == "__main__":
    main()
