"""
==============================================================================
SQMG Main Loop — SOQPSO 單目標優化 + v5 線性相鄰鍵編碼
==============================================================================

主流程使用單目標 SOQPSO 極大化 fitness = validity * uniqueness。
支援 CUDA-Q mqpu 多 GPU 非同步取樣、化學先驗限制、MPI 多 Rank 獨立儲存。
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np


def get_mpi_rank() -> int:
    for env_var in ['OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'MV2_COMM_WORLD_RANK']:
        if env_var in os.environ:
            return int(os.environ[env_var])
    return 0


MPI_RANK = get_mpi_rank()
IS_MAIN_PROCESS = MPI_RANK == 0


def logger_print(*args, **kwargs):
    """僅在主進程 (Rank 0) 輸出，避免多進程重複列印。"""
    if IS_MAIN_PROCESS:
        print(*args, **kwargs)


try:
    import cudaq
except ImportError:
    cudaq = None

try:
    from rdkit import RDLogger

    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    RDLogger = None

if TYPE_CHECKING:
    from evaluator import MoleculeEvaluator
    from molecule_decoder import MoleculeDecoder
    from sqmg_kernel import SQMGKernel


# 修改原因：以 CUDA-Q 官方 mqpu 介面初始化多 GPU，整合單 GPU / 多 GPU 設定。
def configure_cudaq_backend(target: str = 'nvidia-mqpu'):
    if cudaq is None:
        raise RuntimeError('無法匯入 CUDA-Q (cudaq)，請先安裝後再執行主流程。')
    try:
        if target == 'tensornet':
            cudaq.set_target('tensornet')
        elif target in ('nvidia', 'nvidia-mqpu'):
            if target == 'nvidia-mqpu':
                os.environ.setdefault('CUDAQ_MQPU_NGPUS', '8')
                try:
                    cudaq.set_target('nvidia', option='mqpu')
                    n_gpus = cudaq.num_available_gpus() if hasattr(cudaq, 'num_available_gpus') else 1
                    logger_print(f"[CUDA-Q] 多 QPU 模式已啟用，偵測到 {n_gpus} 個 GPU")
                except Exception as mqpu_err:
                    logger_print(f"[CUDA-Q] 多 QPU 模式啟用失敗（{mqpu_err}），降級為單 GPU 模式")
                    cudaq.set_target('nvidia')
            else:
                cudaq.set_target('nvidia')
        else:
            cudaq.set_target(target)
        logger_print(f'[CUDA-Q] 後端已設定為: {target}')
    except Exception as exc:
        logger_print(f"[CUDA-Q] 警告：無法設定後端 '{target}'，改用 qpp-cpu。")
        logger_print(f'        錯誤訊息: {exc}')
        cudaq.set_target('qpp-cpu')
        logger_print('[CUDA-Q] 已降級為 qpp-cpu 後端。')


def create_fitness_function(
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
    verbose_eval: bool = False,
):
    eval_count = [0]

    def fitness_fn(params: np.ndarray) -> Tuple[float, List[str], List[dict]]:
        """
        單目標適應度評估函式。

        流程：
        1. params → CUDA-Q kernel (cudaq.sample)
        2. bit-strings → MoleculeDecoder
        3. 分子 → RDKit → fitness_score = validity * uniqueness
        """
        eval_count[0] += 1
        try:
            counts = kernel.sample(params)
            fitness_score, decoded = decoder.compute_fitness(counts)
            valid_smiles = [
                r['smiles'] for r in decoded
                if r.get('valid') and r.get('smiles') and not r.get('partial_valid')
            ]

            if verbose_eval:
                logger_print(
                    f"    [Eval #{eval_count[0]}] "
                    f"BS={len(decoded)} "
                    f"Valid={len(valid_smiles)} "
                    f"fitness={fitness_score:.4f}"
                )
            return fitness_score, valid_smiles, decoded
        except Exception as e:
            if verbose_eval:
                logger_print(f"    [Eval #{eval_count[0]}] 錯誤: {e}")
            return 0.0, [], []

    return fitness_fn


# 修改原因：callback 全面對齊 MOQPSO iter_record 的實際 key，避免 KeyError 導致歷史記錄全歸零。
def create_iteration_callback(
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
    evaluator: MoleculeEvaluator,
    extended_history: List[Dict],
    all_molecules: List[Dict],
    best_so_far: List[float],
):
    seen_smiles = {molecule['smiles'] for molecule in all_molecules if molecule.get('smiles')}

    def callback(iteration: int, record: dict):
        validity = 0.0
        uniqueness = 0.0
        novelty = 0.0

        decoded = record.get('gbest_decoded', [])
        if decoded:
            try:
                metrics = evaluator.evaluate(decoded)
                validity = metrics.get('validity', 0.0)
                uniqueness = metrics.get('uniqueness', 0.0)
                novelty = float(metrics.get('novelty', 0.0))

                for molecule in decoded:
                    if molecule.get('valid') and molecule.get('smiles') and not molecule.get('partial_valid'):
                        smiles = molecule['smiles']
                        if smiles not in seen_smiles:
                            all_molecules.append({
                                'smiles': smiles,
                                'mol': molecule.get('mol'),
                            })
                            seen_smiles.add(smiles)
            except Exception as exc:
                logger_print(f'  [Callback Iter {iteration + 1}] 評估失敗: {exc}')

        ext_record = {
            'iteration': record['iteration'],
            'gbest_fitness': record.get('gbest_fitness', 0.0),
            'mean_fitness': record.get('mean_fitness', 0.0),
            'max_fitness': record.get('max_fitness', 0.0),
            'min_fitness': 0.0,
            'alpha': record.get('alpha', 0.0),
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'diversity': record.get('diversity', 0.0),
            'stagnation_counter': record.get('stagnation_counter', 0),
            'n_mutated': record.get('n_mutated', 0),
        }
        extended_history.append(ext_record)

        # [Bug6] 突破新高時破例印出（所有 Rank 均可見）
        gbest_fitness = record.get('gbest_fitness', 0.0)
        if gbest_fitness > best_so_far[0]:
            best_so_far[0] = gbest_fitness
            print(
                f"\U0001f525 [Rank {MPI_RANK}] 突破新高! "
                f"Fitness: {gbest_fitness:.4f}  "
                f"Validity: {validity:.4f}  Uniqueness: {uniqueness:.4f}"
            )

    return callback


def export_history_csv(history: List[Dict], filepath: str):
    if not history:
        return

    # [FIX-QED] 移除 Mean_QED 欄位，QED 不是優化目標
    fieldnames = [
        'Iteration', 'Gbest_Fitness', 'Mean_Fitness', 'Alpha',
        'Validity', 'Uniqueness', 'Novelty',
        'Diversity', 'Stagnation', 'N_Mutated',
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow({
                'Iteration': int(record.get('iteration', 0)) + 1,
                'Gbest_Fitness': f"{record.get('gbest_fitness', 0.0):.6f}",
                'Mean_Fitness': f"{record.get('mean_fitness', 0.0):.6f}",
                'Alpha': f"{record.get('alpha', 0.0):.4f}",
                'Validity': f"{record.get('validity', 0.0):.4f}",
                'Uniqueness': f"{record.get('uniqueness', 0.0):.4f}",
                'Novelty': f"{record.get('novelty', 0.0):.4f}",
                # [FIX-QED] 已移除 已移除 的注釋
                'Diversity': f"{record.get('diversity', 0.0):.4f}",
                'Stagnation': int(record.get('stagnation_counter', 0)),
                'N_Mutated': int(record.get('n_mutated', 0)),
            })
    logger_print(f'  歷史指標已匯出至: {filepath}')


def export_molecules_csv(molecules: List[Dict], filepath: str):
    if not molecules:
        return
    with open(filepath, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['SMILES'])
        writer.writeheader()
        for molecule in molecules:
            writer.writerow({
                'SMILES': molecule.get('smiles', ''),
            })
    logger_print(f'  分子列表已匯出至: {filepath}')


def analyze_best_result(best_params: np.ndarray, kernel: SQMGKernel, decoder: MoleculeDecoder):
    if best_params is None or len(best_params) != kernel.n_params:
        logger_print('\n⚠ 無可分析的最佳參數。')
        return []

    logger_print('\n' + '=' * 70)
    logger_print('最終結果分析')
    logger_print('=' * 70)

    original_shots = kernel.shots
    kernel.shots = max(original_shots * 4, 4096)
    counts = kernel.sample(best_params)
    decoded = decoder.decode_counts(counts)
    kernel.shots = original_shots

    valid_results = [item for item in decoded if item.get('valid') and not item.get('partial_valid')]
    logger_print(f'\n不同 bit-string 數 : {len(decoded)}')
    logger_print(f'有效分子數         : {len(valid_results)}')

    if valid_results:
        logger_print('\n所有有效分子：')
        for rank, item in enumerate(valid_results, start=1):
            logger_print(
                f"{rank:4d}  {item.get('smiles', ''):30s}  "
                f"count={item.get('count', 0)}"
            )
    else:
        logger_print('\n⚠ 未找到有效分子。')

    logger_print('\n最佳參數向量：')
    logger_print(f'  np.array({best_params.tolist()})')
    return valid_results


def main():
    if cudaq is None:
        logger_print('=' * 70)
        logger_print('錯誤：無法匯入 CUDA-Q (cudaq)。')
        logger_print('請先安裝 CUDA-Q。')
        logger_print('=' * 70)
        sys.exit(1)
    if RDLogger is None:
        logger_print('=' * 70)
        logger_print('錯誤：無法匯入 RDKit。')
        logger_print('=' * 70)
        sys.exit(1)

    from evaluator import MoleculeEvaluator
    from molecule_decoder import MoleculeDecoder
    from plot_utils import plot_all
    from quantum_optimizer import SOQPSOOptimizer
    from sqmg_kernel import SQMGKernel

    parser = argparse.ArgumentParser(
        description='SQMG — Scalable Quantum Molecular Generation with SOQPSO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--max_atoms', type=int, default=4, help='最大重原子數量 N (default: 4)')
    # [FIX-HYPER] default=30
    parser.add_argument('--particles', type=int, default=30, help='SOQPSO 粒子數量 M (default: 30)')
    # [FIX-HYPER] default=150
    parser.add_argument('--iterations', type=int, default=150, help='SOQPSO 最大迭代次數 T (default: 150)')
    # [FIX-HYPER] default=1024
    parser.add_argument('--shots', type=int, default=1024, help='每次量子取樣的 shots 數 (default: 1024)')
    parser.add_argument('--alpha_max', type=float, default=1.2, help='SOQPSO α 最大值 (default: 1.2)')
    parser.add_argument('--alpha_min', type=float, default=0.4, help='SOQPSO α 最小值 (default: 0.4)')
    parser.add_argument(
        '--backend',
        type=str,
        default='nvidia-mqpu',
        choices=['qpp-cpu', 'nvidia', 'tensornet', 'nvidia-mqpu'],
        help='CUDA-Q 模擬後端 (default: nvidia-mqpu)',
    )
    parser.add_argument('--seed', type=int, default=42, help='隨機數種子 (default: 42)')
    parser.add_argument('--verbose_eval', action='store_true', help='輸出每次適應度評估資訊')
    parser.add_argument('--output_dir', type=str, default='./output', help='結果輸出目錄')
    parser.add_argument('--dimred', type=str, default='pca', choices=['pca', 'tsne'], help='化學空間降維方法')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger_print('╔' + '═' * 68 + '╗')
    logger_print('║  SQMG — 可擴展量子分子生成系統 (SOQPSO + CUDA-Q mqpu)            ║')
    logger_print('╚' + '═' * 68 + '╝')
    logger_print(f"\n啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger_print(f'MPI Rank: {MPI_RANK} (Main Process: {IS_MAIN_PROCESS})')

    logger_print(f"\n{'─' * 70}\nStep 1: 設定 CUDA-Q 模擬後端\n{'─' * 70}")
    configure_cudaq_backend(args.backend)

    logger_print(f"\n{'─' * 70}\nStep 2: 初始化線性相鄰鍵 (Linear Chain) SQMG 量子線路\n{'─' * 70}")
    kernel = SQMGKernel(max_atoms=args.max_atoms, shots=args.shots)
    logger_print(kernel.describe())

    logger_print(f"\n{'─' * 70}\nStep 3: 初始化分子解碼器\n{'─' * 70}")
    decoder = MoleculeDecoder(max_atoms=args.max_atoms)

    logger_print(f"\n{'─' * 70}\nStep 4: 建立單目標適應度函式\n{'─' * 70}")
    fitness_fn = create_fitness_function(kernel, decoder, args.verbose_eval)
    logger_print('  Objective = fitness_score = validity * uniqueness')

    logger_print(f"\n{'─' * 70}\nStep 5: 初始化評估器與迭代回呼\n{'─' * 70}")
    evaluator = MoleculeEvaluator()
    extended_history: List[Dict] = []
    all_molecules: List[Dict] = []
    best_so_far = [-np.inf]  # 用 list 以便在 callback 中修改
    iteration_callback = create_iteration_callback(kernel, decoder, evaluator, extended_history, all_molecules, best_so_far)

    logger_print(f"\n{'─' * 70}\nStep 6: 初始化 SOQPSO 優化器\n{'─' * 70}")
    # [FIX-CHEM] v5 Kernel 每個原子 15 個參數，每個鍵 6 個參數
    n_atom_params = 15 * args.max_atoms
    n_bond_pairs = args.max_atoms - 1
    bond_param_indices = []
    for bp_idx in range(n_bond_pairs):
        base = n_atom_params + 6 * bp_idx
        bond_param_indices.append((base, 'single_double'))
        bond_param_indices.append((base + 1, 'double_triple'))

    # [Bug5] 每個 Rank 使用不同的隨機種子，確保搜索軌跡不重疊
    actual_seed = args.seed + MPI_RANK

    optimizer = SOQPSOOptimizer(
        n_params=kernel.n_params,
        n_particles=args.particles,
        max_iterations=args.iterations,
        fitness_fn=fitness_fn,
        kernel=kernel,
        decoder=decoder,
        shots=args.shots,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        seed=actual_seed,
        verbose=True,
        iteration_callback=iteration_callback,
        bond_param_indices=bond_param_indices,
        use_chem_constraints=True,
        use_async_sampling=True,
    )

    logger_print(f"\n{'─' * 70}\nStep 7: 執行 SOQPSO 優化\n{'─' * 70}")
    start_time = time.time()
    best_params, best_fitness, _history = optimizer.optimize()
    elapsed = time.time() - start_time

    logger_print(f"\n總耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")
    logger_print(f'  最佳 fitness      : {best_fitness:.6f}')

    logger_print(f"\n{'─' * 70}\nStep 8: 分析最佳結果\n{'─' * 70}")
    valid_results = analyze_best_result(best_params, kernel, decoder)
    existing_smiles = {molecule['smiles'] for molecule in all_molecules if molecule.get('smiles')}
    for item in valid_results:
        smiles = item.get('smiles')
        if smiles and smiles not in existing_smiles:
            all_molecules.append({
                'smiles': smiles,
                'mol': item.get('mol'),
            })
            existing_smiles.add(smiles)

    # Final metrics summary
    if IS_MAIN_PROCESS:
        counts_final = kernel.sample(np.asarray(best_params, dtype=np.float64))
        final_decoded = decoder.decode_counts(counts_final)
        final_metrics = evaluator.evaluate(final_decoded)
        logger_print(f"\n{'─' * 70}")
        logger_print("最終評估指標 (Validity / Uniqueness / Novelty)：")
        logger_print(f"{'─' * 70}")
        logger_print(evaluator.format_metrics(final_metrics))

    logger_print(f"\n{'─' * 70}\nStep 9: 匯出結果與視覺化\n{'─' * 70}")
    export_history_csv(
        extended_history,
        os.path.join(args.output_dir, f'history_metrics_rank{MPI_RANK}.csv'),
    )
    export_molecules_csv(
        all_molecules,
        os.path.join(args.output_dir, f'generated_molecules_rank{MPI_RANK}.csv'),
    )

    rank_plot_dir = os.path.join(args.output_dir, f'plots_rank{MPI_RANK}')
    os.makedirs(rank_plot_dir, exist_ok=True)
    try:
        plot_paths = plot_all(
            history=extended_history,
            molecules=all_molecules,
            output_dir=rank_plot_dir,
            show=False,
            dimred_method=args.dimred,
        )
        logger_print(f'\n  已生成 {len(plot_paths)} 張圖表至 {rank_plot_dir}。')
    except Exception as exc:
        logger_print(f'\n  [視覺化警告] 圖表生成失敗: {exc}')

    logger_print(f"\n{'═' * 70}")
    logger_print('SQMG 執行完成。所有結果已輸出至:')
    logger_print(f"  {os.path.abspath(args.output_dir)}")
    logger_print(f"{'═' * 70}")


if __name__ == '__main__':
    main()