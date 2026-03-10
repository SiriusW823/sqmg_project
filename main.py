"""
==============================================================================
SQMG Main Loop — MOQPSO + 完整鄰接矩陣版本
==============================================================================
# MODIFIED: FIX-P0-1, FIX-P0-2, FIX-P1-1, FIX-QED, FIX-HYPER, FIX-CHEM, FIX-GPU

修改原因：
  • 主流程已從單目標 SOQPSO 改為多目標 MOQPSO
  • callback/CSV/圖表欄位需對齊新的 iter_record 結構
  • CUDA-Q 後端改為支援 mqpu 的原生多 GPU 非同步取樣
==============================================================================
"""

from __future__ import annotations

import argparse
import builtins
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

_original_print = builtins.print


def rank0_print(*args, **kwargs):
    if IS_MAIN_PROCESS:
        _original_print(*args, **kwargs)


builtins.print = rank0_print


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


# 修改原因：以 CUDA-Q 官方 mqpu 介面初始化多 GPU，而非使用已棄用 target 名稱。
def configure_cudaq_backend(target: str = 'nvidia-mqpu'):
    if cudaq is None:
        raise RuntimeError('無法匯入 CUDA-Q (cudaq)，請先安裝後再執行主流程。')
    try:
        # [FIX-BACKEND] tensornet 後端需要使用 tensornet 字串
        if target == 'tensornet':
            cudaq.set_target('tensornet')
        elif target == 'nvidia':
            # [FIX-BACKEND] 暫時設定為 nvidia，Step 1 結束後會由 FIX-GPU 升級為 mqpu
            cudaq.set_target('nvidia')
        elif target == 'nvidia-mqpu':
            cudaq.set_target('nvidia')
        else:
            cudaq.set_target(target)
        print(f'[CUDA-Q] 後端已設定為: {target}')
    except Exception as exc:
        print(f"[CUDA-Q] 警告：無法設定後端 '{target}'，改用 qpp-cpu。")
        print(f'        錯誤訊息: {exc}')
        cudaq.set_target('qpp-cpu')
        print('[CUDA-Q] 已降級為 qpp-cpu 後端。')


# 修改原因：fitness_fn 改為回傳 (validity, uniqueness, valid_smiles)，供 archive 正確保存分子資訊。
def create_fitness_function(
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
    verbose_eval: bool = False,
):
    eval_count = [0]

    # [FIX-P0-2] fitness_fn 改為回傳三個值：(validity, uniqueness, valid_smiles_list)
    def fitness_fn(params: np.ndarray) -> Tuple[float, float, List[str]]:
        """
        三目標回傳的適應度評估函式。

        流程：
        1. params → CUDA-Q kernel (cudaq.sample)
        2. bit-strings → MoleculeDecoder
        3. 分子 → RDKit → (validity, uniqueness, valid_smiles)
        """
        eval_count[0] += 1
        try:
            counts = kernel.sample(params)
            (validity, uniqueness), decoded = decoder.compute_fitness(counts)
            # [FIX-P0-2] 提取有效 SMILES 列表，供 archive.try_add() 使用
            valid_smiles = [
                r['smiles'] for r in decoded
                if r.get('valid') and r.get('smiles') and not r.get('partial_valid')
            ]

            if verbose_eval:
                valid_count = len(valid_smiles)
                print(
                    f"    [Eval #{eval_count[0]}] "
                    f"BS={len(decoded)} "
                    f"Valid={valid_count} "
                    f"val={validity:.4f} "
                    f"uniq={uniqueness:.4f}"
                )
            return (validity, uniqueness, valid_smiles)
        except Exception as e:
            if verbose_eval:
                print(f"    [Eval #{eval_count[0]}] 錯誤: {e}")
            return 0.0, 0.0, []

    return fitness_fn


# 修改原因：callback 全面對齊 MOQPSO iter_record 的實際 key，避免 KeyError 導致歷史記錄全歸零。
def create_iteration_callback(
    kernel: SQMGKernel,
    decoder: MoleculeDecoder,
    evaluator: MoleculeEvaluator,
    extended_history: List[Dict],
    all_molecules: List[Dict],
):
    seen_smiles = {molecule['smiles'] for molecule in all_molecules if molecule.get('smiles')}

    def callback(iteration: int, record: dict):
        # [FIX-P0-1] 修正 key 名稱：gbest_params → best_compromise_params
        gbest_params = record.get('best_compromise_params')
        validity = 0.0
        uniqueness = 0.0
        novelty = 0.0
        # [FIX-QED] 已移除 QED

        if gbest_params is None:
            # [FIX-P0-1] 更新警告訊息以符合新 key 名稱
            print(f"  [Callback Iter {iteration + 1}] 警告：best_compromise_params 為 None，Archive 可能為空")
        else:
            try:
                counts = kernel.sample(np.asarray(gbest_params, dtype=np.float64))
                decoded = decoder.decode_counts(counts)
                metrics = evaluator.evaluate(decoded)
                validity = metrics['validity']
                uniqueness = metrics['uniqueness']
                novelty = float(metrics.get('novelty', 0.0))
                # [FIX-QED] 已移除 QED

                for molecule in decoded:
                    if molecule.get('valid') and molecule.get('smiles') and not molecule.get('partial_valid'):
                        smiles = molecule['smiles']
                        if smiles not in seen_smiles:
                            all_molecules.append({
                                'smiles': smiles,
                                'qed': molecule.get('qed', 0.0),
                                'mol': molecule.get('mol'),
                            })
                            seen_smiles.add(smiles)
            except Exception as exc:
                print(f'  [Callback Iter {iteration + 1}] 評估失敗: {exc}')

        # [FIX-P0-1] 修正所有 key 名稱以符合 MOQPSOOptimizer.iter_record 的實際結構
        ext_record = {
            'iteration': record['iteration'],
            'gbest_fitness': record.get('best_validity', 0.0) + record.get('best_uniqueness', 0.0),
            'mean_fitness': record.get('mean_validity', 0.0),
            'max_fitness': record.get('max_validity', 0.0),
            'min_fitness': 0.0,
            'alpha': record.get('alpha', 0.0),
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            # [FIX-QED] 已移除 QED
            'diversity': record.get('diversity', 0.0),
            'stagnation_counter': record.get('stagnation_counter', 0),
            'n_mutated': record.get('n_mutated', 0),
        }
        extended_history.append(ext_record)

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
    print(f'  歷史指標已匯出至: {filepath}')


# 修改原因：Pareto Archive 匯出需保留 Top_SMILES，避免空白欄位。
def export_archive_csv(archive_entries: List[Dict], filepath: str):
    if not archive_entries:
        return

    fieldnames = ['Rank', 'Validity', 'Uniqueness', 'Compromise', 'Top_SMILES', 'Num_SMILES']
    with open(filepath, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, entry in enumerate(archive_entries, start=1):
            smiles_list = entry.get('smiles', []) or []
            writer.writerow({
                'Rank': rank,
                'Validity': f"{entry.get('objectives', {}).get('validity', 0.0):.6f}",
                'Uniqueness': f"{entry.get('objectives', {}).get('uniqueness', 0.0):.6f}",
                'Compromise': f"{entry.get('objectives', {}).get('validity', 0.0) * entry.get('objectives', {}).get('uniqueness', 0.0):.6f}",
                'Top_SMILES': ' | '.join(smiles_list[:5]),
                'Num_SMILES': len(smiles_list),
            })
    print(f'  Pareto Archive 已匯出至: {filepath}')


def export_molecules_csv(molecules: List[Dict], filepath: str):
    if not molecules:
        return
    with open(filepath, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['SMILES', 'QED'])
        writer.writeheader()
        for molecule in sorted(molecules, key=lambda item: -item.get('qed', 0.0)):
            writer.writerow({
                'SMILES': molecule.get('smiles', ''),
                'QED': f"{molecule.get('qed', 0.0):.6f}",
            })
    print(f'  分子列表已匯出至: {filepath}')


def analyze_best_result(best_params: np.ndarray, kernel: SQMGKernel, decoder: MoleculeDecoder):
    if best_params is None or len(best_params) != kernel.n_params:
        print('\n⚠ 無可分析的最佳參數。')
        return []

    print('\n' + '=' * 70)
    print('最終結果分析')
    print('=' * 70)

    original_shots = kernel.shots
    kernel.shots = max(original_shots * 4, 4096)
    counts = kernel.sample(best_params)
    decoded = decoder.decode_counts(counts)
    kernel.shots = original_shots

    valid_results = [item for item in decoded if item.get('valid') and not item.get('partial_valid')]
    print(f'\n不同 bit-string 數 : {len(decoded)}')
    print(f'有效分子數         : {len(valid_results)}')

    if valid_results:
        qeds = [item.get('qed', 0.0) for item in valid_results]
        print(f'平均 QED           : {np.mean(qeds):.4f}')
        print(f'最高 QED           : {np.max(qeds):.4f}')
        print('\n所有有效分子（按 QED 降序）：')
        for rank, item in enumerate(sorted(valid_results, key=lambda row: -row.get('qed', 0.0)), start=1):
            print(
                f"{rank:4d}  {item.get('smiles', ''):30s}  "
                f"QED={item.get('qed', 0.0):.4f}  count={item.get('count', 0)}"
            )
    else:
        print('\n⚠ 未找到有效分子。')

    print('\n最佳參數向量：')
    print(f'  np.array({best_params.tolist()})')
    return valid_results


def build_param_metadata(kernel: SQMGKernel) -> Tuple[List[int], List[Tuple[int, str]]]:
    atom_param_indices = kernel.get_atom_param_indices()
    bond_param_indices = kernel.get_bond_param_indices()
    return atom_param_indices, bond_param_indices


def main():
    if cudaq is None:
        print('=' * 70)
        print('錯誤：無法匯入 CUDA-Q (cudaq)。')
        print('請先安裝 CUDA-Q。')
        print('=' * 70)
        sys.exit(1)
    if RDLogger is None:
        print('=' * 70)
        print('錯誤：無法匯入 RDKit。')
        print('=' * 70)
        sys.exit(1)

    from evaluator import MoleculeEvaluator
    from molecule_decoder import MoleculeDecoder
    from plot_utils import plot_all
    from quantum_optimizer import MOQPSOOptimizer
    from sqmg_kernel import SQMGKernel

    parser = argparse.ArgumentParser(
        description='SQMG — Scalable Quantum Molecular Generation with MOQPSO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--max_atoms', type=int, default=4, help='最大重原子數量 N (default: 4)')
    # [FIX-HYPER] default=30
    parser.add_argument('--particles', type=int, default=30, help='MOQPSO 粒子數量 M (default: 30)')
    # [FIX-HYPER] default=150
    parser.add_argument('--iterations', type=int, default=150, help='MOQPSO 最大迭代次數 T (default: 150)')
    # [FIX-HYPER] default=1024
    parser.add_argument('--shots', type=int, default=1024, help='每次量子取樣的 shots 數 (default: 1024)')
    parser.add_argument('--alpha_max', type=float, default=1.2, help='MOQPSO α 最大值 (default: 1.2)')
    parser.add_argument('--alpha_min', type=float, default=0.4, help='MOQPSO α 最小值 (default: 0.4)')
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

    print('╔' + '═' * 68 + '╗')
    print('║  SQMG — 可擴展量子分子生成系統 (MOQPSO + CUDA-Q mqpu)            ║')
    print('╚' + '═' * 68 + '╝')
    print(f"\n啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'MPI Rank: {MPI_RANK} (Main Process: {IS_MAIN_PROCESS})')

    print(f"\n{'─' * 70}\nStep 1: 設定 CUDA-Q 模擬後端\n{'─' * 70}")
    configure_cudaq_backend(args.backend)
    # [FIX-GPU] 若後端為 nvidia，嘗試啟用多 QPU 模式以利用 8× V100
    if args.backend in {'nvidia', 'nvidia-mqpu'}:
        try:
            import os
            os.environ.setdefault('CUDAQ_MQPU_NGPUS', '8')
            cudaq.set_target("nvidia", option="mqpu")
            n_gpus = cudaq.num_available_gpus() if hasattr(cudaq, 'num_available_gpus') else 1
            print(f"[CUDA-Q] 多 QPU 模式已啟用，偵測到 {n_gpus} 個 GPU")
        except Exception as mqpu_err:
            print(f"[CUDA-Q] 多 QPU 模式啟用失敗（{mqpu_err}），維持單 GPU 模式")

    print(f"\n{'─' * 70}\nStep 2: 初始化完整鄰接矩陣 SQMG 量子線路\n{'─' * 70}")
    kernel = SQMGKernel(max_atoms=args.max_atoms, shots=args.shots)
    print(kernel.describe())

    print(f"\n{'─' * 70}\nStep 3: 初始化分子解碼器\n{'─' * 70}")
    decoder = MoleculeDecoder(max_atoms=args.max_atoms)

    print(f"\n{'─' * 70}\nStep 4: 建立多目標適應度函式\n{'─' * 70}")
    fitness_fn = create_fitness_function(kernel, decoder, args.verbose_eval)
    print('  Objectives = (Validity, Uniqueness)')

    print(f"\n{'─' * 70}\nStep 5: 初始化評估器與迭代回呼\n{'─' * 70}")
    evaluator = MoleculeEvaluator()
    extended_history: List[Dict] = []
    all_molecules: List[Dict] = []
    iteration_callback = create_iteration_callback(kernel, decoder, evaluator, extended_history, all_molecules)

    print(f"\n{'─' * 70}\nStep 6: 初始化 MOQPSO 優化器\n{'─' * 70}")
    # [FIX-CHEM] 建立 Chemistry Constraints 的 bond 參數索引
    # 參數佈局：前 3N 個為 atom 參數，後 N*(N-1) 個為 bond 參數
    # 每組 bond pair 佔 2 個參數：
    #   偏移 0 → 'single_double' 類型，對應 Single/Double 鍵切換，限制在 [0, π]
    #   偏移 1 → 'double_triple' 類型，對應 Double/Triple 鍵切換，限制在 [0, π/2]
    n_atom_params = 3 * args.max_atoms
    n_bond_pairs = args.max_atoms - 1
    bond_param_indices = []
    for bp_idx in range(n_bond_pairs):
        base = n_atom_params + 2 * bp_idx
        bond_param_indices.append((base, 'single_double'))
        bond_param_indices.append((base + 1, 'double_triple'))

    optimizer = MOQPSOOptimizer(
        n_params=kernel.n_params,
        n_particles=args.particles,
        max_iterations=args.iterations,
        fitness_fn=fitness_fn,
        kernel=kernel,
        decoder=decoder,
        shots=args.shots,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        seed=args.seed,
        verbose=True,
        iteration_callback=iteration_callback,
        # [FIX-CHEM] 傳入 Chemistry Constraints 參數
        bond_param_indices=bond_param_indices,
        use_chem_constraints=True,
        use_async_sampling=True,
    )

    print(f"\n{'─' * 70}\nStep 7: 執行 MOQPSO 優化\n{'─' * 70}")
    start_time = time.time()
    best_params, best_objectives, _history = optimizer.optimize()
    elapsed = time.time() - start_time
    best_score = best_objectives.get('validity', 0.0) * best_objectives.get('uniqueness', 0.0)

    print(f"\n總耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")
    print(f"  最佳 validity     : {best_objectives.get('validity', 0.0):.6f}")
    print(f"  最佳 uniqueness   : {best_objectives.get('uniqueness', 0.0):.6f}")
    print(f'  validity × uniqueness: {best_score:.6f}')

    print(f"\n{'─' * 70}\nStep 8: 分析最佳結果\n{'─' * 70}")
    valid_results = analyze_best_result(best_params, kernel, decoder)
    existing_smiles = {molecule['smiles'] for molecule in all_molecules if molecule.get('smiles')}
    for item in valid_results:
        smiles = item.get('smiles')
        if smiles and smiles not in existing_smiles:
            all_molecules.append({
                'smiles': smiles,
                'qed': item.get('qed', 0.0),
                'mol': item.get('mol'),
            })
            existing_smiles.add(smiles)

    if IS_MAIN_PROCESS:
        print(f"\n{'─' * 70}\nStep 9: 匯出結果與視覺化\n{'─' * 70}")
        export_history_csv(extended_history, os.path.join(args.output_dir, 'history_metrics.csv'))
        export_archive_csv(optimizer.archive.as_sorted_list(), os.path.join(args.output_dir, 'pareto_archive.csv'))
        export_molecules_csv(all_molecules, os.path.join(args.output_dir, 'generated_molecules.csv'))

        try:
            plot_paths = plot_all(
                history=extended_history,
                molecules=all_molecules,
                output_dir=args.output_dir,
                show=False,
                dimred_method=args.dimred,
            )
            print(f'\n  已生成 {len(plot_paths)} 張圖表。')
        except Exception as exc:
            print(f'\n  [視覺化警告] 圖表生成失敗: {exc}')

        print(f"\n{'═' * 70}")
        print('SQMG 執行完成。所有結果已輸出至:')
        print(f"  {os.path.abspath(args.output_dir)}")
        print(f"{'═' * 70}")


if __name__ == '__main__':
    main()