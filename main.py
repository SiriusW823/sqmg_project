"""
==============================================================================
SQMG Main Loop — SOQPSO 單目標優化 + v5 全上三角鍵編碼
==============================================================================

主流程使用單目標 SOQPSO 極大化 fitness = validity * uniqueness。
單一 Python 行程 + CUDA-Q tensornet 後端，多 GPU 並行由 cuTensorNet 函式庫內部處理。
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


def configure_cudaq_backend(target: str = 'tensornet'):
    """CUDA-Q 後端初始化。tensornet 多 GPU 由 cuTensorNet 內部處理，對 Python 層透明。"""
    if cudaq is None:
        raise RuntimeError('無法匯入 CUDA-Q (cudaq)，請先安裝後再執行主流程。')
    try:
        if target == 'tensornet':
            # tensornet 多 GPU：設定 splitindex 策略
            # 同時設定兩個變數以兼容不同 CUDA-Q / cuTensorNet 版本
            os.environ.setdefault('CUDAQ_MGPU_STRATEGY', 'splitindex')
            os.environ.setdefault('CUTENSORNET_MGPU_STRATEGY', 'splitindex')
            cudaq.set_target('tensornet')
            try:
                n_gpus = cudaq.num_available_gpus() if hasattr(cudaq, 'num_available_gpus') else 1
                print(f"[CUDA-Q] tensornet 後端已啟用，偵測到 {n_gpus} 個 GPU")
            except Exception:
                pass
        elif target == 'nvidia':
            cudaq.set_target('nvidia')
        else:
            cudaq.set_target(target)
        print(f'[CUDA-Q] 後端已設定為: {target}')
    except Exception as exc:
        print(f"[CUDA-Q] 警告：無法設定後端 '{target}'，改用 qpp-cpu。")
        print(f'        錯誤訊息: {exc}')
        cudaq.set_target('qpp-cpu')
        print('[CUDA-Q] 已降級為 qpp-cpu 後端。')


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
                print(
                    f"    [Eval #{eval_count[0]}] "
                    f"BS={len(decoded)} "
                    f"Valid={len(valid_smiles)} "
                    f"fitness={fitness_score:.4f}"
                )
            return fitness_score, valid_smiles, decoded
        except Exception as e:
            if verbose_eval:
                print(f"    [Eval #{eval_count[0]}] 錯誤: {e}")
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
                print(f'  [Callback Iter {iteration + 1}] 評估失敗: {exc}')

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

        # 突破新高時印出
        gbest_fitness = record.get('gbest_fitness', 0.0)
        if gbest_fitness > best_so_far[0]:
            best_so_far[0] = gbest_fitness
            print(
                f"\U0001f525 突破新高! "
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
    print(f'  歷史指標已匯出至: {filepath}')


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
    print(f'  分子列表已匯出至: {filepath}')


def analyze_best_result(
    best_params: np.ndarray, kernel: SQMGKernel, decoder: MoleculeDecoder,
) -> Tuple[List[Dict], List[Dict]]:
    """回傳 (valid_results, decoded)，讓呼叫端可直接複用高精度取樣結果。"""
    if best_params is None or len(best_params) != kernel.n_params:
        print('\n⚠ 無可分析的最佳參數。')
        return [], []

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
        print('\n所有有效分子：')
        for rank, item in enumerate(valid_results, start=1):
            print(
                f"{rank:4d}  {item.get('smiles', ''):30s}  "
                f"count={item.get('count', 0)}"
            )
    else:
        print('\n⚠ 未找到有效分子。')

    print('\n最佳參數向量：')
    print(f'  np.array({best_params.tolist()})')
    return valid_results, decoded


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
        default='tensornet',
        choices=['qpp-cpu', 'nvidia', 'tensornet'],
        help='CUDA-Q 模擬後端 (default: tensornet)',
    )
    parser.add_argument('--seed', type=int, default=42, help='隨機數種子 (default: 42)')
    parser.add_argument('--verbose_eval', action='store_true', help='輸出每次適應度評估資訊')
    parser.add_argument('--output_dir', type=str, default='./output', help='結果輸出目錄')
    parser.add_argument('--dimred', type=str, default='pca', choices=['pca', 'tsne'], help='化學空間降維方法')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('╔' + '═' * 68 + '╗')
    print('║  SQMG — 可擴展量子分子生成系統 (SOQPSO + CUDA-Q tensornet)       ║')
    print('╚' + '═' * 68 + '╝')
    print(f"\n啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n{'─' * 70}\nStep 1: 設定 CUDA-Q 模擬後端\n{'─' * 70}")
    configure_cudaq_backend(args.backend)

    print(f"\n{'─' * 70}\nStep 2: 初始化全上三角 (Full Upper-Triangular) SQMG 量子線路\n{'─' * 70}")
    kernel = SQMGKernel(max_atoms=args.max_atoms, shots=args.shots)
    print(kernel.describe())

    print(f"\n{'─' * 70}\nStep 3: 初始化分子解碼器\n{'─' * 70}")
    decoder = MoleculeDecoder(max_atoms=args.max_atoms)

    print(f"\n{'─' * 70}\nStep 4: 建立單目標適應度函式\n{'─' * 70}")
    fitness_fn = create_fitness_function(kernel, decoder, args.verbose_eval)
    print('  Objective = fitness_score = validity * uniqueness')

    print(f"\n{'─' * 70}\nStep 5: 初始化評估器與迭代回呼\n{'─' * 70}")
    evaluator = MoleculeEvaluator()
    extended_history: List[Dict] = []
    all_molecules: List[Dict] = []
    best_so_far = [0.0]  # 初始 0.0，使第一次有效 fitness 突破時能正確印出
    iteration_callback = create_iteration_callback(kernel, decoder, evaluator, extended_history, all_molecules, best_so_far)

    print(f"\n{'─' * 70}\nStep 6: 初始化 SOQPSO 優化器\n{'─' * 70}")
    # 從 Kernel 中提取帶有化學先驗限制的 Array 邊界
    lower_bounds, upper_bounds = kernel.get_param_bounds()

    # 全上三角鍵結：N*(N-1)/2 bonds，每個 bond 3 params（v7）
    # Param 0 per bond (base)   → 'bond_existence'    : |00⟩→|10⟩（無鍵→單鍵），[0, π/2]
    # Param 1 per bond (base+1) → 'bond_order'        : |10⟩→|11⟩（單鍵→雙鍵），[0, π/2]
    # Param 2 per bond (base+2) → 'bond_triple_order' : |11⟩→|01⟩（雙鍵→三鍵），[0, π/2]
    n_bonds = kernel.n_bonds
    bond_param_indices = []
    for bp_idx in range(n_bonds):
        base = kernel.n_atom_params + 3 * bp_idx
        bond_param_indices.append((base,     'bond_existence'))
        bond_param_indices.append((base + 1, 'bond_order'))
        bond_param_indices.append((base + 2, 'bond_triple_order'))

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
        param_lower=lower_bounds,
        param_upper=upper_bounds,
        seed=args.seed,
        verbose=True,
        iteration_callback=iteration_callback,
        bond_param_indices=bond_param_indices,
        use_chem_constraints=True,
    )

    print(f"\n{'─' * 70}\nStep 7: 執行 SOQPSO 優化\n{'─' * 70}")
    start_time = time.time()
    best_params, best_fitness, _history = optimizer.optimize()
    elapsed = time.time() - start_time

    print(f"\n總耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")
    print(f'  最佳 fitness      : {best_fitness:.6f}')

    print(f"\n{'─' * 70}\nStep 8: 分析最佳結果\n{'─' * 70}")
    valid_results, final_decoded = analyze_best_result(best_params, kernel, decoder)
    existing_smiles = {molecule['smiles'] for molecule in all_molecules if molecule.get('smiles')}
    for item in valid_results:
        smiles = item.get('smiles')
        if smiles and smiles not in existing_smiles:
            all_molecules.append({
                'smiles': smiles,
                'mol': item.get('mol'),
            })
            existing_smiles.add(smiles)

    # Final metrics summary — 直接複用 analyze_best_result 的 4× shots 高精度取樣
    final_metrics = evaluator.evaluate(final_decoded)
    print(f"\n{'─' * 70}")
    print("最終評估指標 (Validity / Uniqueness / Novelty)：")
    print(f"{'─' * 70}")
    print(evaluator.format_metrics(final_metrics))

    print(f"\n{'─' * 70}\nStep 9: 匯出結果與視覺化\n{'─' * 70}")
    export_history_csv(
        extended_history,
        os.path.join(args.output_dir, 'history_metrics.csv'),
    )
    export_molecules_csv(
        all_molecules,
        os.path.join(args.output_dir, 'generated_molecules.csv'),
    )

    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    try:
        plot_paths = plot_all(
            history=extended_history,
            molecules=all_molecules,
            output_dir=plot_dir,
            show=False,
            dimred_method=args.dimred,
        )
        print(f'\n  已生成 {len(plot_paths)} 張圖表至 {plot_dir}。')
    except Exception as exc:
        print(f'\n  [視覺化警告] 圖表生成失敗: {exc}')

    print(f"\n{'═' * 70}")
    print('SQMG 執行完成。所有結果已輸出至:')
    print(f"  {os.path.abspath(args.output_dir)}")
    print(f"{'═' * 70}")


if __name__ == '__main__':
    main()
