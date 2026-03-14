"""
==============================================================================
SQMG Main Loop — SOQPSO 單目標優化 (v13 closure-kernel 版)
==============================================================================

v13 主要變更：
  - SQMGKernel 改用 closure pattern（sqmg_kernel.py v13）
    build_sqmg_kernel(max_atoms) 在類別初始化時固化所有尺寸常數。
    kernel.sample(params) 只傳 thetas，不再傳 n_atom_qubits / n_bonds。
  - 對應 molecule_decoder.py v8（連通性強制）與 quantum_optimizer.py v9（lb_vec[0] fix）。
  - 執行結束後輸出 bitstring-count 與 shot-weighted 雙重指標對比。

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
    from evaluator       import MoleculeEvaluator
    from molecule_decoder import MoleculeDecoder
    from sqmg_kernel      import SQMGKernel


# ===========================================================================
# CUDA-Q 後端設定
# ===========================================================================

def configure_cudaq_backend(target: str = 'tensornet'):
    """
    初始化 CUDA-Q 後端。

    後端選擇指南：
      qpp-cpu  : N ≤ 5，僅供本地測試
      nvidia   : N ≤ 9，單 GPU statevector（V100 32GB 可行）
      tensornet: N ≤ 40，推薦，cuTensorNet MPS 壓縮，多 GPU 自動並行

    tensornet 多 GPU 由 cuTensorNet 函式庫內部處理（splitindex 策略），
    對 Python 層完全透明。
    """
    if cudaq is None:
        raise RuntimeError('無法匯入 CUDA-Q (cudaq)。')
    try:
        if target == 'tensornet':
            os.environ.setdefault('CUDAQ_MGPU_STRATEGY',   'splitindex')
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
        print(f"[CUDA-Q] 警告：無法設定後端 '{target}'，改用 qpp-cpu。\n        錯誤: {exc}")
        cudaq.set_target('qpp-cpu')
        print('[CUDA-Q] 已降級為 qpp-cpu 後端。')


# ===========================================================================
# 適應度函式工廠
# ===========================================================================

def create_fitness_function(
    kernel:      'SQMGKernel',
    decoder:     'MoleculeDecoder',
    verbose_eval: bool = False,
):
    """
    建立單目標適應度評估函式。

    流程：params → cudaq.sample → MoleculeDecoder → fitness = validity × uniqueness
    """
    eval_count = [0]

    def fitness_fn(params: np.ndarray) -> Tuple[float, List[str], List[dict]]:
        eval_count[0] += 1
        try:
            counts        = kernel.sample(params)
            fitness_score, decoded = decoder.compute_fitness(counts)
            valid_smiles  = [
                r['smiles'] for r in decoded
                if r.get('valid') and r.get('smiles') and not r.get('partial_valid')
            ]
            if verbose_eval:
                print(
                    f"    [Eval #{eval_count[0]}] "
                    f"BS={len(decoded)}  Valid={len(valid_smiles)}  "
                    f"fitness={fitness_score:.4f}"
                )
            return fitness_score, valid_smiles, decoded
        except Exception as e:
            if verbose_eval:
                print(f"    [Eval #{eval_count[0]}] 錯誤: {e}")
            return 0.0, [], []

    return fitness_fn


# ===========================================================================
# 迭代回呼工廠
# ===========================================================================

def create_iteration_callback(
    kernel:           'SQMGKernel',
    decoder:          'MoleculeDecoder',
    evaluator:        'MoleculeEvaluator',
    extended_history: List[Dict],
    all_molecules:    List[Dict],
    best_so_far:      List[float],
):
    seen_smiles = {m['smiles'] for m in all_molecules if m.get('smiles')}

    def callback(iteration: int, record: dict):
        validity = uniqueness = novelty = 0.0
        decoded  = record.get('gbest_decoded', [])

        if decoded:
            try:
                metrics   = evaluator.evaluate(decoded)
                validity  = metrics.get('validity',   0.0)
                uniqueness = metrics.get('uniqueness', 0.0)
                novelty   = float(metrics.get('novelty', 0.0))

                for mol_rec in decoded:
                    smiles = mol_rec.get('smiles')
                    if (mol_rec.get('valid') and smiles
                            and not mol_rec.get('partial_valid')
                            and smiles not in seen_smiles):
                        all_molecules.append({'smiles': smiles, 'mol': mol_rec.get('mol')})
                        seen_smiles.add(smiles)
            except Exception as exc:
                print(f'  [Callback Iter {iteration + 1}] 評估失敗: {exc}')

        extended_history.append({
            'iteration':          record['iteration'],
            'gbest_fitness':      record.get('gbest_fitness', 0.0),
            'mean_fitness':       record.get('mean_fitness',  0.0),
            'max_fitness':        record.get('max_fitness',   0.0),
            'alpha':              record.get('alpha',         0.0),
            'validity':           validity,
            'uniqueness':         uniqueness,
            'novelty':            novelty,
            'diversity':          record.get('diversity',           0.0),
            'stagnation_counter': record.get('stagnation_counter',  0),
            'n_mutated':          record.get('n_mutated',           0),
        })

        gbest_fitness = record.get('gbest_fitness', 0.0)
        if gbest_fitness > best_so_far[0]:
            best_so_far[0] = gbest_fitness
            print(
                f"\U0001f525 突破新高! "
                f"Fitness={gbest_fitness:.4f}  "
                f"Validity={validity:.4f}  Uniqueness={uniqueness:.4f}"
            )

    return callback


# ===========================================================================
# CSV 匯出
# ===========================================================================

def export_history_csv(history: List[Dict], filepath: str):
    if not history:
        return
    fieldnames = [
        'Iteration', 'Gbest_Fitness', 'Mean_Fitness', 'Alpha',
        'Validity', 'Uniqueness', 'Novelty',
        'Diversity', 'Stagnation', 'N_Mutated',
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in history:
            writer.writerow({
                'Iteration':      int(r.get('iteration', 0)) + 1,
                'Gbest_Fitness':  f"{r.get('gbest_fitness',  0.0):.6f}",
                'Mean_Fitness':   f"{r.get('mean_fitness',   0.0):.6f}",
                'Alpha':          f"{r.get('alpha',          0.0):.4f}",
                'Validity':       f"{r.get('validity',       0.0):.4f}",
                'Uniqueness':     f"{r.get('uniqueness',     0.0):.4f}",
                'Novelty':        f"{r.get('novelty',        0.0):.4f}",
                'Diversity':      f"{r.get('diversity',      0.0):.4f}",
                'Stagnation':     int(r.get('stagnation_counter', 0)),
                'N_Mutated':      int(r.get('n_mutated',          0)),
            })
    print(f'  歷史指標已匯出至: {filepath}')


def export_molecules_csv(molecules: List[Dict], filepath: str):
    if not molecules:
        return
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['SMILES'])
        writer.writeheader()
        for m in molecules:
            writer.writerow({'SMILES': m.get('smiles', '')})
    print(f'  分子列表已匯出至: {filepath}')


# ===========================================================================
# 最終結果分析
# ===========================================================================

def analyze_best_result(
    best_params: np.ndarray,
    kernel:      'SQMGKernel',
    decoder:     'MoleculeDecoder',
) -> Tuple[List[Dict], List[Dict]]:
    """
    用 4× shots 高精度取樣分析最佳參數。

    Returns:
        (valid_results, decoded)
    """
    if best_params is None or len(best_params) != kernel.n_params:
        print('\n⚠ 無可分析的最佳參數。')
        return [], []

    print('\n' + '=' * 70)
    print('最終結果分析（4× shots 高精度取樣）')
    print('=' * 70)

    original_shots = kernel.shots
    kernel.shots   = max(original_shots * 4, 4096)
    counts         = kernel.sample(best_params)
    decoded        = decoder.decode_counts(counts)
    kernel.shots   = original_shots

    valid_results  = [r for r in decoded if r.get('valid') and not r.get('partial_valid')]
    print(f'\n不同 bit-string 數 : {len(decoded)}')
    print(f'有效分子數         : {len(valid_results)}')
    print(decoder.summarize(decoded))

    if valid_results:
        print('\n所有有效分子（前 30 個）：')
        for rank, item in enumerate(valid_results[:30], start=1):
            print(f"  {rank:4d}  {item.get('smiles',''):35s}  count={item.get('count', 0)}")
    else:
        print('\n⚠ 未找到有效分子。')

    return valid_results, decoded


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    if cudaq is None:
        print('錯誤：無法匯入 CUDA-Q (cudaq)，請先安裝後再執行。')
        sys.exit(1)
    if RDLogger is None:
        print('錯誤：無法匯入 RDKit，請先安裝後再執行。')
        sys.exit(1)

    from evaluator        import MoleculeEvaluator
    from molecule_decoder import MoleculeDecoder
    from plot_utils       import plot_all
    from quantum_optimizer import SOQPSOOptimizer
    from sqmg_kernel       import SQMGKernel

    parser = argparse.ArgumentParser(
        description='SQMG — Scalable Quantum Molecular Generation (v13)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--max_atoms',  type=int,   default=9,
                        help='最大重原子數量 N（目標 N=9）')
    parser.add_argument('--particles',  type=int,   default=30,
                        help='SOQPSO 粒子數量 M')
    parser.add_argument('--iterations', type=int,   default=150,
                        help='SOQPSO 最大迭代次數 T')
    parser.add_argument('--shots',      type=int,   default=1024,
                        help='每次量子取樣的 shots 數')
    parser.add_argument('--alpha_max',  type=float, default=1.2,
                        help='SOQPSO α 最大值')
    parser.add_argument('--alpha_min',  type=float, default=0.4,
                        help='SOQPSO α 最小值')
    parser.add_argument(
        '--backend', type=str, default='tensornet',
        choices=['qpp-cpu', 'nvidia', 'tensornet'],
        help='CUDA-Q 模擬後端（N=9 推薦 tensornet）',
    )
    parser.add_argument('--seed',        type=int,  default=42)
    parser.add_argument('--verbose_eval', action='store_true',
                        help='輸出每次適應度評估資訊')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--dimred', type=str, default='pca',
                        choices=['pca', 'tsne'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('╔' + '═' * 68 + '╗')
    print('║  SQMG v13 — Scalable Quantum Molecular Generation                 ║')
    print('║  SOQPSO + CUDA-Q tensornet (closure-fixed kernel)                 ║')
    print('╚' + '═' * 68 + '╝')
    print(f"\n啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Step 1: CUDA-Q 後端 ──────────────────────────────────────────
    print(f"\n{'─'*70}\nStep 1: 設定 CUDA-Q 模擬後端\n{'─'*70}")
    configure_cudaq_backend(args.backend)

    # ── Step 2: 初始化 SQMGKernel（v13 closure pattern）─────────────
    print(f"\n{'─'*70}\nStep 2: 初始化量子線路（v13 closure-fixed）\n{'─'*70}")
    kernel = SQMGKernel(max_atoms=args.max_atoms, shots=args.shots)
    print(kernel.describe())

    # ── Step 3: 分子解碼器 ───────────────────────────────────────────
    print(f"\n{'─'*70}\nStep 3: 初始化分子解碼器（v8 連通性修復）\n{'─'*70}")
    decoder = MoleculeDecoder(max_atoms=args.max_atoms)
    print(f"  期望 bitstring 長度: {decoder.expected_length}  (N²+2N = {args.max_atoms**2 + 2*args.max_atoms})")

    # ── Step 4: 適應度函式 ───────────────────────────────────────────
    print(f"\n{'─'*70}\nStep 4: 建立適應度函式\n{'─'*70}")
    fitness_fn = create_fitness_function(kernel, decoder, args.verbose_eval)
    print('  Objective = fitness = validity × uniqueness  (shot-weighted)')

    # ── Step 5: 評估器與回呼 ─────────────────────────────────────────
    print(f"\n{'─'*70}\nStep 5: 初始化評估器\n{'─'*70}")
    evaluator         = MoleculeEvaluator()
    extended_history: List[Dict] = []
    all_molecules:    List[Dict] = []
    best_so_far       = [0.0]
    iteration_callback = create_iteration_callback(
        kernel, decoder, evaluator, extended_history, all_molecules, best_so_far
    )

    # ── Step 6: 初始化 SOQPSO（v9）──────────────────────────────────
    print(f"\n{'─'*70}\nStep 6: 初始化 SOQPSO 優化器（v9）\n{'─'*70}")
    lower_bounds, upper_bounds = kernel.get_param_bounds()

    # bond_param_indices：全上三角，N*(N-1)/2 bonds，每 bond 3 params
    bond_param_indices = []
    for bp_idx in range(kernel.n_bonds):
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

    # ── Step 7: 優化 ────────────────────────────────────────────────
    print(f"\n{'─'*70}\nStep 7: 執行 SOQPSO 優化\n{'─'*70}")
    start_time = time.time()
    best_params, best_fitness, _history = optimizer.optimize()
    elapsed = time.time() - start_time

    print(f"\n總耗時     : {elapsed:.1f} 秒 ({elapsed/60:.1f} 分鐘)")
    print(f'最佳 fitness: {best_fitness:.6f}')

    # ── Step 8: 最終分析 ─────────────────────────────────────────────
    print(f"\n{'─'*70}\nStep 8: 分析最佳結果\n{'─'*70}")
    valid_results, final_decoded = analyze_best_result(best_params, kernel, decoder)

    # 合併最終發現的分子
    seen = {m['smiles'] for m in all_molecules if m.get('smiles')}
    for item in valid_results:
        smiles = item.get('smiles')
        if smiles and smiles not in seen:
            all_molecules.append({'smiles': smiles, 'mol': item.get('mol')})
            seen.add(smiles)

    # ── Step 9: 指標報告（雙重對比）────────────────────────────────
    print(f"\n{'─'*70}")
    print("最終評估指標：")
    print(f"{'─'*70}")

    # Bitstring-count 指標（論文用）
    final_metrics_bc  = evaluator.evaluate(final_decoded)
    print(evaluator.format_metrics(final_metrics_bc, show_shot_weighted=False))

    # Shot-weighted 指標（與 compute_fitness 對比）
    final_metrics_sw  = evaluator.evaluate_shot_weighted(final_decoded)
    print(evaluator.format_metrics(final_metrics_sw, show_shot_weighted=True))

    print(f"\n  Validity × Uniqueness (bitstring-count) : "
          f"{final_metrics_bc['validity'] * final_metrics_bc['uniqueness']:.4f}")
    print(f"  Validity × Uniqueness (shot-weighted)    : "
          f"{final_metrics_sw['validity'] * final_metrics_sw['uniqueness']:.4f}")

    # ── Step 10: 匯出結果 ────────────────────────────────────────────
    print(f"\n{'─'*70}\nStep 10: 匯出結果與視覺化\n{'─'*70}")
    export_history_csv(extended_history, os.path.join(args.output_dir, 'history_metrics.csv'))
    export_molecules_csv(all_molecules,  os.path.join(args.output_dir, 'generated_molecules.csv'))

    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    try:
        from plot_utils import plot_all
        plot_paths = plot_all(
            history=extended_history,
            molecules=all_molecules,
            output_dir=plot_dir,
            show=False,
            dimred_method=args.dimred,
        )
        print(f'  已生成 {len(plot_paths)} 張圖表至 {plot_dir}。')
    except Exception as exc:
        print(f'  [視覺化警告] 圖表生成失敗: {exc}')

    print(f"\n{'═'*70}")
    print('SQMG v13 執行完成。所有結果已輸出至:')
    print(f"  {os.path.abspath(args.output_dir)}")
    print(f"{'═'*70}")


if __name__ == '__main__':
    main()