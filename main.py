"""
==============================================================================
SQMG Main Loop — 可擴展量子分子生成系統 主流程
==============================================================================

本模組整合四大核心元件：
  1. SQMGKernel       — CUDA-Q 3N+2 參數化量子線路
  2. MoleculeDecoder   — Bit-string 到分子結構的解碼器
  3. QuantumOptimizer  — QPSO 量子粒子群優化器
  4. Main_Loop         — 初始化、優化迭代、結果輸出

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
import sys
import time
from datetime import datetime

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
from quantum_optimizer import QuantumOptimizer


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
):
    """
    建立連接 CUDA-Q kernel ↔ MoleculeDecoder ↔ QPSO 的適應度函式。

    閉包 (Closure) 內封裝了 kernel 與 decoder 的參考，
    使 QPSO 只需要傳入參數向量即可得到適應度分數。

    適應度定義：
        fitness = α × validity_ratio + (1 − α) × mean_qed

    Args:
        kernel:       SQMGKernel 實例
        decoder:      MoleculeDecoder 實例
        alpha:        validity_ratio 的權重（0~1）
        verbose_eval: 若 True，每次評估都印出詳細結果

    Returns:
        fitness_fn: Callable[[np.ndarray], float]
    """
    eval_count = [0]  # 使用 list 以便在閉包中修改

    def fitness_fn(params: np.ndarray) -> float:
        """
        實際的適應度評估函式。

        流程：
        1. params → CUDA-Q kernel (cudaq.sample)
        2. bit-strings → MoleculeDecoder
        3. 分子 → RDKit → Validity + QED
        4. → fitness score
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

            return fitness

        except Exception as e:
            # 任何未預期的錯誤都安全地回傳 0 分
            if verbose_eval:
                print(f"    [Eval #{eval_count[0]}] 錯誤: {e}")
            return 0.0

    return fitness_fn


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

    args = parser.parse_args()

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

    fitness_fn = create_fitness_function(
        kernel=kernel,
        decoder=decoder,
        alpha=args.alpha,
        verbose_eval=args.verbose_eval,
    )
    print(f"  適應度公式: fitness = {args.alpha}×validity + "
          f"{1 - args.alpha}×mean_QED")

    # ================================================================
    # Step 5: 初始化 QPSO 優化器
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 5: 初始化 QPSO 量子粒子群優化器")
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
    )

    # ================================================================
    # Step 6: 執行 QPSO 優化
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 6: 執行 QPSO 優化迭代")
    print(f"{'─' * 70}")

    start_time = time.time()
    best_params, best_fitness, history = optimizer.optimize()
    elapsed = time.time() - start_time

    print(f"\n總耗時: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分鐘)")

    # ================================================================
    # Step 7: 分析最佳結果
    # ================================================================
    print(f"\n{'─' * 70}")
    print("Step 7: 分析最佳結果")
    print(f"{'─' * 70}")

    valid_results = analyze_best_result(best_params, kernel, decoder)

    # ── 收斂曲線資料（可用於繪圖）──
    print(f"\n{'─' * 70}")
    print("收斂曲線（gbest fitness vs. iteration）：")
    print(f"{'─' * 70}")
    iters, fitnesses = optimizer.get_convergence_curve()
    for it, fit in zip(iters, fitnesses):
        bar_len = int(fit * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  Iter {it + 1:3d}: {bar} {fit:.4f}")

    print(f"\n{'═' * 70}")
    print("SQMG 執行完成。")
    print(f"{'═' * 70}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
