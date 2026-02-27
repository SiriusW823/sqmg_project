"""
==============================================================================
Plot Utils — SQMG 視覺化模組
==============================================================================

提供四種圖表對應論文標準分析：
  1. 多目標優化軌跡 (Convergence Trajectory)
     — Validity×Uniqueness 與 Mean QED 隨 Iteration 的演進
  2. Pareto 前緣散佈圖 (Pareto Frontier)
     — QED vs. Validity 的 Pareto 最優解可視化
  3. 化學空間分佈 (Chemical Space Mapping)
     — Morgan Fingerprint → t-SNE / PCA 2D 降維散佈圖
  4. 收斂曲線 (Convergence Curve)
     — Gbest Fitness 隨 Iteration 的變化

所有圖表自動存檔為 PNG，同時支援互動式 plt.show()。

依賴：matplotlib, seaborn, scikit-learn, rdkit
==============================================================================
"""

import os
import numpy as np
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # 非互動後端，避免無 GUI 環境報錯
import matplotlib.pyplot as plt
import seaborn as sns

# 設定全域繪圖風格
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)


def plot_convergence_trajectory(
    history: List[Dict],
    output_dir: str = ".",
    filename: str = "convergence_trajectory.png",
    show: bool = False,
) -> str:
    """
    繪製多目標優化軌跡：Validity×Uniqueness 與 Mean QED 隨 Iteration 的演進。

    對應論文中的 Figure: Multi-objective optimization trajectory。

    Args:
        history:    QPSO 歷史紀錄列表（每個元素須包含
                    'iteration', 'validity', 'uniqueness', 'mean_qed'）
        output_dir: 圖表輸出目錄
        filename:   輸出檔名
        show:       是否呼叫 plt.show()

    Returns:
        圖片儲存路徑
    """
    iters = [h['iteration'] + 1 for h in history]

    # Validity × Uniqueness（組合品質指標）
    val_uniq = [h.get('validity', 0) * h.get('uniqueness', 0) for h in history]
    mean_qeds = [h.get('mean_qed', 0) for h in history]
    validities = [h.get('validity', 0) for h in history]
    novelties = [h.get('novelty', 0) for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左 Y 軸：Validity × Uniqueness
    color1 = '#2196F3'
    ax1.set_xlabel('Iteration', fontsize=13)
    ax1.set_ylabel('Validity × Uniqueness', color=color1, fontsize=13)
    line1 = ax1.plot(iters, val_uniq, color=color1, linewidth=2.5,
                     marker='o', markersize=4, label='Validity × Uniqueness')
    line2 = ax1.plot(iters, validities, color='#64B5F6', linewidth=1.5,
                     linestyle='--', alpha=0.7, label='Validity')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.05, 1.1)

    # 右 Y 軸：Mean QED
    ax2 = ax1.twinx()
    color2 = '#FF5722'
    ax2.set_ylabel('Mean QED', color=color2, fontsize=13)
    line3 = ax2.plot(iters, mean_qeds, color=color2, linewidth=2.5,
                     marker='s', markersize=4, label='Mean QED')
    line4 = ax2.plot(iters, novelties, color='#FF8A65', linewidth=1.5,
                     linestyle=':', alpha=0.7, label='Novelty')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.05, 1.1)

    # 合併圖例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=10)

    plt.title('SQMG Multi-Objective Optimization Trajectory', fontsize=15, pad=15)
    fig.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    print(f"  [Plot] 收斂軌跡已儲存至: {filepath}")
    return filepath


def plot_pareto_frontier(
    molecules: List[Dict],
    output_dir: str = ".",
    filename: str = "pareto_frontier.png",
    show: bool = False,
) -> str:
    """
    繪製 Pareto 前緣散佈圖：QED vs. molecule index (或其他指標)。

    找出在 QED 維度上的 Pareto 最優分子並高亮標記。

    Args:
        molecules:  有效分子列表（每個元素須包含 'smiles', 'qed'）
        output_dir: 圖表輸出目錄
        filename:   輸出檔名
        show:       是否呼叫 plt.show()

    Returns:
        圖片儲存路徑
    """
    if not molecules:
        print("  [Plot] 警告：無有效分子可繪製 Pareto 前緣。")
        return ""

    # 為每個分子計算簡單的複雜度指標（SMILES 長度作為代理）
    smiles_list = [m['smiles'] for m in molecules]
    qeds = [m['qed'] for m in molecules]
    complexity = [len(s) for s in smiles_list]

    fig, ax = plt.subplots(figsize=(10, 7))

    # 散佈圖：所有有效分子
    scatter = ax.scatter(complexity, qeds, c=qeds, cmap='RdYlGn',
                        s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    # 找 Pareto 前緣（非支配解）
    # 在 (complexity, QED) 空間中，我們想要低複雜度 + 高 QED
    pareto_indices = _find_pareto_frontier(
        np.array(complexity), np.array(qeds), minimize_x=True, maximize_y=True
    )

    if pareto_indices:
        px = [complexity[i] for i in pareto_indices]
        py = [qeds[i] for i in pareto_indices]
        # 按 x 排序以繪製連線
        sorted_pairs = sorted(zip(px, py))
        px_sorted = [p[0] for p in sorted_pairs]
        py_sorted = [p[1] for p in sorted_pairs]

        ax.scatter(px, py, c='red', s=120, zorder=5,
                  edgecolors='darkred', linewidth=1.5, label='Pareto Frontier')
        ax.plot(px_sorted, py_sorted, 'r--', alpha=0.6, linewidth=1.5)

        # 標註 Pareto 前緣上的 SMILES
        for idx in pareto_indices[:5]:  # 最多標註 5 個
            ax.annotate(
                smiles_list[idx],
                (complexity[idx], qeds[idx]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=7, alpha=0.8,
                arrowprops=dict(arrowstyle='->', alpha=0.4),
            )

    plt.colorbar(scatter, ax=ax, label='QED Score')
    ax.set_xlabel('Molecular Complexity (SMILES Length)', fontsize=13)
    ax.set_ylabel('QED Score', fontsize=13)
    ax.set_title('Generated Molecules: Pareto Frontier\n(Complexity vs. QED)',
                fontsize=14, pad=15)
    ax.legend(fontsize=10)
    fig.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    print(f"  [Plot] Pareto 前緣已儲存至: {filepath}")
    return filepath


def plot_chemical_space(
    molecules: List[Dict],
    output_dir: str = ".",
    filename: str = "chemical_space.png",
    method: str = "tsne",
    show: bool = False,
) -> str:
    """
    繪製化學空間分佈（Chemical Space Mapping）。

    使用 Morgan Fingerprints (Radius=2, 2048 bits) 編碼分子，
    再用 t-SNE 或 PCA 降維至 2D 散佈圖。

    Args:
        molecules:  有效分子列表（每個元素須包含 'smiles', 'qed', 'mol'）
        output_dir: 圖表輸出目錄
        filename:   輸出檔名
        method:     降維方法 ('tsne' 或 'pca')
        show:       是否呼叫 plt.show()

    Returns:
        圖片儲存路徑
    """
    if len(molecules) < 3:
        print(f"  [Plot] 警告：有效分子數不足 ({len(molecules)})，無法生成化學空間圖。")
        return ""

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError as e:
        print(f"  [Plot] 警告：缺少依賴套件 ({e})，跳過化學空間圖。")
        return ""

    # ── 計算 Morgan Fingerprints ──
    fps = []
    valid_mols = []
    valid_qeds = []
    valid_smiles = []

    for m in molecules:
        mol = m.get('mol')
        if mol is None:
            # 嘗試從 SMILES 重建
            smi = m.get('smiles')
            if smi:
                mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
            fps.append(arr)
            valid_mols.append(mol)
            valid_qeds.append(m.get('qed', 0.0))
            valid_smiles.append(m.get('smiles', ''))
        except Exception:
            continue

    if len(fps) < 3:
        print(f"  [Plot] 警告：有效指紋數不足 ({len(fps)})，跳過化學空間圖。")
        return ""

    X = np.array(fps)
    qed_colors = np.array(valid_qeds)

    # ── 降維 ──
    if method.lower() == "tsne":
        perplexity = min(30, len(X) - 1)
        reducer = TSNE(n_components=2, random_state=42,
                      perplexity=max(perplexity, 2))
        X_2d = reducer.fit_transform(X)
        method_label = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        method_label = "PCA"

    # ── 繪圖 ──
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=qed_colors, cmap='viridis',
        s=80, alpha=0.8, edgecolors='white', linewidth=0.5,
    )

    # 標註 QED 最高的前 3 個分子
    top_indices = np.argsort(qed_colors)[-3:]
    for idx in top_indices:
        ax.annotate(
            f"{valid_smiles[idx]}\nQED={qed_colors[idx]:.3f}",
            (X_2d[idx, 0], X_2d[idx, 1]),
            textcoords="offset points", xytext=(10, 10),
            fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', alpha=0.5),
        )

    plt.colorbar(scatter, ax=ax, label='QED Score')
    ax.set_xlabel(f'{method_label} Dimension 1', fontsize=13)
    ax.set_ylabel(f'{method_label} Dimension 2', fontsize=13)
    ax.set_title(
        f'Chemical Space Distribution ({method_label})\n'
        f'Morgan Fingerprints (Radius=2, 2048 bits) → {method_label}',
        fontsize=14, pad=15,
    )
    fig.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    print(f"  [Plot] 化學空間分佈已儲存至: {filepath}")
    return filepath


def plot_convergence_curve(
    history: List[Dict],
    output_dir: str = ".",
    filename: str = "convergence_curve.png",
    show: bool = False,
) -> str:
    """
    繪製 QPSO 收斂曲線（Gbest Fitness vs. Iteration）。

    Args:
        history:    QPSO 歷史紀錄列表
        output_dir: 圖表輸出目錄
        filename:   輸出檔名
        show:       是否呼叫 plt.show()

    Returns:
        圖片儲存路徑
    """
    iters = [h['iteration'] + 1 for h in history]
    gbest = [h['gbest_fitness'] for h in history]
    mean_fit = [h.get('mean_fitness', 0) for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iters, gbest, 'b-', linewidth=2.5, marker='o', markersize=4,
            label='Global Best Fitness')
    ax.fill_between(iters,
                    [h.get('min_fitness', 0) for h in history],
                    [h.get('max_fitness', 0) for h in history],
                    alpha=0.2, color='blue', label='Min-Max Range')
    ax.plot(iters, mean_fit, 'g--', linewidth=1.5, alpha=0.7,
            label='Mean Fitness')

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Fitness Score', fontsize=13)
    ax.set_title('QPSO Convergence Curve', fontsize=15, pad=15)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=-0.05)
    fig.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    print(f"  [Plot] 收斂曲線已儲存至: {filepath}")
    return filepath


def plot_all(
    history: List[Dict],
    molecules: List[Dict],
    output_dir: str = ".",
    show: bool = False,
    dimred_method: str = "pca",
) -> List[str]:
    """
    一次性生成所有 4 種圖表。

    Args:
        history:       QPSO 歷史紀錄列表（含 validity, uniqueness 等指標）
        molecules:     所有有效分子列表
        output_dir:    輸出目錄
        show:          是否呼叫 plt.show()
        dimred_method: 降維方法 ('tsne' 或 'pca')

    Returns:
        儲存的圖片路徑列表
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    print("\n生成視覺化圖表...")

    paths.append(plot_convergence_curve(history, output_dir, show=show))
    paths.append(plot_convergence_trajectory(history, output_dir, show=show))
    paths.append(plot_pareto_frontier(molecules, output_dir, show=show))
    paths.append(plot_chemical_space(molecules, output_dir,
                                     method=dimred_method, show=show))

    return [p for p in paths if p]


# ============================================================================
# 輔助函式
# ============================================================================

def _find_pareto_frontier(
    x: np.ndarray, y: np.ndarray,
    minimize_x: bool = True, maximize_y: bool = True,
) -> List[int]:
    """
    找出 2D 空間中的 Pareto 前緣索引。

    Args:
        x, y:        兩個目標值的陣列
        minimize_x:  是否最小化 x 軸
        maximize_y:  是否最大化 y 軸

    Returns:
        Pareto 前緣上的點索引列表
    """
    n = len(x)
    if n == 0:
        return []

    # 統一為「越大越好」
    x_sign = -1.0 if minimize_x else 1.0
    y_sign = 1.0 if maximize_y else -1.0
    xx = x * x_sign
    yy = y * y_sign

    pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j 支配 i 的條件：j 在兩個目標上都不差，且至少一個更好
            if xx[j] >= xx[i] and yy[j] >= yy[i]:
                if xx[j] > xx[i] or yy[j] > yy[i]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(i)

    return pareto
