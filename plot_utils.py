"""
==============================================================================
Plot Utils — SQMG 視覺化模組
==============================================================================
# MODIFIED: FIX-P0-1, FIX-P0-2, FIX-P1-1, FIX-QED, FIX-HYPER, FIX-CHEM, FIX-GPU

提供三種圖表對應論文標準分析（單目標 v6，已移除 Pareto 前緣圖）：
  1. 收斂曲線 (Convergence Curve)
     — Gbest Fitness 隨 Iteration 的變化
  2. 優化軌跡 (Convergence Trajectory)
     — Validity×Uniqueness 隨 Iteration 的演進
  3. 化學空間分佈 (Chemical Space Mapping)
     — Morgan Fingerprint → t-SNE / PCA 2D 降維散佈圖

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
    繪製 Validity & Uniqueness 收斂軌跡。

    專注於優化目標指標：
    - Validity × Uniqueness（主指標）
    - Validity（單獨追蹤）
    - Novelty（輔助參考，不參與優化）
    """
    iters = [h['iteration'] + 1 for h in history]
    val_uniq = [h.get('validity', 0) * h.get('uniqueness', 0) for h in history]
    validities = [h.get('validity', 0) for h in history]
    uniquenesses = [h.get('uniqueness', 0) for h in history]
    novelties = [h.get('novelty', 0) for h in history]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)

    ax.plot(iters, val_uniq, color='#1565C0', linewidth=2.5,
        marker='o', markersize=4, label='Validity × Uniqueness (目標)')
    ax.plot(iters, validities, color='#42A5F5', linewidth=1.8,
        linestyle='--', alpha=0.85, label='Validity')
    ax.plot(iters, uniquenesses, color='#26A69A', linewidth=1.8,
        linestyle='-.', alpha=0.85, label='Uniqueness')
    ax.plot(iters, novelties, color='#AB47BC', linewidth=1.5,
        linestyle=':', alpha=0.7, label='Novelty (參考)')

    ax.axhline(y=0.8, color='red', linewidth=1.2,
           linestyle='--', alpha=0.6, label='目標線 (0.8)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='lower right', fontsize=10)

    plt.title('Validity & Uniqueness Convergence', fontsize=15, pad=15)
    fig.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    print(f"  [Plot] 收斂軌跡已儲存至: {filepath}")
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
        molecules:  有效分子列表（每個元素須包含 'smiles', 'mol'）
        output_dir: 圖表輸出目錄
        filename:   輸出檔名
        method:     降維方法 ('tsne' 或 'pca')
        show:       是否呼叫 plt.show()

    Returns:
        圖片儲存路徑
    """
    if len(molecules) < 2:
        print(f"  [Plot] 警告：有效分子數不足 ({len(molecules)}<2)，跳過化學空間圖。")
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
            valid_smiles.append(m.get('smiles', ''))
        except Exception:
            continue

    if len(fps) < 2:
        print(f"  [Plot] 警告：唯一有效指紋數不足 ({len(fps)}<2)，跳過化學空間圖。")
        return ""

    X = np.array(fps)

    # ── 降維 ──
    if method.lower() == "tsne":
        n_samples = len(X)
        if n_samples < 2:
            print(f"  [Plot] 警告：t-SNE 樣本數不足 ({n_samples}<2)，跳過化學空間圖。")
            return ""
        # 動態設定 perplexity，確保 perplexity < n_samples
        perplexity = min(30, max(1, n_samples - 1))
        reducer = TSNE(n_components=2, random_state=42,
                      perplexity=perplexity)
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
        c='#42A5F5',
        s=80, alpha=0.8, edgecolors='white', linewidth=0.5,
    )

    # 標註前 3 個分子
    if len(valid_smiles) >= 3:
        top_indices = list(range(min(3, len(valid_smiles))))
        for idx in top_indices:
            ax.annotate(
                valid_smiles[idx],
                (X_2d[idx, 0], X_2d[idx, 1]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', alpha=0.5),
            )
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
    gbest = [h.get('gbest_fitness', 0) for h in history]
    mean_fit = [h.get('mean_fitness', 0) for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(iters, gbest, 'b-', linewidth=2.5, marker='o', markersize=4,
            label='Global Best Fitness')
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
    一次性生成所有 3 種圖表（單目標 v6，已移除 Pareto 前緣圖）。

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
    paths.append(plot_chemical_space(molecules, output_dir,
                                     method=dimred_method, show=show))

    return [p for p in paths if p]
