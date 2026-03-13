# SQMG — 可擴展量子分子生成系統

**Scalable Quantum Molecular Generation with CUDA-Q & QPSO**

基於 NVIDIA CUDA-Q 框架與量子粒子群優化 (QPSO) 的新一代量子分子生成系統。

---

## 架構概覽

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQMG 系統架構                                 │
│                                                                 │
│  ┌──────────────┐    cudaq.sample()   ┌──────────────────────┐  │
│  │  SQMGKernel  │ ──────────────────► │  MoleculeDecoder     │  │
│  │  (CUDA-Q     │    bit-strings      │  (RDKit)             │  │
│  │   3N+2       │                     │  ┌────────────────┐  │  │
│  │   Ansatz)    │                     │  │ Atom: 3-bit    │  │  │
│  │              │                     │  │ Bond: 2-bit    │  │  │
│  │  Atom Reg:   │                     │  │ → SMILES       │  │  │
│  │  3N qubits   │                     │  └────────────────┘  │  │
│  │  Bond Reg:   │                     └──────────┬───────────┘  │
│  │  2 qubits    │                                │              │
│  │  (reused)    │                        fitness  │             │
│  └──────┬───────┘                                │              │
│         │                                        ▼              │
│         │  params (θ)               ┌──────────────────────┐    │
│         └───────────────────────────│  QuantumOptimizer    │    │
│                                     │  (QPSO)              │    │
│                                     │                      │    │
│                                     │  • Delta 勢阱模型     │    │
│                                     │  • mbest 全域吸引     │    │
│                                     │  • α 排程 (1.0→0.5)  │    │
│                                     │  • 無梯度優化         │    │
│                                     └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 3N+2 量子位元架構

本系統採用 **3N+2 Ansatz**（N = 重原子數量上限）：

| 暫存器 | 量子位元數 | 策略 | 說明 |
|--------|-----------|------|------|
| Atom Register | 3N | 靜態分配 | 每個重原子 3 個量子位元，1 層 HEA（RY×3+RZ×3+Ring-CNOT+RY×3），9p/atom |
| Bond Register | 2 | 動態重複使用 | RY + Ctrl-RY(single→double) + Ctrl-RY(double→triple)，3p/bond，N(N-1)/2 bonds，可達態：|00⟩無鍵/|10⟩單鍵/|11⟩雙鍵/|01⟩三鍵 |

> ℹ **Bond subcircuit (v7)**：每個鍵使用 3 個參數（Gate 1: RY bond_existence, Gate 2: CRY single→double, Gate 3: CRY double→triple），可達 4 種鍵結態。總參數量公式為 **9N + 3·N(N-1)/2**。
> SQMG 論文 Table I 的 **N²+9N-1** 對應 9N（HEA）+ (N-1)（Cross-atom CRY）+ 2·N(N-1)/2（2p/bond）；本程式碼改採 3p/bond 實現三鍵，捨棄 Cross-atom CRY，兩者均無法對齊論文參數量。

### 原子類型映射 (8 States)

| 量子態 | 原子 | 說明 |
|--------|------|------|
| \|000⟩ | NONE | 終止符 — 停止分子建構 |
| \|001⟩ | C | 碳 (Carbon) |
| \|010⟩ | O | 氧 (Oxygen) |
| \|011⟩ | N | 氮 (Nitrogen) |
| \|100⟩ | S | 硫 (Sulfur) |
| \|101⟩ | P | 磷 (Phosphorus) |
| \|110⟩ | F | 氟 (Fluorine) |
| \|111⟩ | Cl | 氯 (Chlorine) |

### 鍵結類型映射 (v7：4 種可達態)

Bond 子電路使用 3-gate 設計（對應 Chen et al. QMG Eq.2-3）：

| 量子態 | 可達 | 鍵結 | 產生路徑 |
|--------|------|------|---------|
| \|00⟩ | ✓ | 無鍵 | Gate 1 未激發 |
| \|10⟩ | ✓ | 單鍵 (Single) | Gate 1 激發，Gate 2 未激發 |
| \|11⟩ | ✓ | 雙鍵 (Double) | Gate 1+2 激發，Gate 3 未激發 |
| \|01⟩ | ✓ | 三鍵 (Triple) | Gate 1+2+3 均激發 |

參數邊界（Hierarchical Bounds）：
- Gate 1 `bond_existence`    ∈ [0, π/2]：P(bond) ≤ 50%
- Gate 2 `bond_order`        ∈ [0, π/2]：P(double\|bond) ≤ 50%（對應 Eq.2）
- Gate 3 `bond_triple_order` ∈ [0, π/2]：P(triple\|double) ≤ 50%（等效 Eq.3）

## 與 QMG 原始論文的差異

| 面向 | QMG (原始論文) | SQMG (本專案) |
|------|---------------|---------------|
| 量子框架 | Qiskit / Cirq | NVIDIA CUDA-Q (GPU 加速) |
| 量子位元策略 | 2-qubit atom + 全動態 (QMG) | 3-qubit atom + Atom no reuse + Bond reuse，無 Ancilla (SQMG) |
| 鍵結拓撲 | 全上三角矩陣 N(N-1)/2 bonds | 全上三角矩陣 N(N-1)/2 bonds |
| 優化器 | BO（Chen et al. 用 GPEI/SAASBO；SQMG 論文用 COBYLA + GP-EI） | QPSO 量子粒子群優化（Delta 勢阱，取代兩篇論文的 BO） |
| 模擬加速 | CPU 模擬 | GPU Tensor Network / Statevector |

### Table I — 參數量對照表

> ⚠ **注意**：SQMG 論文 N²+9N-1 = 9N（HEA）+ (N-1)（Cross-atom CRY）+ 2×N(N-1)/2（2p/bond）。
> 本程式碼 v7 改採 **3p/bond** 以支援三鍵，放棄 Cross-atom CRY，
> 公式為 **9N + 3·N(N-1)/2**，與論文及 v6 均不同。

| N | 論文 static qubits | SQMG hybrid qubits | 論文 params (N²+9N-1) | v6 params (N²+8N) | **v7 params (9N+3·N(N-1)/2)** |
|---|--------------------|--------------------|----------------------|-------------------|-------------------------------|
| 2 | 8 | 8 | 21 | 20 | **21** |
| 3 | 15 | 11 | 35 | 33 | **36** |
| 4 | 24 | 14 | 51 | 48 | **54** |
| 5 | 35 | 17 | 69 | 64 | **69** |
| 10 | 120 | 32 | 189 | 180 | **225** |

## Backend 選擇指南

| Backend | 適用 N | 硬體需求 | 備註 |
|---------|--------|---------|------|
| qpp-cpu | N ≤ 5 | CPU only | 僅供測試 |
| nvidia | N ≤ 9 | 1× GPU | 超過 N=9 顯存不足 |
| tensornet | N ≤ 40 | 1-8× GPU | 推薦，cuTensorNet 內部自動多 GPU |

DGX V100 8-GPU 環境下，推薦使用 tensornet：
```bash
python main.py --backend tensornet --max_atoms 8 --particles 30 \
               --iterations 150 --shots 1024
```

tensornet 多 GPU 並行由 cuTensorNet 函式庫內部處理（splitindex 策略），
對 Python 層完全透明，不需要 MPI 或 rank 概念。
根據 SQMG 論文，tensornet 在 N=8 時約 3.45 秒/iteration，
比 CPU 快約 2,200 倍；同時 atom no-reuse 架構比 atom reuse
在 N=40 時快約 1.9 倍（論文 Fig.4）。

## 環境需求

> ⚠ **極度重要**：NumPy 版本必須 < 2.0！  
> NumPy 2.0+ 的 C++ ABI 變更會導致 RDKit 在執行時崩潰。

### 安裝

```bash
# 1. 建立虛擬環境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2. 安裝依賴
pip install "numpy>=1.24,<2.0"
pip install rdkit
pip install cuda-quantum-cu12   # 或 cuda-quantum-cu11 (根據 CUDA 版本)

# 或使用 requirements.txt
pip install -r requirements.txt
```

### 驗證安裝

```python
import numpy as np
print(f"NumPy: {np.__version__}")  # 應顯示 1.x

import cudaq
print(f"CUDA-Q: {cudaq.__version__}")

from rdkit import Chem
print(f"RDKit: OK")
```

## 快速開始

```bash
# 使用預設參數（N=4, 20 particles, 30 iterations）
python main.py

# 自訂參數
python main.py --max_atoms 6 --particles 30 --iterations 100 --shots 1024

# 使用 GPU 加速（需要 NVIDIA GPU）
python main.py --backend nvidia

# 使用張量網路模擬（大 N 推薦）
python main.py --backend tensornet

# 看所有選項
python main.py --help
```

### 命令列參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--max_atoms` | 4 | 最大重原子數量 N |
| `--particles` | 30 | QPSO 粒子數量 M |
| `--iterations` | 150 | QPSO 最大迭代次數 T |
| `--shots` | 1024 | 每次量子取樣的重複次數 |
| `--alpha_max` | 1.2 | QPSO α 初始值 |
| `--alpha_min` | 0.4 | QPSO α 終值 |
| `--backend` | tensornet | CUDA-Q 後端 (qpp-cpu/nvidia/tensornet) |
| `--seed` | 42 | 隨機數種子 |
| `--verbose_eval` | False | 印出每次適應度評估的詳細資訊 |

## 模組說明

### 1. `sqmg_kernel.py` — CUDA-Q 量子線路

- `sqmg_circuit()`: CUDA-Q `@cudaq.kernel` 修飾的量子線路函式
  - 參數化 RY + RZ 旋轉 + CNOT 糾纏
  - Mid-circuit measurement + Reset 實現 Bond Reuse
  - 顯式 `mz()` 防止 CUDA-Q 編譯器吞掉測量結果
- `SQMGKernel`: 封裝類別，提供 `sample()` 介面

### 2. `molecule_decoder.py` — 分子解碼器

- `MoleculeDecoder`: 將 bit-string 轉譯為化學分子
  - `parse_bitstring()`: 拆解為原子碼 + 鍵碼
  - `build_molecule()`: 使用 RDKit `RWMol` 建構分子
  - `compute_fitness()`: 計算組合適應度

### 3. `quantum_optimizer.py` — QPSO 優化器

- `QuantumOptimizer`: 量子粒子群優化演算法
  - Delta 勢阱模型（無梯度）
  - mbest 全域吸引子
  - α 線性遞減排程
  - 收斂曲線輸出

### 4. `main.py` — 主流程

- CUDA-Q 後端組態
- 元件初始化與連接
- QPSO 優化迴圈
- 結果分析與輸出

## QPSO 演算法詳解

量子粒子群優化 (Quantum Particle Swarm Optimization) 的核心方程式：

1. **平均最佳位置 (mbest)**：
   ```
   mbest_d = (1/M) × Σᵢ pbest_{i,d}
   ```

2. **局部吸引子**：
   ```
   p_{i,d} = φ × pbest_{i,d} + (1-φ) × gbest_d,  φ ~ U(0,1)
   ```

3. **位置更新 (Delta 勢阱)**：
   ```
   x_{i,d} = p_{i,d} ± α × |mbest_d - x_{i,d}| × ln(1/u),  u ~ U(0,1)
   ```

4. **α 排程**：
   ```
   α(t) = α_min + 0.5 × (α_max - α_min) × (1 + cos(π × t/T))
   ```

## 參考文獻

- **QMG 原始論文**: PEESEgroup QMG — https://github.com/PEESEgroup/QMG
- **CUDA-Q**: NVIDIA CUDA Quantum — https://nvidia.github.io/cuda-quantum/
- **QPSO**: Sun, J., et al. "Quantum-Behaved Particle Swarm Optimization: Analysis of Individual Particle Behavior and Parameter Selection." *Evolutionary Computation*, 2012.

## 授權

本專案僅供學術研究用途。