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
│  │  3N qubits   │                     │  │ → QED Score    │  │  │
│  │              │                     │  └────────────────┘  │  │
│  │  Bond Reg:   │                     └──────────┬───────────┘  │
│  │  2 qubits    │                                │              │
│  │  (reused)    │                        fitness  │              │
│  └──────┬───────┘                                │              │
│         │                                        ▼              │
│         │  params (θ)               ┌──────────────────────┐    │
│         └───────────────────────────│  QuantumOptimizer    │    │
│                                     │  (QPSO)              │    │
│                                     │                      │    │
│                                     │  • Delta 勢阱模型    │    │
│                                     │  • mbest 全域吸引    │    │
│                                     │  • α 排程 (1.0→0.5)  │    │
│                                     │  • 無梯度優化        │    │
│                                     └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 3N+2 量子位元架構

本系統採用 **3N+2 Ansatz**（N = 重原子數量上限）：

| 暫存器 | 量子位元數 | 策略 | 說明 |
|--------|-----------|------|------|
| Atom Register | 3N | 靜態分配 | 每個重原子 3 顆量子位元 → 8 種原子類型 |
| Bond Register | 2 | 動態重複使用 | 透過 mid-circuit measurement + reset 在 N-1 個鍵結間重複使用 |

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

### 鍵結類型映射

| 量子態 | 鍵結 |
|--------|------|
| \|00⟩ | 無鍵 |
| \|01⟩ | 單鍵 (Single) |
| \|10⟩ | 雙鍵 (Double) |
| \|11⟩ | 參鍵 (Triple) |

## 與 QMG 原始論文的差異

| 面向 | QMG (原始論文) | SQMG (本專案) |
|------|---------------|---------------|
| 量子框架 | Qiskit / Cirq | NVIDIA CUDA-Q (GPU 加速) |
| 量子位元策略 | 固定分配 | Atom no reuse + Bond reuse (3N+2) |
| 優化器 | 貝葉斯優化 (GPEI/SAASBO) | QPSO 量子粒子群優化 |
| 模擬加速 | CPU 模擬 | GPU Tensor Network / Statevector |

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
| `--particles` | 20 | QPSO 粒子數量 M |
| `--iterations` | 30 | QPSO 最大迭代次數 T |
| `--shots` | 512 | 每次量子取樣的重複次數 |
| `--alpha` | 0.4 | 適應度權重 (α×validity + (1-α)×QED) |
| `--alpha_max` | 1.0 | QPSO α 初始值 |
| `--alpha_min` | 0.5 | QPSO α 終值 |
| `--backend` | qpp-cpu | CUDA-Q 後端 (qpp-cpu/nvidia/tensornet) |
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
  - `compute_qed()`: 計算 QED 藥物相似度分數
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
   α(t) = α_max - (α_max - α_min) × (t / T)
   ```

## 參考文獻

- **QMG 原始論文**: PEESEgroup QMG — https://github.com/PEESEgroup/QMG
- **CUDA-Q**: NVIDIA CUDA Quantum — https://nvidia.github.io/cuda-quantum/
- **QPSO**: Sun, J., et al. "Quantum-Behaved Particle Swarm Optimization: Analysis of Individual Particle Behavior and Parameter Selection." *Evolutionary Computation*, 2012.
- **QED**: Bickerton, G.R., et al. "Quantifying the chemical beauty of drugs." *Nature Chemistry*, 2012.

## 授權

本專案僅供學術研究用途。
