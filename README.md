<div align="center">

# 🧠 Tiny Neural Network in C++

![C++](https://img.shields.io/badge/C++-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)
![No Dependencies](https://img.shields.io/badge/Dependencies-None-orange?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS%20|%20Windows-lightgrey?style=for-the-badge)

**A minimal neural network engine implemented from scratch in modern C++20 — no external ML libraries.**

</div>

---

## ✨ Features

- **Matrix Engine** — Dynamic, row-major matrix class with full operator support
- **Matrix Ops** — Multiplication, transpose, scalar operations
- **Activations** — ReLU, Sigmoid (scalar & element-wise matrix versions)
- **Loss Function** — Binary Cross Entropy with gradient
- **Dense Layer** — Fully connected layer with configurable activation
- **Backpropagation** — Gradient descent with chain rule
- **XOR Example** — Classic non-linear classification problem

---

## 🏗️ Architecture

### Network Topology

```mermaid
graph TD
    I1((x₁)) --> H1
    I1 --> H2
    I1 --> H3
    I1 --> H4
    I1 --> H5
    I1 --> H6
    I1 --> H7
    I1 --> H8
    I2((x₂)) --> H1
    I2 --> H2
    I2 --> H3
    I2 --> H4
    I2 --> H5
    I2 --> H6
    I2 --> H7
    I2 --> H8
    H1((h₁)) --> O((ŷ))
    H2((h₂)) --> O
    H3((h₃)) --> O
    H4((h₄)) --> O
    H5((h₅)) --> O
    H6((h₆)) --> O
    H7((h₇)) --> O
    H8((h₈)) --> O

    subgraph Input["Input Layer (2)"]
        I1
        I2
    end

    subgraph Hidden["Hidden Layer (8) · ReLU"]
        H1
        H2
        H3
        H4
        H5
        H6
        H7
        H8
    end

    subgraph Output["Output Layer (1) · Sigmoid"]
        O
    end

    style Input fill:#1e3a5f,color:#fff,stroke:#4a90d9
    style Hidden fill:#1a4731,color:#fff,stroke:#4caf7d
    style Output fill:#4a1942,color:#fff,stroke:#c06dd9
```

### Forward & Backward Pass

```mermaid
sequenceDiagram
    participant IN as Input
    participant L1 as Dense Layer 1<br/>(ReLU)
    participant L2 as Dense Layer 2<br/>(Sigmoid)
    participant LF as Loss (BCE)

    Note over IN,LF: ── Forward Pass ──
    IN->>L1: x (2×1)
    L1->>L2: h = ReLU(W₁x + b₁) (8×1)
    L2->>LF: ŷ = σ(W₂h + b₂) (1×1)
    LF-->>LF: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

    Note over IN,LF: ── Backward Pass ──
    LF->>L2: ∂L/∂ŷ
    L2->>L1: ∂L/∂h (update W₂, b₂)
    L1->>IN: ∂L/∂x (update W₁, b₁)
```

---

## 📁 Project Structure

```mermaid
graph LR
    ROOT[tiny_nn/] --> SRC[src/]
    ROOT --> BUILD[build/]
    ROOT --> CMAKE[CMakeLists.txt]
    ROOT --> README[README.md]
    ROOT --> LICENSE[LICENSE]

    SRC --> MAIN[main.cpp]
    SRC --> MAT_H[matrix.hpp]
    SRC --> MAT_C[matrix.cpp]
    SRC --> LAYER_H[layer.hpp]
    SRC --> LAYER_C[layer.cpp]
    SRC --> ACT[activations.hpp]
    SRC --> LOSS[loss.hpp]

    style ROOT fill:#2d2d2d,color:#fff,stroke:#888
    style SRC fill:#1e3a5f,color:#fff,stroke:#4a90d9
    style BUILD fill:#3a3a1e,color:#fff,stroke:#d9c74a
```

---

## 🚀 Build & Run

```bash
mkdir build
cd build
cmake ..
make
./tiny_nn
```

---

## 📊 Training Flow

```mermaid
flowchart TD
    A([Start]) --> B[Initialize Layers\nXavier Weights]
    B --> C{epoch < 10000?}
    C -- Yes --> D[Sample from XOR Dataset]
    D --> E[Forward Pass\nLayer 1 ReLU]
    E --> F[Forward Pass\nLayer 2 Sigmoid]
    F --> G[Compute BCE Loss]
    G --> H[Compute ∂L/∂ŷ]
    H --> I[Backward Layer 2\nUpdate W₂, b₂]
    I --> J[Backward Layer 1\nUpdate W₁, b₁]
    J --> K{All samples done?}
    K -- No --> D
    K -- Yes --> L{epoch % 1000 == 0?}
    L -- Yes --> M[Print Avg Loss]
    L -- No --> C
    M --> C
    C -- No --> N[Run Final Predictions]
    N --> O([End])

    style A fill:#1a4731,color:#fff,stroke:none
    style O fill:#4a1942,color:#fff,stroke:none
    style G fill:#1e3a5f,color:#fff,stroke:#4a90d9
    style M fill:#3a2a0a,color:#fff,stroke:#d9a84a
```

---

## 📈 Example Output

```
Epoch 0    Loss: 0.7231
Epoch 1000 Loss: 0.6891
Epoch 2000 Loss: 0.5204
Epoch 3000 Loss: 0.3012
...
Epoch 9000 Loss: 0.0183

Final predictions:
[0,0] → 0.02  ✓
[0,1] → 0.97  ✓
[1,0] → 0.97  ✓
[1,1] → 0.03  ✓
```

---

## 🔭 Future Improvements

| Feature | Status |
|---|---|
| Batch training | 🔲 Planned |
| Softmax + multi-class | 🔲 Planned |
| MNIST dataset | 🔲 Planned |
| SIMD optimization | 🔲 Planned |
| OpenMP parallelization | 🔲 Planned |
| CUDA implementation | 🔲 Planned |

---

<div align="center">

**Author:** [@Yahia995](https://github.com/Yahia995) &nbsp;·&nbsp; **License:** MIT

</div>
