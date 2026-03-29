# MVC Project: Multilayer Perceptron (MLP) from Scratch

**Student Roll No:** 25I-3075 | **Section:** SE-D

A complete hand-computed and Python-implemented Multilayer Perceptron (MLP) covering forward pass, loss calculation, backpropagation, weight updates, gradient descent variants, optimizer comparisons, and a full NumPy implementation trained on MNIST.

---

## Repository Structure
```
mvc-mlp-25I-3075/
├── src/
│   └── mlp_25I-3075.ipynb          # Jupyter Notebook with all cells executed
├── data/
│   └── mnist.npz                   # MNIST dataset
├── report/
│   └── MLP_Solution_25I-3075.pdf   # Compiled PDF report
└── README.md
```

---

## Network Architecture
```
Input (x1, x2)  →  Hidden Layer 1 (h1, h2)  →  Hidden Layer 2 (h3, h4)  →  Output (ŷ)
                        sigmoid σ                      sigmoid σ               sigmoid σ
```

| Layer   | Type           | Neurons | Activation |
|---------|----------------|---------|------------|
| Layer 0 | Input          | 2       | —          |
| Layer 1 | Hidden layer 1 | 2       | Sigmoid σ  |
| Layer 2 | Hidden layer 2 | 2       | Sigmoid σ  |
| Layer 3 | Output         | 1       | Sigmoid σ  |

> Shared bias b⁽¹⁾ added to both neurons in hidden layer 1; shared b⁽²⁾ added to both neurons in hidden layer 2. No bias at the output layer.

---

## Assigned Weights & Biases (Roll No: 25I-3075, Sheet: SE-D)

| Parameter | Value | Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|-----------|-------|
| w1  | −0.45 | w5  |  0.18 | w9  | −0.15 |
| w2  | −0.39 | w6  |  0.74 | w10 | −0.66 |
| w3  | −0.11 | w7  |  0.49 | b⁽¹⁾ | −0.12 |
| w4  |  0.29 | w8  | −0.49 | b⁽²⁾ | −0.09 |

---

## Dataset — Student Performance (Binary Classification)

| Sample | x1 (Study Hours) | x2 (Attendance) | y (Pass?) |
|--------|-----------------|-----------------|-----------|
| s⁽¹⁾  | 0.2             | 0.8             | 1         |
| s⁽²⁾  | 0.9             | 0.4             | 1         |
| s⁽³⁾  | 0.1             | 0.2             | 0         |

---

## Task Summary & Results

### Task 1 — Forward Pass

**Symbolic equations (Hidden Layer 1):**
```
z⁽¹⁾₁ = w1·x1 + w3·x2 + b⁽¹⁾        a⁽¹⁾₁ = σ(z⁽¹⁾₁)
z⁽¹⁾₂ = w2·x1 + w4·x2 + b⁽¹⁾        a⁽¹⁾₂ = σ(z⁽¹⁾₂)
```

**Symbolic equations (Hidden Layer 2):**
```
z⁽²⁾₁ = w5·a⁽¹⁾₁ + w7·a⁽¹⁾₂ + b⁽²⁾   a⁽²⁾₁ = σ(z⁽²⁾₁)
z⁽²⁾₂ = w6·a⁽¹⁾₁ + w8·a⁽¹⁾₂ + b⁽²⁾   a⁽²⁾₂ = σ(z⁽²⁾₂)
```

**Output layer (no bias):**
```
z⁽³⁾₁ = w9·a⁽²⁾₁ + w10·a⁽²⁾₂          ŷ = σ(z⁽³⁾₁)
```

**Numerical results for all samples:**

| Sample | x1  | x2  | a⁽¹⁾₁ | a⁽¹⁾₂ | a⁽²⁾₁ | a⁽²⁾₂ | y | ŷ      |
|--------|-----|-----|--------|--------|--------|--------|---|--------|
| s⁽¹⁾  | 0.2 | 0.8 | 0.4260 | 0.5085 | 0.5587 | 0.4940 | 1 | 0.3989 |
| s⁽²⁾  | 0.9 | 0.4 | 0.3615 | 0.4122 | 0.5441 | 0.4939 | 1 | 0.3995 |
| s⁽³⁾  | 0.1 | 0.2 | 0.4534 | 0.4748 | 0.5558 | 0.5032 | 0 | 0.3976 |

---

### Task 2 — Loss Calculation (MSE)
```
L_MSE = (1/3) × [(1 − ŷ⁽¹⁾)² + (1 − ŷ⁽²⁾)² + (0 − ŷ⁽³⁾)²]
      = (1/3) × [0.3613 + 0.3606 + 0.1581]
      = 0.2933
```

**Final Loss: L_MSE = 0.2933**

---

### Task 3 — Backpropagation (Sample s⁽¹⁾)

**Output delta:** δ⁽³⁾₁ = −2(y − ŷ)·ŷ(1−ŷ) = **−0.2883**

| Parameter | Current Value | Gradient  | Parameter | Current Value | Gradient  |
|-----------|--------------|-----------|-----------|--------------|-----------|
| w1        | −0.45        |  0.0012   | w6        |  0.74        |  0.0203   |
| w2        | −0.39        | −0.0008   | w7        |  0.49        |  0.0054   |
| w3        | −0.11        |  0.0049   | w8        | −0.49        |  0.0242   |
| w4        |  0.29        | −0.0031   | w9        | −0.15        | −0.1610   |
| w5        |  0.18        |  0.0045   | w10       | −0.66        | −0.1424   |
| b⁽¹⁾     | −0.12        |  0.0023   | b⁽²⁾     | −0.09        |  0.0582   |

---

### Task 4 — Weight Update & Multiple Iterations (η = 0.1)

| Parameter | Old    | New     |
|-----------|--------|---------|
| w1        | −0.45  | −0.4501 |
| w2        | −0.39  | −0.3899 |
| w3        | −0.11  | −0.1105 |
| w4        |  0.29  |  0.2903 |
| w5        |  0.18  |  0.1795 |
| w6        |  0.74  |  0.7380 |
| w7        |  0.49  |  0.4895 |
| w8        | −0.49  | −0.4924 |
| w9        | −0.15  | −0.1339 |
| w10       | −0.66  | −0.6458 |
| b⁽¹⁾     | −0.12  | −0.1202 |
| b⁽²⁾     | −0.09  | −0.0958 |

| Iteration | MSE Loss | Decreased? |
|-----------|----------|------------|
| 0 (init)  | 0.2933   | —          |
| 1         | 0.2911   | ✓ YES      |
| 2         | 0.2890   | ✓ YES      |
| 3         | 0.2869   | ✓ YES      |
| 4         | 0.2848   | ✓ YES      |
| 5         | 0.2829   | ✓ YES      |

---

### Task 5 — Gradient Descent Variants

| Epoch | Batch | Samples Used | MSE After Update |
|-------|-------|--------------|-----------------|
| 1     | 1     | s⁽¹⁾, s⁽²⁾ | 0.3560          |
| 1     | 2     | s⁽³⁾        | 0.1592          |
| 2     | 1     | s⁽³⁾, s⁽¹⁾ | 0.2593          |
| 2     | 2     | s⁽²⁾        | 0.3533          |
| 3     | 1     | s⁽²⁾, s⁽³⁾ | 0.2581          |
| 3     | 2     | s⁽¹⁾        | 0.3483          |

Updates per epoch with m = 60,000 and B = 32: **⌈60000/32⌉ = 1875 updates/epoch**

---

### Task 6 — Optimizers (η = 0.1, β = 0.9)

| Iter | Plain GD | Momentum | NAG    | Fastest  |
|------|----------|----------|--------|----------|
| 1    | 0.2926   | 0.2930   | 0.2930 | Plain GD |
| 2    | 0.2919   | 0.2927   | 0.2927 | Plain GD |
| 3    | 0.2912   | 0.2923   | 0.2923 | Plain GD |
| 4    | 0.2905   | 0.2918   | 0.2918 | Plain GD |
| 5    | 0.2899   | 0.2913   | 0.2913 | Plain GD |
| 6    | 0.2892   | 0.2907   | 0.2907 | Plain GD |
| 7    | 0.2886   | 0.2902   | 0.2902 | Plain GD |
| 8    | 0.2880   | 0.2896   | 0.2896 | Plain GD |
| 9    | 0.2874   | 0.2890   | 0.2890 | Plain GD |
| 10   | 0.2868   | 0.2884   | 0.2884 | Plain GD |

---

### Task 7 — Python Implementation (MNIST)

| Layer   | Type           | Neurons | Activation |
|---------|----------------|---------|------------|
| Layer 0 | Input          | 784     | —          |
| Layer 1 | Hidden layer 1 | 128     | Sigmoid    |
| Layer 2 | Hidden layer 2 | 64      | Sigmoid    |
| Layer 3 | Output         | 10      | Sigmoid    |

Training config: Mini-batch GD · B = 32 · η = 0.1 · 20 epochs · `np.random.seed(42)`

Required notebook outputs:
1. Training loss curve (MSE vs. epoch)
2. Test accuracy on 10,000 test images
3. Sample predictions — 10 test images with true and predicted labels

---

## How to Run
```bash
pip install numpy matplotlib jupyter
jupyter notebook src/mlp_25I-3075.ipynb
```

Place `mnist.npz` in the `data/` folder before running.

---

## Submission

- **Repository name:** `mvc-mlp-25I-3075` (private — instructor added as collaborator with read access)
- **ZIP submission:** `25I-3075_[Name].ZIP` containing the `.ipynb` and PDF report
- Commit work progressively — do not push everything in one final commitdont add the MLP_Dolution in it16:59markdown# MVC Project: Multilayer Perceptron (MLP) from Scratch

**Student Roll No:** 25I-3075 | **Section:** SE-D

A complete hand-computed and Python-implemented Multilayer Perceptron (MLP) covering forward pass, loss calculation, backpropagation, weight updates, gradient descent variants, optimizer comparisons, and a full NumPy implementation trained on MNIST.

---

## Repository Structure
```
mvc-mlp-25I-3075/
├── src/
│   └── mlp_25I-3075.ipynb    # Jupyter Notebook with all cells executed
├── data/
│   └── mnist.npz             # MNIST dataset
└── README.md
```

---

## Network Architecture
```
Input (x1, x2)  →  Hidden Layer 1 (h1, h2)  →  Hidden Layer 2 (h3, h4)  →  Output (ŷ)
                        sigmoid σ                      sigmoid σ               sigmoid σ
```

| Layer   | Type           | Neurons | Activation |
|---------|----------------|---------|------------|
| Layer 0 | Input          | 2       | —          |
| Layer 1 | Hidden layer 1 | 2       | Sigmoid σ  |
| Layer 2 | Hidden layer 2 | 2       | Sigmoid σ  |
| Layer 3 | Output         | 1       | Sigmoid σ  |

> Shared bias b⁽¹⁾ added to both neurons in hidden layer 1; shared b⁽²⁾ added to both neurons in hidden layer 2. No bias at the output layer.

---

## Assigned Weights & Biases (Roll No: 25I-3075, Sheet: SE-D)

| Parameter | Value | Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|-----------|-------|
| w1  | −0.45 | w5  |  0.18 | w9  | −0.15 |
| w2  | −0.39 | w6  |  0.74 | w10 | −0.66 |
| w3  | −0.11 | w7  |  0.49 | b⁽¹⁾ | −0.12 |
| w4  |  0.29 | w8  | −0.49 | b⁽²⁾ | −0.09 |

---

## Dataset — Student Performance (Binary Classification)

| Sample | x1 (Study Hours) | x2 (Attendance) | y (Pass?) |
|--------|-----------------|-----------------|-----------|
| s⁽¹⁾  | 0.2             | 0.8             | 1         |
| s⁽²⁾  | 0.9             | 0.4             | 1         |
| s⁽³⁾  | 0.1             | 0.2             | 0         |

---

## Task Summary & Results

### Task 1 — Forward Pass

| Sample | x1  | x2  | a⁽¹⁾₁ | a⁽¹⁾₂ | a⁽²⁾₁ | a⁽²⁾₂ | y | ŷ      |
|--------|-----|-----|--------|--------|--------|--------|---|--------|
| s⁽¹⁾  | 0.2 | 0.8 | 0.4260 | 0.5085 | 0.5587 | 0.4940 | 1 | 0.3989 |
| s⁽²⁾  | 0.9 | 0.4 | 0.3615 | 0.4122 | 0.5441 | 0.4939 | 1 | 0.3995 |
| s⁽³⁾  | 0.1 | 0.2 | 0.4534 | 0.4748 | 0.5558 | 0.5032 | 0 | 0.3976 |

---

### Task 2 — Loss Calculation (MSE)

**Final Loss: L_MSE = 0.2933**

---

### Task 3 — Backpropagation (Sample s⁽¹⁾)

| Parameter | Current Value | Gradient  | Parameter | Current Value | Gradient  |
|-----------|--------------|-----------|-----------|--------------|-----------|
| w1        | −0.45        |  0.0012   | w6        |  0.74        |  0.0203   |
| w2        | −0.39        | −0.0008   | w7        |  0.49        |  0.0054   |
| w3        | −0.11        |  0.0049   | w8        | −0.49        |  0.0242   |
| w4        |  0.29        | −0.0031   | w9        | −0.15        | −0.1610   |
| w5        |  0.18        |  0.0045   | w10       | −0.66        | −0.1424   |
| b⁽¹⁾     | −0.12        |  0.0023   | b⁽²⁾     | −0.09        |  0.0582   |

---

### Task 4 — Weight Update & Multiple Iterations (η = 0.1)

| Iteration | MSE Loss | Decreased? |
|-----------|----------|------------|
| 0 (init)  | 0.2933   | —          |
| 1         | 0.2911   | ✓ YES      |
| 2         | 0.2890   | ✓ YES      |
| 3         | 0.2869   | ✓ YES      |
| 4         | 0.2848   | ✓ YES      |
| 5         | 0.2829   | ✓ YES      |

---

### Task 5 — Gradient Descent Variants

| Epoch | Batch | Samples Used | MSE After Update |
|-------|-------|--------------|-----------------|
| 1     | 1     | s⁽¹⁾, s⁽²⁾ | 0.3560          |
| 1     | 2     | s⁽³⁾        | 0.1592          |
| 2     | 1     | s⁽³⁾, s⁽¹⁾ | 0.2593          |
| 2     | 2     | s⁽²⁾        | 0.3533          |
| 3     | 1     | s⁽²⁾, s⁽³⁾ | 0.2581          |
| 3     | 2     | s⁽¹⁾        | 0.3483          |

---

### Task 6 — Optimizers (η = 0.1, β = 0.9)

| Iter | Plain GD | Momentum | NAG    | Fastest  |
|------|----------|----------|--------|----------|
| 1    | 0.2926   | 0.2930   | 0.2930 | Plain GD |
| 2    | 0.2919   | 0.2927   | 0.2927 | Plain GD |
| 3    | 0.2912   | 0.2923   | 0.2923 | Plain GD |
| 4    | 0.2905   | 0.2918   | 0.2918 | Plain GD |
| 5    | 0.2899   | 0.2913   | 0.2913 | Plain GD |
| 6    | 0.2892   | 0.2907   | 0.2907 | Plain GD |
| 7    | 0.2886   | 0.2902   | 0.2902 | Plain GD |
| 8    | 0.2880   | 0.2896   | 0.2896 | Plain GD |
| 9    | 0.2874   | 0.2890   | 0.2890 | Plain GD |
| 10   | 0.2868   | 0.2884   | 0.2884 | Plain GD |

---

### Task 7 — Python Implementation (MNIST)

Training config: Mini-batch GD · B = 32 · η = 0.1 · 20 epochs · `np.random.seed(42)`

Required notebook outputs:
1. Training loss curve (MSE vs. epoch)
2. Test accuracy on 10,000 test images
3. Sample predictions — 10 test images with true and predicted labels

---

## How to Run
```bash
pip install numpy matplotlib jupyter
jupyter notebook src/mlp_25I-3075.ipynb
```

Place `mnist.npz` in the `data/` folder before running.

---

## Submission

- **Repository name:** `mvc-mlp-25I-3075` (private — instructor added as collaborator with read access)
- **ZIP submission:** `25I-3075_[Name].ZIP` containing the `.ipynb` and PDF report
- Commit work progressively — do not push everything in one final commit
