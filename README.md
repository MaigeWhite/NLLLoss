# NLLLoss - Ascend C Custom Operator

## 📝 Operator Overview: NLLLoss (Negative Log Likelihood Loss)

This custom operator implements the forward computation of **Negative Log Likelihood Loss (NLLLoss)**, a commonly used loss function for classification tasks during neural network training on Ascend AI processors.

---

## 🎯 Functionality Description

The operator computes the NLL loss for each sample based on:

- **Input logits (`x`)**
- **Target class indices (`target`)**
- **Class weights (`weight`)**

It supports three reduction modes for flexible output control:

| Mode  | Description            | Output Shape |
|-------|------------------------|--------------|
| **1** | Per-sample loss output | `[N]`        |
| **2** | Weighted average loss  | `[1]`        |
| **3** | Total loss sum         | `[1]`        |

---

## 📥 Input Parameters

| Parameter | Description                       | Shape   |
|-----------|---------------------------------|---------|
| `x`       | Model prediction scores (logits) | `[N, C]` |
| `target`  | Ground truth class index for each sample | `[N]` |
| `weight`  | Weight for each class             | `[C]`   |
| `shape`   | Tensor shape info, typically `[N, C]` | -       |
| `mode`    | Output mode (1/2/3)              | Scalar  |

---

## ⚙️ Core Computation Logic

For each sample:

1. Select the score corresponding to the target class from `x`.
2. Negate the score and apply the class-specific weight.
3. Accumulate and output results according to the selected `mode`.

---

## ✅ Data Type Support

- **Half-precision float (`half`)**
- **Single-precision float (`float`)**

Implemented using C++ templates for flexible data type support.

---


## 📥 Clone the Repository

```bash
git clone https://github.com/MaigeWhite/NLLLoss.git
cd NLLLoss
```

## 📥 Build the Operator

```bash
bash build.sh

This will generate the installer package at
./build_out/custom_opp_ubuntu_aarch64.run
```

## 📥 Install the Operator

```bash
bash ./build_out/custom_opp_ubuntu_aarch64.run

This will install the custom operator to the default Ascend CANN custom operator path.
```

## 📥 Run Test Cases
```bash
After installation, run the provided test cases

bash NLLLossCase/NLLLoss_Case1/run.sh
bash NLLLossCase/NLLLoss_Case2/run.sh
bash NLLLossCase/NLLLoss_Case3/run.sh
bash NLLLossCase/NLLLoss_Case4/run.sh

Each test case verifies the operator under different configurations (input shape, reduction mode).
```
