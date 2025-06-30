# NLLLoss - Ascend C Custom Operator

This project implements a **custom NLLLoss (Negative Log Likelihood Loss) operator** using **Ascend C** for the Ascend AI processor.

---

## 📌 Features

- Implemented with Ascend C kernel
- Supports multiple reduction modes
- Includes multiple test cases for validation

---

## 📥 Clone the Repository

```bash
git clone https://github.com/MaigeWhite/NLLLoss.git
cd NLLLoss
```

## 📥 Build the Operator

```bash
bash build.sh

This will generate the installer package at:
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
