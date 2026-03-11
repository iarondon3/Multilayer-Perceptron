# 🧠 Multilayer Perceptron from Scratch

> A fully interactive multilayer neural network built in pure Python and NumPy, capable of training via backpropagation, saving/loading trained models, and classifying both binary and multi-class datasets.

---

## 📌 Description

This project implements a **Multilayer Perceptron (MLP)** from scratch, without the use of any specialized machine learning frameworks. The network supports configurable architecture, iterative training with backpropagation, forward-pass inference, and persistent model storage — all through a console-based interface.

This project builds upon a [single-layer perceptron](https://github.com/iarondon3/Perceptron) implementation, extending it into a full deep learning pipeline with automatic differentiation via backpropagation.


---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Matrix%20Operations-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![No ML Frameworks](https://img.shields.io/badge/No%20PyTorch%2FTensorFlow-Pure%20Implementation-green)

- **Python 3.x** — core language
- **NumPy** — matrix operations, weight initialization, normalization
- **Matplotlib** — scatter plots, error curves, confusion matrix
- No PyTorch, TensorFlow, or Keras — fully manual implementation
---

## ✨ Features

- 🏗️ Fully configurable **network architecture** (inputs, hidden layers, neurons per layer, outputs)
- 🔄 **Backpropagation** training with configurable learning rate and epochs
- ➡️ **Forward-pass inference** on unseen test data
- 💾 **Save and load** trained models to/from CSV files
- 📊 Automatic generation of **3 visualizations** after training and testing:
  - Scatter plot of real data (2D or 3D projection)
  - Correct vs incorrect predictions per data point
  - Training: error evolution per epoch · Testing: confusion matrix
- 🔢 Supports **binary and multi-class** classification (One-Hot Encoding)
- 🔁 Continuous training loop — keep training the same model across multiple sessions

---

## 🗂️ Project Structure

```
multilayer-perceptron/
│
├── main.py                    # Entry point and main menu logic
├── neural_network.py          # Network core: initialization, forward pass, backpropagation
├── activation_functions.py    # Activation functions (Sigmoid + derivative)
├── data.py                    # CSV loading, normalization, One-Hot encoding, save/load model
├── plots.py                   # Visualizations: scatter plots, error curve, confusion matrix
│
├── 2d_2_class_train.csv       # Training set: 2D input, 2-class output
├── 2d_2_class_test.csv        # Test set: 2D input, 2-class output
├── trained_model_2d_2.csv     # Trained model: 2D / 2-class (ready to load)
├── 3d_4_class_train.csv       # Training set: 3D input, 4-class output
├── 3d_4_class_test.csv        # Test set: 3D input, 4-class output
└── trained_model_3d_4.csv     # Trained model: 3D / 4-class (ready to load)
```
---
## 📊 Visualizations

### During Training
| Plot | Description |
|---|---|
| **1. Real Data** | Scatter plot colored by true class label |
| **2. Predictions** | Green = correct · Red = incorrect |
| **3. Error Curve** | Mean squared error per epoch |

### During Testing
| Plot | Description |
|---|---|
| **1. Real Data** | Scatter plot colored by true class label |
| **2. Predictions** | Green = correct · Red = incorrect |
| **3. Confusion Matrix** | Predicted vs actual class using `matshow` |

> For datasets with 3+ input dimensions, the first three are rendered using a **3D projection**.

---

<details>
<summary>🚀 Getting Started</summary>

### 1. Clone the repository

```bash
git clone https://github.com/iarondon3/multilayer-perceptron.git
cd multilayer-perceptron
```

### 2. Install dependencies

```bash
pip install numpy matplotlib
```

### 3. Run the program

```bash
python main.py
```

### Program Flow

```
1. Choose: create a new network or load a saved model
2. If new → configure: inputs, outputs, hidden layers, neurons per layer
3. Main menu:
   ├── 1. Train  → provide training CSV + number of epochs
   ├── 2. Test   → provide test CSV → see predictions + confusion matrix
   ├── 3. Save   → export trained model to CSV
   └── 4. Exit
4. After training or testing, visualizations are displayed automatically
5. Return to main menu to continue training or testing
```

</details>

---

<details>
<summary>📁 File Formats</summary>

### Data Files (CSV)
One vector per row — last column is the class label:

```csv
x1,x2,class
0.52,0.91,1
0.13,0.47,-1
...
```

### Model Files (CSV)
Trained models are saved with hyperparameters in the first 5 rows, followed by one row per neuron:

```
u, 2          ← input size
v, 1          ← output size
L, 2          ← hidden layers
b, 4          ← neurons per layer
e, 5000       ← epochs trained
h,1,0,w1,...,bias
o,2,0,w1,...,bias
```

</details>

---

<details>
<summary>🧮 Mathematical Foundation</summary>

**Forward Pass (per layer):**
```
Z = X · W + b
A = σ(Z)
```

**Backpropagation:**
```
δ_output = (target - output) · σ'(output)
δ_hidden = (δ_next · Wᵀ) · σ'(activation)
W = W + α · Aᵀ · δ
```

**Activation Function — Sigmoid:**
```
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x) · (1 - σ(x))
```

**Multi-class support via One-Hot Encoding** — automatically applied when more than 2 classes are detected in the dataset.

</details>

---

## 👨‍💻 About the Author
*Isabella Rondón* | ***Business Economist & Data Analyst*** 

[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/isabella-rondon-rojas-/)
[![GitHub Portfolio](https://img.shields.io/badge/Visit-Portfolio-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/iarondon3) 







