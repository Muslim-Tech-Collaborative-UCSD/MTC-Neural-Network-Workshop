# Vibe-Coding Workshop: Build a Neural Network 🧠
---

## Table of Contents

1. [Setup](#setup)
2. [Part 1 — What is a Neural Network?](#part-1--what-is-a-neural-network)
3. [Part 2 — Pick Your Dataset](#part-2--pick-your-dataset)
4. [Part 3 — Building the Network](#part-3--building-the-network)
5. [Part 4 — Training Loop & Backpropagation](#part-4--training-loop--backpropagation)
6. [Part 5 — Experiment & Explore](#part-5--experiment--explore)
7. [Part 6 — Final Push & Wrap-Up](#part-6--final-push--wrap-up)
8. [Full Code](#full-code)
9. [Troubleshooting](#troubleshooting)

---

## Setup

### Prerequisites

- Python and NumPy installed (Google Colab or Jupyter Notebook recommended)
- Optional: `matplotlib` for plotting

Example:

```python
import numpy as np
print(np.__version__)
```

Install matplotlib (optional):

```bash
pip install matplotlib
```

### GitHub Setup

Why GitHub? GitHub is how developers save, share, and collaborate on code. It tracks every change so you can always go back.

1. Create a free account at https://github.com (if you don't have one).
2. Install Git (or confirm it's installed): `git --version`.
3. Create a new repository on GitHub (e.g., `neural-network-from-scratch`) and check "Add a README file".
4. Clone it:

```bash
git clone https://github.com/YOUR-USERNAME/neural-network-from-scratch.git
cd neural-network-from-scratch
```

Create your main Python file:

```bash
touch neural_net.py
```

### Claude Setup (optional)

- Open https://claude.ai and use it as an AI pair programmer for explanations, debugging, and conceptual help.

### First Commit

```bash
git add .
git commit -m "Initial commit - project setup"
git push origin main
```

---

## Part 1 — What is a Neural Network?

~15 minutes · Conceptual overview · No code yet

### The Neuron Analogy

A neuron takes inputs, multiplies by weights, adds a bias, and passes through an activation function:

```
inputs → [weights × inputs + bias] → activation → output
```

Analogy: inputs are ingredients, weights are how much of each you use, and the activation function decides if the result "fires".

### Layers

- **Input layer:** raw data (features)
- **Hidden layer(s):** where the learning happens
- **Output layer:** the prediction

### Key Vocabulary

| Term | Plain English |
| --- | --- |
| Weights | Knobs the network adjusts to learn patterns |
| Bias | An extra knob that shifts the output |
| Activation function | A filter that decides if a neuron "fires" (we'll use Sigmoid and ReLU) |
| Forward pass | Feeding data through the network to get a prediction |
| Loss | How wrong the prediction is (lower is better) |
| Backpropagation | Figuring out which knobs to turn to reduce the loss |
| Learning rate | How big a step we take when adjusting knobs |
| Epoch | One full pass through all the training data |

The big picture loop: Forward pass → Compute loss → Backpropagation → Update weights → Repeat

---

## Part 2 — Pick Your Dataset

~15 minutes · Choose a dataset that keeps training fast (under ~5 min on a laptop)

Constraints:

- Max ~10,000 samples
- Max ~20 features
- Binary classification preferred

Suggested datasets:

| Dataset | Description | Samples | Features | Source |
| --- | --- | ---: | ---: | --- |
| 🌸 Iris | Classify flower species by measurements | 150 | 4 | `sklearn.datasets.load_iris()` |
| 🩺 Breast Cancer | Predict malignant vs. benign tumors | 569 | 30 | `sklearn.datasets.load_breast_cancer()` |
| 🔢 MNIST Digits | Classify handwritten digits (use 2 classes) | ~1,000 | 784 | `sklearn.datasets.load_digits()` |
| 🍷 Wine | Classify wines by chemical properties | 178 | 13 | `sklearn.datasets.load_wine()` |
| 🫀 Heart Disease | Predict heart disease presence | 303 | 13 | UCI ML Repository / Kaggle |
| 🐧 Palmer Penguins | Classify penguin species | 344 | 4 | `pip install palmerpenguins` |

Starter code:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer  # swap for your choice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)  # shape: (n_samples, 1)

# Normalize features (important for neural nets!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Test samples: {X_test.shape[0]}")
```

---

## Part 3 — Building the Network

~50 minutes · Core code-along

### Activation functions

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

### Initialize the network

```python
def initialize_network(layer_sizes):
    """
    layer_sizes: list like [n_features, hidden1, hidden2, ..., n_output]
    Returns a list of (weights, biases) tuples for each layer.
    """
    np.random.seed(42)
    params = []
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
        b = np.zeros((1, layer_sizes[i+1]))
        params.append((w, b))
    return params
```

### Forward pass

```python
def forward(X, params):
    """
    Run input X through all layers.
    Returns list of activations (needed for backprop).
    """
    activations = [X]
    for i, (w, b) in enumerate(params):
        z = activations[-1] @ w + b
        if i == len(params) - 1:
            a = sigmoid(z)       # output layer
        else:
            a = relu(z)          # hidden layer(s)
        activations.append(a)
    return activations
```

### Loss function

```python
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8  # avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
```

---

## Part 4 — Training Loop & Backpropagation

~40 minutes

### Backpropagation

```python
def backward(y, activations, params):
    """
    Compute gradients via backpropagation.
    Returns list of (dw, db) for each layer.
    """
    m = y.shape[0]
    grads = []

    # Output layer error (derivative of BCE + sigmoid combined)
    delta = activations[-1] - y

    for i in reversed(range(len(params))):
        a_prev = activations[i]
        dw = (a_prev.T @ delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        grads.insert(0, (dw, db))

        if i > 0:  # propagate error to previous layer
            delta = (delta @ params[i][0].T) * relu_derivative(activations[i])

    return grads
```

### Update weights

```python
def update_params(params, grads, learning_rate):
    new_params = []
    for (w, b), (dw, db) in zip(params, grads):
        w = w - learning_rate * dw
        b = b - learning_rate * db
        new_params.append((w, b))
    return new_params
```

### Full training loop

```python
def train(X_train, y_train, layer_sizes, epochs=500, learning_rate=0.1):
    params = initialize_network(layer_sizes)
    losses = []

    for epoch in range(epochs):
        activations = forward(X_train, params)
        loss = binary_cross_entropy(y_train, activations[-1])
        losses.append(loss)
        grads = backward(y_train, activations, params)
        params = update_params(params, grads, learning_rate)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    return params, losses
```

### Train and evaluate

```python
n_features = X_train.shape[1]
layer_sizes = [n_features, 16, 1]

params, losses = train(X_train, y_train, layer_sizes, epochs=500, learning_rate=0.1)

def predict(X, params):
    activations = forward(X, params)
    predictions = (activations[-1] >= 0.5).astype(int)
    return predictions

y_pred = predict(X_test, params)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.1f}%")
```

Plot loss (optional):

```python
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()
```

---

## Part 5 — Experiment & Explore

~15 minutes · Try different configurations

- Change the architecture: e.g., `[n_features, 16, 8, 1]`
- Change the number of neurons: e.g., `[n_features, 64, 1]`
- Adjust the learning rate: try `0.01`, `1.0`, `10.0`
- Change epochs: try `100`, `1000`, `2000`
- Try a different dataset
- Stretch goal — multi-class classification with softmax

Discussion questions are listed in the original notes.

---

## Part 6 — Final Push & Wrap-Up

Create a `README.md` for your project. Example structure is in the original notes (title, dataset, architecture, results, what I learned).

---

## Full Code

The full runnable example is included in the original file. Use it as a single script or in a notebook. (See the `examples/` folder if present.)

---

## Troubleshooting

| Problem | Likely Cause | What to Ask Claude |
| --- | --- | --- |
| `NaN` in loss | Learning rate too high, or data not normalized | "My loss is NaN. Here's my code: [paste]. What's wrong?" |
| Accuracy stuck at ~50% | Not enough epochs, bad architecture, or data issue | "My neural network accuracy is stuck at 50%. [paste code]. How can I improve it?" |
| Shape mismatch error | Array dimensions don't align | "I'm getting a shape mismatch error: [paste error]. Here's my code: [paste]" |
| `git push` rejected | Need to pull first, or auth issue | "Git says my push was rejected. Here's the error: [paste]. How do I fix this?" |
| Can't load custom CSV | File path or format issue | "I'm trying to load a CSV for my neural network but getting [error]. The file is at [path]." |

---

## Git Commands Cheat Sheet

| Command | What It Does |
| --- | --- |
| `git clone <url>` | Download a repo to your computer |
| `git status` | See what files have changed |
| `git add .` | Stage all changes for commit |
| `git commit -m "msg"` | Save a snapshot with a description |
| `git push origin main` | Upload your commits to GitHub |
| `git log --oneline` | See your commit history |
