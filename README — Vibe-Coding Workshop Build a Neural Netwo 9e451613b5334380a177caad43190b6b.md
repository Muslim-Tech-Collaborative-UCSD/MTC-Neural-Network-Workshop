# README — Vibe-Coding Workshop: Build a Neural Network

# Vibe-Coding Workshop: Build a Neural Network from Scratch 🧠

A hands-on workshop where you build a fully functioning neural network using **only Python and NumPy** — no frameworks, no magic. You'll also learn Git/GitHub basics and how to use Claude AI as a coding companion.

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
- Optional: matplotlib for plotting

```python
import numpy as np
print(np.__version__)
```

```python
pip install matplotlib
```

### GitHub Setup

> **Why GitHub?** GitHub is how developers save, share, and collaborate on code. Think of it as Google Docs for code — it tracks every change you make so you can always go back.
> 
1. Create a free account at [github.com](http://github.com) (if you don't have one)
2. Install Git (or confirm it's installed): `git --version`
3. Create a new repository on GitHub:
    - Click **"New repository"**
    - Name it `neural-network-from-scratch`
    - Check **"Add a README file"**
    - Click **Create repository**
4. Clone it to your machine:

```bash
git clone https://github.com/YOUR-USERNAME/neural-network-from-scratch.git
cd neural-network-from-scratch
```

1. Create your main Python file:

```bash
touch neural_net.py
```

### Claude Setup

- Open [claude.ai](http://claude.ai) in a browser tab (free tier works fine)
- Use Claude as your AI pair programmer throughout the workshop:
    - 🔍 **Understand code:** Paste any snippet and ask *"Explain this to me like I'm a beginner"*
    - 🐛 **Debug errors:** Paste your error message and code, ask *"What's wrong and how do I fix it?"*
    - 💡 **Explore concepts:** Ask *"What is backpropagation in simple terms?"*
    - 🎯 **Get unstuck:** Paste where you are and ask Claude to help you catch up

### First Commit 🎉

```bash
git add .
git commit -m "Initial commit - project setup"
git push origin main
```

> `git add .` stages all changed files. `git commit -m "message"` saves a snapshot with a description. `git push` uploads it to GitHub. You'll do this after every major section.
> 

---

## Part 1 — What is a Neural Network?

*~15 minutes · Conceptual overview · No code yet*

### The Neuron Analogy

A neuron takes inputs, multiplies by weights, adds a bias, and passes through an activation function:

`inputs → [weights × inputs + bias] → activation → output`

**Analogy:** Think of it like a recipe — inputs are ingredients, weights are how much of each you use, and the activation function decides if the dish is "good enough" to serve.

### Layers

- **Input layer:** Your raw data (features) — like columns in a spreadsheet
- **Hidden layer(s):** Where the "learning" happens — the brain's thinking
- **Output layer:** The prediction — the final answer

### Key Vocabulary

| **Term** | **Plain English** |
| --- | --- |
| Weights | Knobs the network adjusts to learn patterns |
| Bias | An extra knob that shifts the output |
| Activation function | A filter that decides if a neuron "fires" (we'll use Sigmoid and ReLU) |
| Forward pass | Feeding data through the network to get a prediction |
| Loss | How wrong the prediction is (lower is better) |
| Backpropagation | Figuring out which knobs to turn (and how much) to reduce the loss |
| Learning rate | How big a step we take when adjusting knobs |
| Epoch | One full pass through all the training data |

### The Big Picture Loop

**Forward pass → Compute loss → Backpropagation → Update weights → Repeat**

**Analogy:** It's like studying for an exam — you take a practice test (forward pass), check your score (loss), review what you got wrong (backprop), study those topics (update weights), and repeat.

---

## Part 2 — Pick Your Dataset

*~15 minutes · Choose a dataset that interests you*

### Dataset Constraints

To keep training fast (< 5 min on a laptop):

- Max ~10,000 samples (rows)
- Max ~20 features (columns)
- Binary classification preferred

### Suggested Datasets

| **Dataset** | **Description** | **Samples** | **Features** | **Source** |
| --- | --- | --- | --- | --- |
| 🌸 Iris | Classify flower species by measurements | 150 | 4 | `sklearn.datasets.load_iris()` |
| 🩺 Breast Cancer | Predict malignant vs. benign tumors | 569 | 30 | `sklearn.datasets.load_breast_cancer()` |
| 🔢 MNIST Digits | Classify handwritten digits (use 2 classes) | ~1,000 | 784 | `sklearn.datasets.load_digits()` |
| 🍷 Wine | Classify wines by chemical properties | 178 | 13 | `sklearn.datasets.load_wine()` |
| 🫀 Heart Disease | Predict heart disease presence | 303 | 13 | UCI ML Repository / Kaggle |
| 🐧 Palmer Penguins | Classify penguin species | 344 | 4 | `pip install palmerpenguins` |

### Loading Your Dataset — Starter Code

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

```bash
git add .
git commit -m "Add dataset loading and preprocessing"
git push origin main
```

---

## Part 3 — Building the Network

*~50 minutes · Core code-along*

### Step 3.1 — Activation Functions

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

### Step 3.2 — Initialize the Network

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

### Step 3.3 — Forward Pass

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

### Step 3.4 — Loss Function

```python
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8  # avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
```

```bash
git add .
git commit -m "Add activation functions, network init, forward pass, and loss function"
git push origin main
```

---

## Part 4 — Training Loop & Backpropagation

*~40 minutes*

### Step 4.1 — Backpropagation

```python
def backward(y, activations, params):
    """
    Compute gradients via backpropagation.
    Returns list of (dw, db) for each layer.
    """
    m = y.shape[0]
    grads = []
    
    # Output layer error
    delta = activations[-1] - y  # derivative of BCE + sigmoid combined
    
    for i in reversed(range(len(params))):
        a_prev = activations[i]
        dw = (a_prev.T @ delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        grads.insert(0, (dw, db))
        
        if i > 0:  # propagate error to previous layer
            delta = (delta @ params[i][0].T) * relu_derivative(activations[i])
    
    return grads
```

### Step 4.2 — Update Weights

```python
def update_params(params, grads, learning_rate):
    new_params = []
    for (w, b), (dw, db) in zip(params, grads):
        w = w - learning_rate * dw
        b = b - learning_rate * db
        new_params.append((w, b))
    return new_params
```

### Step 4.3 — The Full Training Loop

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

### Step 4.4 — Train It!

```python
n_features = X_train.shape[1]
layer_sizes = [n_features, 16, 1]

params, losses = train(X_train, y_train, layer_sizes, epochs=500, learning_rate=0.1)
```

### Step 4.5 — Evaluate

```python
def predict(X, params):
    activations = forward(X, params)
    predictions = (activations[-1] >= 0.5).astype(int)
    return predictions

y_pred = predict(X_test, params)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.1f}%")
```

### Step 4.6 — Plot the Loss Curve (Optional)

```python
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()
```

```bash
git add .
git commit -m "Add backprop, training loop, and evaluation - model works!"
git push origin main
```

---

## Part 5 — Experiment & Explore

*~15 minutes · Try different configurations*

- [ ]  **Change the architecture:** Add another hidden layer → `[n_features, 16, 8, 1]`
- [ ]  **Change the number of neurons:** Try `[n_features, 64, 1]` or `[n_features, 4, 1]`
- [ ]  **Adjust the learning rate:** What happens at `0.01`? At `1.0`? At `10.0`?
- [ ]  **Change epochs:** Try `100` vs `1000` vs `2000`
- [ ]  **Try a different dataset:** Swap out the dataset and re-run
- [ ]  **Stretch goal — Multi-class:** Modify for multi-class classification with softmax

### Discussion Questions

- What happened when you made the learning rate too high?
- Did adding more layers always help?
- How does the size of your dataset affect accuracy?
- What's the difference between memorizing the data and actually learning patterns? *(intro to overfitting)*

```bash
git add .
git commit -m "Experiment with different architectures and hyperparameters"
git push origin main
```

---

## Part 6 — Final Push & Wrap-Up

*~15 minutes*

Create a `README.md` for your GitHub repo. Example structure:

```markdown
# Neural Network from Scratch 🧠

A simple feedforward neural network built with only Python and NumPy.
Built during the Vibe-Coding Workshop.

## Dataset
- **Dataset:** Breast Cancer Wisconsin
- **Task:** Binary classification (malignant vs. benign)

## Architecture
- Input: 30 features
- Hidden: 16 neurons (ReLU)
- Output: 1 neuron (Sigmoid)

## Results
- **Test Accuracy:** 97.4%
- **Epochs:** 500
- **Learning Rate:** 0.1

## What I Learned
- How forward propagation works
- How backpropagation adjusts weights
- How to use Git and GitHub
- How to use Claude as a coding companion
```

```bash
git add .
git commit -m "Add README and final project code"
git push origin main
```

---

## Full Code

- Click to expand the complete code
    
    ```python
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    # ---- DATA ----
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ---- ACTIVATION FUNCTIONS ----
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(a):
        return a * (1 - a)
    
    def relu(z):
        return np.maximum(0, z)
    
    def relu_derivative(z):
        return (z > 0).astype(float)
    
    # ---- NETWORK ----
    def initialize_network(layer_sizes):
        np.random.seed(42)
        params = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            params.append((w, b))
        return params
    
    def forward(X, params):
        activations = [X]
        for i, (w, b) in enumerate(params):
            z = activations[-1] @ w + b
            if i == len(params) - 1:
                a = sigmoid(z)
            else:
                a = relu(z)
            activations.append(a)
        return activations
    
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(y, activations, params):
        m = y.shape[0]
        grads = []
        delta = activations[-1] - y
        for i in reversed(range(len(params))):
            a_prev = activations[i]
            dw = (a_prev.T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            grads.insert(0, (dw, db))
            if i > 0:
                delta = (delta @ params[i][0].T) * relu_derivative(activations[i])
        return grads
    
    def update_params(params, grads, learning_rate):
        return [(w - learning_rate * dw, b - learning_rate * db)
                for (w, b), (dw, db) in zip(params, grads)]
    
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
    
    def predict(X, params):
        activations = forward(X, params)
        return (activations[-1] >= 0.5).astype(int)
    
    # ---- RUN ----
    n_features = X_train.shape[1]
    params, losses = train(X_train, y_train, [n_features, 16, 1], epochs=500, learning_rate=0.1)
    
    y_pred = predict(X_test, params)
    print(f"Test Accuracy: {np.mean(y_pred == y_test) * 100:.1f}%")
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    ```
    

---

## Troubleshooting

| **Problem** | **Likely Cause** | **What to Ask Claude** |
| --- | --- | --- |
| `NaN` in loss | Learning rate too high, or data not normalized | *"My loss is NaN. Here's my code: [paste]. What's wrong?"* |
| Accuracy stuck at ~50% | Not enough epochs, bad architecture, or data issue | *"My neural network accuracy is stuck at 50%. [paste code]. How can I improve it?"* |
| Shape mismatch error | Array dimensions don't align | *"I'm getting a shape mismatch error: [paste error]. Here's my code: [paste]"* |
| `git push` rejected | Need to pull first, or auth issue | *"Git says my push was rejected. Here's the error: [paste]. How do I fix this?"* |
| Can't load custom CSV | File path or format issue | *"I'm trying to load a CSV for my neural network but getting [error]. The file is at [path]."* |

---

## Git Commands Cheat Sheet

| **Command** | **What It Does** |
| --- | --- |
| `git clone <url>` | Download a repo to your computer |
| `git status` | See what files have changed |
| `git add .` | Stage all changes for commit |
| `git commit -m "msg"` | Save a snapshot with a description |
| `git push origin main` | Upload your commits to GitHub |
| `git log --oneline` | See your commit history |