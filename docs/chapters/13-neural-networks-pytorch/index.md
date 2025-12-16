# Neural Networks and PyTorch

---
title: Neural Networks and PyTorch
description: The ultimate data science superpower - teaching machines to think
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

## Summary

This comprehensive chapter introduces neural networks and deep learning using PyTorch. Students will learn neural network architecture including neurons, layers, activation functions, and propagation algorithms. The chapter covers PyTorch fundamentals including tensors, autograd, and building neural network modules. Students will implement complete training loops with optimizers and loss functions. The chapter concludes with best practices for model interpretability, documentation, reproducibility, ethics, and capstone project development. By the end of this chapter, students will be able to build, train, and deploy neural network models while following professional best practices.

## Concepts Covered

This chapter covers the following 55 concepts from the learning graph:

### Neural Networks (20 concepts)

1. Neural Networks
2. Artificial Neuron
3. Perceptron
4. Activation Function
5. Sigmoid Function
6. ReLU Function
7. Input Layer
8. Hidden Layer
9. Output Layer
10. Weights
11. Biases
12. Forward Propagation
13. Backpropagation
14. Deep Learning
15. Network Architecture
16. Epochs
17. Batch Size
18. Mini-batch
19. Stochastic Gradient
20. Vanishing Gradient

### PyTorch (20 concepts)

21. PyTorch Library
22. Tensors
23. Tensor Operations
24. Autograd
25. Automatic Differentiation
26. Computational Graph
27. Neural Network Module
28. Sequential Model
29. Linear Layer
30. Loss Functions PyTorch
31. Optimizer
32. SGD Optimizer
33. Adam Optimizer
34. Training Loop
35. Model Evaluation PyTorch
36. GPU Computing
37. CUDA
38. Model Saving
39. Model Loading
40. Transfer Learning

### Best Practices (10 concepts)

41. Explainable AI
42. Model Interpretability
43. Feature Importance Analysis
44. SHAP Values
45. Model Documentation
46. Reproducibility
47. Random Seed
48. Version Control
49. Git
50. Data Ethics

### Projects (5 concepts)

51. Capstone Project
52. End-to-End Pipeline
53. Model Deployment
54. Results Communication
55. Data-Driven Decisions

## Prerequisites

This chapter builds on concepts from:

- [Chapter 10: NumPy and Numerical Computing](../10-numpy-computing/index.md)
- [Chapter 11: Non-linear Models and Regularization](../11-nonlinear-models-regularization/index.md)
- [Chapter 12: Introduction to Machine Learning](../12-intro-to-machine-learning/index.md)

---

## Introduction: Welcome to the Deep End

You've arrived at the most exciting chapter in this entire book. Everything you've learned—data structures, visualization, statistics, regression, model evaluation, NumPy, optimization—has been preparing you for this moment. **Neural networks** are the technology behind self-driving cars, language translation, image recognition, and AI assistants. And now you're going to build them yourself.

Neural networks aren't magic, even though they sometimes feel like it. They're just the concepts you already know—gradient descent, loss functions, matrix multiplication—stacked together in clever ways. If you understood the last chapter, you have everything you need to understand neural networks.

By the end of this chapter, you'll have:

- Built neural networks from scratch and with PyTorch
- Trained models using professional techniques
- Learned best practices for real-world deployment
- Prepared for your capstone project

Let's unlock your ultimate data science superpower.

---

# Part 1: Neural Network Fundamentals

## What Are Neural Networks?

**Neural networks** are computing systems loosely inspired by biological brains. They consist of interconnected nodes (neurons) organized in layers that learn to transform inputs into outputs through training.

The key insight: neural networks are *universal function approximators*. Given enough neurons and data, they can learn virtually any pattern—recognizing faces, translating languages, playing chess, or predicting stock prices.

```python
# A neural network is just layers of transformations
# Input → Transform → Transform → ... → Output

# Conceptually:
def neural_network(x):
    h1 = activation(x @ W1 + b1)      # First hidden layer
    h2 = activation(h1 @ W2 + b2)     # Second hidden layer
    output = h2 @ W3 + b3              # Output layer
    return output
```

What makes neural networks special:

| Feature | Traditional ML | Neural Networks |
|---------|---------------|-----------------|
| Feature engineering | Manual, requires expertise | Automatic, learned from data |
| Complexity | Limited by model type | Unlimited (add more layers) |
| Data requirements | Works with less data | Needs lots of data |
| Interpretability | Often clear | Often opaque (black box) |

## The Artificial Neuron: The Basic Unit

An **artificial neuron** (or node) is the fundamental building block of neural networks. It takes multiple inputs, multiplies each by a weight, adds them up with a bias, and passes the result through an activation function.

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w} \cdot \mathbf{x} + b)$$

Where:

- $x_i$ are the inputs
- $w_i$ are the weights (learned parameters)
- $b$ is the bias (learned parameter)
- $f$ is the activation function
- $y$ is the output

```python
import numpy as np

def artificial_neuron(inputs, weights, bias, activation_fn):
    """
    A single artificial neuron
    """
    # Weighted sum
    z = np.dot(inputs, weights) + bias

    # Apply activation function
    output = activation_fn(z)

    return output

# Example
inputs = np.array([0.5, 0.3, 0.2])
weights = np.array([0.4, 0.6, 0.8])
bias = 0.1

output = artificial_neuron(inputs, weights, bias, lambda x: max(0, x))
print(f"Neuron output: {output:.4f}")
```

## The Perceptron: The Simplest Neural Network

The **perceptron** is the simplest neural network—just a single neuron with a step function as its activation. It was invented in 1958 and could learn to classify linearly separable data.

```python
def perceptron(x, weights, bias):
    """
    Original perceptron: weighted sum + step function
    """
    z = np.dot(x, weights) + bias
    return 1 if z > 0 else 0

# A perceptron can learn simple patterns
# But it can't learn XOR - this limitation led to the "AI winter"
```

The perceptron's limitations sparked the development of multi-layer networks with non-linear activation functions—the neural networks we use today.

## Activation Functions: Adding Non-Linearity

**Activation functions** introduce non-linearity into neural networks. Without them, stacking layers would be pointless—a sequence of linear transformations is just one linear transformation. Activation functions allow networks to learn complex, non-linear patterns.

### Sigmoid Function

The **sigmoid function** squashes any input to a value between 0 and 1:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output range: (0, 1)
# Good for: output layer in binary classification (probability)
# Problem: vanishing gradients for very large/small inputs
```

### ReLU Function

The **ReLU (Rectified Linear Unit)** function is the most popular activation in modern networks:

$$\text{ReLU}(x) = \max(0, x)$$

```python
def relu(x):
    return np.maximum(0, x)

# Output range: [0, ∞)
# Good for: hidden layers, fast to compute
# Problem: "dying ReLU" - neurons can get stuck at 0
```

| Activation | Formula | Range | Use Case |
|------------|---------|-------|----------|
| Sigmoid | $1/(1+e^{-x})$ | (0, 1) | Binary classification output |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | (-1, 1) | Hidden layers (centered) |
| ReLU | $\max(0, x)$ | [0, ∞) | Hidden layers (default choice) |
| Softmax | $e^{x_i}/\sum e^{x_j}$ | (0, 1), sums to 1 | Multi-class output |

#### Diagram: Activation Function Explorer

<details markdown="1">
<summary>Activation Function Explorer</summary>
Type: microsim

Bloom Taxonomy: Understand, Apply

Learning Objective: Visualize different activation functions and understand why non-linearity is essential for neural networks

Canvas Layout (850x550):
- Main area (850x400): Graph showing activation function curves
- Bottom area (850x150): Controls and information panel

Main Visualization:
- X-axis range: -5 to 5
- Y-axis range: -2 to 2 (adjustable)
- Multiple activation functions plotted (selectable)
- Derivative shown as dashed line (optional)
- Current function highlighted prominently

Activation Functions to Include:
1. Linear (y = x) - shows why this is useless
2. Step function - original perceptron
3. Sigmoid - smooth S-curve
4. Tanh - centered sigmoid
5. ReLU - simple but powerful
6. Leaky ReLU - fixes dying ReLU
7. Softmax - for probabilities (1D simplified)

Interactive Controls:
- Checkboxes: Select which functions to display
- Toggle: "Show derivatives"
- Input field: Enter x value, see f(x) for each function
- Slider: Adjust x to see moving point on each curve

Educational Annotations:
- Point out vanishing gradient regions (sigmoid extremes)
- Show where ReLU has zero gradient
- Demonstrate why non-linearity enables complex patterns

Demo: "Why Non-linearity Matters"
- Button to show: linear combinations of linear = still linear
- Animation showing stacked linear layers collapsing to one

Implementation: p5.js with multiple function plots
</details>

## Network Architecture: Layers and Depth

**Network architecture** describes how neurons are organized into layers and connected. The architecture determines what patterns the network can learn.

### Input Layer

The **input layer** receives the raw data. It has one neuron per feature—no computation happens here, just data entry.

### Hidden Layers

**Hidden layers** perform the actual computation. They're "hidden" because we don't directly observe their outputs. More hidden layers = deeper network = more complex patterns.

### Output Layer

The **output layer** produces the final prediction. Its structure depends on the task:

- Regression: 1 neuron, no activation (or linear)
- Binary classification: 1 neuron, sigmoid activation
- Multi-class: N neurons, softmax activation

```python
# A simple architecture
"""
Input (4 features) → Hidden (8 neurons, ReLU) → Hidden (4 neurons, ReLU) → Output (1 neuron)

Layer sizes: [4, 8, 4, 1]
Total parameters: (4×8 + 8) + (8×4 + 4) + (4×1 + 1) = 40 + 36 + 5 = 81
"""
```

**Deep learning** refers to neural networks with many hidden layers. Depth allows networks to learn hierarchical features—simple patterns in early layers, complex patterns in later layers.

#### Diagram: Neural Network Architecture Builder

<details markdown="1">
<summary>Neural Network Architecture Builder</summary>
Type: microsim

Bloom Taxonomy: Apply, Create

Learning Objective: Build and visualize neural network architectures, understanding how layer sizes and depth affect the network

Canvas Layout (900x600):
- Main area (650x600): Network visualization
- Right panel (250x600): Architecture controls

Network Visualization:
- Circles represent neurons arranged in vertical layers
- Lines connect neurons between adjacent layers
- Line thickness proportional to weight magnitude (after training)
- Neurons colored by activation value during forward pass
- Labels showing layer names and sizes

Layer Representation:
- Input layer on left (green circles)
- Hidden layers in middle (blue circles)
- Output layer on right (orange circles)
- If too many neurons, show sample with "..." indicator

Interactive Controls:
- Slider: Number of hidden layers (1-5)
- Slider for each hidden layer: Number of neurons (1-128)
- Dropdown: Input size (preset options or custom)
- Dropdown: Output size (1 for regression, N for classification)
- Dropdown: Activation function per layer

Parameter Counter:
- Total weights: calculated live
- Total biases: calculated live
- Total parameters: sum

Forward Pass Animation:
- Button: "Run Forward Pass"
- Watch activations flow through network
- Color intensity shows activation magnitude
- Step-by-step or continuous animation

Preset Architectures:
- "Simple" [4, 8, 1]
- "Deep" [4, 32, 16, 8, 1]
- "Wide" [4, 128, 1]
- "Classification" [4, 16, 8, 3]

Implementation: p5.js with animated data flow
</details>

## Weights and Biases: The Learnable Parameters

**Weights** and **biases** are the parameters that the network learns during training.

- **Weights** determine how strongly each input affects the output. Large positive weights amplify the input; negative weights invert it.
- **Biases** allow neurons to activate even when inputs are zero. They shift the activation function left or right.

```python
# For a layer with 10 inputs and 5 outputs:
# Weights: 10 × 5 = 50 parameters
# Biases: 5 parameters
# Total: 55 parameters

import numpy as np

# Initialize weights (many strategies exist)
n_in, n_out = 10, 5

# Xavier/Glorot initialization (good for sigmoid/tanh)
weights_xavier = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)

# He initialization (good for ReLU)
weights_he = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

# Biases usually start at zero
biases = np.zeros(n_out)
```

The number of parameters in a network:

$$\text{Parameters} = \sum_{\ell=1}^{L} (n_{\ell-1} \times n_\ell + n_\ell)$$

Where $n_\ell$ is the number of neurons in layer $\ell$.

## Forward Propagation: Making Predictions

**Forward propagation** is the process of passing inputs through the network to get outputs. Data flows forward from input to output, layer by layer.

```python
def forward_propagation(X, weights, biases, activations):
    """
    Forward pass through a neural network
    """
    a = X  # Input is the first "activation"

    for W, b, activation in zip(weights, biases, activations):
        z = a @ W + b           # Linear transformation
        a = activation(z)        # Non-linear activation

    return a  # Final output

# Example: 3-layer network
weights = [W1, W2, W3]
biases = [b1, b2, b3]
activations = [relu, relu, sigmoid]

predictions = forward_propagation(X, weights, biases, activations)
```

Forward propagation is just matrix multiplications and function applications—exactly what NumPy and PyTorch are optimized for.

## Backpropagation: Learning from Errors

**Backpropagation** is the algorithm that computes gradients for training neural networks. It works backward from the output, propagating error signals to update all weights and biases.

The key insight: use the chain rule from calculus. If error depends on output, and output depends on weights, we can compute how error depends on weights.

$$\frac{\partial \text{Loss}}{\partial w} = \frac{\partial \text{Loss}}{\partial \text{output}} \times \frac{\partial \text{output}}{\partial w}$$

```python
# Backprop computes gradients layer by layer, from output to input
# We won't implement it manually - PyTorch does it automatically!

# But conceptually:
# 1. Compute loss at output
# 2. Compute gradient of loss w.r.t. output layer weights
# 3. Propagate gradient backward to previous layer
# 4. Repeat until input layer
# 5. Update all weights using gradients
```

The good news: you rarely implement backpropagation manually. Modern frameworks like PyTorch compute gradients automatically.

## Training Concepts: Epochs, Batches, and Stochastic Gradient Descent

When training neural networks, we don't process the entire dataset at once. Instead, we use batches and multiple passes.

**Epoch**: One complete pass through the entire training dataset.

**Batch size**: Number of samples processed before updating weights.

**Mini-batch**: A subset of the training data used for one gradient update.

**Stochastic Gradient Descent (SGD)**: Using random mini-batches instead of the full dataset for each update.

| Approach | Batch Size | Pros | Cons |
|----------|-----------|------|------|
| Batch GD | Entire dataset | Stable, accurate gradients | Slow, memory intensive |
| Stochastic GD | 1 sample | Fast updates, escapes local minima | Noisy, unstable |
| Mini-batch GD | 32-256 samples | Best of both worlds | Sweet spot |

```python
# Typical training configuration
epochs = 100          # 100 passes through the data
batch_size = 32       # Process 32 samples at a time
n_samples = 10000     # Total training samples

batches_per_epoch = n_samples // batch_size  # 312 batches
total_updates = epochs * batches_per_epoch    # 31,200 weight updates
```

## The Vanishing Gradient Problem

The **vanishing gradient problem** occurs when gradients become extremely small in deep networks, causing early layers to learn very slowly (or not at all).

Why it happens: Sigmoid and tanh saturate for large inputs, producing gradients near zero. When you multiply many small gradients together through backpropagation, the result vanishes.

Solutions:

- Use ReLU activation (gradients are 1 for positive inputs)
- Use batch normalization
- Use skip connections (ResNets)
- Careful weight initialization

```python
# Sigmoid gradient is at most 0.25
# After 10 layers: 0.25^10 = 0.000001
# Gradients essentially disappear!

# ReLU gradient is 1 for positive inputs
# Gradients flow through unchanged
```

---

# Part 2: PyTorch Fundamentals

## The PyTorch Library

**PyTorch** is a deep learning framework created by Facebook's AI Research lab. It's the most popular framework for research and increasingly popular in industry.

Why PyTorch:

- **Pythonic**: Feels like natural Python code
- **Dynamic graphs**: Build networks on-the-fly
- **Easy debugging**: Use standard Python debugger
- **GPU acceleration**: Automatic CUDA support
- **Rich ecosystem**: torchvision, torchaudio, transformers

```python
import torch
import torch.nn as nn
import torch.optim as optim

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Tensors: PyTorch's Data Structure

**Tensors** are PyTorch's core data structure—like NumPy arrays but with GPU support and automatic differentiation.

```python
import torch

# Create tensors
a = torch.tensor([1, 2, 3, 4])                    # From list
b = torch.zeros(3, 4)                              # 3x4 zeros
c = torch.ones(2, 3)                               # 2x3 ones
d = torch.randn(5, 5)                              # Random normal
e = torch.arange(0, 10, 2)                         # Range

# From NumPy
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)

# Tensor properties
print(f"Shape: {d.shape}")
print(f"Data type: {d.dtype}")
print(f"Device: {d.device}")
```

## Tensor Operations

**Tensor operations** work similarly to NumPy but run on GPU when available.

```python
# Basic operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise
print(a + b)           # Addition
print(a * b)           # Multiplication
print(a ** 2)          # Power

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B              # Matrix multiplication
print(f"Result shape: {C.shape}")  # (3, 5)

# Aggregations
print(A.sum())         # Sum all elements
print(A.mean(dim=0))   # Mean along dimension 0
print(A.max())         # Maximum value

# Reshaping
D = torch.arange(12)
print(D.reshape(3, 4))
print(D.view(2, 6))    # view is like reshape but shares memory
```

## Autograd: Automatic Differentiation

**Autograd** is PyTorch's automatic differentiation engine. It computes gradients automatically—no manual backpropagation needed!

```python
# Enable gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Perform operations
y = x ** 2
z = y.sum()  # z = x[0]² + x[1]² = 4 + 9 = 13

# Compute gradients
z.backward()

# dz/dx = 2x
print(f"x: {x}")
print(f"Gradient: {x.grad}")  # [4.0, 6.0] = 2*[2, 3]
```

**Automatic differentiation** builds a **computational graph** as you perform operations. When you call `.backward()`, it traverses this graph in reverse to compute all gradients.

```python
# Every operation builds the graph
a = torch.tensor(2.0, requires_grad=True)
b = a * 3        # Graph: a → multiply → b
c = b + 5        # Graph: b → add → c
d = c ** 2       # Graph: c → power → d

d.backward()     # Traverse graph backward
print(f"Gradient of d w.r.t. a: {a.grad}")  # Chain rule: 2*c * 3 = 2*(a*3+5) * 3
```

## Neural Network Modules

PyTorch provides `nn.Module` as the base class for all neural networks. You define layers in `__init__` and the forward pass in `forward`.

```python
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Create model
model = SimpleNetwork(input_size=10, hidden_size=32, output_size=1)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

## Sequential Models

For simple architectures, `nn.Sequential` provides a convenient shortcut:

```python
# Same network using Sequential
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Make prediction
x = torch.randn(5, 10)  # 5 samples, 10 features
output = model(x)
print(f"Output shape: {output.shape}")  # (5, 1)
```

The **Linear layer** (`nn.Linear`) performs $y = xW^T + b$—exactly the weighted sum we discussed earlier.

## Loss Functions in PyTorch

PyTorch provides common **loss functions** ready to use:

```python
# Regression
mse_loss = nn.MSELoss()           # Mean Squared Error
mae_loss = nn.L1Loss()            # Mean Absolute Error
huber_loss = nn.HuberLoss()       # Huber Loss

# Classification
bce_loss = nn.BCELoss()                    # Binary Cross-Entropy (after sigmoid)
bce_logits = nn.BCEWithLogitsLoss()        # BCE with built-in sigmoid
ce_loss = nn.CrossEntropyLoss()            # Multi-class (includes softmax)

# Example
predictions = torch.tensor([0.8, 0.2, 0.9])
targets = torch.tensor([1.0, 0.0, 1.0])

loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")
```

## Optimizers: SGD and Adam

**Optimizers** update model weights based on gradients. PyTorch provides many optimizers:

```python
# SGD Optimizer - simple, reliable
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam Optimizer - adaptive learning rates, usually works well
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=10, gamma=0.1)
```

**SGD (Stochastic Gradient Descent)** is the classic optimizer. Add momentum for smoother updates.

**Adam** adapts the learning rate for each parameter. It's often the default choice for neural networks.

## The Training Loop

The **training loop** is where learning happens. It's the heart of neural network training:

```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()  # Set to training mode

    for epoch in range(epochs):
        total_loss = 0

        for batch_x, batch_y in train_loader:
            # 1. Zero gradients from previous step
            optimizer.zero_grad()

            # 2. Forward pass
            predictions = model(batch_x)

            # 3. Compute loss
            loss = criterion(predictions, batch_y)

            # 4. Backward pass (compute gradients)
            loss.backward()

            # 5. Update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model
```

The five essential steps:

1. **Zero gradients**: Clear gradients from the previous iteration
2. **Forward pass**: Compute predictions
3. **Compute loss**: Measure how wrong we are
4. **Backward pass**: Compute gradients via backpropagation
5. **Update weights**: Apply gradients using optimizer

#### Diagram: Training Loop Visualizer

<details markdown="1">
<summary>Training Loop Visualizer</summary>
Type: microsim

Bloom Taxonomy: Apply, Analyze

Learning Objective: Understand the five steps of the training loop and see how weights update over iterations

Canvas Layout (900x600):
- Left panel (450x600): Training loop steps with code
- Right panel (450x600): Loss curve and weight visualization

Left Panel - Step-by-Step:
- Five cards showing each training step
- Current step highlighted
- Code snippet for each step
- Arrow showing data/gradient flow

Steps Display:
1. "Zero Gradients" - optimizer.zero_grad()
2. "Forward Pass" - predictions = model(x)
3. "Compute Loss" - loss = criterion(pred, y)
4. "Backward Pass" - loss.backward()
5. "Update Weights" - optimizer.step()

Right Panel - Visualizations:
- Top: Live loss curve (updates each iteration)
- Bottom: Weight histogram or specific weight values

Animation:
- Watch data flow forward through network
- See loss computed at output
- Watch gradient flow backward
- See weights shift after update
- Loss decreases over iterations

Interactive Controls:
- Button: "Step" - advance one step
- Button: "Complete Iteration" - run all 5 steps
- Button: "Run Epoch" - run full epoch
- Slider: Learning rate
- Slider: Animation speed

Metrics Display:
- Current iteration number
- Current batch loss
- Running average loss
- Number of weight updates

Implementation: p5.js with synchronized animation
</details>

## Model Evaluation in PyTorch

Evaluating models requires disabling gradient computation and switching to eval mode:

```python
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

            # For classification
            predicted_class = (predictions > 0.5).float()
            correct += (predicted_class == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy

# Usage
test_loss, test_acc = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2%}")
```

## GPU Computing with CUDA

**GPU computing** accelerates training dramatically. PyTorch makes it easy with **CUDA** support:

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

# Move data to GPU
x = torch.randn(64, 10).to(device)
y = torch.randn(64, 1).to(device)

# Training works the same way
output = model(x)  # Computation happens on GPU
```

GPU speedups depend on:

- Network size (larger = more benefit)
- Batch size (larger = more benefit)
- Operation type (matrix operations benefit most)

## Saving and Loading Models

**Model saving** preserves your trained models for later use:

```python
# Save model weights only (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Save entire model (includes architecture)
torch.save(model, 'full_model.pth')

# Save checkpoint (weights, optimizer, epoch, loss)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

**Model loading** restores saved models:

```python
# Load weights into existing model
model = SimpleNetwork(10, 32, 1)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Load from checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

## Transfer Learning

**Transfer learning** uses a model trained on one task as the starting point for another task. This leverages knowledge learned from large datasets.

```python
import torchvision.models as models

# Load pre-trained model
resnet = models.resnet18(pretrained=True)

# Freeze early layers (don't update their weights)
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Only train the new layer
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
```

Transfer learning is powerful because:

- Pre-trained models learned general features from millions of images
- You only need a small dataset for your specific task
- Training is much faster

---

# Part 3: Best Practices

## Explainable AI and Model Interpretability

**Explainable AI (XAI)** and **model interpretability** help us understand *why* models make their predictions. This is crucial for trust, debugging, and ethics.

Methods for interpretability:

```python
# Feature importance (for tree-based models)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_

# For neural networks, use specialized libraries
# SHAP (SHapley Additive exPlanations)
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

**SHAP values** attribute each feature's contribution to a prediction, based on game theory. They show:

- Which features pushed the prediction higher/lower
- Feature importance across the dataset
- Interaction effects between features

## Model Documentation

**Model documentation** records everything needed to understand, reproduce, and maintain your model:

Essential documentation:

- **Model card**: Purpose, training data, performance, limitations
- **Data documentation**: Sources, preprocessing, quality issues
- **Code documentation**: Comments, docstrings, README
- **Experiment logs**: Hyperparameters, metrics, decisions

```markdown
# Model Card: House Price Predictor

## Model Details
- Architecture: 3-layer neural network [10, 64, 32, 1]
- Training data: 10,000 house sales from 2020-2023
- Validation R²: 0.87

## Intended Use
- Estimate house prices for real estate listings
- NOT for mortgage underwriting decisions

## Limitations
- Trained only on suburban properties
- May not generalize to urban/rural markets
- Does not account for market volatility

## Ethical Considerations
- Remove protected attributes (race, religion)
- Monitor for disparate impact
```

## Reproducibility

**Reproducibility** ensures others (including future you) can recreate your results exactly.

Key practices:

```python
# 1. Set random seeds everywhere
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. Record all hyperparameters
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'hidden_sizes': [64, 32],
    'optimizer': 'Adam',
    'seed': 42
}

# 3. Use version control for code (Git)
# 4. Version your data
# 5. Log all experiments
```

## Version Control with Git

**Version control** tracks changes to your code over time. **Git** is the industry standard:

```bash
# Initialize a repository
git init

# Add files to staging
git add model.py data_processing.py

# Commit changes
git commit -m "Add initial neural network model"

# Create a branch for experiments
git checkout -b experiment/larger-network

# Push to remote (GitHub, GitLab)
git push origin main
```

Git benefits:

- Track all changes with history
- Collaborate with teammates
- Revert to previous versions
- Branch for experiments without breaking main code

## Data Ethics

**Data ethics** ensures your work respects privacy, fairness, and societal impact:

Key principles:

| Principle | Description | Example |
|-----------|-------------|---------|
| Privacy | Protect personal information | Anonymize before training |
| Fairness | Avoid bias against groups | Test for disparate impact |
| Transparency | Explain how decisions are made | Provide model cards |
| Consent | Use data as authorized | Respect terms of service |
| Accountability | Take responsibility for outcomes | Monitor deployed models |

```python
# Check for protected attribute correlation
protected_attrs = ['race', 'gender', 'age']
for attr in protected_attrs:
    if attr in df.columns:
        print(f"Warning: {attr} in dataset - ensure it's not used improperly")

# Test for fairness
from fairlearn.metrics import MetricFrame
metric_frame = MetricFrame(
    metrics={'accuracy': accuracy_score},
    y_true=y_test,
    y_pred=predictions,
    sensitive_features=sensitive_test
)
print(metric_frame.by_group)
```

---

# Part 4: Capstone Projects

## The End-to-End Pipeline

A **capstone project** demonstrates everything you've learned by building a complete **end-to-end pipeline**:

```python
# Complete data science pipeline
class DataSciencePipeline:
    def __init__(self):
        self.scaler = None
        self.model = None

    def load_data(self, path):
        """1. Data Collection"""
        self.df = pd.read_csv(path)
        return self

    def clean_data(self):
        """2. Data Cleaning"""
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        return self

    def engineer_features(self):
        """3. Feature Engineering"""
        self.df['feature_ratio'] = self.df['a'] / self.df['b']
        return self

    def prepare_data(self):
        """4. Train/Test Split"""
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2)
        return self

    def train_model(self):
        """5. Model Training"""
        self.model = NeuralNetwork()
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate(self):
        """6. Model Evaluation"""
        predictions = self.model.predict(self.X_test)
        print(f"R² Score: {r2_score(self.y_test, predictions):.4f}")
        return self

    def save(self, path):
        """7. Model Deployment"""
        torch.save(self.model.state_dict(), path)
        return self

# Run the complete pipeline
pipeline = DataSciencePipeline()
pipeline.load_data('data.csv') \
        .clean_data() \
        .engineer_features() \
        .prepare_data() \
        .train_model() \
        .evaluate() \
        .save('model.pth')
```

## Model Deployment

**Model deployment** makes your trained model available for real-world use:

Deployment options:

| Option | Use Case | Complexity |
|--------|----------|------------|
| Flask/FastAPI | Simple web API | Low |
| Docker | Containerized deployment | Medium |
| Cloud (AWS, GCP, Azure) | Production scale | Medium-High |
| Edge devices | Mobile, IoT | High |

```python
# Simple Flask API for model serving
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Load model once at startup
model = torch.load('model.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = torch.tensor(data['features']).float()

    with torch.no_grad():
        prediction = model(features)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Results Communication

**Results communication** translates technical findings into insights that stakeholders can act on:

```python
# Create executive summary visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=['Model Performance', 'Feature Importance',
                                   'Predictions vs Actual', 'Error Distribution'])

# Add visualizations that tell the story
# ... (detailed plotting code)

fig.update_layout(title='House Price Prediction Model - Executive Summary')
fig.write_html('results_dashboard.html')
```

Key communication principles:

- Lead with the **business impact**, not technical details
- Use **visualizations** over tables of numbers
- Quantify **uncertainty** (confidence intervals, error ranges)
- Provide **actionable recommendations**
- Be honest about **limitations**

## Data-Driven Decisions

The ultimate goal of data science is **data-driven decisions**—using evidence to guide action:

```python
# From prediction to decision
def recommend_action(prediction, threshold=0.7):
    """
    Convert model output to business recommendation
    """
    if prediction['churn_probability'] > threshold:
        return {
            'action': 'HIGH PRIORITY: Retention intervention needed',
            'confidence': prediction['churn_probability'],
            'suggested_offers': get_retention_offers(prediction)
        }
    else:
        return {
            'action': 'Standard engagement',
            'confidence': 1 - prediction['churn_probability'],
            'suggested_offers': []
        }

# Example output
recommendation = recommend_action({'churn_probability': 0.85})
print(recommendation)
# {'action': 'HIGH PRIORITY: Retention intervention needed',
#  'confidence': 0.85,
#  'suggested_offers': ['20% discount', 'Free month', 'Upgrade offer']}
```

---

## Complete Example: Building a Neural Network in PyTorch

Here's a complete, working example that ties everything together:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Generate synthetic data
n_samples = 1000
X = np.random.randn(n_samples, 10)
y = (3*X[:, 0] - 2*X[:, 1] + X[:, 2]**2 + np.random.randn(n_samples)*0.5).reshape(-1, 1)

# 2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 3. Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# 4. Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 6. Create model, loss, optimizer
model = NeuralNetwork(input_size=10, hidden_sizes=[64, 32], output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
epochs = 100
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 8. Evaluate
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t)
    test_loss = criterion(test_pred, y_test_t)
    print(f"\nTest Loss: {test_loss.item():.4f}")

# 9. Visualize training
fig = px.line(y=train_losses, title='Training Loss Over Time',
              labels={'x': 'Epoch', 'y': 'Loss'})
fig.show()

# 10. Save model
torch.save(model.state_dict(), 'trained_model.pth')
print("Model saved!")
```

---

## Summary: Your Complete Data Science Toolkit

Congratulations! You've now learned the complete data science toolkit:

**Neural Network Fundamentals:**
- Artificial neurons, activation functions (ReLU, Sigmoid)
- Network architecture (input, hidden, output layers)
- Weights, biases, forward propagation, backpropagation
- Training with epochs, batches, and gradient descent

**PyTorch Skills:**
- Tensors and tensor operations
- Autograd for automatic differentiation
- Building models with nn.Module and Sequential
- Training loops with optimizers (SGD, Adam)
- GPU acceleration with CUDA
- Saving and loading models

**Professional Best Practices:**
- Model interpretability and SHAP values
- Documentation and reproducibility
- Version control with Git
- Data ethics and fairness

**Project Skills:**
- End-to-end pipelines
- Model deployment
- Results communication
- Data-driven decision making

You now have every tool you need to tackle real-world data science problems.

---

## Key Takeaways

- Neural networks are universal function approximators built from simple neurons
- Activation functions add non-linearity; ReLU is the default choice for hidden layers
- Backpropagation computes gradients; PyTorch handles this automatically
- The training loop: zero gradients → forward → loss → backward → update
- PyTorch tensors are like NumPy arrays with GPU support and autograd
- Always use train/eval modes and disable gradients during evaluation
- Document everything: code, data, decisions, limitations
- Set random seeds for reproducibility
- Consider ethics: privacy, fairness, transparency
- Deploy models to create real-world impact

---

## Your Capstone Project Awaits

You've completed an incredible journey from Python basics to building neural networks. You've learned to:

- Clean and explore data
- Create stunning visualizations
- Apply statistical analysis
- Build regression and classification models
- Evaluate and validate models
- Construct and train neural networks
- Follow professional best practices

Now it's time to apply everything you've learned.

!!! question "What Will Your Capstone Project Be?"
    Think about a problem you care about solving. Consider:

    - **Personal interests**: Sports analytics? Music recommendations? Climate data?
    - **Social impact**: Healthcare predictions? Educational outcomes? Environmental monitoring?
    - **Career goals**: Financial analysis? Customer behavior? Manufacturing optimization?
    - **Local community**: Traffic patterns? Local business trends? Public health?

    Your capstone project is your chance to demonstrate your new superpowers. You'll build an end-to-end pipeline, from raw data to deployed model, solving a problem that matters to you.

    **So, what will YOU build?**