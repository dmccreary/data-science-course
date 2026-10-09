---
title: Activation Function Explorer
description: Choose an activation function, move the input along its curve, and read the output and gradient to see where gradients vanish or die and why a network needs a non-linear activation.
image: /sims/activation-function-explorer/activation-function-explorer.png
og:image: /sims/activation-function-explorer/activation-function-explorer.png
twitter:image: /sims/activation-function-explorer/activation-function-explorer.png
social:
   cards: false
quality_score: 0
---

# Activation Function Explorer

<iframe src="main.html" height="592" width="100%" scrolling="no"></iframe>

[Run the Activation Function Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The graph shows one activation function $f(x)$ as a solid curve and its derivative $f'(x)$ as a dashed curve for inputs from −5 to 5. Both axes are fixed, so all six functions are drawn to the same scale. A dot marks the output at the chosen input and a ring marks the gradient there.

- **Sigmoid:** $f(x) = 1/(1 + e^{-x})$, $f'(x) = f(x)(1 - f(x))$, range $(0, 1)$
- **Tanh:** $f(x) = \tanh(x)$, $f'(x) = 1 - f(x)^2$, range $(-1, 1)$
- **ReLU:** $f(x) = \max(0, x)$, $f'(x) = 1$ for $x > 0$ and $0$ otherwise, range $[0, \infty)$
- **Leaky ReLU:** $f(x) = x$ for $x > 0$ and $0.1x$ otherwise, so $f'(x)$ is $1$ or $0.1$
- **Step:** $1$ for $x > 0$ and $0$ otherwise, the original perceptron; its derivative is 0 wherever it exists
- **Linear:** $f(x) = x$, $f'(x) = 1$

Shading marks the inputs where learning stalls: orange where the gradient is below 0.05 (a vanishing gradient) and red where it is exactly 0. The yellow panel gives $f(x)$ and $f'(x)$ at the chosen input, says what that gradient means, and multiplies it through 10 layers the way backpropagation does. For the sigmoid at $x = 0$ this is $0.25^{10} \approx 0.000001$, the number quoted in the chapter.

The small plot shows the output of a network with two hidden neurons, $y = f(x + 2) - f(2x - 2)$. With any non-linear activation the output bends. With the linear activation it collapses to the single straight line $y = 4 - x$, which is why stacking linear layers adds nothing.

Not included: softmax, which acts on a whole vector of outputs and cannot be drawn as one curve. The Leaky ReLU slope is 0.1 so that the leak can be seen (PyTorch's default is 0.01). At $x = 0$ the slope of ReLU is undefined, and the sim reports 0 as PyTorch does.

## How to Use

1. Read the default view: the sigmoid, its dashed derivative, and the orange bands where the gradient is below 0.05.
2. Drag the **Input x** slider, or click the graph, to move into an orange band. Read $f'(x)$ and the 10-layer product in the yellow panel.
3. Choose **ReLU** in the first menu. Move x below 0 and then above 0 and compare the gradient.
4. Set the second menu to **vs Leaky ReLU** to draw both curves and read both sets of values at the same input.
5. Choose **Linear** and read the two-neuron network panel, then switch back to another function and compare the shape.
6. Uncheck **Show derivative** to see the function alone.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/activation-function-explorer/main.html"
        height="592"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The artificial neuron (a weighted sum plus a bias), reading the graph of a function, and the derivative as the slope of a curve.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to determine the output and gradient of an activation function at a given input, identify the inputs where its gradient vanishes or is zero, and explain why a network needs a non-linear activation.

### Activities

1. **Predict** (3 min): Before moving the slider, students sketch where they expect the sigmoid's gradient to be largest and where it is close to zero, then check their sketch against the dashed curve and the orange bands.
2. **Measure** (6 min): Students record $f(x)$, $f'(x)$, and the 10-layer product at x = −4, 0, and 4 for Sigmoid, Tanh, ReLU, and Leaky ReLU, and mark each entry as healthy, vanishing, or zero.
3. **Explain** (5 min): Students select Linear, read the two-neuron network panel, and write two sentences on why a network whose activations are all linear can do no more than a single linear layer.

### Assessment
For the input x = −3, students state without the sim which of Sigmoid, ReLU, and Leaky ReLU passes back the largest gradient and which passes back none, then explain why ReLU is still the usual choice for deep hidden layers.

## References

1. Wikipedia. [Activation function](https://en.wikipedia.org/wiki/Activation_function).
2. Wikipedia. [Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).
3. PyTorch documentation. [torch.nn.LeakyReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html).
