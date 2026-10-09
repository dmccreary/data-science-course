---
title: Training Loop Visualizer
description: Step through the five lines of a PyTorch training loop on a one-neuron model and watch the weight, bias, gradients, and loss change after each line.
image: /sims/training-loop-visualizer/training-loop-visualizer.png
og:image: /sims/training-loop-visualizer/training-loop-visualizer.png
twitter:image: /sims/training-loop-visualizer/training-loop-visualizer.png
social:
   cards: false
quality_score: 0
---

# Training Loop Visualizer

<iframe src="main.html" height="622" width="100%" scrolling="no"></iframe>

[Run the Training Loop Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The model is a single neuron, `nn.Linear(1, 1)`, which predicts $wx + b$. It is trained on 8 points that lie near the line $y = 2x + 1$, all 8 in one batch, with `nn.MSELoss` and `optim.SGD`. Every number on screen is computed the way PyTorch computes it:

1. **Zero gradients**, `optimizer.zero_grad()`, clears the stored gradients. `backward()` adds each new gradient to whatever is already in `.grad`, so without this line the gradients of earlier iterations pile up.
2. **Forward pass**, `pred = model(x)`, computes $wx + b$ for every point.
3. **Compute loss**, `loss = criterion(pred, y)`, averages the squared errors $(\text{pred} - y)^2$.
4. **Backward pass**, `loss.backward()`, stores the slopes of the loss: the mean of $2(\text{pred} - y)x$ in `w.grad` and the mean of $2(\text{pred} - y)$ in `b.grad`. The weight and bias do not change.
5. **Update weights**, `optimizer.step()`, replaces `w` with `w - lr * w.grad` and `b` with `b - lr * b.grad`.

The panels show the line that just ran, an explanation with the actual numbers, the batch as a table, the model state, the data with the model's line, and the loss in every iteration. The dashed line on the loss plot is the lowest loss any straight line can reach on these points (the least-squares fit, 0.0706). The loss curve can approach it but cannot go below it.

After `zero_grad()` the sim shows `None` for the gradients, which is what PyTorch 2 stores (earlier versions stored zeros). The chapter's loop takes mini-batches from a data loader. Here one batch holds all the data, so one iteration is also one epoch.

Not included: animated data flow, mini-batches, a running-average loss, and an animation speed control. Nothing moves until a button is pressed.

## How to Use

1. Press **Step** five times. After each press read which line ran, the explanation, and the highlighted rows of the Model state panel.
2. Check the result of step 5 by hand: new w = old w − lr × w.grad.
3. Press **Run 10 Iterations** twice. The model line turns to fit the points and the loss curve falls toward the dashed line.
4. Press **Reset**, set the **Learning rate** to 0.01, and run 20 iterations. Repeat with 0.50, 0.70, and 0.80 and compare the loss curves. At 0.80 keep running until the sim stops.
5. Press **Reset**, set the learning rate back to 0.10, check **Skip zero_grad()**, and run 30 iterations. Then step through one iteration and read how w.grad is built in step 4.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/training-loop-visualizer/main.html"
        height="622"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Simple linear regression, mean squared error, and gradient descent as repeated steps downhill on the loss.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to trace the five steps of a PyTorch training loop, state what each step changes, and analyze how the learning rate and a missing zero_grad() call affect the loss curve.

### Activities

1. **Trace** (5 min): Students step through the first iteration and fill in a table of w, b, w.grad, b.grad, and loss after each of the five lines, marking the values each line changed.
2. **Compare** (7 min): Students run 20 iterations at learning rates 0.01, 0.10, 0.50, and 0.70, pressing Reset between runs. They record the last loss of each run and describe the shape of each loss curve.
3. **Diagnose** (5 min): Students check Skip zero_grad(), run 30 iterations, and use the result lines of step 4 to explain why the loss rises and falls instead of settling.

### Assessment
Students are shown a training loop whose loss swings up and down and never settles although the learning rate is small. They name the missing line, explain what PyTorch does with gradients when it is missing, and write the five steps in a correct order.

## References

1. PyTorch Tutorials. [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
2. PyTorch documentation. [torch.optim.Optimizer.zero_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html).
3. Wikipedia. [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).
