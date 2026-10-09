---
title: Gradient Descent Visualizer
description: Take gradient descent steps on a contour map of a cost function, read the gradient and the next position at each step, and change the learning rate to see slow progress, zigzag, and divergence.
image: /sims/gradient-descent-visualizer/gradient-descent-visualizer.png
og:image: /sims/gradient-descent-visualizer/gradient-descent-visualizer.png
twitter:image: /sims/gradient-descent-visualizer/gradient-descent-visualizer.png
social:
   cards: false
quality_score: 0
---

# Gradient Descent Visualizer

<iframe src="main.html" height="547" width="100%" scrolling="no"></iframe>

[Run the Gradient Descent Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The colored map is a cost function $J(w_1, w_2)$ seen from above: light bands are low cost and dark bands are high cost. The red dot is the current position, the red trail is every step taken so far, and the black arrow is the step that comes next. Each step applies the update rule from the chapter,

$$w_{new} = w_{old} - \eta \, \nabla J(w_{old})$$

with the exact gradient of the selected function. The panel beside the map lists the position, the cost, the gradient, its length, and the next step, so any step can be checked by hand. The chart plots the cost after every step.

Three cost surfaces are available:

- **Simple bowl**, $J = w_1^2 + w_2^2$. Any learning rate below 1 converges. At $\eta = 0.5$ one step lands exactly on the minimum, and above 1 every step overshoots farther than the last.
- **Elongated valley**, $J = 0.2w_1^2 + 2w_2^2$. The surface curves ten times more sharply across the valley than along it, so a rate that only crawls along the floor already zigzags across it. Above $\eta = 0.5$ it diverges.
- **Two valleys**, $J = (w_1^2 - 1)^2 + 0.3w_1 + w_2^2$. There is a global minimum near $w_1 = -1.04$ and a local minimum near $w_1 = 0.96$. Where the path ends depends on where it starts.

A run stops when the gradient is nearly zero (length below 0.001), when the cost passes 10,000, or after 300 steps.

Not included: the 3D surface view and the two display checkboxes from the original specification. The path and the next-step arrow are always shown.

## How to Use

1. Press **Step** once and check the new position by hand. For the bowl at $w = (-2, 1.5)$ the gradient is $(-4, 3)$, so with $\eta = 0.1$ the step is $(0.4, -0.3)$.
2. Press **Start** to let the steps run. Watch them shrink as the gradient shrinks near the minimum.
3. Press **Reset**, move the **Learning rate** slider to 0.9, and run again. Then try 1.05 and read the status message and the cost chart.
4. Choose **Elongated valley** and compare learning rates 0.05, 0.3, and 0.45. Then find a rate that diverges.
5. Choose **Two valleys** and click the map to start from several places. Note which starts end in the local minimum.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/gradient-descent-visualizer/main.html"
        height="547"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Functions of two variables, slope as a rate of change, and the cost function from earlier in the chapter.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to apply the gradient descent update rule to calculate the next parameter values and predict how the learning rate changes the path to the minimum.

### Activities

1. **Compute a step** (4 min): On the simple bowl students compute the first two steps by hand from the listed gradient and learning rate, then press Step twice to check both positions and costs.
2. **Tune the learning rate** (6 min): On the elongated valley students record the number of steps to converge at learning rates 0.1, 0.2, and 0.45, try 0.05 to see a run that is too slow to finish, and find the smallest rate on the slider that diverges.
3. **Starting point** (4 min): On the two valleys surface students click five different starting points, record which minimum each run reaches, and sketch the boundary between the two sets of starts.

### Assessment
Given $J = w_1^2 + w_2^2$, a starting point, and a learning rate, students compute two gradient descent steps by hand and explain what would happen if the learning rate were 1.2.

## References

1. Wikipedia. [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).
2. Wikipedia. [Learning rate](https://en.wikipedia.org/wiki/Learning_rate).
3. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly. Chapter 4, Training Models.
