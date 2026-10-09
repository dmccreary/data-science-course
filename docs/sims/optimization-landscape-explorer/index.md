---
title: Optimization Landscape Explorer
description: Predict where gradient descent will stop on a cost curve with several valleys, run it from starting points you choose, and compare how often plain descent and momentum reach the global minimum.
image: /sims/optimization-landscape-explorer/optimization-landscape-explorer.png
og:image: /sims/optimization-landscape-explorer/optimization-landscape-explorer.png
twitter:image: /sims/optimization-landscape-explorer/optimization-landscape-explorer.png
social:
   cards: false
quality_score: 0
---

# Optimization Landscape Explorer

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Optimization Landscape Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The blue curve is a cost function $J(x)$ of a single parameter, and the red ball is the optimizer. Each step moves the ball downhill by the learning rate times the slope, $x \leftarrow x - \eta J'(x)$, and it stops where the slope is nearly zero. Triangles mark every minimum with its cost: green for the **global minimum** (the lowest point of the whole curve) and orange for each **local minimum** (lower than its neighbors, but not the lowest).

Four landscapes are available:

- **Convex bowl**, $J = 0.5x^2$. There is one valley, so every start reaches the global minimum.
- **Two valleys**, $J = x^4 - 3x^2 + 0.5x$, the function plotted in the chapter. The global minimum is at $x \approx -1.26$ ($J \approx -2.87$) and the local minimum at $x \approx 1.18$ ($J \approx -1.65$).
- **Many valleys**, $J = 0.2x^2 - 0.7\cos(3(x - 0.4))$. It has five minima; the global one is at $x \approx 0.38$.
- **Plateau**, $J = x^4/4 - 2x^3/3$. It has one minimum at $x = 2$ and a flat spot at $x = 0$, where the slope is zero but the curve goes on down afterwards. On a curve this plays the part that a saddle point plays on a surface.

With **Add momentum** checked the ball keeps 90% of its previous move: $v \leftarrow 0.9v - \eta J'(x)$, then $x \leftarrow x + v$. That lets it coast through shallow dips and across flat ground. It is not a guarantee: many starts still end in a local minimum.

The two strips under the plot record where each finished run started, green if it reached the global minimum and orange if it did not, with one strip for plain descent and one for momentum. The result panel keeps the count.

Not included: the learning rate and noise sliders from the original specification. Each landscape uses one fixed learning rate, shown under the title.

## How to Use

1. Read the prompt, predict where the ball will stop, then press **Start**. Use **Step** instead to advance one update at a time.
2. Click the curve to choose a new starting point and run again. Repeat until the *plain* strip shows which starts reach the global minimum.
3. Check **Add momentum** and press **Start** to repeat the same start with momentum. Compare the two strips.
4. Choose **Many valleys** and look for starts that momentum rescues and starts that it does not.
5. Choose **Plateau**, start on the left slope, and compare plain descent with momentum. **Reset** clears the strips.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/optimization-landscape-explorer/main.html"
        height="582"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Gradient descent, the learning rate, and reading the graph of a function.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to distinguish local minima from the global minimum and compare how the starting point and momentum affect which minimum gradient descent finds.

### Activities

1. **Predict and run** (4 min): On the two valleys landscape students predict the stopping point for three starts of their choice, run each one, and record whether the prediction was right.
2. **Map the starts** (6 min): On the many valleys landscape students run at least eight starts spread across the curve with plain descent, repeat the same starts with momentum, and describe which starts changed outcome.
3. **Flat is not a minimum** (4 min): On the plateau students run plain descent from the left slope and read the slope and the cost above the global minimum where it stalls, then repeat with momentum and explain the difference.

### Assessment
Shown a sketch of a cost curve with three valleys and a marked starting point, students say where plain gradient descent will stop, whether that point is a local or the global minimum, and name one strategy from the chapter that could change the outcome.

## References

1. Wikipedia. [Maxima and minima](https://en.wikipedia.org/wiki/Maxima_and_minima) (local and global minima).
2. Wikipedia. [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (section on momentum).
3. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly. Chapter 11, Training Deep Neural Networks (momentum optimization).
