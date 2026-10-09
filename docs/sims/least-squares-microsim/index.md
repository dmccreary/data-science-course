---
title: Least Squares MicroSim
description: Move a line with slope and intercept sliders and watch each residual, its square, and the sum of squared errors change, then reveal the least-squares line that makes the total area of the squares as small as possible.
image: /sims/least-squares-microsim/least-squares-microsim.png
og:image: /sims/least-squares-microsim/least-squares-microsim.png
twitter:image: /sims/least-squares-microsim/least-squares-microsim.png
social:
   cards: false
quality_score: 0
---

# Least Squares MicroSim

<iframe src="main.html" height="657" width="100%" scrolling="no"></iframe>

[Run the Least Squares MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The plot shows hours studied and exam scores for ten students, with a blue line that you control. A red segment joins each point to the line (its residual), and a square is drawn on each segment. The side of a square is the size of the residual, so its area is the squared error for that point.

The table lists every point with its predicted value $\hat{y}$, its residual $y - \hat{y}$, and the squared residual. The last column adds up to the **sum of squared errors**:

$$\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

The meter under the plot shows SSE as a number and as a bar that turns from red to green as the line improves. **Show Best Fit** moves the sliders to the ordinary least squares solution, $\hat{y} = 47.5 + 5.5x$, where SSE reaches its minimum of 161.50. No other straight line gives a smaller total area. The dashed green line stays on the plot afterward so you can compare your own line with it.

The data are the same ten students as in the Regression Line Anatomy MicroSim. Not included: dragging the line with the mouse and an animated move to the best fit. The sliders and the button do both jobs.

## How to Use

1. Look at the starting line (slope 2.0, intercept 60.0). Find the largest square and the row of the table it belongs to.
2. Move the **Slope** slider and watch the squares grow and shrink.
3. Move the **Intercept** slider to raise or lower the whole line.
4. Make SSE as small as you can. The black marker on the meter remembers the lowest value you have reached.
5. Press **Show Best Fit** to see the least-squares line and the minimum SSE, then move the sliders away and read how far above the minimum you are.
6. Uncheck **Show squares** to see only the residuals. **Reset** returns to the starting line and hides the best fit.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/least-squares-microsim/main.html"
        height="657"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The regression equation $\hat{y} = \beta_0 + \beta_1 x$ and residuals as actual minus predicted.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how the least squares method chooses the regression line by minimizing the sum of squared errors.

### Activities

1. **Predict** (3 min): Before moving anything, students say whether the starting line is too steep or too flat and which students have the largest squared errors, then check the table.
2. **Minimize** (6 min): Students adjust both sliders to get SSE as low as they can and record their best slope, intercept, and SSE. They then press **Show Best Fit** and compare.
3. **Explain** (4 min): Students find a line for which every residual is positive and explain why its SSE is large. They explain in one sentence why the squares, not the residuals themselves, are added.

### Assessment
Given five points and two candidate lines, students compute the residuals and SSE for each line, decide which line fits better, and explain why the sum of the plain residuals would be a poor measure of fit.

## References

1. Wikipedia. [Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares).
2. Wikipedia. [Residual sum of squares](https://en.wikipedia.org/wiki/Residual_sum_of_squares).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 3, Linear Regression.
