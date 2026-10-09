---
title: Error Types Visualizer
description: Fit a polynomial of adjustable degree to training points, compare its training error with its error on unseen test points, judge whether it is underfitting or overfitting, and then reveal the diagnosis.
image: /sims/error-types-visualizer/error-types-visualizer.png
og:image: /sims/error-types-visualizer/error-types-visualizer.png
twitter:image: /sims/error-types-visualizer/error-types-visualizer.png
social:
   cards: false
quality_score: 0
---

# Error Types Visualizer

<iframe src="main.html" height="652" width="100%" scrolling="no"></iframe>

[Run the Error Types Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The same fitted curve is drawn on two scatter plots. On the left are the **training points** the polynomial was fit to. On the right are 40 **test points** the fit never saw. Thin vertical lines are the residuals, and each panel reports its mean squared error, $\text{MSE} = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$.

The data come from one cubic pattern (dashed gray) plus random noise. The bars compare the two errors at the current degree and mark the noise variance $\sigma^2$, the error that even a perfect model would make on average on new data. As the degree rises the training error can only fall, because a more flexible curve can always get closer to its own points. The test error falls at first and then climbs when the curve starts to follow the noise.

The diagnosis is hidden until you ask for it. Checking the box reveals the training and test MSE for every degree from 1 to 12 and a verdict computed from them: **good fit** when the test MSE is within 25% of the lowest test MSE any degree reaches, **underfitting** below that degree, and **overfitting** above it.

The fits are ordinary least squares. Test errors above 1000 are shown as "over 1000": a degree 12 curve fit to 15 points can miss a test point by a very wide margin.

## How to Use

1. Start at degree 1. Compare the two MSE values and the bars, and decide: underfitting, good fit, or overfitting?
2. Raise the **Polynomial degree** one step at a time. Watch the training MSE fall while the test MSE falls and then rises.
3. Pick the degree you judge best, then check **Show diagnosis and error curves** to see the verdict and where the test error is lowest.
4. Set the degree to 10 and move **Training set size** from 15 to 60. More training data pulls the wild curve back toward the pattern.
5. Change the **Noise level** and press **New Data** to see how much the best degree and the two errors depend on the particular sample.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/error-types-visualizer/main.html"
        height="652"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Scatter plots, residuals, mean squared error, polynomial regression, and the train-test split.

### Bloom's Taxonomy Level
Evaluate (L5)

### Learning Objective
Students will be able to judge whether a model is underfitting, well fit, or overfitting by comparing its training error with its test error.

### Activities

1. **Judge, then check** (5 min): With the diagnosis hidden, students record the training and test MSE for degrees 1, 3, 6, and 10, label each as underfitting, good fit, or overfitting, then reveal the diagnosis and compare.
2. **Find the pattern** (5 min): With the curves revealed, students describe the shape of each curve in one sentence and explain why the training curve never rises.
3. **More data, more noise** (5 min): Students fix the degree at 10 and record the test MSE at training sizes 15, 30, and 60, then repeat at a higher noise level and state what each change did to the gap between the two errors.

### Assessment
Given a table of training and test MSE for five models of increasing complexity, students label each model as underfitting, good fit, or overfitting, choose the model to use, and justify the choice with the gap between the two errors.

## References

1. scikit-learn documentation. [Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html).
2. Wikipedia. [Overfitting](https://en.wikipedia.org/wiki/Overfitting).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 2, Statistical Learning (assessing model accuracy).
