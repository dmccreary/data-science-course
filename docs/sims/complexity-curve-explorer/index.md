---
title: Complexity Curve Explorer
description: Set the degree of a polynomial fit to noisy data, compare its training and test error, and judge where the model moves from underfitting through a sweet spot to overfitting.
image: /sims/complexity-curve-explorer/complexity-curve-explorer.png
og:image: /sims/complexity-curve-explorer/complexity-curve-explorer.png
twitter:image: /sims/complexity-curve-explorer/complexity-curve-explorer.png
social:
   cards: false
quality_score: 0
---

# Complexity Curve Explorer

<iframe src="main.html" height="617" width="100%" scrolling="no"></iframe>

[Run the Complexity Curve Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Thirty points follow a cubic curve with random noise added. Twenty are **training points** (blue circles) and ten are **test points** (orange squares). The slider sets the degree of a polynomial that is fit by least squares to the training points only.

The lower chart shows the mean squared error (MSE) for every degree from 1 to 15:

- **Training MSE** (blue) is measured on the 20 points used for the fit. It never goes up as the degree rises.
- **Test MSE** (orange) is measured on the 10 points the fit never saw. It falls while the model is learning the pattern and rises again when the extra flexibility is spent on noise. Orange arrows at the top mark values that are off the chart.

The shading is computed from the test error. A degree is in the **sweet spot** when its test MSE is within 25% of the lowest test MSE. Degrees simpler than the first sweet-spot degree are **underfitting**, and the remaining degrees are **overfitting**. The panel beside the plot reports both errors, their gap, and a verdict for the chosen degree.

Every curve and every error value comes from a real least-squares fit computed in the browser. At high degrees the curve swings far outside the plot between training points. That is what a polynomial of degree 12 to 15 does with 20 points, not a drawing error.

With only ten test points the picture changes from sample to sample. **New Data** usually gives a sweet spot that starts at degree 3, the true shape, but now and then a much higher degree, or even a straight line, tests best by luck. That weakness of a single small test set is what cross-validation, the next topic in the chapter, is designed to reduce.

## How to Use

1. Start at degree 1. The straight line misses the bends in the data, and both errors are high.
2. Drag the **Polynomial degree** slider up one step at a time. Watch the curve in the top plot and the two large dots on the chart.
3. Before going further, decide which degree you think is best and why. Then press **Find Best Degree** to jump to the degree with the lowest test MSE.
4. Keep increasing the degree. Note what happens to the training MSE, the test MSE, and the gap between them.
5. Uncheck **Show test error**. With only the training error visible, which degree looks best? Check the box again to see why that choice fails.
6. Press **New Data** for a new random sample and repeat. Compare the best degree across several samples.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/complexity-curve-explorer/main.html"
        height="617"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Mean squared error, the train-test split, and the idea that a polynomial of higher degree can bend more.

### Bloom's Taxonomy Level
Evaluate (L5)

### Learning Objective
Students will be able to judge whether a model is underfitting, well fit, or overfitting by comparing its training and test error, and justify a choice of model complexity.

### Activities

1. **Predict** (3 min): With **Show test error** unchecked, students step from degree 1 to degree 15 and choose the degree they would use from the training error and the look of the curve. They write down their choice and a reason.
2. **Check** (5 min): Students turn the test error back on, record the training MSE, test MSE, and gap for degrees 1, 3, 6, 10, and 15, and compare their choice with **Find Best Degree**.
3. **Justify** (5 min): Students press **New Data** four times, recording the degree with the lowest test error each time. They argue which degree they would use for this kind of data and explain why the answers are not always the same.

### Assessment
Given a table of training and test MSE for five models of increasing complexity, students label each model as underfitting, a good fit, or overfitting, choose one model, and justify the choice by referring to both errors and to model simplicity.

## References

1. scikit-learn examples. [Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html).
2. Wikipedia. [Overfitting](https://en.wikipedia.org/wiki/Overfitting).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 2, Statistical Learning.
