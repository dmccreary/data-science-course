---
title: Polynomial Degree Explorer
description: Set the degree of a polynomial fit for five data shapes and several noise levels, watch the fitted curve and its uncertainty band, and judge which degree gives the lowest expected test error by reading how the error splits into noise, bias, and variance.
image: /sims/polynomial-degree-explorer/polynomial-degree-explorer.png
og:image: /sims/polynomial-degree-explorer/polynomial-degree-explorer.png
twitter:image: /sims/polynomial-degree-explorer/polynomial-degree-explorer.png
social:
   cards: false
quality_score: 0
---

# Polynomial Degree Explorer

<iframe src="main.html" height="587" width="100%" scrolling="no"></iframe>

[Run the Polynomial Degree Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Twenty training points (blue) lie at evenly spaced $x$ values from 0 to 10, with $y$ equal to a true curve (dashed) plus random noise. A least-squares polynomial of the chosen degree is fit to them. Twenty test points (orange) come from the same curve and are not used in the fit.

The gray band is the fitted curve plus and minus two standard deviations of the fit. It shows how far the curve would move if the noise were drawn again, which is what **New Data** does. A stiff curve has a narrow band even when it misses the true curve. A flexible curve has a wide band, widest near the ends.

The panel reports $R^2$ on the training and test points and then splits the expected test error into three parts:

$$\text{expected test MSE} = \sigma^2 + \text{bias}^2 + \text{variance}$$

Noise $\sigma^2$ is fixed by the slider. Bias$^2$ is how far the average fitted curve stays from the true curve, and it falls as the degree rises. Variance is how much the fitted curve moves from sample to sample, and it rises with the degree. These are computed exactly from the model, not estimated from one sample, so they do not jump when **New Data** is pressed. The black mark on the bar is the lowest total that any degree from 1 to 15 reaches. A degree within 5% of that mark is called a good trade-off.

Not included from the original design: residual lines, a display of the individual coefficients, and a checkbox for the train and test split (the test points are always shown).

## How to Use

1. Start with the cubic data at degree 1. The line misses the dashed curve, and the bar is mostly blue (bias).
2. Drag **Polynomial degree** upward one step at a time. Watch the blue part shrink and the orange part (variance) grow, and stop where the bar reaches the black mark.
3. Go on to degree 12 and then 15. Press **New Data** a few times and watch how far the curve moves inside its gray band.
4. Change **Noise level (SD)**. Find the best degree at noise 0, 2, and 6.
5. Choose other shapes from the menu. Before moving the slider, predict the best degree for each one, then check.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/polynomial-degree-explorer/main.html"
        height="587"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Polynomial regression, R-squared, mean squared error, and the idea of training and test data.

### Bloom's Taxonomy Level
Evaluate (L5)

### Learning Objective
Students will be able to judge which polynomial degree gives the best trade-off between bias and variance for a given data shape and noise level, and justify the choice from the parts of the expected test error.

### Activities

1. **Find the trade-off** (5 min): On the cubic data with noise 2, students record bias², variance, and expected test MSE for degrees 1, 2, 3, 6, 10, and 15, and mark the degree where the total is lowest.
2. **Change the noise** (5 min): Students repeat the search for the sine wave at noise 0.5, 2, and 4 and describe how the best degree moves as the noise grows. They then set the noise to 0 and explain why no degree overfits.
3. **Predict, then check** (6 min): For the linear, quadratic, and step shapes, students write down the degree they expect to be best and one sentence of reasoning before touching the slider, then compare with the verdict panel. They explain why no degree does well on the step.

### Assessment
Students are shown two fitted curves for the same data, one of degree 2 with a narrow band that misses the pattern and one of degree 14 with a wide band, and explain which has high bias, which has high variance, and what would happen to each if new data were collected.

## References

1. Wikipedia. [Bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).
2. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 2, Statistical Learning (the bias-variance trade-off).
3. scikit-learn documentation. [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html).
