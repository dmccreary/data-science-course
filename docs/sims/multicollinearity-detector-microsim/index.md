---
title: Multicollinearity Detector MicroSim
description: Raise the correlation between square feet and rooms, read the correlation matrix and each feature's variance inflation factor, and watch the coefficient confidence intervals widen while R-squared barely moves.
image: /sims/multicollinearity-detector-microsim/multicollinearity-detector-microsim.png
og:image: /sims/multicollinearity-detector-microsim/multicollinearity-detector-microsim.png
twitter:image: /sims/multicollinearity-detector-microsim/multicollinearity-detector-microsim.png
social:
   cards: false
quality_score: 0
---

# Multicollinearity Detector MicroSim

<iframe src="main.html" height="632" width="100%" scrolling="no"></iframe>

[Run the Multicollinearity Detector MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Sixty simulated houses have three features, square feet, rooms, and age, and a price that depends on all three. The slider sets how closely rooms follows square feet. Age is unrelated to both.

The left panel is a scatter plot matrix. The cells below the diagonal plot each pair of features. The cells above it give the correlation $r$ of the same pair, shaded by strength and flagged when it is beyond ±0.7.

The upper right panel shows each feature's **variance inflation factor**, $\text{VIF}_j = 1 / (1 - R_j^2)$, where $R_j^2$ comes from regressing feature $j$ on the other features. The bars are green below 5, orange from 5 to 10, and red above 10, the thresholds used in the chapter. The panel under it shows each fitted coefficient with its 95% confidence interval, as the change in price (in thousands of dollars) for a fixed step of the feature: 500 more square feet, 2 more rooms, or 15 more years. The black triangle marks the true value, which is known only because the data are simulated.

Everything is computed from the 60 houses on screen: the correlations, the least-squares fit, the standard errors, the VIFs, $R^2$, and adjusted $R^2$. A VIF of 16 means the variance of that coefficient is 16 times what it would be if the feature were uncorrelated with the others, so its standard error is 4 times as large.

## How to Use

1. Read the default view. The correlation setting is 0.97, the VIFs of square feet and rooms are above 10, and their intervals are several times as wide as the interval for age.
2. Drag the correlation slider down to 0. The VIF bars shrink toward 1 and the two wide intervals narrow, while $R^2$ changes only a little.
3. Return the slider to 0.97 and press **New Sample** several times. The square feet and rooms coefficients swing from sample to sample, and sometimes one has the wrong sign. The age coefficient moves much less.
4. Press **Resample ×100** to refit the model to 100 new samples. Each faint tick is one estimate. Compare the spread of the ticks (SD) with the standard error (SE) from the single sample.
5. Uncheck **Rooms in the model** to try one remedy, removing a correlated feature. Read what happens to the VIFs, to the interval, and to the value of the square feet coefficient.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/multicollinearity-detector-microsim/main.html"
        height="632"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Multiple linear regression, the correlation coefficient, $R^2$, and the idea of a confidence interval.

### Bloom's Taxonomy Level
Evaluate (L5)

### Learning Objective
Students will be able to diagnose multicollinearity from a correlation matrix and variance inflation factors, judge whether individual coefficients can be trusted, and evaluate the trade-off in removing a correlated feature.

### Activities

1. **Find the thresholds** (5 min): Students lower the correlation slider from 0.99 in steps and record the correlation $r$ and the VIF of square feet at each step. They report the correlation at which the VIF first falls below 10 and below 5.
2. **Stability test** (5 min): At a setting of 0.97 and again at 0.30, students press **New Sample** five times and write down the rooms coefficient each time, then press **Resample ×100** and compare the SD values. They describe the difference in one sentence.
3. **Judge the remedy** (5 min): Students remove rooms from the model at a setting of 0.97 and list what improved and what was lost. They decide whether they would keep or drop rooms if the goal were prediction, and again if the goal were to explain the effect of square feet.

### Assessment
Shown a correlation matrix and a table of VIFs for a new set of features, students identify the features with a multicollinearity problem, state what a VIF of 9 implies about the standard error of that coefficient (it is 3 times as large), and recommend one remedy together with its cost.

## References

1. Wikipedia. [Variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor).
2. Wikipedia. [Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 3, Linear Regression.
