---
title: Regression Assumptions Checker MicroSim
description: Examine four coordinated diagnostic plots for five data sets, decide which regression assumption each one violates, and then reveal traffic-light checks for linearity, homoscedasticity, and normality of residuals.
image: /sims/regression-assumptions-checker-microsim/regression-assumptions-checker-microsim.png
og:image: /sims/regression-assumptions-checker-microsim/regression-assumptions-checker-microsim.png
twitter:image: /sims/regression-assumptions-checker-microsim/regression-assumptions-checker-microsim.png
social:
   cards: false
quality_score: 0
---

# Regression Assumptions Checker MicroSim

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Regression Assumptions Checker MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Each data set has 90 points and is fitted with a least-squares line. Four plots show the same fit from different angles:

- **Data and fitted line**: the scatter plot with the regression line
- **Residuals vs fitted values**: each residual $y - \hat{y}$ against its fitted value, with a dashed line at zero
- **Histogram of residuals**: the standardized residuals with a normal curve for comparison
- **Normal Q-Q plot**: the sorted standardized residuals against the quantiles a normal distribution would give

Five data sets are available: one that meets the assumptions, one with a curved relationship, one whose spread grows with $x$ (heteroscedastic), one with right-skewed errors, and one with two outliers. **New Sample** draws another sample of the same kind.

With **Show diagnosis** checked, each assumption gets a green, yellow, or red light and the plots are annotated: an orange curve for a missed bend, purple boxes for the middle half of the residuals in the left, middle, and right third of the residual plot, and red rings around residuals more than 3 standard deviations from zero. The rules behind the lights are:

- **Linearity**: the share of the residual variation that an added $x^2$ term would explain. Green below 10%, yellow below 25%, red otherwise.
- **Independence**: no light. It cannot be judged from these plots. It depends on how the data were collected.
- **Homoscedasticity**: the interquartile range of the residuals in the right third divided by the interquartile range in the left third. Green when the larger spread is less than 2 times the smaller, yellow when it is less than 2.5 times, red otherwise.
- **Normality**: the Jarque-Bera statistic $\frac{n}{6}\left(S^2 + \frac{K^2}{4}\right)$, where $S$ is the skewness and $K$ the excess kurtosis of the residuals. Green below 9.21, yellow below 25, red otherwise.

These thresholds are rules of thumb chosen for this MicroSim, not formal tests. The data are random samples, so a light sometimes disagrees with the name of the data set. In particular, strong heteroscedasticity also stretches the tails of the residual histogram, so the heteroscedastic data set turns the normality light yellow or red in about four samples out of ten.

Not included: a custom mode with draggable points.

## How to Use

1. Start with **Good data** and study the four plots. This is what a healthy fit looks like.
2. Choose another data set. Before checking anything, decide which plot looks different from the good one and which assumption that points to.
3. Check **Show diagnosis** to see the lights, the statistic behind each one, and the annotations on the plots.
4. Press **New Sample** several times to see how much the plots change from sample to sample for the same kind of data.
5. Uncheck **Show diagnosis**, switch to another data set, and repeat.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/regression-assumptions-checker-microsim/main.html"
        height="582"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Fitting a regression line, residuals, histograms, and the shape of the normal distribution.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to diagnose violations of the linearity, homoscedasticity, and normality assumptions from a residual plot, a residual histogram, and a Q-Q plot.

### Activities

1. **Establish the baseline** (3 min): Students describe each of the four plots for **Good data** in one sentence and draw three new samples to see ordinary sample-to-sample variation.
2. **Diagnose** (8 min): For each of the other four data sets students record which plot shows the problem, what the pattern looks like, and which assumption fails, then check with **Show diagnosis**.
3. **Compare** (5 min): Students compare **Non-normal residuals** with **Outliers present**. Both usually turn the normality light red. They explain how the histogram and the Q-Q plot tell the two apart.

### Assessment
Given printed residual-versus-fitted and Q-Q plots from three unknown data sets, students name the assumption that is violated in each (or state that none is), point to the feature of the plot that shows it, and explain why independence cannot be checked this way.

## References

1. Wikipedia. [Q–Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot).
2. Wikipedia. [Jarque–Bera test](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 3, Linear Regression.
