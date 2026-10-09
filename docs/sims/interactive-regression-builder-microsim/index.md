---
title: Interactive Regression Builder MicroSim
description: Build a data set by clicking, dragging, and double-clicking points and watch the least-squares line, its equation, the meaning of its slope and intercept, R squared, RMSE, a residual plot, and a prediction update after every change.
image: /sims/interactive-regression-builder-microsim/interactive-regression-builder-microsim.png
og:image: /sims/interactive-regression-builder-microsim/interactive-regression-builder-microsim.png
twitter:image: /sims/interactive-regression-builder-microsim/interactive-regression-builder-microsim.png
social:
   cards: false
quality_score: 0
---

# Interactive Regression Builder MicroSim

<iframe src="main.html" height="622" width="100%" scrolling="no"></iframe>

[Run the Interactive Regression Builder MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The scatter plot is yours to edit. Click to add a point, drag a point to move it, and double-click a point to remove it. After every change the model is fitted again by ordinary least squares, the same calculation that scikit-learn's `LinearRegression` performs.

The panel reports:

- the **equation** $\hat{y} = \beta_0 + \beta_1 x$ with the current coefficients
- the **slope** and the **intercept** in plain language with units, and a caution when $x = 0$ is far outside the data
- **R²**, the share of the variation in $y$ that the line explains, $R^2 = 1 - \text{SSE}/\text{SST}$, with a bar from 0 to 1
- **RMSE**, $\sqrt{\text{SSE}/n}$, the typical size of a prediction error (the value of `np.sqrt(mean_squared_error(y, y_pred))`)
- a **prediction** for the $x$ you type, with a warning when that $x$ is outside your data (extrapolation)

The residual plot under the scatter plot shows $y - \hat{y}$ at each $x$ on the same horizontal scale.

Four starting data sets are provided. *Study hours vs score* is the chapter's eight-student example, and its fit matches what scikit-learn returns for that data: $\hat{y} = 47.04 + 5.71x$ with $R^2 = 0.997$. *House size vs price*, *Car age vs value*, and *Random* come from a seeded generator, so they are the same on every load. In the two money data sets the y-axis is in thousands of dollars.

Not included: a confidence band, a confidence interval for the prediction, a noise slider, and assumption indicators. The residual plot uses $x$ on its horizontal axis. With one predictor this shows the same pattern as a plot against fitted values.

## How to Use

1. Choose a **Data set**. Read the equation and the two interpretation sentences.
2. Drag one point far from the line and watch the slope, R², and RMSE change. Drag it back.
3. Click empty space to add a point. Double-click a point to remove it. **Clear All** gives a blank plot for your own data.
4. Type a value in **Predict at x** to see the predicted value as a green diamond on the line. Try a value outside your data to see the extrapolation warning.
5. Uncheck **Show residuals** to hide the red residual segments in the scatter plot.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/interactive-regression-builder-microsim/main.html"
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
The regression equation, residuals, and the idea that least squares picks the line with the smallest sum of squared errors.

### Bloom's Taxonomy Level
Create (L6)

### Learning Objective
Students will be able to construct data sets with a required slope, strength of fit, or outlier, and interpret the fitted model's coefficients, R², RMSE, and predictions.

### Activities

1. **Interpret** (4 min): Students load each data set and write the slope and intercept interpretations in their own words, noting where the intercept has no real meaning.
2. **Build** (8 min): Starting from **Clear All**, students build three data sets: one with R² above 0.9, one with a negative slope, and one with R² below 0.1. They sketch each one and record its equation.
3. **Break** (5 min): Students load the study-hours data, add one outlier, and record how far the slope, R², and RMSE move. They compare an outlier near the middle of the x range with one at the end.

### Assessment
Students build a data set of at least eight points whose fitted slope is between 2 and 3 and whose R² is above 0.8. They write the equation, interpret both coefficients, and explain whether a prediction at an x value of their choice is an interpolation or an extrapolation.

## References

1. scikit-learn documentation. [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
2. scikit-learn documentation. [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).
3. Wikipedia. [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).
