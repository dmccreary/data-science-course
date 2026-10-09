---
title: Metrics Comparison MicroSim
description: Drag data points or add outliers to a scatter plot and compare how MAE, MSE, RMSE, and R-squared respond to the same prediction errors.
image: /sims/metrics-comparison-microsim/metrics-comparison-microsim.png
og:image: /sims/metrics-comparison-microsim/metrics-comparison-microsim.png
twitter:image: /sims/metrics-comparison-microsim/metrics-comparison-microsim.png
social:
   cards: false
quality_score: 0
---

# Metrics Comparison MicroSim

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Metrics Comparison MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Ten points follow $y = 2x + 1$ with random noise, and a least-squares line is fit to them. Each point's **residual** (actual minus predicted) is drawn two ways: as a colored vertical segment, and as a square whose side is that segment. The color shows the size of the residual: green under 2, orange from 2 to 5, red over 5.

- **MAE** is the average length of the segments.
- **MSE** is the average area of the squares, in squared units.
- **RMSE** is the square root of MSE, which brings it back to the units of $y$.

The panel on the right recomputes the fitted line, $R^2$, adjusted $R^2$, MAE, MSE, and RMSE every time a point moves. The two bars put MAE and RMSE on one scale, since both are in the units of $y$. A black mark on each bar shows its value for the starting points, and the factor at the right says how many times larger it is now.

Below the bars, one sentence reports how much of the absolute-error total and how much of the squared-error total comes from the single largest residual. The difference between those two percentages is why RMSE reacts more strongly than MAE to one large miss.

The line is refit after every change, as it would be if you trained the model again. An outlier therefore pulls the line toward itself and changes the other residuals as well.

## How to Use

1. Read the starting values. $R^2$ is about 0.87, and RMSE is a little larger than MAE.
2. Drag one point straight up until it is far from the line. Watch the two bars and the growth factors beside them.
3. Press **Reset to Default**, then press **Add Outlier** once. Compare the largest residual's share of the absolute-error total with its share of the squared-error total.
4. Add a second and a third outlier. Check whether the gap between the two growth factors keeps widening, and read the note at the bottom of the panel.
5. Uncheck **Show squared residuals** or **Show absolute residuals** to study one picture at a time.
6. Hover over any point to read its coordinates, its predicted value, its residual $e$, and $e^2$.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/metrics-comparison-microsim/main.html"
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
Scatter plots, the least-squares regression line, and the definition of a residual.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to compare how MAE, MSE, RMSE, and R-squared respond to the same set of prediction errors and explain why RMSE is more sensitive to outliers than MAE.

### Activities

1. **Predict** (3 min): Before touching the sim, students write down which of MAE and RMSE they expect to grow by the larger factor when one point is moved 10 units from the line, and why.
2. **Measure** (6 min): Students press **Add Outlier** once and record MAE, RMSE, $R^2$, both growth factors, and the two percentages for the largest residual. They repeat with two and three outliers and describe how RMSE ÷ MAE changes.
3. **Explain** (5 min): After **Reset to Default**, students drag points so that every residual is about the same size and watch RMSE ÷ MAE move toward 1. They write two sentences on when they would report RMSE and when MAE.

### Assessment
Given the residuals 1, −1, 2, −2, and 10, students compute MAE, MSE, and RMSE by hand, state which metric the largest residual affects most, and justify the answer using its share of each total.

## References

1. scikit-learn User Guide. [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
2. Wikipedia. [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error).
3. Wikipedia. [Root mean square deviation](https://en.wikipedia.org/wiki/Root_mean_square_deviation).
