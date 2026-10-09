---
title: Correlation Visualizer MicroSim
description: Set the correlation of a scatter plot, switch to curved patterns, and drag points to see how Pearson r, R squared, and Spearman rank correlation respond.
image: /sims/correlation-visualizer-microsim/correlation-visualizer-microsim.png
og:image: /sims/correlation-visualizer-microsim/correlation-visualizer-microsim.png
twitter:image: /sims/correlation-visualizer-microsim/correlation-visualizer-microsim.png
social:
   cards: false
quality_score: 0
---

# Correlation Visualizer MicroSim

<iframe src="main.html" height="587" width="100%" scrolling="no"></iframe>

[Run the Correlation Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The scatter plot shows 40 points. The panel beside it reports three numbers calculated from the points:

- **Pearson r** measures the strength and direction of a straight-line relationship: $r = \dfrac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$
- **R²** is $r^2$, the share of the variance in $y$ that the best-fit line accounts for
- **Spearman ρ** is Pearson r calculated on the ranks of $x$ and $y$. It measures whether $y$ rises or falls steadily with $x$, whether or not the path is straight.

Both correlations are marked on a scale from −1 to +1. The blue triangle is Pearson r and the orange ring is Spearman ρ.

Four patterns are available. The **linear cloud** is built so that its Pearson r equals the slider value exactly. **Line with one outlier** adds a single point far from the trend. The **U-shaped curve** has a strong relationship and an r near 0. The **steady curved rise** climbs along a curve, so Spearman ρ is close to +1 while Pearson r is lower. Any point can be dragged, and all three numbers update as it moves.

The words used for the size of r (weak, moderate, strong) follow a common rule of thumb with cut points at 0.1, 0.4, and 0.7. Other books draw the lines elsewhere.

Not included: the p-value, the confidence band, and a slider for the number of points.

## How to Use

1. Move the **Target r** slider from −1 to +1 and watch the cloud tighten around a falling line, spread into a round cloud near 0, and tighten around a rising line.
2. Choose **Line with one outlier** and compare both correlations with the linear cloud at the same slider value.
3. Choose **U-shaped curve** and **Steady curved rise**. For each, compare Pearson r with Spearman ρ and with what your eyes tell you.
4. Drag one point of a linear cloud to a far corner. Note how much r changes, then drag it back.
5. Press **New Sample** for a different set of points with the same settings. Uncheck **Best-fit line** to judge the pattern without it.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/correlation-visualizer-microsim/main.html"
        height="587"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Reading a scatter plot, and the mean and standard deviation of a variable.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to compare Pearson and Spearman correlation across linear, curved, and outlier-affected scatter plots and explain what each number does and does not reveal about the relationship.

### Activities

1. **Estimate** (4 min): A partner sets the slider while the panel is covered. Students estimate r from the plot alone, then uncover the panel and compare.
2. **Break the correlation** (5 min): Starting from a linear cloud with r = 0.8, students drag a single point to change r by at least 0.2 and record the change in Spearman ρ for the same move.
3. **Compare patterns** (6 min): Students record Pearson r, R², and Spearman ρ for all four patterns in a table and explain each case where the two correlations disagree, or where r is near 0 although a pattern is visible.

### Assessment
Students are shown three scatter plots (a tight falling line, a U shape, and a rising curve with one outlier). For each they decide whether Pearson r is positive, negative, or near zero, whether Spearman ρ would be noticeably different, and why.

## References

1. Wikipedia. [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
2. Wikipedia. [Spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).
3. pandas documentation. [pandas.DataFrame.corr](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html).
