---
title: Feature Scaling Comparison MicroSim
description: Compare a histogram and its summary statistics before and after min-max, standard, robust, and log scaling to see which methods change only the numbers and which change the shape.
image: /sims/feature-scaling-comparison-microsim/feature-scaling-comparison-microsim.png
og:image: /sims/feature-scaling-comparison-microsim/feature-scaling-comparison-microsim.png
twitter:image: /sims/feature-scaling-comparison-microsim/feature-scaling-comparison-microsim.png
social:
   cards: false
quality_score: 0
---

# Feature Scaling Comparison MicroSim

<iframe src="main.html" height="552" width="100%" scrolling="no"></iframe>

[Run the Feature Scaling Comparison MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Feature scaling puts columns with very different ranges onto comparable scales. This MicroSim shows one column of 200 values before and after scaling, as two histograms and a table of statistics.

- **Min-Max**: $x' = (x - \min) / (\max - \min)$. Every value lands between 0 and 1.
- **Standard (Z-score)**: $x' = (x - \text{mean}) / \text{std}$. The mean becomes 0 and the standard deviation 1.
- **Robust**: $x' = (x - \text{median}) / IQR$. The median becomes 0 and the IQR 1.
- **Log transform**: $x' = \ln(1 + x)$.

The first three methods only shift and stretch the axis, so the bars of the lower histogram keep exactly the shape of the upper one. The log transform is the only one that changes the shape. The table highlights the statistics that each method fixes at 0 or 1. A gold band above each histogram marks the middle 50% of the data, which shows how tightly min-max scaling packs the bulk of the data when outliers take the far end of the range.

Outliers are the values more than 1.5 × IQR beyond the quartiles of the original data. With **Mark outliers** on they are red in both histograms. The Outliers row counts the values beyond those fences before and after scaling. The standard deviation is the population value, as scikit-learn's `StandardScaler` uses. The pandas `.std()` in the chapter's hand-written formula divides by $n - 1$, so its result differs very slightly.

Not included: an animated transition between methods and an overlay of the two histograms.

## How to Use

1. Start with the prices data and **Min-Max**. Compare the two histograms, then read the highlighted rows of the table.
2. Switch **Scaling** to Standard and then Robust. Notice that the bars do not move while the axis numbers and the table change.
3. Choose **Log transform** and describe how the shape changed.
4. Change **Dataset** and try each method again. Use the right-skewed incomes to see what the log transform is for.
5. Turn **Mark outliers** off and on to follow the outliers through each transformation.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/feature-scaling-comparison-microsim/main.html"
        height="552"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Reading a histogram, and the meaning of mean, median, standard deviation, and interquartile range.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how min-max, standard, robust, and log scaling each change the values and the shape of a distribution, and how outliers affect each method.

### Activities

1. **Same shape, new numbers** (4 min): Students cycle through Min-Max, Standard, and Robust on one dataset and record which statistics each method sets to 0 or 1.
2. **Outliers** (5 min): On the prices data students record the middle 50% range under Min-Max and under Robust, and explain the difference.
3. **Changing the shape** (5 min): On the incomes data students compare the Outliers row before and after the log transform, then try the same transform on the heights data and explain why little changes.

### Assessment
Students are shown an unlabeled before and after pair of histograms with their statistics and must name the scaling method used and give the evidence. They then recommend a method for a column with a few very large values and justify it.

## References

1. scikit-learn documentation. [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html).
2. scikit-learn documentation. [Compare the effect of different scalers on data with outliers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html).
3. Wikipedia. [Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling).
