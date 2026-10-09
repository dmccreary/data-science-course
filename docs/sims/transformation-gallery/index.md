---
title: Transformation Gallery
description: Compare six common transformations on synthetic data, study the before and after plots of each one, tune the Box-Cox lambda, and add zeros and negative values to see where each transformation is undefined.
image: /sims/transformation-gallery/transformation-gallery.png
og:image: /sims/transformation-gallery/transformation-gallery.png
twitter:image: /sims/transformation-gallery/transformation-gallery.png
social:
   cards: false
quality_score: 0
---

# Transformation Gallery

<iframe src="main.html" height="642" width="100%" scrolling="no"></iframe>

[Run the Transformation Gallery MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Six tiles show the log, square root, reciprocal, square, Box-Cox, and standardizing transformations. Each tile has a small before and after plot of its own data set, so the six effects can be compared at a glance. Clicking a tile opens large **Before** and **After** plots with the formula, the kind of data the transformation is for, and statistics computed from the 200 values on screen.

| Tile | Data | What the numbers show |
|------|------|-----------------------|
| Log, $y' = \ln y$ | right-skewed incomes | skewness falls to about 0 |
| Square root, $y' = \sqrt{y}$ | daily counts at five shops | the spread within a shop stops growing with the mean |
| Reciprocal, $y' = 1/y$ | driving time against speed | a curve becomes a straight line ($R^2$ rises) |
| Square, $y' = y^2$ | pendulum period against length | a flattening curve becomes a straight line |
| Box-Cox, $y' = (y^\lambda - 1)/\lambda$ | skewed waiting times | skewness for the $\lambda$ on the slider, and the maximum-likelihood $\lambda$ |
| Standardize, $z = (x - \mu)/\sigma$ | income in dollars and age in years | both features get mean 0 and SD 1, and the skewness does not change |

The checkbox adds 12 problem values (6 zeros and 6 negatives) to every data set. Values a transformation cannot take are drawn in red, counted on its tile, and left out of the After plot. The log and Box-Cox reject all 12, the square root rejects the negatives, the reciprocal rejects the zeros, and the square and standardizing accept everything. The statistics are computed from the 200 original values, except for standardizing, which uses every value it is given.

Skewness and $\sigma$ use the population formulas (`ddof=0`), which is what `scipy.stats.skew` and `StandardScaler` compute. The maximum-likelihood $\lambda$ is found by a grid search on the same log-likelihood that `scipy.stats.boxcox` maximizes. Not included: uploading your own CSV file.

## How to Use

1. Read the six tiles. In each one the left plot is the data before the transformation and the right plot is the same data after it.
2. Click a tile to open it. Compare the **Before** and **After** plots and read the green line of statistics computed from the data.
3. Open **Box-Cox** and drag the **Box-Cox λ** slider. Try 1, 0.5, 0, and −1, and find the λ that brings the skewness closest to 0. Compare it with the maximum-likelihood λ on screen.
4. Check **Add zeros and negative values**. The red count on each tile shows how many of the 12 added values that transformation cannot take. Open each tile to see where the added values go.
5. Press **New Data** to draw new data sets and see which numbers change and which conclusions stay the same.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/transformation-gallery/main.html"
        height="642"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Histograms and scatter plots, skewness, mean and standard deviation, and the natural logarithm.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to compare the effects of the log, square root, reciprocal, square, Box-Cox, and standardizing transformations on the shape of data and state where each transformation is defined.

### Activities

1. **Survey** (4 min): Students open each tile in turn and record the statistic it reports before and after (skewness, spread, or R² of a straight line). They sort the six into those that change the shape of the data and those that do not.
2. **Tune Box-Cox** (5 min): Students move the λ slider to 1, 0.5, 0, and −1 and record the skewness each time, then find the λ with skewness closest to 0 and compare it with the maximum-likelihood value. They press **New Data** and repeat once.
3. **Find the limits** (5 min): Students check **Add zeros and negative values** and make a table of how many of the 12 added values each transformation rejects, with the reason. They name one fix for a column with zeros that needs a log.

### Assessment
Given three short descriptions (monthly sales that grow by a fixed percentage, counts of website errors per hour, and two features measured in dollars and in years that will go into a ridge model), students choose a transformation for each, justify it, and say what would happen if the column contained a zero.

## References

1. Wikipedia. [Data transformation (statistics)](https://en.wikipedia.org/wiki/Data_transformation_(statistics)).
2. Wikipedia. [Power transform](https://en.wikipedia.org/wiki/Power_transform) (Box-Cox and Yeo-Johnson).
3. scikit-learn documentation. [sklearn.preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html).
