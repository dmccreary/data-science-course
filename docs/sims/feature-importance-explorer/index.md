---
title: Feature Importance Explorer
description: Rank five housing features by standardized coefficient, permutation importance, and drop-column importance, and find out why the three rankings disagree when features are correlated.
image: /sims/feature-importance-explorer/feature-importance-explorer.png
og:image: /sims/feature-importance-explorer/feature-importance-explorer.png
twitter:image: /sims/feature-importance-explorer/feature-importance-explorer.png
social:
   cards: false
quality_score: 0
---

# Feature Importance Explorer

<iframe src="main.html" height="612" width="100%" scrolling="no"></iframe>

[Run the Feature Importance Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

One linear regression is fitted to 200 simulated training houses and scored on 100 test houses. The grid ranks its five features in three ways, one column per method, with the features lined up in rows.

- **Standardized coefficient.** Each feature is scaled to mean 0 and standard deviation 1 before fitting, as `StandardScaler` does, so every coefficient is the price change in thousands of dollars for a change of one standard deviation. This is what feature importance means for a linear model. Raw coefficients cannot be compared, because their units differ.
- **Permutation importance.** One column of the test data is shuffled and the drop in test $R^2$ is recorded. The bar is the mean of 30 shuffles and the error bar is their standard deviation.
- **Drop-column importance.** The model is refitted without the feature and the drop in test $R^2$ is recorded.

The number in each small box is the feature's rank under that method. A box is orange when the three methods do not give the feature the same rank. Clicking a row opens that feature below: a scatter plot against price with two lines, the fit that uses this feature alone (dashed) and the multiple regression's line with the other four features held constant (red), next to the feature's numbers.

With the default correlated features, square_feet is first by standardized coefficient and by permutation but second by drop-column, because bedrooms and bathrooms carry part of the same information and can stand in for it when the model is refitted. No ranking is reliable when features are strongly correlated. Uncheck **Correlated features** and the three methods agree.

## How to Use

1. Read the default view, sorted by standardized coefficient. Find the two features with orange rank boxes and read their ranks across the three columns.
2. Change **Sort by** to **Drop-column importance**. The rows reorder and a different feature comes out on top.
3. Click **square_feet**, then **bedrooms**. Compare the dashed line (the feature alone) with the red line (the other features held constant) and read the VIF.
4. Press **Run Permutation Test** a few times. The purple bars move by about the length of their error bars. Check whether any rank changes.
5. Uncheck **Correlated features**. The same kind of model is fitted to houses whose features are unrelated to each other. Look for orange boxes and compare the two lines in the scatter plot again.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/feature-importance-explorer/main.html"
        height="612"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Multiple linear regression, standardizing a feature, $R^2$ on a test set, and multicollinearity.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to compare standardized coefficients, permutation importance, and drop-column importance for one model, identify the features on which the methods disagree, and explain the disagreement in terms of correlation between features.

### Activities

1. **Compare rankings** (5 min): Students copy the three rankings into a table and circle the features whose rank is not the same in every column. They sort by each method in turn to check the table.
2. **Find the reason** (5 min): Students inspect square_feet and bedrooms and record the slope of each line and the VIF. They then uncheck **Correlated features**, record the same numbers, and state what changed.
3. **Judge** (5 min): In pairs, students decide which feature they would call the most important in the correlated data and defend the choice, naming the method they relied on and its weakness.

### Assessment
Students explain why a feature can have a large permutation importance and a small drop-column importance in the same model, and state what kind of data makes all three methods agree.

## References

1. scikit-learn documentation. [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
2. scikit-learn documentation. [Common pitfalls in the interpretation of coefficients of linear models](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html).
3. Wikipedia. [Standardized coefficient](https://en.wikipedia.org/wiki/Standardized_coefficient).
