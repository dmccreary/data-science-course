---
title: Model Selection Dashboard
description: Train polynomial models of different degrees, compare their cross-validation scores, declare a winner, and only then unlock the test set to check your choice.
image: /sims/model-selection-dashboard/model-selection-dashboard.png
og:image: /sims/model-selection-dashboard/model-selection-dashboard.png
twitter:image: /sims/model-selection-dashboard/model-selection-dashboard.png
social:
   cards: false
quality_score: 0
---

# Model Selection Dashboard

<iframe src="main.html" height="717" width="100%" scrolling="no"></iframe>

[Run the Model Selection Dashboard MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The dashboard runs model selection on one data set of 75 points: 60 training points and 15 test points that stay locked away. The slider sets the degree of a polynomial model, which is fit to the training points only. The line under the plot title gives that model's **Train R²** and its **5-fold cross-validation R²** as the mean ± the standard deviation of the five fold scores.

**Add to Comparison** records the model in the table. Each row has a bar for the CV mean on a scale from 0 to 1, with a whisker of one standard deviation on each side. An orange triangle marks a model whose Train R² is over 0.20 and more than 0.20 above its CV mean, which is a sign of overfitting. The Test R² column shows padlocks.

When you have compared enough models, put your choice on the slider and press **Declare Degree … the Winner**. Only then are the test points drawn and the Test R² column filled in. The bars are recolored (green for the highest CV mean, gold for models within one standard deviation of it, red for the rest), a star marks the CV winner, and the verdict panel compares your pick with it. A simpler model within one standard deviation of the best counts as a sound choice. After the declaration the comparison is closed, because test scores that have been seen can no longer be used to choose a model. **New Data** starts a new problem.

Every score comes from real least-squares fits computed in the browser, using the same definitions as scikit-learn's `cross_val_score` with `cv=5` and `scoring='r2'`. Scores can be negative, and with only 15 test points the test score is noisy too. Not included: a slider for the split ratio and a choice of the number of folds. They are fixed at 80/20 and 5 folds, as in the chapter's code.

## How to Use

1. The table starts with degree 1 and degree 10. Compare their Train R² and their CV mean. Which column would mislead you?
2. Move the **Polynomial degree** slider and watch the curve. Press **Add to Comparison** for each degree you want to consider.
3. Look for a high CV mean with a short whisker, and note which rows have an orange triangle.
4. Put your choice on the slider and press **Declare Degree … the Winner**.
5. Read the verdict. Compare your pick's Test R² with its CV mean, and with the Test R² of the other models.
6. Choose another **Data set** or press **New Data** and repeat. Either one clears the table, because scores from different data cannot be compared.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/model-selection-dashboard/main.html"
        height="717"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
R-squared, the train-test split, K-fold cross-validation, and the idea that a polynomial of higher degree is a more flexible model.

### Bloom's Taxonomy Level
Evaluate (L5)

### Learning Objective
Students will be able to select a model from several candidates using cross-validation scores and their variability, and justify the choice before confirming it on held-out test data.

### Activities

1. **Compare** (5 min): On the Quadratic data set, students add at least five degrees to the table and record Train R², CV mean, and CV standard deviation for each.
2. **Decide** (5 min): Students write down their winner and one sentence of justification, then declare it and compare their reasoning with the verdict.
3. **Generalize** (8 min): Students repeat on the Sine wave and Noisy line data sets. They describe how the best degree and the size of the standard deviations change, and explain why Train R² would have chosen degree 10 every time.

### Assessment
Given a table of five models with Train R², CV mean, and CV standard deviation, students choose a model, justify the choice in terms of CV mean, variability, and simplicity, and explain why the test set was not used to make the choice.

## References

1. scikit-learn User Guide. [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).
2. scikit-learn examples. [Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 5, Resampling Methods.
