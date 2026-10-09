---
title: Ridge vs Lasso Comparison
description: Move one lambda for ridge and lasso models fitted to the same standardized data, compare their coefficient paths and bar charts, and judge which penalty suits each of three data sets with the help of cross-validation.
image: /sims/ridge-vs-lasso-comparison/ridge-vs-lasso-comparison.png
og:image: /sims/ridge-vs-lasso-comparison/ridge-vs-lasso-comparison.png
twitter:image: /sims/ridge-vs-lasso-comparison/ridge-vs-lasso-comparison.png
social:
   cards: false
quality_score: 0
---

# Ridge vs Lasso Comparison

<iframe src="main.html" height="602" width="100%" scrolling="no"></iframe>

[Run the Ridge vs Lasso Comparison MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Two models are fit to the same 50 rows and 8 features. The top plots are **coefficient paths**: each colored line is one coefficient, traced as $\lambda$ grows from 0.1 to 10,000 on a log scale. The dashed vertical line marks the $\lambda$ on the slider, and the bar chart below compares the two sets of coefficients at that $\lambda$. The black line over each pair of bars is the coefficient with no penalty (ordinary least squares).

- **Ridge** shrinks every coefficient smoothly toward 0 and never reaches it.
- **Lasso** bends each path to exactly 0 at some $\lambda$ and keeps it there. Those features have left the model. They are marked with a hollow dot and a gray name.

Both models use the chapter's form of the loss, the residual sum of squares plus $\lambda$ times the penalty, on features and a target that are standardized. That lets one slider drive both. In scikit-learn the same fits are `Ridge(alpha=λ)` and `Lasso(alpha=λ/(2n))` with $n = 50$, because scikit-learn's Lasso divides the residual sum of squares by $2n$. Ridge is solved from $(X^TX + \lambda I)\beta = X^Ty$ and Lasso by coordinate descent with soft-thresholding. Both were checked against scikit-learn 1.8.

The three data sets are synthetic and built to differ. **Housing** has three size features that rise and fall together, and three features (beds, door number, and listing day) with no effect of their own. **Synthetic (sparse)** has three real effects and five noise features. **Medical** has three nearly identical body measurements and a real effect for every feature. The checkbox marks the $\lambda$ with the lowest 5-fold cross-validation error for each method and reports which method predicts better.

Not included: the contour picture of the L1 diamond and the L2 circle.

## How to Use

1. Read the two path plots from left to right. Find a ridge path that is still above 0 at λ = 1000 and a lasso path that has already reached 0 by λ = 100.
2. Drag the **λ** slider slowly to the right. Watch the lasso bars vanish one at a time while the ridge bars only get shorter.
3. Read the panel on the right. It counts the coefficients lasso has set to 0 and reports how far each method has shrunk the largest coefficient.
4. Choose **Synthetic (sparse)** and then **Medical** from the menu. In the medical data, compare what ridge and lasso do with weight, BMI, and waist.
5. Check **Show cross-validation best λ**. For each data set, decide which method you expect to predict better before you read the green line.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/ridge-vs-lasso-comparison/main.html"
        height="602"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Multiple regression coefficients, standardized features, the idea of a penalty on coefficient size, and cross-validation.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to compare how the ridge and lasso penalties change regression coefficients as lambda grows and judge which penalty suits a data set with few important features or with correlated features.

### Activities

1. **Trace the paths** (5 min): On the housing data students record, at λ = 1, 10, 31.6, and 100, how many lasso coefficients are exactly 0 and the percentage by which ridge has shrunk sq ft. They state the difference between the two shapes of path in one sentence.
2. **Correlated features** (5 min): On the medical data students compare the no-penalty coefficients of weight, BMI, and waist with the ridge and lasso coefficients at λ = 31.6 and describe what each method does with three features that measure nearly the same thing.
3. **Judge the method** (6 min): For each data set students predict whether ridge or lasso will have the lower cross-validation error, give a reason from the description of the data, and then check the box to compare.

### Assessment
Students are given a coefficient path plot without a title in which several lines sit exactly on 0 for large λ, identify the method, and explain what a manager should conclude about the features whose lines reached 0 first.

## References

1. scikit-learn documentation. [Linear Models: Ridge regression and Lasso](https://scikit-learn.org/stable/modules/linear_model.html).
2. Wikipedia. [Lasso (statistics)](https://en.wikipedia.org/wiki/Lasso_(statistics)).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 6, Linear Model Selection and Regularization.
