---
title: K-Fold Cross-Validation Animator
description: Step through the rounds of K-fold cross-validation on 50 rows of data, see which fold is held out each time, and compare the score of each round with the mean of all K.
image: /sims/k-fold-cross-validation-animator/k-fold-cross-validation-animator.png
og:image: /sims/k-fold-cross-validation-animator/k-fold-cross-validation-animator.png
twitter:image: /sims/k-fold-cross-validation-animator/k-fold-cross-validation-animator.png
social:
   cards: false
quality_score: 0
---

# K-Fold Cross-Validation Animator

<iframe src="main.html" height="622" width="100%" scrolling="no"></iframe>

[Run the K-Fold Cross-Validation Animator MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Each row of the grid is one round of cross-validation, and each small bar is one of 50 rows of housing data (size and price). In a round, the blue fold is held out as the test set and the green folds are used to fit a least-squares line. The bar and the number at the right give that round's $R^2$ on the held-out fold. The dashed red line marks the mean of the rounds run so far.

The scatter plot shows the highlighted round. Green dots are the training rows, blue squares are the test rows, the black line is the fit from the green dots only, and the short blue segments are the errors on the test rows that the score is built from.

Every number is computed from the data on screen. The folds are cut the way scikit-learn's `KFold` cuts them (consecutive blocks of rows, in order), so with the rows in file order the scores equal what `cross_val_score(model, X, y, cv=K, scoring='r2')` returns for this data. The standard deviation divides by K, as `cv_scores.std()` does.

Two details differ from the chapter text. The chapter's list holds out fold 5 first. scikit-learn, and this MicroSim, hold out fold 1 first; the scores are the same either way. **Shuffle Rows** reorders the rows with the MicroSim's own seeded shuffle before the folds are cut, which is what `KFold(K, shuffle=True, random_state=...)` does, but the rows will not match scikit-learn's for the same number.

## How to Use

1. Read Round 1. Fold 1 (blue) is held out and the line is fit on the other 40 rows. Its score appears at the right.
2. Press **Next Fold** to run the next round, or **Start** to play the remaining rounds. **Start** becomes **Pause** while it runs.
3. Watch the blue block move one fold to the right each round. When all the rounds are done, every row has been a test row exactly once.
4. Click any finished round to see its fit again in the scatter plot.
5. Change **Number of folds (K)** to 3 or 10 and run the rounds again. With K = 10 each test fold has only 5 rows, so the single-fold scores vary much more.
6. With all rounds finished, press **Shuffle Rows** several times. Compare how much the score of Round 1 changes with how much the mean changes. **Reset** returns to file order and Round 1.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/k-fold-cross-validation-animator/main.html"
        height="622"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Train-test split, fitting a regression line, and $R^2$ as a test score.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how K-fold cross-validation rotates the test fold through the data and why the mean of the fold scores is a more reliable estimate than the score from a single train-test split.

### Activities

1. **Trace the rotation** (4 min): With K = 5 students press **Next Fold** four times, recording which fold is tested in each round and that round's R². They confirm that the five test folds cover all 50 rows with no overlap.
2. **Calculate** (4 min): Students compute the mean of the five fold scores by hand and compare it with the Mean R² shown. They name the lowest and highest fold score and state what a single split could have reported.
3. **Compare estimates** (5 min): With all rounds finished, students press **Shuffle Rows** six times, recording the score of Round 1 and the mean each time. They compare the spread of the two columns and explain the difference.

### Assessment
Students are given five fold scores (for example 0.62, 0.71, 0.55, 0.68, 0.74). They compute the mean, state how many models were trained to get it, and explain in two or three sentences why the mean is a better estimate of performance than the 0.55 or the 0.74 that a single split might have produced.

## References

1. scikit-learn User Guide. [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).
2. scikit-learn documentation. [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html).
3. Wikipedia. [Cross-validation (statistics)](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).
