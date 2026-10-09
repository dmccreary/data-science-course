---
title: Feature Engineering Laboratory
description: Build new features from five raw housing columns with a product, ratio, sum, difference, square, or log, preview each one, and see the real change in cross-validated R squared when it joins a linear regression model.
image: /sims/feature-engineering-laboratory/feature-engineering-laboratory.png
og:image: /sims/feature-engineering-laboratory/feature-engineering-laboratory.png
twitter:image: /sims/feature-engineering-laboratory/feature-engineering-laboratory.png
social:
   cards: false
quality_score: 0
---

# Feature Engineering Laboratory

<iframe src="main.html" height="622" width="100%" scrolling="no"></iframe>

[Run the Feature Engineering Laboratory MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The laboratory holds 300 simulated houses with five raw columns (`sqft`, `beds`, `age`, `lot`, `school`) and a `price` in thousands of dollars. A linear regression on the five raw columns is the **baseline**. Your job is to engineer features that lift the model's score toward the gold **goal** mark, which is the score of the formula that generated the prices.

Two dropdowns choose the columns **A** and **B** and a third chooses the operation. The **Preview** panel shows what you have set up before you commit to it:

- the pandas expression and the first eight rows, with the new column in purple
- a histogram of column A and of the new feature, each with its skew
- the correlation $r$ of each column with price
- the cross-validated R² the model would have if the feature were added, also drawn as a purple outline on the score bar

**Add Feature** puts the feature into the model. The **Model performance** panel compares the baseline with your model: Train R² (fit and scored on all 300 rows) and CV R² (the mean of 5 folds, as `cross_val_score` with `cv=5` computes it). The list at the lower right gives each engineered feature's **worth**, which is the CV R² the model would lose if that one feature were removed. A feature worth less than 0.005 is flagged in red, and its ✕ removes it.

Every number comes from a real least-squares fit in the browser. Things worth finding out:

- Train R² never goes down when a feature is added. CV R² can.
- A sum or a difference of two columns that are already in the model changes nothing, because a linear model can already weight them separately.
- A feature can have $r$ near zero with price and still be valuable next to the column it was built from (try `age²`).
- Two features that carry the same information each look nearly worthless while the other is in the model (add `sqft ÷ lot` and then `school ÷ lot`, which both stand in for `log(lot)`).

Skew is the third moment divided by the 1.5 power of the second, the default of `scipy.stats.skew`. Large values are shown in millions with an M. On a narrow screen the preview drops the table and shows one histogram. Not included: binning, a text box for naming features (names are generated), and preset buttons. The lab holds at most six engineered features.

## How to Use

1. Read the baseline row of the **Model performance** panel and find the black baseline mark and the gold goal mark on the bar.
2. The lab opens on `log(lot)`. Compare the two histograms and their skews, then read the **If added** line. Press **Add Feature**.
3. Set up a product, such as `sqft` × `school`, and read the preview before you add it. Does the value of a square foot depend on the school rating?
4. Try a sum or a difference of two columns and explain what the preview says.
5. Keep adding features that you expect to help. Watch the **Worth** column, and remove any feature flagged as low with its red ✕.
6. Try to reach the goal mark with four engineered features. **Reset** clears your features.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/feature-engineering-laboratory/main.html"
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
Multiple linear regression, R-squared on training data and under cross-validation, and the idea of interaction terms, polynomial terms, and log transforms.

### Bloom's Taxonomy Level
Create (L6)

### Learning Objective
Students will be able to construct new features from existing columns and use cross-validated R squared to decide which of them improve a linear regression model.

### Activities

1. **Explore one transform** (5 min): Students preview the log, the square, and the product with school for each raw column and record which previews show a gain above 0.005.
2. **Build a model** (8 min): Students build a model of at most four engineered features that reaches the goal mark, recording CV R² after each addition and the worth of each feature at the end.
3. **Explain the surprises** (5 min): Students explain why `sqft + lot` adds nothing, why `age²` helps although its correlation with price is near zero, and why a feature's worth changed when another feature was added.

### Assessment
Students propose one engineered feature for a new data set, justify it from domain knowledge, and state which number they would check to decide whether to keep it and why Train R² is not that number.

## References

1. scikit-learn User Guide. [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html).
2. scikit-learn API Reference. [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html).
3. Wikipedia. [Feature engineering](https://en.wikipedia.org/wiki/Feature_engineering).
4. VanderPlas, J. *Python Data Science Handbook*. O'Reilly. Chapter 5, Machine Learning, section Feature Engineering.
