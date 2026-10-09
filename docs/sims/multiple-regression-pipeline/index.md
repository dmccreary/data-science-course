---
title: Multiple Regression Pipeline
description: Step through the ten stages of a multiple regression workflow on 500 simulated houses, follow the shape of the data from stage to stage, and read the real result and the most common mistake at each stage.
image: /sims/multiple-regression-pipeline/multiple-regression-pipeline.png
og:image: /sims/multiple-regression-pipeline/multiple-regression-pipeline.png
twitter:image: /sims/multiple-regression-pipeline/multiple-regression-pipeline.png
social:
   cards: false
quality_score: 0
---

# Multiple Regression Pipeline

<iframe src="main.html" height="627" width="100%" scrolling="no"></iframe>

[Run the Multiple Regression Pipeline MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The flowchart shows a complete multiple regression workflow as ten stages. Each box carries the shape of the data that leaves the stage (rows × feature columns), so you can watch the columns grow from 8 to 10 in feature engineering and to 12 in preprocessing, then shrink to 11 after the multicollinearity check and to 7 after feature selection.

Below the flowchart, the left panel explains the selected stage, shows the Python that carries it out, and names the mistake that most often goes wrong there. The right panel shows the **result on this data**. Nothing in it is typed in. The page generates 500 houses and runs the whole pipeline in the browser:

1. **Raw data**: 6 numeric and 2 categorical columns, with gaps in `lot_size`.
2. **Feature engineering**: `log_lot` and `is_new`, two row-by-row formulas.
3. **Train/test split**: a shuffled 80/20 split into 400 and 100 rows.
4. **Preprocessing**: median imputation, standardization, and one-hot encoding, all learned from the 400 training rows.
5. **Multicollinearity**: $\text{VIF} = 1 / (1 - R^2)$ for each numeric column. `rooms` is nearly `bedrooms + bathrooms + 2`, so it is dropped.
6. **Feature selection**: forward selection on 5-fold cross-validated R², with a gain of 0.005 required.
7. **Model training**: ordinary least squares on the selected columns.
8. **Cross-validation**: the five fold scores, their mean, and their spread.
9. **Final evaluation**: R² and RMSE on the 100 test rows, with a plot of predicted against actual price.
10. **Feature importance**: the coefficients ranked by size.

The scores of stages 8 and 9 appear in the flowchart only when you reach those stages, because the test set is looked at once, at the end.

The workflow differs from the chapter's code in two places, on purpose. The chapter's listing computes `price_per_sqft` from the target. Stage 2 lists a feature like that as its mistake, because it hands the model the answer. The chapter also engineers `age_squared`, which is strongly correlated with `age` by construction and would dominate the VIF check, so this pipeline engineers a log and a flag instead. VIF is computed on the standardized training columns, where it equals $1 / (1 - R^2)$ with an intercept.

## How to Use

1. Read stage 1, then press **Next** to move through the pipeline. You can also click any box in the flowchart.
2. At each stage, compare the shape in its box with the shape in the box before it and say what added or removed rows or columns.
3. Read the **Result on this data** panel. At stages 3, 5, 6, 7, 8, and 10 read the bars, and at stage 9 read the plot.
4. Read **What can go wrong** for the stage. Untick the checkbox if you want to try naming the mistake before you see it.
5. At stage 9, compare the test R² with the cross-validation estimate from stage 8 and the training R² from stage 7.
6. At stage 10, compare the ranking with the order in which forward selection added the features at stage 6.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/multiple-regression-pipeline/main.html"
        height="627"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Multiple linear regression, the train-test split, standardization and one-hot encoding, VIF, feature selection, and cross-validation, each of which appears earlier in the chapter.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to carry out the stages of a multiple regression workflow in order, explain how each stage changes the shape of the data, and identify the mistake each stage must avoid.

### Activities

1. **Trace the shapes** (5 min): Students step through the ten stages and fill in a table of rows and feature columns after each stage, with one phrase for what caused each change.
2. **Name the mistake** (5 min): With the checkbox unticked, students write down what could go wrong at stages 2, 3, 4, and 9, then tick the box and compare.
3. **Order matters** (8 min): Students explain what would be wrong with three reorderings: scaling before the split, selecting features after looking at the test score, and computing the test score before cross-validation.

### Assessment
Given a short script that scales all rows before splitting and chooses features by test score, students identify both leaks, rewrite the order of the steps, and state which reported number can no longer be trusted.

## References

1. scikit-learn User Guide. [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
2. scikit-learn User Guide. [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
3. Wikipedia. [Variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor).
