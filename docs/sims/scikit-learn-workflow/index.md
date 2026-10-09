---
title: Scikit-learn Workflow
description: Step through the six-step scikit-learn pattern of import, prepare data, create, fit, predict, and evaluate on the chapter's study-hours data, choose the value to predict for, and switch on the single-brackets mistake to see where the script fails.
image: /sims/scikit-learn-workflow/scikit-learn-workflow.png
og:image: /sims/scikit-learn-workflow/scikit-learn-workflow.png
twitter:image: /sims/scikit-learn-workflow/scikit-learn-workflow.png
social:
   cards: false
quality_score: 0
---

# Scikit-learn Workflow

<iframe src="main.html" height="562" width="100%" scrolling="no"></iframe>

[Run the Scikit-learn Workflow MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The flowchart shows the pattern used with every scikit-learn model:

1. **Import** the model class
2. **Prepare data**: a 2-D `X` (features) and a 1-D `y` (target)
3. **Create** the model object
4. **Fit**: `model.fit(X, y)` learns the coefficients and stores them in the model
5. **Predict**: `model.predict(X_new)` applies them to new data
6. **Evaluate**: `model.score(X, y)` measures the fit

Below the flowchart is the complete script. The lines of the selected step are highlighted, and a panel explains the step and shows its result for the chapter's eight students (study hours 1 to 8, exam scores 52 to 92). The small plot shows the data, then the fitted line once `fit` has run, then the prediction.

The results are computed in the MicroSim with the formulas scikit-learn uses, and they agree with scikit-learn 1.8 on the same data: `coef_` is 5.7143, `intercept_` is 47.0357, and `score` returns $R^2 = 0.9970$. For a regression model `score` is $R^2$, not accuracy.

The **Mistake** checkbox changes step 2 to `X = df['study_hours']`. Single brackets give a 1-D Series, and `fit` stops with a `ValueError`. Recent versions of scikit-learn word it as shown in the MicroSim ("Expected a 2-dimensional container but got ... Series instead"). Older versions, and NumPy arrays, give "Expected 2D array, got 1D array instead". Either way the fix is double brackets, `df[['study_hours']]`, or `df['study_hours'].values.reshape(-1, 1)`.

Not included: a switch to other model classes. Only the import line and the create line would change.

## How to Use

1. Press **Next** to walk through the six steps in order, or click any box of the flowchart or any line of the script.
2. At each step read the explanation, then the **Result** panel, which shows what Python holds after that line has run.
3. At step 5, move the **Hours for X_new** slider. Watch the predicted score, and read the warning when the hours go outside 1 to 8.
4. Check **Mistake: single brackets on X** and step through again to see which line fails and why the later lines never run.
5. Uncheck the mistake, cover the script, and write the six steps from memory.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/scikit-learn-workflow/main.html"
        height="562"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The regression equation, the meaning of slope, intercept, and R², and selecting DataFrame columns in pandas.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to apply the scikit-learn pattern of import, prepare data, create, fit, predict, and evaluate to train a linear regression model and make a prediction.

### Activities

1. **Walk through** (4 min): Students step through the six steps and write down, for each one, the line of code and what exists afterward: a class, X and y, an untrained model, coefficients, a prediction, a score.
2. **Predict** (4 min): At step 5 students predict the score for 6.5 hours by hand from the coefficients and check with the slider. They then find the smallest slider value for which the model predicts more than 100 and explain why that prediction cannot be trusted.
3. **Debug** (4 min): Students switch on the mistake, find the step where the script stops, and write the corrected line in two different ways.

### Assessment
Given a DataFrame with a column sqft and a column price, students write the lines that train a LinearRegression model and predict the price of a 1,500 square foot house, and explain what model.score(X, y) returns.

## References

1. scikit-learn documentation. [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
2. scikit-learn documentation. [Getting Started](https://scikit-learn.org/stable/getting_started.html).
3. VanderPlas, J. *Python Data Science Handbook*. O'Reilly. The chapter Introducing Scikit-Learn.
