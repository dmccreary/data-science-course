---
title: Model Evaluation Workflow
description: Step through the ten-step model evaluation workflow on a real data set, see which scores exist at each step, and compare the honest order of operations with one that leaks the test data.
image: /sims/model-evaluation-workflow/model-evaluation-workflow.png
og:image: /sims/model-evaluation-workflow/model-evaluation-workflow.png
twitter:image: /sims/model-evaluation-workflow/model-evaluation-workflow.png
social:
   cards: false
quality_score: 0
---

# Model Evaluation Workflow

<iframe src="main.html" height="652" width="100%" scrolling="no"></iframe>

[Run the Model Evaluation Workflow MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The flowchart shows the ten steps of an honest model evaluation, arranged in four swimlanes by the data each step may touch: the full dataset, the training data, the validation or cross-validation folds inside the training data, and the test data. A dashed gray line carries the held-out test rows from the split straight to step 8, across a red **point of no return**. The padlock on the Test Data lane opens only at step 8.

The workflow is run on a real data set of 100 rows (80 training, 20 test), with polynomial degrees 1 to 5 as the candidate models. That is the same grid as the chapter's `GridSearchCV` example. As you step through, the panel on the right says what the step computes on this data, and the table fills in with the scores that exist at that point: Train R² and cross-validation scores during development, and a single Test R², for the selected model only, at the end.

The checkbox **Use the test data to choose the model (a leak)** switches to a leaky version of the same workflow. Red arrows show test rows flowing into the two steps where a choice is made, every candidate's Test R² appears early, and the model with the highest test score is selected. Step 10 then sets the leaky report beside the honest one.

The leaky score can never be lower than the honest one, because it is the highest of the five test scores and the honest model is one of the five. With only five candidates the gap is small, and on some samples both routes pick the same model. The gap grows as more models and settings are compared. In every case the leaked score has stopped being an independent check, which is the reason for the rule.

## How to Use

1. Press **Next** to move through the steps, or click any step in the flowchart. For each step, note which lane it sits in.
2. Watch the table under the description. At which step does each row first get a number?
3. At step 7, note which degree is selected and which score decided it. At step 8, the padlock opens and one Test R² appears.
4. Go back to step 1 and check **Use the test data to choose the model (a leak)**. Step through again and find where the red arrows and warning signs appear.
5. At step 10 with the leak on, compare the leaky report with the honest one.
6. Stay on step 10 and press **New Data** several times. Record both test scores each time.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/model-evaluation-workflow/main.html"
        height="652"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The train-test split, K-fold cross-validation, R-squared, and hyperparameters such as polynomial degree.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to place each step of the model evaluation workflow in the data subset it may use and compare an honest workflow with a leaky one, explaining why test data must not influence any choice.

### Activities

1. **Trace** (4 min): Students step through the honest workflow and list, for each of the ten steps, which rows of data it uses.
2. **Find the leak** (4 min): With the checkbox on, students identify the steps where test data enters, and state what would have to change in the code to remove the leak.
3. **Measure the bias** (5 min): Staying on step 10 with the leak on, students press **New Data** ten times and tally how often the leaky test score is higher than, equal to, or lower than the honest one. They explain the result.

### Assessment
Given a short description of a project in which the analyst scaled the features using all rows, tried eight models, and reported the best test score, students mark each point where test data leaked and rewrite the steps in an order that avoids it.

## References

1. scikit-learn User Guide. [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
2. scikit-learn User Guide. [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html).
3. Wikipedia. [Leakage (machine learning)](https://en.wikipedia.org/wiki/Leakage_(machine_learning)).
