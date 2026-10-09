---
title: Regularization Decision Tree
description: Walk a decision tree with the facts of six modeling problems and choose between no penalty, ridge, lasso, and elastic net, with feedback on each choice.
image: /sims/regularization-decision-tree/regularization-decision-tree.png
og:image: /sims/regularization-decision-tree/regularization-decision-tree.png
twitter:image: /sims/regularization-decision-tree/regularization-decision-tree.png
social:
   cards: false
quality_score: 0
---

# Regularization Decision Tree

<iframe src="main.html" height="627" width="100%" scrolling="no"></iframe>

[Run the Regularization Decision Tree MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The tree on the left asks five questions in order and ends at one of four results: no penalty (`LinearRegression`), `Ridge`, `Lasso`, or `ElasticNet`. The panel on the right describes a modeling problem and gives its numbers: the samples $n$, the features $p$, and the training and validation $R^2$. The two quantities the tree needs, $p/n$ and the gap between the two scores, are worked out from them.

1. **Is the relationship a straight line?** If not, add polynomial features first.
2. **Does the model overfit?** The tree calls a gap of more than 0.05 between training and validation $R^2$ overfitting. With no gap, no penalty is needed.
3. **How many features for the number of samples?** This sets how strong $\lambda$ should be to start with: strong when $p/n$ is above 0.5, moderate from 0.05 to 0.5, light below 0.05.
4. **Should the model drop some features?** If every feature is to be kept, use Ridge.
5. **Are some features highly correlated?** If not, Lasso can drop features safely. If so, Elastic Net keeps correlated features together.

The path for each case is computed from its facts with these rules. The thresholds are rules of thumb for teaching. In practice the strength of $\lambda$ is always settled by cross-validation.

Case A opens with its path shown as a worked example. For the other cases the path is hidden until you ask for it. The design's separate question about interpretability is folded into question 4, since a model that drops features is the one that is easier to explain.

## How to Use

1. Read case A and follow the green path through the tree. Check each answer against the numbers in the panel.
2. Choose case B from the menu. The path is now hidden. Answer the questions one at a time from the facts of the case.
3. Click the result you would choose. Read the feedback, and if it is not the best fit, use the hint and try again.
4. Click any blue question to read what it asks and why it matters. Click it again to close the explanation.
5. Check **Show the path** to see the route the tree takes and the reason for it.
6. Work through cases C to F the same way.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/regularization-decision-tree/main.html"
        height="627"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Overfitting, training and validation scores, and what the ridge, lasso, and elastic net penalties do to coefficients.

### Bloom's Taxonomy Level
Evaluate (L5)

### Learning Objective
Students will be able to choose between no penalty, ridge, lasso, and elastic net for a described modeling problem and justify the choice from the size of the data, the gap between training and validation scores, the goal, and the correlation among features.

### Activities

1. **Worked example** (3 min): Students follow the highlighted path for case A and explain why a penalty is not needed when the two scores are almost equal.
2. **Decide and check** (8 min): For cases B to F students write down their answer to each question and their chosen method before clicking, then record whether the feedback agreed and what they missed.
3. **Change one fact** (4 min): Students pick one case and state one fact that would have to change for the tree to end at a different result, for example what would move case D from Elastic Net to Lasso.

### Assessment
Given a new description (500 patients, 40 lab measurements of which several are highly correlated, training R² 0.80 and validation R² 0.62, and a doctor who wants a short list of measurements to order), students name the method and the starting strength of λ and justify each step of the path.

## References

1. scikit-learn documentation. [Linear Models: Ridge regression, Lasso, and Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html).
2. Wikipedia. [Elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 6, Linear Model Selection and Regularization.
