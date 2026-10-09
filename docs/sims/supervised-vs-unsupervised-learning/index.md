---
title: Supervised vs Unsupervised Learning
description: Step through training data, training, and new data for a spam classifier and a customer clustering model side by side, and compare what each kind of learning is given, what it learns, and what it outputs.
image: /sims/supervised-vs-unsupervised-learning/supervised-vs-unsupervised-learning.png
og:image: /sims/supervised-vs-unsupervised-learning/supervised-vs-unsupervised-learning.png
twitter:image: /sims/supervised-vs-unsupervised-learning/supervised-vs-unsupervised-learning.png
social:
   cards: false
quality_score: 0
---

# Supervised vs Unsupervised Learning

<iframe src="main.html" height="622" width="100%" scrolling="no"></iframe>

[Run the Supervised vs Unsupervised Learning MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Two panels run the same three stages on two small data sets.

- **Supervised learning (green):** 40 emails, each described by two features (percent of words in ALL CAPS and exclamation marks per 100 words) and each carrying a label, *spam* or *not spam*. A logistic regression model is trained on the labels and draws the boundary where $P(\text{spam}) = 0.5$.
- **Unsupervised learning (blue):** 42 customers described by annual spending and visits per month, with no labels at all. K-Means with $k = 3$ groups the customers that sit close together and marks each group center.

The table under the panels lines up the difference stage by stage: the training data ($X$ and $y$, or $X$ only), what is learned (a rule from $X$ to $y$, or groups of similar rows), and the output (a predicted label, or a group number that a person still has to interpret). The row for the current stage is highlighted.

Both models are really fit to the seeded data on screen, so the numbers in the status lines (training emails on the correct side of the boundary, customers in each group, the probability for the new email) are computed, not typed in.

Not included: the quiz mode from the original specification.

## How to Use

1. Read stage 1, **Training data**. Both plots have two features per row, but only the emails are colored by a label.
2. Press **Next** to train both models. Compare the boundary learned from the labels with the three groups K-Means finds without any.
3. Press **Next** again. A new email and a new customer appear as diamonds. Click inside either plot to move the diamond and watch the predicted label and $P(\text{spam})$, or the nearest group, change.
4. Point at (or tap) a use case under the table to read why it belongs to that kind of learning.
5. Check **Show math notation** to see the same three table rows written formally.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/supervised-vs-unsupervised-learning/main.html"
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
Features and labels (X and y), scatter plots, and fitting a model with `model.fit()`.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to compare supervised and unsupervised learning by explaining what data each one is given, what it learns, and what kind of output it produces.

### Activities

1. **Step through** (4 min): Students go through the three stages and write one sentence per stage describing what differs between the two panels.
2. **Move the new point** (5 min): At stage 3 students click to place the new email on each side of the boundary and close to it, and record the predicted label and P(spam). They then place the new customer between two groups and note that the answer is only a group letter.
3. **Sort scenarios** (4 min): Students read the six use cases, then sort four new scenarios supplied by the teacher (for example predicting tomorrow's temperature, or grouping news articles) into supervised or unsupervised, and justify each choice by asking whether labeled answers exist.

### Assessment
Given a short description of a data set and a goal, students state whether the task is supervised or unsupervised, name what plays the role of X and (if present) y, and describe the form of the model's output.

## References

1. scikit-learn documentation. [Supervised learning](https://scikit-learn.org/stable/supervised_learning.html) and [Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html).
2. Wikipedia. [Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) and [Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 2, Statistical Learning (supervised versus unsupervised learning).
