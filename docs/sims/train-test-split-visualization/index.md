---
title: Train-Test Split Visualization
description: Move a slider to split 100 rows of housing data into a training set and a test set, see which rows land on each side of the wall, and compare the model's score on rows it has seen with its score on rows it has not.
image: /sims/train-test-split-visualization/train-test-split-visualization.png
og:image: /sims/train-test-split-visualization/train-test-split-visualization.png
twitter:image: /sims/train-test-split-visualization/train-test-split-visualization.png
social:
   cards: false
quality_score: 0
---

# Train-Test Split Visualization

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Train-Test Split Visualization MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

One hundred squares stand for the rows of a housing data set (square feet and price). The top strip shows the rows in file order. Below it the rows have been shuffled and cut once. The green squares to the left of the wall are the **training set** and the blue squares to the right are the **test set**. The number in each square is the row's position in the original file, so you can see that the test rows come from all over the data set and not from the end of it.

The eye marks the rows the model studies when `model.fit(X_train, y_train)` runs. The crossed-out eye marks the rows that stay hidden until the final check with `model.score(X_test, y_test)`.

The two scores in the lower panel are computed, not typed in. A least-squares line (price from square feet) is fit on the training rows only, and $R^2$ is then measured twice: on the training rows the model has already seen and on the test rows it has never seen. Only the second number is a fair estimate of how the model will do on new houses. The advice panel warns when fewer than 60% of the rows are used for training (less to learn from) or more than 90% (too few test rows for a steady score).

The shuffle uses the MicroSim's own seeded generator. The seed plays the role of `random_state`: the same seed always gives the same split, but the rows chosen will not match the rows scikit-learn picks for that number.

## How to Use

1. Read the default view, an 80/20 split of 100 rows. There are 16 columns of five training rows to the left of the wall and 4 columns of five test rows to the right.
2. Drag the **Training share** slider from 50% to 95%. The wall moves, the sample counts change, and so does `test_size` in the code line under the title.
3. Hover over any square to read that row's square feet and price and to find the same row in the other picture.
4. Press **New Shuffle** several times. Compare how much the training $R^2$ and the test $R^2$ change from one shuffle to the next.
5. Set the slider to 95% and press **New Shuffle** again. With only 5 test rows, the test $R^2$ swings more widely than it did at 80%.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/train-test-split-visualization/main.html"
        height="582"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Rows and columns of a data set, fitting a regression line, and $R^2$ as a score for a fit.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how a train-test split divides a data set and why the test rows must stay hidden from the model until the final evaluation.

### Activities

1. **Explore** (4 min): Students move the slider across its range and record the number of training and testing samples at 50%, 70%, 80%, and 95%. They hover over three blue squares and note that the test rows come from different parts of the file.
2. **Compare scores** (5 min): At 80% students press **New Shuffle** five times and write down the training and test R² each time, then repeat at 95%. They describe which score is steadier and at which setting.
3. **Explain** (4 min): In pairs, students explain in two sentences why the score on the training rows cannot serve as the estimate for new data, and what would go wrong if test rows were used during fitting.

### Assessment
Given a data set of 500 rows and `test_size=0.3`, students state how many rows are in each set, say which set `fit` and `score` should each receive, and explain why a 99/1 split would give an unreliable test score.

## References

1. scikit-learn documentation. [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
2. Wikipedia. [Training, validation, and test data sets](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 5, Resampling Methods.
