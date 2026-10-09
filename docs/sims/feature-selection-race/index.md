---
title: Feature Selection Race
description: Step forward selection, backward elimination, and stepwise selection through the same data, predict each next move from the cross-validated scores, and compare the features, score, and work of the three results.
image: /sims/feature-selection-race/feature-selection-race.png
og:image: /sims/feature-selection-race/feature-selection-race.png
twitter:image: /sims/feature-selection-race/feature-selection-race.png
social:
   cards: false
quality_score: 0
---

# Feature Selection Race

<iframe src="main.html" height="712" width="100%" scrolling="no"></iframe>

[Run the Feature Selection Race MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Three feature selection methods work on the same data at the same time: 200 simulated rows with 8 candidate features. Each method has a track of eight chips. A colored chip is a feature that is in that method's model, and a gray chip is a feature that is left out.

Every score is a 5-fold cross-validated R² (CV R²) from real least-squares fits computed in the browser, the same number that `cross_val_score(LinearRegression(), X[features], y, cv=5, scoring='r2').mean()` returns. The number on a chip is the change in CV R² if that method flipped the chip. One rule decides every move: a feature has to earn 0.005 of CV R².

- **Forward selection** starts empty and adds the feature with the largest gain. It stops when no addition gains more than 0.005.
- **Backward elimination** starts with all 8 features and removes the one whose removal costs the least. It stops when every removal would cost more than 0.005.
- **Stepwise selection** starts empty and makes the single addition or removal that is best under the same rule, so it can undo an earlier choice.

The chip with the black outline is the move a method makes next. Under the chips is the path so far, with the score after each move. The table compares the features in each model, the CV R², and the number of models each method has had to score, which stands in for running time. When all three have stopped, the winner is named: the highest CV R², where scores within 0.005 count as a tie that goes to fewer features and then to fewer models scored. Because the data is simulated, the features the target was really built from are known. They are revealed at the finish with gold bars, and the last table column says whether each method found them.

The four data sets show where the methods differ. In **Few useful features** and **Most features useful** all three usually agree, and the difference is the work. In **A redundant stand-in**, `rooms` is close to `beds + baths`, so it looks strong on its own and stops helping once both are in the model. In **A pair that works together**, profit depends on `sales − costs`, two columns that are almost perfectly correlated, so neither one helps alone.

Nothing is scripted. **New Data** draws a new sample, and the paths and the winner can change. The empty model starts slightly below zero because a model that only predicts the mean of its training folds does a little worse than the mean of the fold it is scored on.

The chapter describes backward elimination with p-values. Here all three methods use the same CV R² rule, as the chapter's forward selection code does, so that their results can be compared. Not included: a speed slider (the race advances one move every 1.4 seconds, or one move per press of Next Step).

## How to Use

1. Read the numbers on the chips of each track. For each method, find the move it will make and check it against the chip with the black outline.
2. Press **Next Step**. All three methods make one move. Read the new path line and the new numbers on the chips.
3. Keep stepping, or press **Start Race** to let the methods run. A method that has no move worth 0.005 stops.
4. When all three have stopped, compare the rows of the table and read the result panel. Check each method's features against the gold bars.
5. Choose another **Data** set and repeat. Before you step, predict which method will need the fewest models and which will end with the best model.
6. Press **New Data** a few times on the last two data sets. Note when the outcome changes and when it does not.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/feature-selection-race/main.html"
        height="712"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Multiple linear regression, R-squared, K-fold cross-validation, and the definitions of forward selection, backward elimination, and stepwise selection.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to carry out forward selection, backward elimination, and stepwise selection one move at a time and compare the feature sets, scores, and amount of work they produce on the same data.

### Activities

1. **Predict and step** (5 min): On Few useful features, students write down each method's next move before every press of Next Step and record how many models each method scored by the finish.
2. **Compare the work** (5 min): Students run Most features useful and explain, from the two model counts, why backward elimination is cheaper there and forward selection is cheaper on the first data set.
3. **Explain the disagreement** (8 min): Students run A redundant stand-in and A pair that works together. For each, they explain in two sentences why forward selection ends with a different model than backward elimination, using the numbers on the chips as evidence.

### Assessment
Given the chip values of one track at one step, students state the method's next move or explain why it stops. They also describe one kind of data on which forward selection keeps a feature it does not need and one on which it misses features that backward elimination keeps.

## References

1. Wikipedia. [Stepwise regression](https://en.wikipedia.org/wiki/Stepwise_regression).
2. scikit-learn User Guide. [Feature selection, Sequential Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 6, Linear Model Selection and Regularization.
