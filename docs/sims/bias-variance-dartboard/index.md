---
title: Bias-Variance Dartboard
description: Compare four reference dartboards for low and high bias and variance, then set a model complexity, throw darts, and read the bias, variance, and total error measured from where the darts land.
image: /sims/bias-variance-dartboard/bias-variance-dartboard.png
og:image: /sims/bias-variance-dartboard/bias-variance-dartboard.png
twitter:image: /sims/bias-variance-dartboard/bias-variance-dartboard.png
social:
   cards: false
quality_score: 0
---

# Bias-Variance Dartboard

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Bias-Variance Dartboard MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The bullseye is the true value and each dart is one prediction. The four small boards show the four combinations of bias and variance:

- **Low bias, low variance**: tightly grouped on the bullseye
- **Low bias, high variance**: centered on the bullseye on average, but scattered
- **High bias, low variance**: tightly grouped, but off target in the same direction
- **High bias, high variance**: off target and scattered

On every board the gold ✕ is the average landing point, and the orange segment from the bullseye to the ✕ is the **bias**. The **variance** is the average squared distance of the darts from the ✕. Both are measured from the darts that are drawn, on a board whose radius is 10 units. For any set of darts, $\text{total error} = \text{bias}^2 + \text{variance}$, where the total error is the average squared distance of the darts from the bullseye. That is why the third bar is the first two placed end to end.

The **Model complexity** slider is an analogy, not a fitted model. A low setting moves the thrower's aim away from the bullseye and tightens the grouping, like a simple model with high bias and low variance. A high setting centers the aim and widens the scatter, like a complex model with low bias and high variance. The gray text under the heading gives the true bias and variance of the setting. The measured values differ from them because they come from a limited number of throws.

No setting reaches the low-bias, low-variance board. Along this slider, lowering one error raises the other. That is the tradeoff.

## How to Use

1. Study the four reference boards and the bias and variance printed under each.
2. Move the **Model complexity** slider. Ten darts are thrown at each new setting. When one kind of error is at least twice the other, the reference board that the darts resemble is outlined in gold.
3. Read the three bars: bias squared (orange), variance (purple), and total error (the two stacked).
4. Press **Throw 10 Darts** to add darts at the same setting, up to 200. Watch the measured bias and variance settle toward the true values of the setting.
5. Step through the settings from 1 to 10 and note the total error at each one. Find the settings where it is smallest.
6. **Clear Board** removes the darts so you can start a setting again.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/bias-variance-dartboard/main.html"
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
The mean of a set of values, squared distance, and the ideas of underfitting and overfitting.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to interpret a pattern of predictions as high or low bias and high or low variance, and explain how increasing model complexity trades bias for variance.

### Activities

1. **Describe** (3 min): Students describe each reference board in their own words and match it to a thrower, for example a steady thrower with a bent dart for high bias and low variance.
2. **Measure** (5 min): Students set the complexity to 2, 5, and 9, throw 50 darts at each, and record bias², variance, and total error in a table. They check that the first two columns add up to the third.
3. **Explain the tradeoff** (5 min): Using the table, students explain which error falls and which rises as complexity increases, and identify the range of settings with the lowest total error.

### Assessment
Students are shown three unlabeled dart patterns. For each one they state whether the bias and the variance are high or low, name the modeling problem it stands for (underfitting, overfitting, or neither), and say whether a more complex or a simpler model would help.

## References

1. Wikipedia. [Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).
2. Wikipedia. [Accuracy and precision](https://en.wikipedia.org/wiki/Accuracy_and_precision).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 2, Statistical Learning.
