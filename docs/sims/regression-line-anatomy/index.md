---
title: Regression Line Anatomy
description: Hover over or click the five numbered parts of a fitted regression line (intercept, slope, predicted value, actual value, and residual) to read what each one means and see its value for ten students.
image: /sims/regression-line-anatomy/regression-line-anatomy.png
og:image: /sims/regression-line-anatomy/regression-line-anatomy.png
twitter:image: /sims/regression-line-anatomy/regression-line-anatomy.png
social:
   cards: false
quality_score: 0
---

# Regression Line Anatomy

<iframe src="main.html" height="587" width="100%" scrolling="no"></iframe>

[Run the Regression Line Anatomy MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

A scatter plot shows hours studied and exam scores for ten students, with the least-squares regression line drawn through them. Five numbered callouts mark the parts of the model:

1. **Intercept** ($\beta_0$): where the line crosses the y-axis, the predicted $y$ when $x = 0$
2. **Slope** ($\beta_1$): rise over run, the change in the predicted $y$ for each one-unit increase in $x$
3. **Predicted value** ($\hat{y}$): the model's answer for a given $x$, always on the line
4. **Actual value** ($y$): an observed data point, usually not on the line
5. **Residual** ($y - \hat{y}$): the vertical distance from the actual value to the predicted value

The panel explains the selected part and gives its value in this data. The equation under the plot, $\hat{y} = 47.5 + 5.5x$, uses the same colors, so each number in the equation can be matched to a part of the picture. Selecting the residual keeps the actual and predicted values lit, because the residual is the gap between them.

The slope and intercept are computed from the ten points by ordinary least squares. The points were chosen so that the line is exactly $\hat{y} = 47.5 + 5.5x$, the model the chapter interprets. They are a separate data set from the chapter's eight-student example, whose least-squares line is $\hat{y} = 47.04 + 5.71x$.

## How to Use

1. Read the plot first. Each dot is a student and the black line is the regression line.
2. Hover over a numbered callout, a row of the list, or a colored number in the equation to highlight that part and read about it.
3. Click a part to keep it selected. Click it again to release it.
4. Check **Show all residuals** to draw the residual of every student, not only the featured one.
5. Uncheck **Show names**, name each numbered part from memory, and hover or click to check.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/regression-line-anatomy/main.html"
        height="587"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
5-10 minutes

### Prerequisites
Reading a scatter plot and the slope-intercept form of a line, $y = mx + b$.

### Bloom's Taxonomy Level
Remember (L1)

### Learning Objective
Students will be able to identify the intercept, slope, predicted value, actual value, and residual on a regression plot and in the regression equation.

### Activities

1. **Explore** (3 min): Students hover over the five parts in order and write each name, symbol, and meaning in a table.
2. **Connect to the equation** (3 min): Students hover over each colored number in the equation and point to the same part in the plot. They compute the predicted score for 3 hours by hand and compare it with part 3.
3. **Self-test** (3 min): With **Show names** unchecked, students name each numbered part and click it to check. They then turn on **Show all residuals** and list the students whose scores the model predicted too high.

### Assessment
Given an unlabeled scatter plot with a regression line and its equation, students label the intercept, the slope triangle, one predicted value, one actual value, and the residual between them, and state the sign of that residual.

## References

1. Wikipedia. [Simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression).
2. Wikipedia. [Errors and residuals](https://en.wikipedia.org/wiki/Errors_and_residuals).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 3, Linear Regression.
