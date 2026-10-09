---
title: Multiple Regression Anatomy
description: Move sliders for square feet, bedrooms, and age and watch the intercept and each coefficient-times-feature term add up to a predicted house price.
image: /sims/multiple-regression-anatomy/multiple-regression-anatomy.png
og:image: /sims/multiple-regression-anatomy/multiple-regression-anatomy.png
twitter:image: /sims/multiple-regression-anatomy/multiple-regression-anatomy.png
social:
   cards: false
quality_score: 0
---

# Multiple Regression Anatomy

<iframe src="main.html" height="607" width="100%" scrolling="no"></iframe>

[Run the Multiple Regression Anatomy MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The equation $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$ is drawn across the top with five numbered parts: the intercept, one term for each feature, and the prediction. Under each symbol is the number it stands for right now, for example 150 × 1,500 for the square feet term.

The chart below it is a waterfall. It starts with the intercept, adds each term where the last one ended, and finishes with a gold bar for the predicted price. Terms that raise the prediction are green and terms that lower it are red.

The coefficients are not typed in. The MicroSim fits a least-squares regression to 40 simulated houses and shows the result with its $R^2$ and adjusted $R^2$. The houses were built so that the fit comes out exactly as the chapter's model, Price = 50,000 + 150 × SqFt + 10,000 × Bedrooms − 1,000 × Age.

## How to Use

1. Read the default view. A 1,500 square foot house with 3 bedrooms that is 20 years old is predicted at $285,000, which is 50,000 + 225,000 + 30,000 − 20,000.
2. Drag the **Square feet** slider. Only the square feet term and the prediction change. The bedrooms and age terms hold still.
3. Add one bedroom and check that the prediction rises by exactly $10,000. Repeat at a different square footage and confirm that the rise is the same.
4. Drag **Age** to 0 and then to 60 and watch the red bar pull the running total back.
5. Hover over a numbered part, in the equation or in the chart, to read what it means. Click to keep it selected and click again to release it.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/multiple-regression-anatomy/main.html"
        height="607"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Simple linear regression, the meaning of slope and intercept, and $R^2$.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how the intercept and each coefficient-times-feature term combine into one prediction, and interpret a coefficient as the change in the prediction for a one-unit increase in its feature with the other features held constant.

### Activities

1. **Explore** (4 min): Students move each slider in turn and record which numbers on screen change. They state in one sentence what stays the same when only one feature moves.
2. **One-unit changes** (5 min): Students work out by hand the price change for 100 more square feet, 1 more bedroom, and 10 more years of age, then check each answer with the sliders.
3. **Explain** (4 min): In pairs, students click each numbered part and restate its explanation in their own words, including why the intercept is not the price of a real house.

### Assessment
Given the model Price = 50,000 + 150 × SqFt + 10,000 × Bedrooms − 1,000 × Age, students compute the prediction for a 2,000 square foot, 4-bedroom, 10-year-old house and explain what the coefficient −1,000 means, using the phrase "holding the other features constant".

## References

1. Wikipedia. [Linear regression](https://en.wikipedia.org/wiki/Linear_regression).
2. scikit-learn documentation. [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 3, Linear Regression.
