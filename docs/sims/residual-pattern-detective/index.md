---
title: Residual Pattern Detective
description: Study four unlabeled residual plots, decide what each one says about a linear model, and get feedback on your diagnosis.
image: /sims/residual-pattern-detective/residual-pattern-detective.png
og:image: /sims/residual-pattern-detective/residual-pattern-detective.png
twitter:image: /sims/residual-pattern-detective/residual-pattern-detective.png
social:
   cards: false
quality_score: 0
---

# Residual Pattern Detective

<iframe src="main.html" height="637" width="100%" scrolling="no"></iframe>

[Run the Residual Pattern Detective MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

A **residual plot** shows what a model left unexplained. Each point is one row of data, placed at its predicted value (horizontal) and its residual, actual minus predicted (vertical). The dashed red line marks a residual of 0.

The four plots come from four different data sets of 60 points, and a straight line was fit to each one by least squares. One data set really is a straight line with even noise. In the other three, the true relationship is curved, the noise grows with the size of the target, or a hidden group shifts some of the rows by a constant amount. The plots are unlabeled and appear in shuffled order.

Select a plot, decide what it tells you about the model, and press one of the four diagnosis buttons. A wrong guess is answered with a description of what that diagnosis would look like, so you can compare it with the plot in front of you. A correct diagnosis labels the plot with its pattern name and icon and opens an explanation: what the pattern means, what to do about it, and numbers computed from the residuals that support it.

When all four are solved, the grid is a reference chart of the four patterns from the chapter: healthy residuals, a curved pattern, a funnel shape, and clustered groups. **New Cases** draws new data and reshuffles the plots.

## How to Use

1. Case A is selected (blue border). Check two things: are the points balanced above and below the dashed line at every predicted value, and is the vertical spread the same from left to right?
2. Press the diagnosis you think fits. The notes panel responds.
3. After a wrong guess, read what that diagnosis would look like, look at the plot again, and choose another.
4. After a correct diagnosis, read what the pattern means, what to do about it, and the evidence line. Then click another plot.
5. Solve all four with as few wrong guesses as you can. The count is shown under the buttons.
6. Press **New Cases** and try to solve the new set with no wrong guesses.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/residual-pattern-detective/main.html"
        height="637"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Simple linear regression and the definition of a residual as actual minus predicted.

### Bloom's Taxonomy Level
Evaluate (L5)

### Learning Objective
Students will be able to judge from a residual plot whether a linear model is adequate, identify curved, funnel-shaped, and clustered patterns, and recommend a fix for each.

### Activities

1. **Diagnose** (5 min): Students solve the four cases, writing one sentence of evidence for each diagnosis before pressing a button.
2. **Check the evidence** (5 min): For each solved case, students read the evidence line and explain how the means or standard deviations of the three thirds support the diagnosis. They explain why the thirds cannot reveal the clustered case.
3. **Transfer** (5 min): Students press **New Cases** and solve the new set with no wrong guesses, then sketch the residual plot they would expect after applying each fix.

### Assessment
Shown a residual plot they have not seen before, students name the pattern, state what it implies about the model, and recommend a next step, citing what they see in the plot as evidence.

## References

1. Wikipedia. [Errors and residuals](https://en.wikipedia.org/wiki/Errors_and_residuals).
2. Wikipedia. [Homoscedasticity and heteroscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity).
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 3, Linear Regression.
