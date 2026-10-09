---
title: Lambda Tuning Playground
description: Tune the regularization strength of a degree-10 polynomial model fitted with ridge, lasso, or elastic net, read the training and cross-validation scores and the coefficient sizes, and check the choice against the cross-validation best.
image: /sims/lambda-tuning-playground/lambda-tuning-playground.png
og:image: /sims/lambda-tuning-playground/lambda-tuning-playground.png
twitter:image: /sims/lambda-tuning-playground/lambda-tuning-playground.png
social:
   cards: false
quality_score: 0
---

# Lambda Tuning Playground

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Lambda Tuning Playground MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Fifteen points follow the chapter's cubic curve (dashed) plus noise. The model is a degree-10 polynomial, so it has 10 coefficients and an intercept for 15 points and can bend far more than the data justify. The slider sets $\lambda$, the strength of the penalty that holds it back.

Four linked views update together:

- **Fit**: the fitted curve at the current $\lambda$. It turns red when $\lambda$ is too small, green in the sweet spot, and orange when $\lambda$ is too large.
- **Scores**: training $R^2$, 5-fold cross-validation $R^2$, the number of non-zero coefficients, and a verdict. The panel title is the scikit-learn call that gives the same model.
- **Score against λ**: both scores for every $\lambda$ from 0.0001 to 1000. Training $R^2$ is highest at the far left. The star marks the highest cross-validation $R^2$.
- **Coefficient size**: the size of each standardized coefficient on a log scale. A coefficient that lasso or elastic net has set to exactly 0 has no bar.

Here $\lambda$ is scikit-learn's `alpha`. The pipeline rescales $x$ to $t = (x - 5)/5$, builds the powers $t, t^2, \dots, t^{10}$, standardizes them, and fits `Ridge(alpha=λ)`, `Lasso(alpha=λ)`, or `ElasticNet(alpha=λ, l1_ratio=0.5)`. The scaler is refit inside each fold. Coefficients and scores were checked against scikit-learn 1.8. The best $\lambda$ is not comparable between methods: the penalties differ, and scikit-learn divides the squared error by $2n$ for lasso and elastic net but not for ridge.

The test $R^2$ (200 new points) appears only when $\lambda$ is at the cross-validation best. A test set is for the final check, not for tuning.

Not included: sliders for the polynomial degree and for `l1_ratio` (fixed at 10 and 0.5), and an animated search (**Find Best λ** moves straight to the answer).

## How to Use

1. Read the opening view. λ is very small, the curve wiggles through the points, and cross-validation R² is far below training R².
2. Drag the **λ (alpha)** slider to the right. Watch the curve smooth out, the bars shrink, and the green cross-validation dot climb toward the star.
3. Keep going past the star until the verdict changes to an underfitting warning and the curve flattens toward a horizontal line.
4. Bring λ back to the sweet spot by eye, then press **Find Best λ** to see how close you were. Read the test R² that appears.
5. Choose **Lasso** and then **Elastic Net** and tune each one. Count the non-zero coefficients at the best λ for each method.
6. Press **New Data** and tune again. Compare the best λ with the one you found before.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/lambda-tuning-playground/main.html"
        height="582"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Polynomial regression, overfitting and underfitting, ridge and lasso penalties, and k-fold cross-validation.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to use cross-validation scores, the fitted curve, and the coefficient sizes to tune the regularization strength of ridge, lasso, and elastic net models and to recognize a lambda that is too small or too large.

### Activities

1. **Tune by eye** (5 min): With ridge selected, students move λ until they judge the fit to be best, write down their λ and its cross-validation R², then press **Find Best λ** and record the best λ, its cross-validation R², and the test R².
2. **Read the warning signs** (5 min): Students record training R², cross-validation R², and the largest coefficient size at λ = 0.0001, at the best λ, and at λ = 100, and describe in one sentence each what the curve looks like.
3. **Compare methods** (6 min): Students find the best λ for lasso and elastic net, record the number of non-zero coefficients and which powers survive, and explain why the best λ is not the same number for the three methods.

### Assessment
Students are shown a score-against-λ chart with the current λ marked far to the left of the star, state whether the model is overfitting or underfitting, say what they expect the fitted curve and the coefficient sizes to look like, and name the direction to move λ.

## References

1. scikit-learn documentation. [sklearn.linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html).
2. scikit-learn documentation. [sklearn.linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).
3. Wikipedia. [Cross-validation (statistics)](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).
4. James, G., Witten, D., Hastie, T., and Tibshirani, R. *An Introduction to Statistical Learning*. Springer. Chapter 6, Linear Model Selection and Regularization.
