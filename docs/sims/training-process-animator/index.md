---
title: Training Process Animator
description: Step through gradient descent as it fits a line to study-hours data, and watch the weight, the bias, the gradients, and the training and validation error change at every iteration.
image: /sims/training-process-animator/training-process-animator.png
og:image: /sims/training-process-animator/training-process-animator.png
twitter:image: /sims/training-process-animator/training-process-animator.png
social:
   cards: false
quality_score: 0
---

# Training Process Animator

<iframe src="main.html" height="537" width="100%" scrolling="no"></iframe>

[Run the Training Process Animator MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Each blue dot is one of 30 training rows: a student's study hours (standardized) and quiz score. The ten orange rings are validation rows that the training loop never uses. The model is a line, $\hat{y} = wz + b$, and training repeats the loop from the chapter:

1. **Predict** with the current $w$ and $b$.
2. **Measure the error**: $J = \frac{1}{n}\sum(\hat{y}_i - y_i)^2$, the mean squared error on the training rows.
3. **Compute the gradients**: $\frac{\partial J}{\partial w} = \frac{2}{n}\sum(\hat{y}_i - y_i)z_i$ and $\frac{\partial J}{\partial b} = \frac{2}{n}\sum(\hat{y}_i - y_i)$.
4. **Update**: $w \leftarrow w - \eta\,\frac{\partial J}{\partial w}$ and $b \leftarrow b - \eta\,\frac{\partial J}{\partial b}$.

Every press of **Step** runs that loop once on all 30 training rows. The line moves, the faint gray lines show where it was on earlier iterations, and the thin vertical segments are the residuals being squared. The line turns from red to green as its error approaches the lowest error possible. The chart plots the training MSE (blue) and the validation MSE (orange) after every iteration.

Study hours are standardized with the training mean and standard deviation, $z = (\text{hours} - \text{mean})/\text{sd}$, as the chapter's pipeline does with `StandardScaler`. Training stops when the gradient is nearly zero, and the status line then shows the answer from the least-squares formula for comparison. It also stops after 200 iterations, or if the error passes 1,000,000.

## How to Use

1. Read the starting state. With $w = 0$ and $b = 0$ the line lies along the bottom of the plot and the error is large.
2. Press **Step** several times. After each press read the new $w$, $b$, and training MSE, and note how much the error dropped.
3. Press **Start** to let training run until it converges, then compare the final $w$ and $b$ with the least-squares values in the status line.
4. Press **Reset**, set the **Learning rate** to 0.02, and run again. Then try 0.5, 0.9, and 1.05.
5. Choose a different starting line from the menu and check that training ends at the same $w$ and $b$.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/training-process-animator/main.html"
        height="537"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The equation of a line, residuals and mean squared error, and the gradient descent update rule.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how each iteration of training uses the error and its gradients to adjust a model's parameters, and predict how the learning rate changes the way training proceeds.

### Activities

1. **Trace the loop** (5 min): Students press Step five times from the default start and record w, b, and the training MSE in a table, then describe how the size of the change in w and b relates to the size of the gradients.
2. **Learning rate** (5 min): Students run training to the end at learning rates 0.02, 0.1, 0.5, 0.9, and 1.05 and record the number of iterations and the outcome for each.
3. **Same destination** (4 min): Students train from each of the three starting lines and compare the final w, b, training MSE, and validation MSE.

### Assessment
Students are given w, b, both gradients, and a learning rate for one iteration, compute the updated w and b, and explain in two sentences why the validation MSE is tracked even though it is never used in the update.

## References

1. Wikipedia. [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).
2. scikit-learn documentation. [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html) and [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
3. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly. Chapter 4, Training Models.
