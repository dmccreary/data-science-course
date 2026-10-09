---
title: Normal Distribution Explorer MicroSim
description: Set the mean and standard deviation of a normal distribution and read the intervals that hold 68, 95, and 99.7 percent of the values.
image: /sims/normal-distribution-explorer-microsim/normal-distribution-explorer-microsim.png
og:image: /sims/normal-distribution-explorer-microsim/normal-distribution-explorer-microsim.png
twitter:image: /sims/normal-distribution-explorer-microsim/normal-distribution-explorer-microsim.png
social:
   cards: false
quality_score: 0
---

# Normal Distribution Explorer MicroSim

<iframe src="main.html" height="597" width="100%" scrolling="no"></iframe>

[Run the Normal Distribution Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The blue curve is the normal density

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

drawn on fixed axes, so moving a slider changes the curve and not the scale. The mean $\mu$ sets where the curve is centered. The standard deviation $\sigma$ sets how spread out it is. The total area under the curve is always 1, so a narrower curve has to be taller.

The three shaded bands mark the values within 1, 2, and 3 standard deviations of the mean. The table lists each interval and the area under the curve inside it, calculated from the density by numerical integration: 68.27%, 95.45%, and 99.73%. These percentages are the same for every normal distribution, which is why the 68-95-99.7 rule works for any $\mu$ and $\sigma$.

The yellow panel shows the equation with the current values put in and the height of the peak. Pointing at the plot reads off the z-score $z = (x - \mu)/\sigma$ and the density at that $x$. The density is a height, not a probability. Probabilities are areas under the curve.

The standard deviation slider runs from 8 to 50 so that every curve fits on the fixed vertical axis.

## How to Use

1. Move the **Mean** slider and watch the curve slide left and right without changing shape.
2. Move the **Std. deviation** slider and watch the curve widen and flatten, or narrow and grow taller. Read the three intervals in the table as they change.
3. Choose a preset (IQ scores, test scores, or heights) and use the table to answer questions such as: between which two IQ scores do about 95% of people fall?
4. Press **Pin Curve** to keep the current curve as a dashed outline, then change one slider and compare the two curves. Press **Unpin Curve** to remove it.
5. Point at the plot to read the z-score and the density at any x. Uncheck **68-95-99.7 regions** to see the curve alone.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/normal-distribution-explorer-microsim/main.html"
        height="597"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Mean and standard deviation of a dataset, and reading a histogram.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to use the mean and standard deviation of a normal distribution to find the intervals that contain about 68%, 95%, and 99.7% of the values.

### Activities

1. **Explore** (4 min): Students change each slider on its own and write one sentence for what μ does to the curve and one for what σ does.
2. **Apply the rule** (6 min): Students solve three problems with the sliders: find σ so that about 95% of values fall between 60 and 140 when μ = 100, find the interval that holds about 68% of test scores for μ = 75 and σ = 10, and find the height that is 2 standard deviations above the mean in the heights preset.
3. **Compare** (4 min): Students pin the IQ curve, make a curve with the same mean and twice the standard deviation, and explain why the new peak is half as high.

### Assessment
Without the MicroSim, students state the interval that holds about 95% of the values of a normal distribution with a given mean and standard deviation (for example μ = 500 and σ = 100) and sketch how the curve changes if σ is halved.

## References

1. Wikipedia. [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution).
2. Wikipedia. [68–95–99.7 rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule).
3. NumPy documentation. [numpy.random.normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html).
