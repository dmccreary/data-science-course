---
title: Central Limit Theorem Simulator MicroSim
description: Draw repeated samples from a normal, uniform, right-skewed, or bimodal population and compare the histogram of sample means with the normal curve that the central limit theorem predicts.
image: /sims/central-limit-theorem-simulator-microsim/central-limit-theorem-simulator-microsim.png
og:image: /sims/central-limit-theorem-simulator-microsim/central-limit-theorem-simulator-microsim.png
twitter:image: /sims/central-limit-theorem-simulator-microsim/central-limit-theorem-simulator-microsim.png
social:
   cards: false
quality_score: 0
---

# Central Limit Theorem Simulator MicroSim

<iframe src="main.html" height="587" width="100%" scrolling="no"></iframe>

[Run the Central Limit Theorem Simulator MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The top chart shows a population distribution with its mean $\mu$ and standard deviation $\sigma$. A sample is $n$ values drawn at random from that population. The orange dots are the values of the latest sample, and the dashed line carries their mean down to the bottom chart, where it is counted in a histogram. Both charts share the same axis from 0 to 100.

The central limit theorem says that, for large enough $n$, the sample means are approximately normally distributed with

$$\text{mean} = \mu \qquad \text{standard deviation} = \frac{\sigma}{\sqrt{n}}$$

whatever the shape of the population. The second quantity is called the standard error. The red curve is that normal distribution, scaled to the number of samples taken. The table compares the mean and standard deviation of the sample means actually drawn with the two predicted values.

The MicroSim opens with 1000 samples of size 30 from a uniform population, the same setup as the code in the chapter. Changing the population or the sample size draws a fresh 1000 samples. The random numbers come from a seeded generator, so the same settings always give the same samples. The standard deviation of the sample means is computed with $n - 1$ in the denominator (`ddof=1`).

## How to Use

1. Compare the two charts as they open. The population is flat, but the sample means pile up in a bell shape around 50.
2. Move the **Sample size** slider through 2, 5, 10, 30, and 100. Watch the histogram narrow, and compare the observed SD of the sample means with σ/√n in the table.
3. Choose each **Population** in turn. For the right-skewed and bimodal populations, find the smallest sample size at which the histogram follows the red curve well.
4. Press **Clear**, then **Take 1 Sample** several times to see how one sample becomes one block of the histogram.
5. Press **Start** to keep sampling (up to 5000 samples) and **Pause** to stop.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/central-limit-theorem-simulator-microsim/main.html"
        height="587"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Mean and standard deviation, the difference between a population and a sample, and the shape of the normal distribution.

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to compare the distribution of sample means with the population distribution and analyze how the sample size changes its shape and spread.

### Activities

1. **Observe one sample** (4 min): Students clear the samples, take five single samples, and record each sample mean. They explain why the means vary less than the individual orange dots do.
2. **Compare shapes** (7 min): For each of the four populations, students set n to 2 and then to 30 and describe how well the histogram of sample means follows the red curve.
3. **Test the formula** (5 min): Students fill in a table of n, σ/√n, and the observed SD of the sample means for n = 2, 5, 10, 30, and 100, then work out how much larger n must be to halve the standard error.

### Assessment
Students predict the mean and the standard deviation of the sample means for samples of size 25 from a population with μ = 80 and σ = 20, and explain whether the histogram of those sample means would look normal if the population were strongly skewed.

## References

1. Wikipedia. [Central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem).
2. Wikipedia. [Standard error](https://en.wikipedia.org/wiki/Standard_error).
3. NumPy documentation. [numpy.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html).
