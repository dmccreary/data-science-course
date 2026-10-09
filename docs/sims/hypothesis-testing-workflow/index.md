---
title: Hypothesis Testing Workflow
description: Step through the flowchart of a hypothesis test with a coin-flip example, set the number of heads and the significance level, and see which decision the p-value leads to.
image: /sims/hypothesis-testing-workflow/hypothesis-testing-workflow.png
og:image: /sims/hypothesis-testing-workflow/hypothesis-testing-workflow.png
twitter:image: /sims/hypothesis-testing-workflow/hypothesis-testing-workflow.png
social:
   cards: false
quality_score: 0
---

# Hypothesis Testing Workflow

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Hypothesis Testing Workflow MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The flowchart on the left shows the eight steps of a hypothesis test:

1. Ask a research question
2. State the null hypothesis $H_0$ and the alternative $H_1$
3. Choose the significance level $\alpha$
4. Collect data and compute a test statistic
5. Calculate the p-value
6. Decide: is the p-value below $\alpha$?
7. Reject $H_0$, or fail to reject $H_0$
8. Report the results

The panel on the right explains the selected step, applies it to one example, and shows the matching Python code. The example is the one from the chapter: a coin is flipped 100 times to test whether it is fair. $H_0$ says $P(\text{heads}) = 0.5$.

The chart at the bottom of the panel shows how likely each number of heads is when $H_0$ is true. The red bars are the outcomes at least as extreme as the observed count, and their total probability is the p-value. This is the exact two-sided binomial test, the same calculation as `scipy.stats.binomtest`. With 60 heads the p-value is 0.0569, which is not below 0.05, so the test fails to reject $H_0$. With 61 heads it is 0.0352 and the test rejects.

The p-value is the probability of data at least this extreme, calculated on the assumption that $H_0$ is true. It is not the probability that $H_0$ is true. Failing to reject $H_0$ does not prove that the coin is fair. The last step reports the sample proportion and an exact (Clopper-Pearson) confidence interval at level $1 - \alpha$.

## How to Use

1. Press **Next** to move through the steps in order, or click any box in the flowchart. Read the explanation, the coin example, and the code for each step.
2. At step 5, find the red bars in the chart and relate them to the p-value.
3. At step 6, predict the answer before pressing **Next**. The flowchart then highlights the branch that the data lead to.
4. Move the **Heads in 100 flips** slider to find the smallest number of heads above 50 that rejects H₀ at α = 0.05.
5. Change **Significance level α** to 0.01 and to 0.10 and see how the decision for the same data changes.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/hypothesis-testing-workflow/main.html"
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
Probability as a number from 0 to 1, the difference between a population and a sample, and reading a probability distribution.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to apply the steps of a hypothesis test to coin-flip data and decide whether to reject the null hypothesis by comparing the p-value with the significance level.

### Activities

1. **Walk through** (5 min): Students step through all eight steps for 60 heads at α = 0.05 and write the hypotheses, the p-value, and the decision in their own words.
2. **Find the boundary** (5 min): For each α in the menu, students use the slider to find the smallest number of heads above 50 that leads to rejecting H₀ and record the three p-values.
3. **Interpret** (4 min): Students explain why 60 heads does not prove the coin is fair, and name the type of error that has been made if the coin is in fact unfair.

### Assessment
Students are told that a coin landed heads 38 times in 100 flips, with a two-sided p-value of 0.021. They state H₀ and H₁, make the decision at α = 0.05 and at α = 0.01, and write one sentence that interprets the p-value correctly.

## References

1. Wikipedia. [Statistical hypothesis test](https://en.wikipedia.org/wiki/Statistical_hypothesis_test).
2. Wikipedia. [p-value](https://en.wikipedia.org/wiki/P-value).
3. SciPy documentation. [scipy.stats.binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
