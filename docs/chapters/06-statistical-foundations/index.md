---
title: Statistical Foundations
description: Master the mathematical language that powers all of data science
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

# Statistical Foundations

## Summary

This chapter establishes the statistical foundation essential for data science and machine learning. Students will learn descriptive statistics (mean, median, mode, variance, standard deviation), understand distributions and probability, and explore sampling concepts. The chapter covers hypothesis testing, confidence intervals, and measures of association including correlation and covariance. By the end of this chapter, students will be able to summarize datasets statistically, understand the relationship between variables, and make probabilistic inferences.

## Concepts Covered

This chapter covers the following 30 concepts from the learning graph:

1. Descriptive Statistics
2. Mean
3. Median
4. Mode
5. Range
6. Variance
7. Standard Deviation
8. Quartiles
9. Percentiles
10. Interquartile Range
11. Skewness
12. Kurtosis
13. Distribution
14. Normal Distribution
15. Probability
16. Random Variables
17. Expected Value
18. Sample
19. Population
20. Sampling
21. Central Limit Theorem
22. Confidence Interval
23. Hypothesis Testing
24. P-Value
25. Statistical Significance
26. Correlation
27. Covariance
28. Pearson Correlation
29. Spearman Correlation
30. Correlation Matrix

## Prerequisites

This chapter builds on concepts from:

- [Chapter 1: Introduction to Data Science](../01-intro-to-data-science/index.md)
- [Chapter 5: Data Visualization with Matplotlib](../05-data-visualization/index.md)

---

## The Language of Uncertainty

Here's a secret that surprises most people: the world runs on statistics. Every medical treatment you take was proven effective through statistics. Every recommendation Netflix makes uses statistical patterns. Every weather forecast is a statistical prediction. Every poll predicting election results? Statistics.

Statistics is the mathematical language for dealing with uncertainty and variation. And in a world drowning in data, those who speak this language fluently have an incredible advantage.

This chapter gives you that fluency. You'll learn to summarize thousands of data points with a handful of numbers, understand how data is distributed, measure relationships between variables, and make confident statements about populations based on samples. These aren't just academic exercises—they're the core tools that power everything from A/B testing at tech companies to clinical trials for new medicines.

Fair warning: this chapter is packed with concepts. But don't worry—each one builds on the last, and by the end, you'll have a complete statistical toolkit. Let's start with the basics.

## Descriptive Statistics: Summarizing Data

**Descriptive statistics** are numbers that summarize and describe a dataset. Instead of looking at thousands of individual values, descriptive statistics give you the big picture in just a few numbers.

Think of it like this: if someone asks "How tall are the students in your school?", you don't list every student's height. You say something like "The average is 5'7", ranging from 4'11" to 6'4"." That's descriptive statistics in action.

There are two main categories:

- **Measures of central tendency**: Where is the "center" of the data? (mean, median, mode)
- **Measures of spread**: How spread out is the data? (range, variance, standard deviation)

```python
import pandas as pd
import numpy as np

# Sample dataset: test scores
scores = [72, 85, 90, 78, 88, 92, 76, 84, 89, 95, 70, 82, 87, 91, 79]

# Quick descriptive statistics with pandas
df = pd.DataFrame({'score': scores})
print(df.describe())
```

Output:
```
            score
count   15.000000
mean    83.866667
std      7.577722
min     70.000000
25%     78.000000
50%     85.000000
75%     90.000000
max     95.000000
```

In one command, you get count, mean, standard deviation, min, max, and quartiles. Let's understand each of these.

## Measures of Central Tendency

### Mean: The Arithmetic Average

The **mean** is what most people call "the average." Add up all the values and divide by how many there are:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + ... + x_n}{n}$$

```python
scores = [72, 85, 90, 78, 88]

# Calculate mean
mean_score = sum(scores) / len(scores)
print(f"Mean: {mean_score}")  # Output: 82.6

# Or use numpy/pandas
import numpy as np
print(f"Mean: {np.mean(scores)}")  # Output: 82.6
```

The mean is intuitive and mathematically convenient, but it has a weakness: it's sensitive to outliers.

```python
# What happens with an outlier?
scores_with_outlier = [72, 85, 90, 78, 88, 250]  # Someone scored 250? (data error)
print(f"Mean with outlier: {np.mean(scores_with_outlier)}")  # Output: 110.5 (misleading!)
```

One extreme value pulled the mean way up. For this reason, we sometimes use the median instead.

### Median: The Middle Value

The **median** is the middle value when data is sorted. Half the values are below it, half are above.

```python
scores = [72, 85, 90, 78, 88]

# Sort: [72, 78, 85, 88, 90]
# Middle value: 85

print(f"Median: {np.median(scores)}")  # Output: 85.0

# With the outlier
scores_with_outlier = [72, 85, 90, 78, 88, 250]
# Sort: [72, 78, 85, 88, 90, 250]
# Middle: average of 85 and 88 = 86.5

print(f"Median with outlier: {np.median(scores_with_outlier)}")  # Output: 86.5 (much more reasonable!)
```

The median is **robust** to outliers—extreme values don't affect it much. Use median when your data might have outliers or is skewed.

### Mode: The Most Common Value

The **mode** is the value that appears most frequently. It's the only measure of central tendency that works for categorical data.

```python
from scipy import stats

# Numeric mode
test_scores = [85, 90, 85, 78, 85, 92, 90, 85]
print(f"Mode: {stats.mode(test_scores, keepdims=True).mode[0]}")  # Output: 85

# Categorical mode
favorite_colors = ['blue', 'red', 'blue', 'green', 'blue', 'red']
print(f"Mode: {stats.mode(favorite_colors, keepdims=True).mode[0]}")  # Output: blue
```

Data can have multiple modes (bimodal, multimodal) or no mode at all if every value appears once.

| Measure | Best For | Sensitive to Outliers? |
|---------|----------|------------------------|
| Mean | Symmetric data, further calculations | Yes |
| Median | Skewed data, outliers present | No |
| Mode | Categorical data, finding most common | No |

#### Diagram: Central Tendency Comparison MicroSim

<details markdown="1">
    <summary>Mean, Median, Mode Interactive Explorer</summary>
    Type: microsim

    Bloom Taxonomy: Understand (L2)

    Learning Objective: Help students visualize how mean, median, and mode respond differently to data changes and outliers

    Canvas layout (800x500px):
    - Top (800x300): Interactive histogram with draggable data points
    - Bottom (800x200): Statistics display and controls

    Visual elements:
    - Histogram showing data distribution
    - Vertical lines for mean (red), median (green), mode (blue)
    - Individual data points displayed as draggable circles below histogram
    - Statistics panel showing current values

    Interactive controls:
    - Draggable data points: Click and drag any point to change its value
    - "Add Point" button: Add new data point
    - "Add Outlier" button: Add extreme value
    - "Remove Point" button: Click to remove
    - "Reset" button: Return to original dataset
    - Dropdown: Preset distributions (symmetric, left-skewed, right-skewed, bimodal)

    Initial dataset:
    - 20 points normally distributed around 50

    Behavior:
    - All three measures update in real-time as points are dragged
    - Visual indication when mean and median diverge significantly
    - Highlight which measure is "best" for current distribution
    - Animation when adding outliers to show mean shifting

    Educational annotations:
    - "Notice how the mean moves toward the outlier"
    - "The median stays stable!"
    - "Mode shows the peak of the distribution"

    Challenge tasks:
    - "Make the mean equal to the median"
    - "Create a distribution where mode ≠ median ≠ mean"
    - "Add an outlier that changes the mean by at least 10"

    Visual style: Clean statistical visualization with color-coded measures

    Implementation: p5.js with real-time statistical calculations
</details>

## Measures of Spread

Knowing the center isn't enough—you also need to know how spread out the data is. Two datasets can have the same mean but very different spreads.

### Range: Simplest Spread Measure

The **range** is simply the difference between the maximum and minimum values:

$$\text{Range} = \text{Max} - \text{Min}$$

```python
scores = [72, 85, 90, 78, 88, 92, 76]

range_value = max(scores) - min(scores)
print(f"Range: {range_value}")  # Output: 20
```

Range is easy to understand but has limitations: it only uses two values and is very sensitive to outliers.

### Variance: Average Squared Deviation

**Variance** measures how far each value is from the mean, on average. It squares the deviations (so negatives don't cancel positives):

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

```python
import numpy as np

scores = [72, 85, 90, 78, 88]
mean = np.mean(scores)

# Calculate variance step by step
deviations = [(x - mean) for x in scores]  # How far from mean
squared_deviations = [d**2 for d in deviations]  # Square them
variance = sum(squared_deviations) / len(scores)  # Average

print(f"Variance: {variance}")  # Output: 41.04

# Or simply:
print(f"Variance: {np.var(scores)}")  # Output: 41.04
```

The problem with variance? The units are squared. If your data is in meters, variance is in meters². That's hard to interpret.

### Standard Deviation: The Useful Spread Measure

**Standard deviation** is the square root of variance, bringing us back to the original units:

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

```python
import numpy as np

scores = [72, 85, 90, 78, 88]

std_dev = np.std(scores)
print(f"Standard Deviation: {std_dev:.2f}")  # Output: 6.41
```

Standard deviation tells you how much values typically deviate from the mean. In a normal distribution:

- ~68% of data falls within 1 standard deviation of the mean
- ~95% falls within 2 standard deviations
- ~99.7% falls within 3 standard deviations

This is called the **68-95-99.7 rule** (or empirical rule).

!!! tip "Population vs Sample"
    When calculating variance/standard deviation for a **sample** (subset of data), divide by $n-1$ instead of $n$. This corrects for bias. Use `np.var(data, ddof=1)` or `np.std(data, ddof=1)` for sample statistics.

```python
# Population standard deviation (divide by n)
np.std(scores, ddof=0)

# Sample standard deviation (divide by n-1) - use this most of the time!
np.std(scores, ddof=1)
```

## Quartiles, Percentiles, and IQR

### Quartiles: Dividing Data into Fourths

**Quartiles** divide sorted data into four equal parts:

- **Q1 (25th percentile)**: 25% of data is below this value
- **Q2 (50th percentile)**: The median—50% below, 50% above
- **Q3 (75th percentile)**: 75% of data is below this value

```python
import numpy as np

scores = [72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98]

q1 = np.percentile(scores, 25)
q2 = np.percentile(scores, 50)  # Same as median
q3 = np.percentile(scores, 75)

print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")  # Output: Q1: 78.5, Q2: 85.0, Q3: 92.5
```

### Percentiles: Any Division You Want

**Percentiles** generalize quartiles—the Pth percentile is the value below which P% of the data falls.

```python
# What score puts you in the top 10%?
top_10_cutoff = np.percentile(scores, 90)
print(f"90th percentile: {top_10_cutoff}")

# What percentile is a score of 85?
# Use scipy for this
from scipy import stats
percentile_of_85 = stats.percentileofscore(scores, 85)
print(f"85 is at the {percentile_of_85}th percentile")
```

### Interquartile Range (IQR)

The **interquartile range** is the range of the middle 50% of data:

$$\text{IQR} = Q3 - Q1$$

```python
iqr = q3 - q1
print(f"IQR: {iqr}")  # Output: 14.0

# Or use scipy
from scipy.stats import iqr as calc_iqr
print(f"IQR: {calc_iqr(scores)}")
```

IQR is robust to outliers and is used to detect them: any value below $Q1 - 1.5 \times IQR$ or above $Q3 + 1.5 \times IQR$ is considered an outlier.

```python
# Outlier detection using IQR
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = [x for x in scores if x < lower_fence or x > upper_fence]
print(f"Outliers: {outliers}")
```

#### Diagram: Box Plot Anatomy

<details markdown="1">
    <summary>Interactive Box Plot Anatomy</summary>
    Type: infographic

    Bloom Taxonomy: Remember (L1)

    Learning Objective: Help students identify and remember the components of a box plot

    Purpose: Visual breakdown of box plot structure with labeled components

    Layout: Central box plot with callouts pointing to each component

    Main visual: A horizontal box plot with sample data showing:
    - Whisker extending left to minimum (non-outlier)
    - Box from Q1 to Q3
    - Median line inside box
    - Whisker extending right to maximum (non-outlier)
    - Two outlier points beyond whiskers

    Callouts (numbered with leader lines):

    1. MINIMUM (pointing to left whisker end)
       - "Smallest non-outlier value"
       - "= Q1 - 1.5×IQR or actual min, whichever is larger"
       - Color: Blue

    2. Q1 / FIRST QUARTILE (pointing to left edge of box)
       - "25% of data below this"
       - "Left edge of box"
       - Color: Green

    3. MEDIAN / Q2 (pointing to line inside box)
       - "50% of data below this"
       - "Center line in box"
       - Color: Red

    4. Q3 / THIRD QUARTILE (pointing to right edge of box)
       - "75% of data below this"
       - "Right edge of box"
       - Color: Green

    5. MAXIMUM (pointing to right whisker end)
       - "Largest non-outlier value"
       - "= Q3 + 1.5×IQR or actual max, whichever is smaller"
       - Color: Blue

    6. IQR (bracket spanning the box)
       - "Interquartile Range = Q3 - Q1"
       - "Contains middle 50% of data"
       - Color: Orange

    7. OUTLIERS (pointing to dots beyond whiskers)
       - "Values beyond 1.5×IQR from box"
       - "Shown as individual points"
       - Color: Purple

    Bottom section: "What box plots tell you at a glance"
    - Center (median position)
    - Spread (box width)
    - Symmetry (median position within box)
    - Outliers (individual points)

    Interactive elements:
    - Hover over each component to highlight it
    - Click to see formula or code to calculate
    - Toggle between horizontal and vertical orientation

    Implementation: SVG with CSS hover effects and JavaScript interactivity
</details>

## Distribution Shape: Skewness and Kurtosis

### Skewness: Leaning Left or Right

**Skewness** measures asymmetry in a distribution:

- **Negative skew (left-skewed)**: Tail extends to the left; mean < median
- **Zero skew**: Symmetric; mean ≈ median
- **Positive skew (right-skewed)**: Tail extends to the right; mean > median

```python
from scipy.stats import skew

# Symmetric data
symmetric = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Symmetric skewness: {skew(symmetric):.3f}")  # Close to 0

# Right-skewed (like income data)
right_skewed = [1, 2, 2, 3, 3, 3, 4, 4, 5, 10, 15, 20]
print(f"Right-skewed: {skew(right_skewed):.3f}")  # Positive

# Left-skewed
left_skewed = [1, 5, 10, 15, 16, 16, 17, 17, 17, 18, 18, 19]
print(f"Left-skewed: {skew(left_skewed):.3f}")  # Negative
```

Real-world examples:

- **Right-skewed**: Income, house prices, social media followers
- **Left-skewed**: Age at death in developed countries, exam scores (if test is easy)

### Kurtosis: Tails and Peaks

**Kurtosis** measures the "tailedness" of a distribution—how much data is in the extreme tails versus the center:

- **Positive kurtosis (leptokurtic)**: Heavy tails, sharp peak, more outliers
- **Zero kurtosis (mesokurtic)**: Normal distribution
- **Negative kurtosis (platykurtic)**: Light tails, flat peak, fewer outliers

```python
from scipy.stats import kurtosis

# Normal distribution has kurtosis ≈ 0 (with Fisher's definition)
normal_data = np.random.normal(0, 1, 10000)
print(f"Normal kurtosis: {kurtosis(normal_data):.3f}")  # Close to 0

# Heavy tails (more extreme values)
heavy_tails = np.concatenate([np.random.normal(0, 1, 9000),
                              np.random.normal(0, 5, 1000)])
print(f"Heavy tails kurtosis: {kurtosis(heavy_tails):.3f}")  # Positive
```

| Skewness | Distribution Shape | Mean vs Median |
|----------|-------------------|----------------|
| Negative | Left tail longer | Mean < Median |
| Zero | Symmetric | Mean ≈ Median |
| Positive | Right tail longer | Mean > Median |

## Understanding Distributions

A **distribution** describes how values in a dataset are spread across different possible values. It shows the frequency or probability of each value occurring.

### The Normal Distribution

The **normal distribution** (also called Gaussian or bell curve) is the most important distribution in statistics. It appears everywhere:

- Heights of people
- Measurement errors
- Test scores (often)
- Many natural phenomena

```python
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Generate normal distribution
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=10000)  # Mean=100, StdDev=15

# Interactive histogram with Plotly
fig = px.histogram(data, nbins=50, title='Normal Distribution (μ=100, σ=15)')
fig.update_layout(
    xaxis_title='Value',
    yaxis_title='Frequency',
    showlegend=False
)
fig.show()
```

Key properties of normal distributions:

- Symmetric around the mean
- Mean = Median = Mode
- Defined by two parameters: mean (μ) and standard deviation (σ)
- The 68-95-99.7 rule applies

```python
# Visualize the 68-95-99.7 rule
mean, std = 100, 15

within_1_std = np.sum((data >= mean - std) & (data <= mean + std)) / len(data)
within_2_std = np.sum((data >= mean - 2*std) & (data <= mean + 2*std)) / len(data)
within_3_std = np.sum((data >= mean - 3*std) & (data <= mean + 3*std)) / len(data)

print(f"Within 1 std: {within_1_std:.1%}")  # ~68%
print(f"Within 2 std: {within_2_std:.1%}")  # ~95%
print(f"Within 3 std: {within_3_std:.1%}")  # ~99.7%
```

#### Diagram: Normal Distribution Explorer MicroSim

<details markdown="1">
    <summary>Interactive Normal Distribution Explorer</summary>
    Type: microsim

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Let students manipulate mean and standard deviation to understand how they affect the normal distribution shape

    Canvas layout (850x550px):
    - Main area (850x400): Interactive normal distribution plot
    - Control panel (850x150): Sliders and statistics

    Visual elements:
    - Smooth normal distribution curve
    - Shaded regions showing 1σ, 2σ, 3σ areas
    - Vertical line at mean
    - Axis labels and tick marks
    - Current μ and σ displayed prominently

    Interactive controls:
    - Slider: Mean (μ) range: 0 to 200, default: 100
    - Slider: Standard Deviation (σ) range: 1 to 50, default: 15
    - Toggle: Show 68-95-99.7 regions
    - Toggle: Show probability density values
    - Button: "Add second distribution" (for comparison)
    - Dropdown: Preset examples (IQ scores, heights, test scores)

    Display panels:
    - Probability within 1σ: 68.27%
    - Probability within 2σ: 95.45%
    - Probability within 3σ: 99.73%
    - Current curve equation

    Behavior:
    - Curve updates smoothly as sliders move
    - Shaded regions resize with σ changes
    - Curve shifts horizontally with μ changes
    - Comparison mode overlays two distributions

    Educational annotations:
    - "Larger σ = wider, flatter curve"
    - "Smaller σ = narrower, taller curve"
    - "μ shifts the center, σ changes the spread"

    Challenge tasks:
    - "Set parameters to match IQ distribution (μ=100, σ=15)"
    - "What σ makes 95% fall between 60 and 140?"
    - "Compare two distributions: same mean, different spread"

    Visual style: Clean mathematical visualization with Plotly-like aesthetics

    Implementation: p5.js or Plotly.js with real-time updates
</details>

## Probability Fundamentals

**Probability** is the mathematical framework for quantifying uncertainty. It assigns a number between 0 and 1 to events:

- **P = 0**: Impossible
- **P = 1**: Certain
- **P = 0.5**: Equal chance of happening or not

```python
# Probability of rolling a 6 on a fair die
p_roll_6 = 1 / 6
print(f"P(roll 6) = {p_roll_6:.4f}")  # 0.1667

# Probability of flipping heads
p_heads = 1 / 2
print(f"P(heads) = {p_heads}")  # 0.5
```

### Random Variables and Expected Value

A **random variable** is a variable whose value depends on random outcomes. It can be:

- **Discrete**: Takes specific values (dice roll: 1, 2, 3, 4, 5, 6)
- **Continuous**: Takes any value in a range (height: 5.5, 5.51, 5.512...)

The **expected value** (E[X]) is the long-run average—what you'd expect on average over many repetitions:

$$E[X] = \sum_{i} x_i \cdot P(x_i)$$

```python
# Expected value of a fair die roll
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

expected_value = sum(x * p for x, p in zip(outcomes, probabilities))
print(f"Expected value of die roll: {expected_value:.2f}")  # 3.5

# Verify with simulation
rolls = np.random.randint(1, 7, 100000)
print(f"Simulated average: {rolls.mean():.2f}")  # ~3.5
```

The expected value of a fair die is 3.5—you can never actually roll 3.5, but it's the average outcome over time.

## Sampling: From Population to Sample

### Population vs Sample

A **population** is the entire group you want to study. A **sample** is a subset you actually measure.

- **Population**: All high school students in the US
- **Sample**: 1,000 randomly selected high school students

**Sampling** is the process of selecting a sample from a population. Good sampling is crucial—a biased sample leads to wrong conclusions.

```python
import numpy as np

# Population: All test scores (imagine this is millions of values)
np.random.seed(42)
population = np.random.normal(75, 10, 1000000)  # Mean=75, StdDev=10

# Sample: We can only survey 100 students
sample = np.random.choice(population, size=100, replace=False)

print(f"Population mean: {population.mean():.2f}")
print(f"Sample mean: {sample.mean():.2f}")
print(f"Difference: {abs(population.mean() - sample.mean()):.2f}")
```

The sample mean estimates the population mean, but there's always some error. That's where the Central Limit Theorem helps.

### The Central Limit Theorem (CLT)

The **Central Limit Theorem** is one of the most important results in statistics. It says:

> When you take many random samples from ANY population and calculate the mean of each sample, those sample means will be approximately normally distributed—regardless of the original population's distribution.

This is magical because:

1. It works for any population shape (uniform, skewed, bimodal...)
2. The distribution of sample means gets more normal as sample size increases
3. It lets us make probability statements about sample means

```python
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

np.random.seed(42)

# Start with a NON-normal population (uniform distribution)
population = np.random.uniform(0, 100, 100000)

# Take many samples and calculate their means
sample_size = 30
num_samples = 1000

sample_means = [np.random.choice(population, size=sample_size).mean()
                for _ in range(num_samples)]

# The sample means are normally distributed!
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Original Population (Uniform)',
                                   'Distribution of Sample Means (Normal!)'])

fig.add_trace(go.Histogram(x=population, nbinsx=50, name='Population'),
              row=1, col=1)
fig.add_trace(go.Histogram(x=sample_means, nbinsx=30, name='Sample Means'),
              row=1, col=2)

fig.update_layout(title='Central Limit Theorem in Action')
fig.show()
```

#### Diagram: Central Limit Theorem Simulator MicroSim

<details markdown="1">
    <summary>Central Limit Theorem Interactive Demonstration</summary>
    Type: microsim

    Bloom Taxonomy: Analyze (L4)

    Learning Objective: Help students understand the CLT by visualizing how sample means become normally distributed regardless of population shape

    Canvas layout (900x600px):
    - Left panel (450x600): Population distribution
    - Right panel (450x600): Distribution of sample means
    - Bottom strip (900x100): Controls

    Visual elements:
    - Left: Histogram of original population
    - Right: Histogram of sample means (builds up over time)
    - Normal curve overlay on right panel
    - Running statistics display

    Interactive controls:
    - Dropdown: Population distribution type
      - Normal
      - Uniform
      - Exponential (right-skewed)
      - Bimodal
      - Custom (draw your own!)
    - Slider: Sample size (5, 10, 30, 50, 100)
    - Button: "Take One Sample" (animated)
    - Button: "Take 100 Samples" (fast)
    - Button: "Take 1000 Samples" (bulk)
    - Button: "Reset"
    - Slider: Animation speed

    Display panels:
    - Population mean and std
    - Mean of sample means
    - Std of sample means (should ≈ σ/√n)
    - Number of samples taken

    Behavior:
    - "Take One Sample" animates: highlight sample from population, calculate mean, add to right histogram
    - Sample means histogram builds up gradually
    - Normal curve overlay adjusts to fit data
    - Show how larger sample sizes make sample means distribution narrower

    Educational annotations:
    - "Notice: Even though population is [skewed/uniform], sample means are normal!"
    - "Larger samples → narrower distribution of means"
    - "Standard error = σ/√n"

    Challenge tasks:
    - "Which sample size makes sample means most normal?"
    - "Predict the std of sample means for n=100"
    - "Try the most extreme distribution—CLT still works!"

    Visual style: Side-by-side comparison with animation

    Implementation: p5.js with smooth animations and Plotly for histograms
</details>

## Confidence Intervals: Quantifying Uncertainty

A **confidence interval** gives a range that likely contains the true population parameter. Instead of saying "the average is 75," you say "I'm 95% confident the average is between 72 and 78."

```python
import numpy as np
from scipy import stats

# Sample data
sample = np.random.normal(75, 10, 100)

# Calculate 95% confidence interval for the mean
confidence = 0.95
mean = sample.mean()
se = stats.sem(sample)  # Standard error
ci = stats.t.interval(confidence, len(sample)-1, loc=mean, scale=se)

print(f"Sample mean: {mean:.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
```

The interpretation: If we repeated this sampling process many times, 95% of the confidence intervals we calculate would contain the true population mean.

!!! warning "Common Misconception"
    A 95% confidence interval does NOT mean "there's a 95% probability the true value is in this range." The true value either is or isn't in the range—we just don't know which. The 95% refers to the long-run success rate of the method.

## Hypothesis Testing: Making Decisions with Data

**Hypothesis testing** is a framework for making decisions based on data. You start with a hypothesis and use data to evaluate whether the evidence supports it.

### The Process

1. **State the null hypothesis (H₀)**: The default assumption (usually "no effect" or "no difference")
2. **State the alternative hypothesis (H₁)**: What you're testing for
3. **Collect data and calculate a test statistic**
4. **Calculate the p-value**
5. **Make a decision** based on significance level

### P-Value: The Evidence Measure

The **p-value** is the probability of seeing results at least as extreme as yours, assuming the null hypothesis is true.

- **Small p-value** (< 0.05): Evidence against H₀; reject it
- **Large p-value** (≥ 0.05): Not enough evidence; fail to reject H₀

```python
from scipy import stats

# Example: Testing if a coin is fair
# Flip 100 times, get 60 heads
# H₀: coin is fair (p = 0.5)
# H₁: coin is not fair (p ≠ 0.5)

n_flips = 100
n_heads = 60

# Binomial test
result = stats.binomtest(n_heads, n_flips, p=0.5, alternative='two-sided')
print(f"P-value: {result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("Reject H₀: The coin appears to be unfair")
else:
    print("Fail to reject H₀: No evidence the coin is unfair")
```

### Statistical Significance

**Statistical significance** means the p-value is below a predetermined threshold (usually 0.05). It indicates the result is unlikely to have occurred by chance alone.

| P-value | Interpretation |
|---------|----------------|
| < 0.001 | Very strong evidence against H₀ |
| < 0.01 | Strong evidence against H₀ |
| < 0.05 | Evidence against H₀ (significant) |
| ≥ 0.05 | Insufficient evidence against H₀ |

!!! tip "Statistical vs Practical Significance"
    A result can be statistically significant but practically meaningless. If a drug reduces blood pressure by 0.1 mmHg and it's significant with p < 0.001, so what? That's too small to matter clinically. Always consider effect size, not just p-values.

#### Diagram: Hypothesis Testing Workflow

<details markdown="1">
    <summary>Hypothesis Testing Decision Flowchart</summary>
    Type: workflow

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Guide students through the hypothesis testing process step by step

    Purpose: Visual decision tree for conducting hypothesis tests

    Visual style: Vertical flowchart with decision diamonds and process rectangles

    Steps (top to bottom):

    1. START: "Research Question"
       Hover text: "What are you trying to determine?"
       Color: Blue

    2. PROCESS: "State Hypotheses"
       - H₀: Null hypothesis (no effect/difference)
       - H₁: Alternative hypothesis (effect exists)
       Hover text: "H₀ is what you're trying to disprove"
       Color: Green

    3. PROCESS: "Choose Significance Level (α)"
       - Usually α = 0.05
       Hover text: "This is your threshold for 'unlikely'"
       Color: Green

    4. PROCESS: "Collect Data & Calculate Test Statistic"
       Hover text: "t-test, chi-square, etc. depending on your data"
       Color: Green

    5. PROCESS: "Calculate P-value"
       Hover text: "Probability of seeing this result if H₀ is true"
       Color: Orange

    6. DECISION: "Is p-value < α?"
       Color: Yellow

    7a. YES PATH: "Reject H₀"
       - "Results are statistically significant"
       - "Evidence supports H₁"
       Hover text: "But also check effect size!"
       Color: Red

    7b. NO PATH: "Fail to Reject H₀"
       - "Results are not statistically significant"
       - "Insufficient evidence for H₁"
       Hover text: "This doesn't prove H₀ is true!"
       Color: Gray

    8. END: "Report Results"
       - Include: test statistic, p-value, effect size, confidence interval
       Color: Blue

    Side annotations:
    - "Type I Error (α): Rejecting H₀ when it's actually true"
    - "Type II Error (β): Failing to reject H₀ when it's actually false"

    Interactive elements:
    - Hover over each step for detailed explanation
    - Click to see Python code for that step
    - Example problems that walk through the flowchart

    Implementation: SVG with JavaScript interactivity
</details>

## Correlation: Measuring Relationships

**Correlation** measures the strength and direction of a linear relationship between two variables. It ranges from -1 to +1:

- **+1**: Perfect positive correlation (as X increases, Y increases)
- **0**: No linear correlation
- **-1**: Perfect negative correlation (as X increases, Y decreases)

### Covariance: The Building Block

**Covariance** measures how two variables change together:

$$\text{Cov}(X, Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

```python
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

covariance = np.cov(x, y)[0, 1]
print(f"Covariance: {covariance:.2f}")
```

The problem with covariance: it's affected by the scale of the variables. Covariance between height in inches and weight in pounds will be different from height in centimeters and weight in kilograms.

### Pearson Correlation: The Standard Measure

**Pearson correlation** standardizes covariance to a -1 to +1 scale:

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

```python
import numpy as np
from scipy import stats

# Example: Study hours vs exam score
study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores = [50, 55, 65, 70, 72, 80, 85, 90]

# Calculate Pearson correlation
r, p_value = stats.pearsonr(study_hours, exam_scores)
print(f"Pearson r: {r:.3f}")
print(f"P-value: {p_value:.4f}")
```

Pearson correlation assumes:

- Linear relationship
- Both variables are continuous
- Data is normally distributed (roughly)

### Spearman Correlation: For Non-Linear Relationships

**Spearman correlation** uses ranks instead of raw values, making it robust to:

- Non-linear relationships (as long as monotonic)
- Outliers
- Non-normal distributions

```python
from scipy import stats

# Data with a monotonic but non-linear relationship
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [1, 2, 4, 8, 16, 32, 64, 128]  # Exponential

pearson_r, _ = stats.pearsonr(x, y)
spearman_r, _ = stats.spearmanr(x, y)

print(f"Pearson r: {pearson_r:.3f}")   # Lower because relationship isn't linear
print(f"Spearman r: {spearman_r:.3f}")  # 1.0 because relationship is monotonic
```

| Measure | Measures | Assumptions | Best For |
|---------|----------|-------------|----------|
| Pearson | Linear relationship | Normal, continuous | Linear relationships |
| Spearman | Monotonic relationship | Ordinal or continuous | Non-linear, ordinal data |

### Correlation Matrix: Many Variables at Once

A **correlation matrix** shows correlations between all pairs of variables:

```python
import pandas as pd
import numpy as np
import plotly.express as px

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'hours_studied': np.random.normal(5, 2, 100),
    'hours_sleep': np.random.normal(7, 1, 100),
    'exam_score': np.random.normal(75, 10, 100)
})
# Add correlations
df['exam_score'] = df['exam_score'] + df['hours_studied'] * 3 - df['hours_sleep'] * 0.5

# Calculate correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Interactive heatmap with Plotly
fig = px.imshow(corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu',
                title='Correlation Matrix')
fig.show()
```

#### Diagram: Correlation Visualizer MicroSim

<details markdown="1">
    <summary>Interactive Correlation Explorer</summary>
    Type: microsim

    Bloom Taxonomy: Analyze (L4)

    Learning Objective: Help students understand correlation through interactive visualization of scatter plots with different correlation strengths

    Canvas layout (850x550px):
    - Main area (550x500): Interactive scatter plot
    - Right panel (300x500): Controls and statistics
    - Bottom strip (850x50): Correlation strength indicator

    Visual elements:
    - Scatter plot with data points
    - Best-fit line (toggleable)
    - Correlation coefficient displayed prominently
    - Correlation strength meter (-1 to +1 scale)

    Interactive controls:
    - Slider: Target correlation (-1.0 to +1.0)
    - Button: "Generate Data" with current correlation
    - Slider: Number of points (20-200)
    - Slider: Noise level
    - Toggle: Show regression line
    - Toggle: Show confidence band
    - Dropdown: Preset examples (perfect positive, perfect negative, no correlation, moderate)
    - Draggable points: Move individual points to see effect

    Display panels:
    - Pearson r
    - Spearman r
    - P-value
    - R² (coefficient of determination)
    - Sample size

    Behavior:
    - Adjusting correlation slider regenerates data with target correlation
    - Dragging individual points updates all statistics in real-time
    - Adding outliers shows how they affect Pearson vs Spearman
    - Noise slider shows how correlation degrades with noise

    Educational annotations:
    - "r = 0.8 means strong positive relationship"
    - "Notice Spearman handles the outlier better!"
    - "R² = 0.64 means 64% of variance in Y is explained by X"

    Challenge tasks:
    - "Create data with r ≈ 0 but clear pattern (try a curve!)"
    - "Add an outlier that changes r by at least 0.2"
    - "Find the minimum sample size for statistical significance"

    Visual style: Clean Plotly-like scatter plot with interactive elements

    Implementation: p5.js with statistical calculations
</details>

!!! warning "Correlation ≠ Causation"
    The most important rule in statistics: **correlation does not imply causation**. Ice cream sales and drowning deaths are correlated (both increase in summer), but ice cream doesn't cause drowning. Always consider confounding variables and look for experimental evidence before claiming causation.

## Putting Statistics into Practice

Let's combine everything in a real analysis:

```python
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

# Load or create dataset
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age': np.random.normal(35, 10, n).clip(18, 65).astype(int),
    'income': np.random.normal(50000, 15000, n).clip(20000, 150000),
    'education_years': np.random.normal(14, 3, n).clip(8, 22).astype(int),
    'satisfaction': np.random.normal(7, 1.5, n).clip(1, 10)
})
# Add relationships
df['income'] = df['income'] + df['education_years'] * 2000 + np.random.normal(0, 5000, n)

# 1. DESCRIPTIVE STATISTICS
print("=== Descriptive Statistics ===")
print(df.describe())
print(f"\nSkewness:\n{df.apply(stats.skew)}")

# 2. DISTRIBUTION VISUALIZATION
fig = px.histogram(df, x='income', marginal='box',
                   title='Income Distribution with Box Plot')
fig.show()

# 3. CORRELATION ANALYSIS
print("\n=== Correlation Matrix ===")
print(df.corr().round(3))

# Interactive correlation heatmap
fig = px.imshow(df.corr(), text_auto='.2f',
                color_continuous_scale='RdBu',
                title='Variable Correlations')
fig.show()

# 4. HYPOTHESIS TEST
# Is there a significant correlation between education and income?
r, p = stats.pearsonr(df['education_years'], df['income'])
print(f"\n=== Hypothesis Test: Education vs Income ===")
print(f"Pearson r: {r:.3f}")
print(f"P-value: {p:.4e}")
print(f"Result: {'Significant' if p < 0.05 else 'Not significant'} correlation")

# 5. CONFIDENCE INTERVAL for mean income
mean_income = df['income'].mean()
se = stats.sem(df['income'])
ci = stats.t.interval(0.95, len(df)-1, loc=mean_income, scale=se)
print(f"\n=== 95% CI for Mean Income ===")
print(f"Mean: ${mean_income:,.0f}")
print(f"95% CI: (${ci[0]:,.0f}, ${ci[1]:,.0f})")
```

This workflow demonstrates the complete statistical analysis pipeline:

1. Summarize with descriptive statistics
2. Visualize distributions
3. Explore relationships with correlations
4. Test hypotheses about relationships
5. Quantify uncertainty with confidence intervals

??? question "Chapter 6 Checkpoint: Test Your Understanding"
    **Question 1:** A dataset has mean = 50 and median = 65. What can you infer about the distribution?

    **Question 2:** You test whether a new teaching method improves scores. The p-value is 0.03. What do you conclude at α = 0.05?

    **Question 3:** Two variables have Pearson r = 0.85 and Spearman r = 0.60. What might explain this difference?

    **Click to reveal answers:**

    **Answer 1:** The distribution is left-skewed (negative skew). When mean < median, the tail extends to the left, pulling the mean down.

    **Answer 2:** Since p = 0.03 < α = 0.05, you reject the null hypothesis. There is statistically significant evidence that the new teaching method affects scores. But check the effect size to see if the improvement is practically meaningful!

    **Answer 3:** The relationship is likely non-linear. Pearson measures linear correlation (strong here), but Spearman measures monotonic correlation (weaker). There might be outliers affecting Spearman, or the relationship curves rather than being perfectly monotonic.

!!! success "Achievement Unlocked: Statistical Thinker"
    You now speak the language of uncertainty. You can summarize data with the right measures, understand distributions, measure relationships, and make probabilistic inferences. These skills separate people who "look at data" from people who truly understand what data is telling them.

## Key Takeaways

1. **Descriptive statistics** summarize data: measures of central tendency (mean, median, mode) and spread (range, variance, standard deviation).

2. **Mean** is sensitive to outliers; **median** is robust. Choose based on your data.

3. **Standard deviation** measures typical distance from the mean; use the 68-95-99.7 rule for normal distributions.

4. **Quartiles** and **IQR** divide data into parts and help identify outliers.

5. **Skewness** measures asymmetry; **kurtosis** measures tail heaviness.

6. The **normal distribution** is central to statistics—the bell curve appears everywhere.

7. **Probability** quantifies uncertainty; **expected value** is the long-run average.

8. **Samples** estimate **population** parameters; **sampling** must be done carefully to avoid bias.

9. The **Central Limit Theorem** says sample means are approximately normal, regardless of population shape.

10. **Confidence intervals** quantify uncertainty about estimates; **p-values** measure evidence against hypotheses.

11. **Correlation** measures relationship strength; Pearson for linear, Spearman for monotonic. Remember: correlation ≠ causation!

12. A **correlation matrix** shows all pairwise relationships at once.

You've built a solid statistical foundation. In the next chapter, you'll use these concepts to build your first predictive model with linear regression—where the statistical concepts you just learned become the engine for making predictions!
