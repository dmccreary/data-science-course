---
title: Measurement Scales Pyramid
description: An interactive pyramid visualization showing the four measurement scales (Nominal, Ordinal, Interval, Ratio) and what operations each allows.
image: /sims/measurement-scales-pyramid/measurement-scales-pyramid.png
og:image: /sims/measurement-scales-pyramid/measurement-scales-pyramid.png
twitter:image: /sims/measurement-scales-pyramid/measurement-scales-pyramid.png
social:
   cards: false
---

# Measurement Scales Pyramid

<iframe src="main.html" height="502px" width="100%" scrolling="no"></iframe>

[Run the Measurement Scales Pyramid Fullscreen](./main.html){ .md-button .md-button--primary }

## About This MicroSim

This interactive infographic visualizes the four measurement scales as a hierarchy where each level adds mathematical capabilities. The pyramid shape reinforces that:

- **More variables** fit into the lower categories (wider base)
- **More operations** are possible at higher levels (narrower top = more exclusive)

## How to Use

- **Hover** over any layer to see 5 example variables of that measurement type
- **Click** a layer to see which statistical tests are appropriate for that scale
- Click again to deselect and return to hover mode

## The Four Measurement Scales

### 1. Nominal (Base)
- **Operations**: Equality only (= ≠)
- **Example**: Jersey numbers, eye color, blood type
- **Statistics**: Mode, chi-square tests, frequency tables

### 2. Ordinal
- **Operations**: Equality + ordering (= ≠ < >)
- **Example**: Race positions, Likert scales, pain ratings
- **Statistics**: Median, Spearman correlation, Mann-Whitney U

### 3. Interval
- **Operations**: Equality + ordering + equal intervals (= ≠ < > + −)
- **Example**: Temperature in Fahrenheit, IQ scores, calendar dates
- **Statistics**: Mean, t-tests, Pearson correlation, regression

### 4. Ratio (Top)
- **Operations**: Full arithmetic (= ≠ < > + − × ÷)
- **Example**: Height, weight, age, income
- **Statistics**: All statistical tests, geometric mean, coefficient of variation

## Why This Matters

Choosing the wrong statistical test for your measurement scale can lead to meaningless results. For example:

- Computing the mean of jersey numbers is meaningless (nominal data)
- You can't say "80°F is twice as hot as 40°F" (interval, not ratio)
- Race positions 1st, 2nd, 3rd don't tell you the time gaps (ordinal)

## Learning Objectives

After exploring this visualization, students should be able to:

1. **Remember** (L1): Name the four measurement scales in order
2. **Understand** (L1): Explain what operations are valid for each scale
3. **Apply** (L2): Classify any variable into the correct measurement scale
4. **Analyze** (L3): Select appropriate statistical tests based on measurement scale

## Related Resources

- [Chapter 1: Introduction to Data Science](../../chapters/01-intro-to-data-science/)
- [Variable Types Decision Tree](../variable-types-decision-tree/)
- [Four Types of Data](../four-types-of-data/)

## Embed This MicroSim

You can include this MicroSim on your website using the following iframe:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/measurement-scales-pyramid/main.html"
        height="502px"
        width="100%"
        scrolling="no"></iframe>
```
