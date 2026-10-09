---
title: Outlier Detection Methods MicroSim
description: Apply the Z-score rule and the IQR rule to the same data, move the two thresholds, and see which points each rule flags and where the rules disagree.
image: /sims/outlier-detection-methods-microsim/outlier-detection-methods-microsim.png
og:image: /sims/outlier-detection-methods-microsim/outlier-detection-methods-microsim.png
twitter:image: /sims/outlier-detection-methods-microsim/outlier-detection-methods-microsim.png
social:
   cards: false
quality_score: 0
---

# Outlier Detection Methods MicroSim

<iframe src="main.html" height="587" width="100%" scrolling="no"></iframe>

[Run the Outlier Detection Methods MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Two common rules for flagging outliers can give different answers on the same data. This MicroSim draws the data twice, as two dot plots on the same axis, and applies one rule to each.

- **Z-score rule**: $z = (x - \text{mean}) / \text{std}$. A value is flagged when $|z|$ is larger than the threshold (3.0 by default).
- **IQR rule**: a value is flagged when it lies below $Q_1 - k \cdot IQR$ or above $Q_3 + k \cdot IQR$, with $k = 1.5$ by default.

Dashed lines mark the limits and the shaded zones show where values are flagged. Flagged points are red. A point with a dark ring is flagged by that rule only. The upper plot marks the mean, and the lower plot draws the box from $Q_1$ to $Q_3$ with the median. The panel lists the statistics, the limits, the flagged values, the mean with and without the flagged points, and a sentence comparing the rules.

There are four datasets: bell-shaped heights with three extreme values, right-skewed incomes that are all valid, commute times in two groups, and ages with entry errors. In the ages data the largest error inflates the standard deviation so much that the Z-score rule misses impossible ages that the IQR rule catches. The Z-score uses the population standard deviation, as `scipy.stats.zscore` does in the chapter code, and quartiles are interpolated as `Series.quantile` does.

Not included: the custom minimum and maximum (domain rule) method, and a list of points to investigate.

## How to Use

1. Look at the two plots for the heights data. Find the point with a dark ring: only one rule flags it.
2. Drag **Z-score threshold** down from 3.0 and watch the limits move in and more points turn red.
3. Drag **IQR multiplier** between 1.0 and 3.0 and compare the count with the Z-score count.
4. Point at any dot to read its value and its z-score.
5. Switch **Dataset** and repeat. For each dataset decide which flagged points are errors and which are valid extremes.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/outlier-detection-methods-microsim/main.html"
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
Mean, standard deviation, quartiles, and reading a dot plot or box plot.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to apply the Z-score rule and the IQR rule to a dataset, adjust their thresholds, and explain why the two rules can flag different points.

### Activities

1. **Compare the rules** (5 min): With the default settings students record the number of points each rule flags in all four datasets and note where the counts differ.
2. **Tune the thresholds** (7 min): Students find the Z-score threshold at which the Z-score rule flags the same heights as the IQR rule at 1.5, and explain what lowering a threshold does.
3. **Judge the flags** (6 min): For the incomes and the ages data students decide which flagged values are errors and which are valid, and say which rule served each dataset better.

### Assessment
Given a list of about ten numbers with one extreme value, students compute the IQR fences, state which values the IQR rule flags, and explain in words why a Z-score threshold of 3 might fail to flag a second, smaller outlier.

## References

1. Wikipedia. [Outlier](https://en.wikipedia.org/wiki/Outlier).
2. Wikipedia. [Standard score](https://en.wikipedia.org/wiki/Standard_score).
3. pandas documentation. [pandas.Series.quantile](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html).
