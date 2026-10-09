---
title: Box Plot Anatomy
description: Hover over or click the seven numbered parts of a box plot of 21 quiz scores to read what each part means, its value in the data, and the pandas code that computes it.
image: /sims/box-plot-anatomy/box-plot-anatomy.png
og:image: /sims/box-plot-anatomy/box-plot-anatomy.png
twitter:image: /sims/box-plot-anatomy/box-plot-anatomy.png
social:
   cards: false
quality_score: 0
---

# Box Plot Anatomy

<iframe src="main.html" height="517" width="100%" scrolling="no"></iframe>

[Run the Box Plot Anatomy MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

A box plot summarizes a set of numbers with five values and marks any outliers. This MicroSim draws the box plot of 21 quiz scores and numbers its parts:

1. **Minimum**: the end of the lower whisker, the smallest value that is not an outlier
2. **Q1**: the first quartile, the edge of the box on the low side
3. **Median**: the line inside the box
4. **Q3**: the third quartile, the edge of the box on the high side
5. **Maximum**: the end of the upper whisker, the largest value that is not an outlier
6. **IQR**: the interquartile range, the length of the box, $Q3 - Q1$
7. **Outliers**: values beyond the fences at $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$

The gray dots under the box are the 21 scores themselves, so each part can be checked against the data. The panel below the plot gives the meaning of the selected part, the calculation with the numbers of this data, and one line of pandas. Quartiles use linear interpolation, the default of `Series.quantile` and `np.percentile`.

The whiskers follow the matplotlib and pandas convention: a whisker ends at the most extreme data value that is still inside the fence, not at the fence itself. Here the lower fence is at 28, but the whisker stops at 41, the smallest score that is not below 28. Selecting the minimum, the maximum, or the outliers draws both fences as dashed lines.

## How to Use

1. Read the summary panel, then hover over each numbered badge (or its name) to highlight that part and read about it.
2. Click a part to keep its explanation on screen. Click it again, or click empty space, to return to the summary.
3. Select **Minimum**, **Maximum**, or **Outlier** to see the two fences as dashed lines and compare them with the whisker ends.
4. Check **Vertical** to turn the plot the way `df.boxplot()` draws it by default.
5. Uncheck **Show names**, name each numbered part from memory, and click the badge to check yourself.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/box-plot-anatomy/main.html"
        height="517"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
5-10 minutes

### Prerequisites
Sorting a list of numbers, the median, and the idea of a percentile.

### Bloom's Taxonomy Level
Remember (L1)

### Learning Objective
Students will be able to identify the seven parts of a box plot and recall what each one represents.

### Activities

1. **Explore** (3 min): Students hover over the seven parts in order and write each name and its meaning in a table.
2. **Check against the data** (3 min): Students count the gray dots below Q1, below the median, and below Q3 and compare the counts with 25%, 50%, and 75% of 21.
3. **Self-test** (3 min): With **Show names** unchecked and the plot vertical, students name each numbered part and click it to check.

### Assessment
Given an unlabeled box plot, students label the median, Q1, Q3, both whisker ends, the IQR, and any outliers, and write the two fence formulas.

## References

1. Wikipedia. [Box plot](https://en.wikipedia.org/wiki/Box_plot).
2. pandas documentation. [pandas.DataFrame.boxplot](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html).
3. Matplotlib documentation. [matplotlib.pyplot.boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html).
