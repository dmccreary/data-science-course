---
title: Chart Type Selection Guide
description: Follow a decision tree from what you want to show to the chart that fits your data, then practice on ten short data scenarios with immediate feedback.
image: /sims/chart-type-selection-guide/chart-type-selection-guide.png
og:image: /sims/chart-type-selection-guide/chart-type-selection-guide.png
twitter:image: /sims/chart-type-selection-guide/chart-type-selection-guide.png
social:
   cards: false
quality_score: 0
---

# Chart Type Selection Guide

<iframe src="main.html" height="595" width="100%" scrolling="no"></iframe>

[Run the Chart Type Selection Guide MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Choosing a chart starts with a question, not with a favorite chart type. This MicroSim is a decision tree with two steps.

1. **What do you want to show?** Pick one of five goals: comparison, distribution, relationship, composition, or trend.
2. **What does your data look like?** Each goal offers three descriptions, such as *few categories* or *over time*, and each one leads to a chart type.

The panel under the tree shows a larger example of the selected chart, a sentence on when to use it, the Python call that draws it, and a warning where one applies (pie charts with too many slices, 3D charts, and dual y-axes). Every small chart is drawn from data. The bar chart and the pie chart use the examples from this chapter, and the histogram and the KDE plot show the same 200 test scores.

The **Practice** strip turns the guide into an exercise. Each scenario describes some data and a question. You answer by clicking a goal and then a chart, and the strip tells you whether the chart fits, whether you have the right goal but the wrong chart, or whether to rethink the goal.

The code lines use Matplotlib where it has a direct function, seaborn for the KDE plot and the pair plot, and Plotly Express for the treemap.

## How to Use

1. Read the question at the top, then click one of the five goals. The three charts for that goal appear below it.
2. Click a chart card to see a larger example, when to use it, and the code that draws it.
3. Look for the red **Warning** line on the pie chart, the vertical bar chart, and the multiple-line chart.
4. Press **New scenario**. Read the scenario, decide the goal first, then click the chart that fits the data.
5. Read the feedback. If it says you have the right goal, compare the three descriptions and try again. Press **New scenario** for the next one (there are ten).

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/chart-type-selection-guide/main.html"
        height="595"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The six basic plot types from this chapter (line, scatter, bar, histogram, box, pie) and the difference between categorical and numeric data.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to choose an appropriate chart type for a dataset by identifying the goal of the visualization and the structure of the data.

### Activities

1. **Tour the tree** (4 min): Students open each of the five goals and say in one sentence what the three charts under it have in common.
2. **Practice scenarios** (7 min): Students work through the ten scenarios. For each one they name the goal aloud before clicking, and note any scenario they missed on the first try.
3. **Write a scenario** (4 min): Each student writes one new scenario for a chart of their choice and trades it with a partner, who uses the tree to answer it.

### Assessment
Given four new descriptions of data and a question about each, students name the goal, the chart type, and the reason, and identify one chart that would be a poor choice for each and why.

## References

1. Matplotlib documentation. [Plot types](https://matplotlib.org/stable/plot_types/index.html).
2. Wikipedia. [Data and information visualization](https://en.wikipedia.org/wiki/Data_and_information_visualization).
3. VanderPlas, Jake. *Python Data Science Handbook*, 2nd ed. O'Reilly Media, 2022. Part IV, Visualization with Matplotlib.
