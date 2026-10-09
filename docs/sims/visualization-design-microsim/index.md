---
title: Visualization Design MicroSim
description: Design your own chart of a restaurant tips dataset by choosing the chart type, the columns, a color grouping, and a title, and watch the preview, the Plotly Express code, and a design check update together.
image: /sims/visualization-design-microsim/visualization-design-microsim.png
og:image: /sims/visualization-design-microsim/visualization-design-microsim.png
twitter:image: /sims/visualization-design-microsim/visualization-design-microsim.png
social:
   cards: false
quality_score: 0
---

# Visualization Design MicroSim

<iframe src="main.html" height="572" width="100%" scrolling="no"></iframe>

[Run the Visualization Design MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This playground lets you design a chart and see three things update together.

- **The preview** draws your chart from 60 sample rows shaped like Plotly's built-in tips dataset (`total_bill`, `tip`, `size`, `day`, `time`, `smoker`).
- **The code panel** shows the Plotly Express call that makes the same chart. You can type it into a notebook. It loads the real 244-row tips dataset, so the chart you get has more points than the preview.
- **The design check** comments on your choices. A green tick means the choice suits the data. An orange mark is a warning with a suggestion.

The preview follows what Plotly Express really does with each combination, including the ones that do not work well. A bar chart stacks rows that share an x value, so each bar is a total. A histogram counts rows and ignores y. A line chart joins the rows in table order, which gives a meaningless zigzag here because the rows have no time order. A box plot with a numeric x gives every row its own box. Box plots use quartiles with whiskers that reach the last point within 1.5 IQR, and points beyond that are drawn as outliers.

Not included: marker size, line width, a grid toggle, and file downloads. Zoom and hover are covered in the Plotly Interactive Features MicroSim.

## How to Use

1. Start with the default scatter plot of `tip` against `total_bill`. Read the code panel and match each argument to something in the preview.
2. Change **Chart** to each type in turn and read the first line of the design check each time.
3. Change **x** to `day` and try Bar, Box, and Scatter. Decide which one tells you the most.
4. Use **Color** to split the data by `time` or `smoker`, and look for the legend and the new `color=` argument.
5. Type your own **Title**. A good title says what the reader should notice. Delete the title and see what the design check says.
6. Build a chart with three green ticks that answers a question of your own, then type the code into a notebook and run it.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/visualization-design-microsim/main.html"
        height="572"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
The basic plot types from this chapter and a first look at Plotly Express.

### Bloom's Taxonomy Level
Create (L6)

### Learning Objective
Students will be able to design a chart for a question about a dataset by selecting a chart type, columns, color grouping, and title, and justify the design.

### Activities

1. **Explore the options** (5 min): Students try every chart type with the default columns and record which ones earn a warning and why.
2. **Design to a brief** (8 min): Students build a chart for each brief and write down the code. (1) Do people tip more at dinner than at lunch? (2) Which day brings in the most money? (3) How are bill sizes distributed?
3. **Critique** (5 min): Partners compare their charts for one brief, discuss whose title and chart type make the answer easier to see, and revise.

### Assessment
Students write their own question about the tips data, build a chart that answers it with no warnings in the design check, and hand in the generated code with two sentences that justify the chart type and the title.

## References

1. Plotly documentation. [Plotly Express in Python](https://plotly.com/python/plotly-express/).
2. Plotly documentation. [Box plots in Python](https://plotly.com/python/box-plots/).
3. McKinney, Wes. *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022. Chapter 9, Plotting and Visualization.
