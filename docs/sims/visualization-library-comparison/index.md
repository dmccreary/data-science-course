---
title: Visualization Library Comparison
description: Compare Matplotlib, Seaborn, and Plotly side by side, pick a need from a list to see which library fits and why, and view a typical chart of the same data in each library's default look.
image: /sims/visualization-library-comparison/visualization-library-comparison.png
og:image: /sims/visualization-library-comparison/visualization-library-comparison.png
twitter:image: /sims/visualization-library-comparison/visualization-library-comparison.png
social:
   cards: false
quality_score: 0
---

# Visualization Library Comparison

<iframe src="main.html" height="622" width="100%" scrolling="no"></iframe>

[Run the Visualization Library Comparison MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Python has several plotting libraries, and this chapter uses three of them. This MicroSim puts them side by side.

- **Three cards** list each library's strengths, what it is best for, its learning curve, and how interactive its charts are.
- **The "I need" list** is the decision guide. Choose what you need, such as a figure for a printed paper or a chart people can explore on a web page. The library that fits is marked with a green bar and the strip at the bottom gives the reason.
- **The example panel** shows a typical chart from the selected library together with the code that makes it. All three charts use the same data: 40 restaurant bills and tips.

The three examples show what each library does with little effort. The Matplotlib chart is a plain static image where you set the title and the axis labels yourself. The Seaborn chart comes from a single `regplot` call that fits the regression line and shades its 95% confidence band. The band is computed here from the data with the usual formula for the confidence interval of a fitted line. The Plotly chart is interactive: point at a dot in the Plotly example and a tooltip appears, which does not happen in the other two.

The libraries are not rivals. Seaborn is built on Matplotlib, so a Seaborn chart can be adjusted with Matplotlib commands, and many projects use all three.

## How to Use

1. Read the three cards and find one strength that only one library has.
2. Open the **I need** list and choose a need. Note which card gets the green **Fits what you need** bar, then read the reason in the strip at the bottom.
3. Click each library card in turn to see its example chart and code. Compare the number of lines of code and how the charts look.
4. Click **Plotly** and move the mouse over the dots. Then click **Matplotlib** and try the same thing.
5. Before you choose the next need, predict which library will be marked.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/visualization-library-comparison/main.html"
        height="622"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10 minutes

### Prerequisites
Knowing what a scatter plot shows and that Python libraries are imported before use.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to compare Matplotlib, Seaborn, and Plotly and explain which library suits a given visualization need.

### Activities

1. **Compare the cards** (3 min): Students list one strength and one best use for each library in their own words.
2. **Predict and check** (4 min): For each of the five needs, students predict the library before selecting the need, then read the reason given.
3. **Same data, three charts** (4 min): Students view the three example charts and describe two differences in the code and two differences in the output.

### Assessment
Students are given three short project descriptions (a figure for a science fair poster, a first look at a survey dataset, and a chart for a class web page) and name the library they would use for each with a one-sentence reason.

## References

1. Matplotlib documentation. [Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html).
2. seaborn documentation. [An introduction to seaborn](https://seaborn.pydata.org/tutorial/introduction.html).
3. Plotly documentation. [Plotly Express in Python](https://plotly.com/python/plotly-express/).
