---
title: Plotly Interactive Features MicroSim
description: Practice the eight interactive features of a real Plotly chart (hover, zoom, pan, box select, lasso select, reset, download, and legend toggling) while a tracker ticks off each one as you use it.
image: /sims/plotly-interactive-features-microsim/plotly-interactive-features-microsim.png
og:image: /sims/plotly-interactive-features-microsim/plotly-interactive-features-microsim.png
twitter:image: /sims/plotly-interactive-features-microsim/plotly-interactive-features-microsim.png
social:
   cards: false
quality_score: 0
---

# Plotly Interactive Features MicroSim

<iframe src="main.html" height="577" width="100%" scrolling="no"></iframe>

[Run the Plotly Interactive Features MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }

## About This MicroSim

Every Plotly chart is interactive without any extra code. This MicroSim is a real Plotly.js scatter plot of 61 points in three groups, the chart that `px.scatter(df, x='x', y='y', color='group')` would make. Nothing is simulated: the toolbar, tooltips, and legend are Plotly's own.

The panel under the chart lists eight features. Each one is ticked the first time you use it, because the MicroSim listens to the events the chart emits:

- **Hover**: move the mouse over a point to see a tooltip with its group, x, and y.
- **Zoom**: drag a box on the chart and the axes rescale to it.
- **Pan**: hold Shift and drag, or choose Pan in the toolbar, to slide the view without changing its scale.
- **Box select** and **Lasso select**: choose the tool in the toolbar, then drag a box or draw a loop. Points outside the selection fade.
- **Reset**: double-click the chart to bring back the full view.
- **Download**: click the camera in the toolbar to save the chart as a PNG file.
- **Legend**: click a group in the legend to hide it, and click again to show it.

Four challenges ask you to use the features to answer a question about the data. They are checked automatically too. The toolbar appears at the top right of the chart when the mouse is over it.

## How to Use

1. Move the mouse over the dots and read the tooltips. Click any feature name in the panel to see how to use it.
2. Follow the **Try next** hint until all eight boxes are ticked.
3. Drag a box around the cluster in the upper right to zoom in, then double-click to reset.
4. Choose **Lasso Select** in the toolbar and draw a loop around every group A point and no others.
5. Find the point with the highest y value and hover over it.
6. Click the camera in the toolbar to download the chart as a PNG.
7. Press **Restart** to clear the ticks and redraw the chart.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/plotly-interactive-features-microsim/main.html"
        height="577"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10 minutes

### Prerequisites
Reading a scatter plot with a legend, and a first Plotly Express chart made with `px.scatter()`.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to use the hover, zoom, pan, box select, lasso select, reset, download, and legend features of a Plotly chart to inspect a dataset.

### Activities

1. **Explore** (4 min): Students follow the Try next hints until the tracker reads 8 of 8.
2. **Challenges** (4 min): Students complete the four challenges and write down the coordinates of the outlier.
3. **Transfer** (3 min): Students open any Plotly chart from the chapter in a notebook and repeat the eight actions on it.

### Assessment
On a new Plotly chart, students demonstrate each of the eight features on request and explain how they would find the exact value of one point and isolate one group.

## References

1. Plotly documentation. [Hover text and formatting in Python](https://plotly.com/python/hover-text-and-formatting/).
2. Plotly documentation. [Legends in Python](https://plotly.com/python/legend/).
3. Plotly documentation. [Configuration in Python](https://plotly.com/python/configuration-options/).
