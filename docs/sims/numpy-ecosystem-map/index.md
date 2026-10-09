---
title: NumPy Ecosystem Map
description: Click through a hub-and-spoke map with NumPy at the center to read how seven data science libraries use NumPy arrays and the code that moves data between them.
image: /sims/numpy-ecosystem-map/numpy-ecosystem-map.png
og:image: /sims/numpy-ecosystem-map/numpy-ecosystem-map.png
twitter:image: /sims/numpy-ecosystem-map/numpy-ecosystem-map.png
social:
   cards: false
quality_score: 0
---

# NumPy Ecosystem Map

<iframe src="main.html" height="567" width="100%" scrolling="no"></iframe>

[Run the NumPy Ecosystem Map MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

NumPy sits in the middle of the Python data science tools. This map puts the NumPy array in the center and seven libraries around it. Two kinds of link are drawn.

- **Built on NumPy** (solid line): NumPy is a required dependency and the library computes with arrays inside. These are pandas, SciPy, scikit-learn and Matplotlib.
- **Works with NumPy** (dashed line): the library has its own engine or tensor type, and it accepts arrays or converts to and from them. These are Plotly, PyTorch and TensorFlow.

An arrowhead at the library means arrays go in. A second arrowhead at NumPy means results come back as arrays. The two charting libraries have one arrowhead, because a picture comes out, not data.

Selecting a library shows what it is for, how it uses NumPy, two lines of code that move data in each direction, and one thing worth remembering. For example, `torch.from_numpy(arr)` shares memory with the array on the CPU, so a change to one shows up in the other, and scikit-learn needs `X` to be 2-D, which is why `X.reshape(-1, 1)` appears so often.

Not included: the animation of data moving along the spokes, from the original specification. Memory sharing is stated only where the library's documentation is explicit about it.

## How to Use

1. Start with NumPy selected and read why it is in the middle.
2. Click a library, or press **Next** and **Previous**, and read its panel. Say the two code lines out loud as array in and array out.
3. Compare a solid link with a dashed link. Use the menu to show only one kind at a time.
4. Find the libraries with one arrowhead and explain why nothing comes back to NumPy.
5. For each library, name one place earlier in the course where you have already used it with an array.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/numpy-ecosystem-map/main.html"
        height="567"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Creating NumPy arrays, and some experience with at least one of pandas, Matplotlib or scikit-learn.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to describe how the main Python data science libraries depend on or exchange data with NumPy arrays, and give the call that converts between an array and each library's own data type.

### Activities

1. **Sort** (4 min): Before using the menu, students sort the seven libraries into built on NumPy and works with NumPy, then check with the filter.
2. **Read and record** (6 min): For each library students copy the two code lines into a table with the columns library, array in, and array out.
3. **Trace a workflow** (4 min): Students trace one path through the map for a small project: load a table with pandas, convert to an array, fit a scikit-learn model, and plot the predictions.

### Assessment
Students write the line of code that turns a DataFrame into a NumPy array, the line that turns an array into a PyTorch tensor, and one sentence explaining why a single well-understood array type makes it possible to use these libraries together.

## References

1. Wikipedia. [NumPy](https://en.wikipedia.org/wiki/NumPy).
2. pandas documentation. [pandas.DataFrame.to_numpy](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html).
3. PyTorch documentation. [torch.from_numpy](https://pytorch.org/docs/stable/generated/torch.from_numpy.html).
