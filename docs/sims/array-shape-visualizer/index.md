---
title: Array Shape Visualizer
description: Type a new shape for an array of up to 12 numbers and see the same numbers laid out as a row, a table, or a stack of tables, with the NumPy error when the shape is impossible.
image: /sims/array-shape-visualizer/array-shape-visualizer.png
og:image: /sims/array-shape-visualizer/array-shape-visualizer.png
twitter:image: /sims/array-shape-visualizer/array-shape-visualizer.png
social:
   cards: false
quality_score: 0
---

# Array Shape Visualizer

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Array Shape Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The **shape** of a NumPy array says how many elements lie along each axis. This MicroSim starts with the flat array `a = np.arange(n)` and shows what `b = a.reshape(...)` produces for the shape you type.

- One size, such as `12`, gives a row. Two sizes, such as `3, 4`, give a table with 3 rows and 4 columns. Three sizes, such as `2, 2, 3`, give a stack of 2 tables.
- The panel reports `b.shape`, `b.ndim` (the number of axes) and `b.size` (the number of elements).
- Brackets under the flat array show how it is cut into runs. Each run fills the last axis of `b`, so the numbers keep their order.
- One element has a red frame in both views. The panel gives its index in `b` and the arithmetic behind it, for example $7 = 1 \times 4 + 3$ for `b[1, 3]`.

A reshape only works when the sizes multiply to `a.size`. One size may be `-1`, which tells NumPy to work that size out. When the request is impossible the MicroSim shows the message NumPy raises, for example `ValueError: cannot reshape array of size 12 into shape (5,3)`, and draws the requested grid with its empty slots so you can see the mismatch.

Not included: the rotating 3-D view and the animated transition in the original specification. A stack of tables is drawn side by side, which is how NumPy prints a 3-D array.

## How to Use

1. Look at the default, `a.reshape(3, 4)`. Match each bracket under the flat array to a row of the table.
2. Type a new shape in the **a.reshape( )** box, for example `4, 3`, `2, 6` or `2, 2, 3`. Separate the sizes with commas.
3. Click any cell to follow that element. Read where it lands and check the arithmetic.
4. Try `4, -1` and `-1, 1`, then read how NumPy worked out the missing size.
5. Try `5, 3` and `5, -1` and read the error message. Count the empty slots.
6. Drag **Elements in a** to 7 or 11 and find every shape that still works. The **Examples** menu jumps to cases from the chapter.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/array-shape-visualizer/main.html"
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
Creating arrays with `np.arange`, and multiplying whole numbers to find factor pairs.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to use `reshape` to rearrange an array into a given number of dimensions, predict the resulting `shape`, `ndim` and `size`, and determine whether a requested shape is possible.

### Activities

1. **Predict** (3 min): Students list every 2-D shape they think 12 numbers can take, then test each one in the reshape box.
2. **Follow an element** (5 min): Students pick a position such as 10 and predict its index in shapes (3, 4), (4, 3) and (2, 2, 3) before clicking it to check.
3. **Break it** (5 min): Students find three requests that fail, copy each error message, and explain in a sentence what went wrong. They then set the slider to 7 and explain why so few shapes work.

### Assessment
Without the MicroSim, students state the result of `np.arange(24).reshape(4, -1).shape`, give the index of the value 17 in that array, and say whether `np.arange(24).reshape(5, -1)` works and why.

## References

1. NumPy documentation. [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html).
2. NumPy documentation. [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html).
3. Wikipedia. [Row- and column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
