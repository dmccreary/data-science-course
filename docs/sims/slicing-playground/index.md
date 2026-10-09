---
title: Slicing Playground
description: Type a row slice and a column slice for a 6 by 8 array to see which elements are selected, what shape the result has, and how a slice shares memory with the original array.
image: /sims/slicing-playground/slicing-playground.png
og:image: /sims/slicing-playground/slicing-playground.png
twitter:image: /sims/slicing-playground/slicing-playground.png
social:
   cards: false
quality_score: 0
---

# Slicing Playground

<iframe src="main.html" height="592" width="100%" scrolling="no"></iframe>

[Run the Slicing Playground MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Slicing picks out part of an array with `start:stop:step` notation, one slice per axis. This MicroSim shows `a = np.arange(48).reshape(6, 8)` and lets you type both parts of `b = a[rows, columns]`.

- The selected elements turn blue in `a`. The result `b` and its shape are drawn in the panel.
- Each slice is explained as a start, a stop that is not included, and a step, with the missing values and negative numbers filled in. `-2:` on 6 rows, for example, becomes start 4, stop before 6.
- The gray labels on the right and bottom edges show every position counted from the end, which is what a negative index means.
- A single number such as `-1` is an index, not a slice. It selects one row or column and removes that axis, so `a[:, -1]` has shape `(6,)` while `a[:, -1:]` has shape `(6, 1)`.

A slice is a **view**: `b` uses the same memory as `a`. Press **b[:] = 0** to write zeros through the slice and watch `a` change. Use `a[...].copy()` when you need an independent array.

Slices never raise an error for being out of range, they are clipped, and an empty result is allowed. A single index that is out of range raises `IndexError`, and a step of 0 raises `ValueError: slice step cannot be zero`. The MicroSim shows these messages as NumPy words them.

Not included: selecting by dragging on the grid, from the original specification. The three challenges replace the separate Check Answer buttons.

## How to Use

1. Read the default `a[1:4, ::2]`. Match the blue cells to the two explanations in the panel.
2. Change the two boxes. Try `:3`, `2:`, `::-1`, `-2:` and `1:6:2` for the rows, and watch the start, stop and step that NumPy uses.
3. Type a single index such as `2` or `-1` in one box and compare `b.shape` with the slice `2:3`.
4. Press **b[:] = 0** and look at `a`. Press **Reset a** to restore the numbers.
5. Choose a **Challenge** from the menu. The target cells get orange frames. Type slices until the panel says the challenge is solved.
6. Try a step of 0 and an index of 9 to see the two error messages.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/slicing-playground/main.html"
        height="592"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Indexing a 2-D array with `a[row, column]` and slicing a Python list with `start:stop`.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to apply `start:stop:step` slicing along both axes of a 2-D array to select a required set of elements, predict the shape of the result, and explain why changing a slice changes the original array.

### Activities

1. **Predict** (3 min): For `a[2:5, 1:4]`, `a[:, ::3]` and `a[::-1, 0]`, students write the shape and the first value of the result before typing each one.
2. **Challenges** (7 min): Students solve the three challenges and write down the slices they used. Pairs compare answers, since more than one slice can select the same cells.
3. **Views** (4 min): Students select a block, press b[:] = 0, and explain what happened to `a`. They then describe how `.copy()` would change the outcome.

### Assessment
Given `m = np.arange(20).reshape(4, 5)` from the chapter, students write the slices that return `[[7 8 9] [12 13 14]]`, the last row reversed, and every other column, and state the shape of each result. They also explain what `m[1:3, 2:5][:] = 0` does to `m`.

## References

1. NumPy documentation. [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html).
2. NumPy documentation. [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html).
3. McKinney, Wes. *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022. Chapter 4, NumPy Basics: Arrays and Vectorized Computation.
