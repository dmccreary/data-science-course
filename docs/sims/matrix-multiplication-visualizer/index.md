---
title: Matrix Multiplication Visualizer
description: Step through the cells of a matrix product to see each one built from a row of A and a column of B, and compare it with element-wise multiplication.
image: /sims/matrix-multiplication-visualizer/matrix-multiplication-visualizer.png
og:image: /sims/matrix-multiplication-visualizer/matrix-multiplication-visualizer.png
twitter:image: /sims/matrix-multiplication-visualizer/matrix-multiplication-visualizer.png
social:
   cards: false
quality_score: 0
---

# Matrix Multiplication Visualizer

<iframe src="main.html" height="472" width="100%" scrolling="no"></iframe>

[Run the Matrix Multiplication Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

For a matrix $A$ of shape $(m, n)$ and a matrix $B$ of shape $(n, p)$, the product `C = A @ B` has shape $(m, p)$, and every cell is a dot product:

$$C_{ij} = \sum_{k} A_{ik} B_{kj}$$

This MicroSim shows `A`, `B` and `C` as grids. For the current cell of `C` (gold), row $i$ of `A` is blue and column $j$ of `B` is green. Small circled numbers mark the pairs that are multiplied, and the panel writes the calculation term by term, for example `C[1, 2] = 3×9 + 4×12 = 27 + 48 = 75`. The default matrices are the chapter's example, so the finished `C` is `[[27 30 33] [61 68 75] [95 106 117]]`.

The line under the title shows the shape rule. The two **inner** sizes, the columns of `A` and the rows of `B`, must be equal. The two **outer** sizes give the shape of `C`. When the inner sizes differ there is no result, and the panel shows the error that NumPy raises for `A @ B` together with a plain explanation.

The operator menu switches to `A * B`. This is element-wise multiplication: each cell uses only the two numbers in the same position, and the shapes must be the same or broadcast to the same shape. It is a different operation from the matrix product, which is a common source of mistakes.

Not included: the size sliders, Play button and speed control from the original specification. Shapes are chosen from two menus so that mismatched shapes can be shown, and the student steps through the cells.

## How to Use

1. Read the shape line for the default, (3, 2) @ (2, 3). Find the two inner sizes and the two outer sizes.
2. Look at the gold cell `C[0, 0]`. Match the circled numbers in the blue row and the green column to the terms in the calculation.
3. Predict the next cell, then press **Next**. Use **Previous** to go back, or click any cell of `C`.
4. Change the **A** and **B** shapes. Find a pair whose inner sizes differ and read the error.
5. Set both shapes to (2, 2), switch the operator menu to element-wise `*`, and compare the result with the matrix product.
6. Set both shapes to (3, 2) and try both operators. One fails and one works. Explain why.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/matrix-multiplication-visualizer/main.html"
        height="472"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
The dot product of two vectors, array shapes, and element-wise multiplication of two arrays.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how each cell of a matrix product is computed from a row and a column, determine from two shapes whether `A @ B` is defined and what shape it has, and distinguish it from element-wise multiplication.

### Activities

1. **Predict a cell** (4 min): With the chapter's matrices, students compute `C[0, 0]` and `C[2, 1]` by hand before stepping to those cells to check.
2. **Shape rule** (5 min): Students test all the shape pairs the menus allow, sort them into works and fails for `@`, and state the rule in their own words.
3. **Two kinds of multiply** (5 min): Using two (2, 2) matrices, students record `A @ B` and `A * B` side by side and explain why the two results differ.

### Assessment
Given `A = [[1, 2], [3, 4]]` and `B = [[5, 6], [7, 8]]`, students compute `A @ B` and `A * B` by hand. They then state whether a (4, 3) matrix can be multiplied by a (4, 3) matrix with `@`, and what transposing the second matrix changes.

## References

1. NumPy documentation. [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
2. Wikipedia. [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication).
3. McKinney, Wes. *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022. Chapter 4, NumPy Basics: Arrays and Vectorized Computation. See the section Linear Algebra.
