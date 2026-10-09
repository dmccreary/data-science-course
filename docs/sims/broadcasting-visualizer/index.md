---
title: Broadcasting Visualizer
description: Choose the shapes of two arrays and an operation to see how NumPy stretches the smaller array to match the larger one, and which pairs of shapes cannot be combined.
image: /sims/broadcasting-visualizer/broadcasting-visualizer.png
og:image: /sims/broadcasting-visualizer/broadcasting-visualizer.png
twitter:image: /sims/broadcasting-visualizer/broadcasting-visualizer.png
social:
   cards: false
quality_score: 0
---

# Broadcasting Visualizer

<iframe src="main.html" height="552" width="100%" scrolling="no"></iframe>

[Run the Broadcasting Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

**Broadcasting** is the rule NumPy uses when an operation involves two arrays of different shapes. This MicroSim shows array `A`, array `B`, and the result of `A + B`, `A - B`, `A * B` or `A / B`.

The rule is checked in the table under the grids.

1. Write the two shapes one above the other, lined up from the **right**.
2. A missing size counts as 1. It is shown in the table as *(1)*.
3. Each pair of sizes must be equal, or one of them must be 1. The result takes the larger size.

An axis of size 1 is stretched: its single row or column is used again at every position. The stretched copies are drawn faded with dashed borders, and the title above the array says what shape it acts like. NumPy does not really build these copies in memory. It reuses the values as it loops.

One cell has a red frame in all three grids, and the line under the grids spells out its calculation, such as `A[1, 2] + B[2] = 6 + 30 = 36`.

If a pair of sizes fails the rule, there is no result, the failing column of the table turns red, and the panel shows the message NumPy raises: `ValueError: operands could not be broadcast together with shapes (3,4) (2,4)`.

Not included: typing your own values and the animation from the original specification. The checkbox switches between the arrays as written and the arrays as stretched.

## How to Use

1. Read the default, a (3, 3) matrix plus a (3,) row. Find the faded copies of the row and the *(1)* in the table.
2. Untick **Show stretched copies** to see the arrays as they were written, then tick it again.
3. Click any cell to move the red frame and read which element of `A` and which element of `B` it combines.
4. Use the **A** and **B** menus to try other shapes. Predict the result shape before you look.
5. Choose **Column and row: both stretch** from the examples and explain where the 3 by 4 result comes from.
6. Choose the two **Mismatch** examples. Read the error message and find the red column that causes it.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/broadcasting-visualizer/main.html"
        height="552"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Array shapes, and element-wise arithmetic on two arrays of the same shape.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to predict whether two array shapes can be broadcast together, state the shape of the result, and explain which array is stretched along which axis.

### Activities

1. **Rule check** (4 min): Students copy the chapter's four-row table of shapes and use the MicroSim to confirm all four rows, noting which axis stretches in each.
2. **Predict then test** (6 min): For six pairs of shapes chosen by the teacher, students predict works or fails and the result shape on paper, then test each pair.
3. **Explain the trap** (4 min): Students explain why (3, 4) with (3,) fails although both contain a 3, and name a shape for B with three values that does work.

### Assessment
Students decide, without the MicroSim, whether shapes (4, 3) and (3,), (4, 3) and (4,), and (4, 1) and (1, 5) can be broadcast, give each result shape, and write the error message NumPy would show for the pair that fails.

## References

1. NumPy documentation. [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
2. VanderPlas, Jake. *Python Data Science Handbook*, 2nd ed. O'Reilly Media, 2022. Part II, Introduction to NumPy. See the chapter Computation on Arrays: Broadcasting.
3. McKinney, Wes. *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022. Appendix A, Advanced NumPy (Broadcasting).
