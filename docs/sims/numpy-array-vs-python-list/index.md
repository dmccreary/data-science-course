---
title: NumPy Array vs Python List
description: Step through doubling five numbers to see why a Python list of scattered objects is processed one element at a time while a NumPy array is one block of values handled in a single call.
image: /sims/numpy-array-vs-python-list/numpy-array-vs-python-list.png
og:image: /sims/numpy-array-vs-python-list/numpy-array-vs-python-list.png
twitter:image: /sims/numpy-array-vs-python-list/numpy-array-vs-python-list.png
social:
   cards: false
quality_score: 0
---

# NumPy Array vs Python List

<iframe src="main.html" height="616" width="100%" scrolling="no"></iframe>

[Run the NumPy Array vs Python List MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

A Python list and a NumPy array can hold the same five numbers, but they store them in very different ways. This MicroSim draws both.

- **Python list** (left): the list object is a row of five pointers. Each pointer leads to a separate `int` object somewhere else in memory.
- **NumPy array** (right): a small header records the `dtype` and the `shape`, and the five values sit side by side in one block, 8 bytes each.

Pressing **Next** doubles the numbers. The list needs one pass through the Python interpreter for every element: follow the pointer, check the type, multiply, store a pointer to the result. The array is finished after one step, because `arr * 2` is a single call whose loop runs in compiled C code along the block.

The bottom panel scales the comparison up. Memory is computed from the sizes that CPython reports on a 64-bit machine: a list costs 56 bytes plus 8 bytes per pointer and 28 bytes per `int` object, and an `int64` array costs 8 bytes per value. For one million numbers that is 36.0 MB against 8.0 MB, the same figures the chapter's `sys.getsizeof` code prints. The speed statement is a typical range, not a measurement, because real timings depend on the operation and the computer. The addresses on screen are made up for illustration.

## How to Use

1. Read the two panels at step 0. Find the five pointers in the list and the five values in the array.
2. Press **Next** and watch which cells turn gold. Read the sentence under each panel.
3. Keep pressing **Next** until the list is done. Count how many steps each side needed.
4. Untick **Show addresses**, then tick it again. Compare the scattered addresses of the `int` objects with the evenly spaced addresses of the array values.
5. Drag the **Array size n** slider and watch the memory bars and the number of interpreter passes. Press **Reset** to start again.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/numpy-array-vs-python-list/main.html"
        height="616"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Python lists and `for` loops, and the idea that a program's data lives in the computer's memory.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how a Python list and a NumPy array store the same numbers and why that difference makes whole-array NumPy operations use less memory and run faster.

### Activities

1. **Predict** (3 min): Before pressing Next, students predict how many steps the list and the array will each need to double five numbers, and say why.
2. **Step through** (6 min): Students step from 0 to 5, describing in their own words what the interpreter does on each pass and what NumPy did at step 1.
3. **Scale up** (5 min): Students set the slider to 1,000 and to 1,000,000 and record the memory for each structure, then work out where the 36 bytes per list element come from.

### Assessment
Students draw their own picture of the list `[10, 20, 30]` and the array `np.array([10, 20, 30])` in memory and write two sentences explaining why `arr * 2` is faster than `[x * 2 for x in nums]` on a large array, and why the gain disappears on a tiny one.

## References

1. NumPy documentation. [What is NumPy?](https://numpy.org/doc/stable/user/whatisnumpy.html).
2. VanderPlas, Jake. *Python Data Science Handbook*, 2nd ed. O'Reilly Media, 2022. Part II, Introduction to NumPy. See the chapter Understanding Data Types in Python.
3. Wikipedia. [NumPy](https://en.wikipedia.org/wiki/NumPy).
