---
title: DataFrame Anatomy
description: Point at or click the parts of a small pandas DataFrame to learn the names index, columns, row, column, cell, and values, and the code that returns each one.
image: /sims/dataframe-anatomy/dataframe-anatomy.png
og:image: /sims/dataframe-anatomy/dataframe-anatomy.png
twitter:image: /sims/dataframe-anatomy/dataframe-anatomy.png
social:
   cards: false
quality_score: 0
---

# DataFrame Anatomy

<iframe src="main.html" height="507" width="100%" scrolling="no"></iframe>

[Run the DataFrame Anatomy MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

A DataFrame is a two-dimensional table with labeled rows and columns. This MicroSim draws the chapter's 4 × 3 example table and names its six parts with numbered, color-coded callouts:

- **Index** (blue): the row labels. `df.index`
- **Columns** (green): the column names. `df.columns`
- **Row** (orange): one observation. `df.loc['row_1']` or `df.iloc[1]`
- **Column** (purple): one variable, returned as a Series. `df['Col_B']`
- **Cell** (red): one value. `df.loc['row_1', 'Col_B']`
- **Values** (gray): the data without its labels, a NumPy array. `df.values`

When a part is highlighted, the panel gives its name, a one-sentence meaning, the pandas expression that returns it, and the result laid out the way pandas prints it. The results are built from the table on screen.

## How to Use

1. Read the six callouts around the table and the matching list in the panel.
2. Point at a part of the table, a callout, or a line of the list to highlight that part. Click to keep it selected. Click it again, or click empty space, to clear it.
3. Use the **Highlight** menu to choose a part by name instead of pointing.
4. Turn **Show code** off, say how you would access the highlighted part, then turn it back on to check.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/dataframe-anatomy/main.html"
        height="507"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Python lists and dictionaries, and the idea of a table with rows and columns.

### Bloom's Taxonomy Level
Remember (L1)

### Learning Objective
Students will be able to identify the index, the columns, a row, a column, a cell, and the values of a pandas DataFrame and recall the expression that accesses each one.

### Activities

1. **Explore** (4 min): Students point at each of the six parts in turn and read its name, meaning, and memory tip.
2. **Recall** (5 min): With Show code off, students write the access expression for each part from memory, then turn Show code on and check all six.
3. **Transfer** (4 min): Students label the same six parts on the chapter's table of Alice, Bob, and Charlie's grades and write the code that returns Bob's row and the Science column.

### Assessment
Given a printed DataFrame they have not seen, students label its index, columns, one row, one column, and one cell, and write the pandas expression that returns each part. All parts are named correctly and at least five of the six expressions are correct.

## References

1. pandas documentation. [Intro to data structures](https://pandas.pydata.org/docs/user_guide/dsintro.html).
2. pandas documentation. [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
3. McKinney, Wes. *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022.
