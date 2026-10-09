---
title: Data Inspection Command Center
description: Choose a pandas inspection command and read its output for an eight-row DataFrame, with the cells the command used highlighted.
image: /sims/data-inspection-command-center/data-inspection-command-center.png
og:image: /sims/data-inspection-command-center/data-inspection-command-center.png
twitter:image: /sims/data-inspection-command-center/data-inspection-command-center.png
social:
   cards: false
quality_score: 0
---

# Data Inspection Command Center

<iframe src="main.html" height="582" width="100%" scrolling="no"></iframe>

[Run the Data Inspection Command Center MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The left panel holds a DataFrame `df` of eight students with five columns. One age and one score are missing. The right panel is a notebook cell: the command you chose and the output pandas gives for it, with a note on how to read that output. Seven commands are available:

- `df.head(n)` and `df.tail(n)`: the first or last n rows
- `df.shape`: the pair (rows, columns)
- `df.info()`: column names, non-null counts, data types, and memory use
- `df.describe()`: summary statistics for the numeric columns
- `df.columns` and `df.dtypes`: the column names, and the data type of each column

The table is highlighted to show what each command looked at, for example the first n rows for `head`, the numeric columns for `describe`, and the `NaN` cells for `info`.

Every output is computed from the table. `describe()` uses the sample standard deviation and linearly interpolated quartiles, as pandas does, and the outputs were checked against pandas 2.3. Text columns have the dtype `object`, as in the chapter. From pandas 3 on they are labeled `str` instead.

## How to Use

1. Pick a command from the **Command** menu. The notebook cell shows the code and its output.
2. For `df.head(n)` and `df.tail(n)`, drag the **Rows n** slider from 1 to 8 and watch which rows are highlighted in `df`.
3. Before choosing `df.shape`, `df.columns`, or `df.dtypes`, predict the output from the table, then check.
4. Choose `df.info()` and find the two columns whose non-null count is below 8. Find the matching NaN cells in the table.
5. Choose `df.describe()` and work out which columns are missing from the output, and why.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/data-inspection-command-center/main.html"
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
The parts of a DataFrame, loading a CSV file with read_csv, and the meaning of mean and quartiles.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to use head, tail, shape, info, describe, columns, and dtypes to inspect a DataFrame and interpret each output, including finding missing values from the non-null counts.

### Activities

1. **Predict and check** (5 min): For shape, columns, and dtypes, students write the output they expect from the table before selecting the command.
2. **Head and tail** (4 min): Students set n to 3 and compare head with tail, then find the smallest n for which head shows a NaN.
3. **Find the missing data** (6 min): Using info() and describe(), students state how many values are missing in each column and explain why count is 7 rather than 8.

### Assessment
Students are shown the info() output from the chapter (1000 entries, four columns). They state the shape of the DataFrame, name the columns with missing values and how many are missing in each, and say which columns describe() would summarize.

## References

1. pandas documentation. [Essential basic functionality](https://pandas.pydata.org/docs/user_guide/basics.html).
2. pandas documentation. [pandas.DataFrame.info](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html).
3. pandas documentation. [pandas.DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html).
