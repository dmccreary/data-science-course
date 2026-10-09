---
title: Data Loading Workflow
description: Step through what pd.read_csv() does to a small CSV file, from raw text on disk to a typed DataFrame in memory, with the common error at each stage.
image: /sims/data-loading-workflow/data-loading-workflow.png
og:image: /sims/data-loading-workflow/data-loading-workflow.png
twitter:image: /sims/data-loading-workflow/data-loading-workflow.png
social:
   cards: false
quality_score: 0
---

# Data Loading Workflow

<iframe src="main.html" height="527" width="100%" scrolling="no"></iframe>

[Run the Data Loading Workflow MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The flowchart at the top has five stages: a CSV file on disk, the call to `pd.read_csv()`, parsing, the finished DataFrame, and analysis. Parsing is split into the four jobs the chapter lists (delimiter, header row, data types, missing values), so there are eight steps in all.

For every step the left panel shows the data as it looks at that moment: raw lines of text, the same lines with each comma marked, a grid of fields, the grid with a data type under each column, and finally the DataFrame with its index. The right panel says what pandas is doing, and the red box names the most common thing that goes wrong at that stage, such as `FileNotFoundError` or `ParserError`.

The file is the chapter's `students.csv` with one score left blank, so that the missing-value step has something to handle. The text is really parsed: the field count, the data types, the `NaN`, the shape, and the two results on the last step are all computed from it. Because of the blank, `score` is `float64` and prints as `85.0`, where the chapter's complete file gives `85`.

## How to Use

1. Press **Next** to move forward one step and **Previous** to go back. The dots in the Parsing box show which of its four steps you are on.
2. Click any stage of the flowchart to jump to it.
3. Before each press of Next, predict what the left panel will look like after the step.
4. Read the red box for the common error at each stage. Uncheck **Show common errors** to hide it.
5. Press **Reset** to return to step 1.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/data-loading-workflow/main.html"
        height="527"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
What a DataFrame is, how to import pandas, and how files are found by name and folder.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain, in order, the stages that turn a CSV file into a pandas DataFrame, including how the delimiter, the header row, data types, and missing values are handled.

### Activities

1. **Predict** (4 min): Looking only at step 1, students write down what the column names and data types will be, then step through to check.
2. **Step through** (6 min): Students press Next through all eight steps and state in one sentence what changed in the left panel at each step.
3. **Diagnose** (5 min): With Show common errors on, students match four situations (a misspelled file name, a line with an extra comma, a word in a number column, a file larger than memory) to the stage where each one shows up.

### Assessment
Students are given a five-line CSV file on paper in which one numeric column has a blank field and another numeric column contains the word unknown. They write the column names, the dtype pandas will infer for each column with a reason, and where NaN will appear.

## References

1. pandas documentation. [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).
2. pandas documentation. [How do I read and write tabular data?](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html)
3. Wikipedia. [Comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values).
