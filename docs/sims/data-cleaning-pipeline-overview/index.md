---
title: Data Cleaning Pipeline Overview
description: Step through the eight stages of a data cleaning pipeline and watch a small messy table become analysis-ready, with the pandas code and the data quality report entry for each stage.
image: /sims/data-cleaning-pipeline-overview/data-cleaning-pipeline-overview.png
og:image: /sims/data-cleaning-pipeline-overview/data-cleaning-pipeline-overview.png
twitter:image: /sims/data-cleaning-pipeline-overview/data-cleaning-pipeline-overview.png
social:
   cards: false
quality_score: 0
---

# Data Cleaning Pipeline Overview

<iframe src="main.html" height="595" width="100%" scrolling="no"></iframe>

[Run the Data Cleaning Pipeline Overview MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Real data arrives with gaps, repeated rows, impossible values, and the wrong types. A cleaning pipeline deals with these problems in a fixed order. This MicroSim runs a seven-row membership table through eight stages:

1. **Raw Data**: the table as loaded.
2. **Missing Values**: the placeholder -999 is turned into NaN, then every NaN is filled with its column median.
3. **Duplicates**: a row that exactly copies an earlier row is dropped.
4. **Outliers**: the 1.5 × IQR rule flags an age of 230, which is replaced by the median.
5. **Data Types**: `age` becomes `int64` and `joined` becomes `datetime64`.
6. **Validation**: a row whose score breaks the rule "0 to 100" is removed.
7. **Transformation**: a min-max scaled copy of `score` is added.
8. **Clean Data**: the finished table and the full data quality report.

The table on the left always shows the data after the current stage. Cells that the stage changed are tinted, and a row that the stage removes is struck out. The panel on the right explains the stage, shows the pandas code, and shows the line written to the data quality report. Every number is computed from the table, so the stages really do depend on each other. The duplicate row, for example, only becomes an exact copy after the missing-value stage has replaced -999.

## How to Use

1. Read the raw table at step 1 and list the problems you can see before going on.
2. Press **Next** to move one stage forward and **Previous** to go back. You can also click any stage in the pipeline.
3. At each stage, find the tinted cells or the struck-out row, then read the explanation and the code that made the change.
4. Watch `df.shape` above the table and the dtype row under the column names as the stages go by.
5. At step 8, read the data quality report and match each line to the stage that wrote it. Press **Reset** to start again.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/data-cleaning-pipeline-overview/main.html"
        height="595"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Loading a CSV file into a pandas DataFrame and reading its rows, columns, and dtypes.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain what each stage of a data cleaning pipeline does to a dataset and why the order of the stages matters.

### Activities

1. **Predict** (3 min): At step 1, students write down every problem they can find in the raw table and which stage they expect to fix it.
2. **Step through** (7 min): Students step through all eight stages. For each one they say in a sentence what changed in the table and check it against the panel.
3. **Explain the order** (5 min): Students explain why row 3 was not a duplicate until the missing values were handled, and why the score of 104 was caught by validation and not by the outlier stage.

### Assessment
Given the eight stage names in random order, students put them in a sensible order and write one sentence for each stage that says what kind of problem it catches. They also give one example of two stages whose order changes the result.

## References

1. pandas documentation. [Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html).
2. Wikipedia. [Data cleansing](https://en.wikipedia.org/wiki/Data_cleansing).
3. McKinney, Wes. *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022. Chapter 7, Data Cleaning and Preparation.
