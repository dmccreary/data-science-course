---
title: Data Selection Playground MicroSim
description: Practice four ways to select data from a pandas DataFrame (columns, rows by position, rows by label, and a Boolean filter) and see the code, the selected cells, and the result update together.
image: /sims/data-selection-playground-microsim/data-selection-playground-microsim.png
og:image: /sims/data-selection-playground-microsim/data-selection-playground-microsim.png
twitter:image: /sims/data-selection-playground-microsim/data-selection-playground-microsim.png
social:
   cards: false
quality_score: 0
---

# Data Selection Playground MicroSim

<iframe src="main.html" height="597" width="100%" scrolling="no"></iframe>

[Run the Data Selection Playground MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The table `df` holds ten students. Their names are the index, so selecting by label (`loc`) and selecting by position (`iloc`) look different. Four selection methods from the chapter are available:

- **Columns**, as in `df[["age", "score"]]`: click column names to build the list
- **Rows by position**, as in `df.iloc[0:3]`: set the start and stop positions
- **Rows by label**, as in `df.loc[["Alice", "Charlie"]]`: click names to build the list
- **Boolean filter**, as in `df[df["score"] >= 90]`: choose a column, a comparison, and a value

Cells that end up in the result are gold in `df`. The second panel shows the exact pandas code, the DataFrame it returns, and the shape of that DataFrame. For a Boolean filter, an extra column shows the True or False the test gives for each row. An empty result is explained in a red note.

Not included: combining two conditions with `&` or `|`, and `isin()` filters on the city column.

## How to Use

1. Choose a method from the **Select by** menu.
2. **Columns** and **Rows by label**: click the blue-outlined column names or row labels in `df` to add them to the list or remove them. The result keeps the order in which you clicked.
3. **Rows by position**: drag **start** and **stop**. Notice that the stop position is not included.
4. **Boolean filter**: pick a column and a comparison in the **Test** row and drag **value**. Watch the True/False column and the gold rows.
5. Try these: show only city and score, get the last three rows, get Diana and then Bob, find everyone who passed (score >= 70), and find the youngest student.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/data-selection-playground-microsim/main.html"
        height="597"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
The parts of a DataFrame, Python list and slice syntax, and comparison operators.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to apply column selection, iloc, loc, and Boolean filtering to extract a required subset of a DataFrame and predict the shape of the result.

### Activities

1. **Predict the shape** (5 min): For each method's starting selection, students predict the shape of the result before reading it, then change one control and predict again.
2. **Position versus label** (5 min): Students produce the rows for Charlie, Diana, and Eve twice, once with iloc and once with loc, and write down both lines of code.
3. **Answer with a filter** (6 min): Students find who passed (score >= 70), who is younger than 22, and the single youngest student, recording the code for each.

### Assessment
Without the MicroSim, students write one line of pandas for each task on the same table and give the shape of each result. The tasks are the city and score columns only, the first four rows, the rows for Grace and Ivy, and the students aged 28 or older.

## References

1. pandas documentation. [Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html).
2. pandas documentation. [How do I select a subset of a DataFrame?](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)
3. VanderPlas, Jake. *Python Data Science Handbook*, 2nd ed. O'Reilly Media, 2023.
