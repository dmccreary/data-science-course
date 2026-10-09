---
title: One-Hot Encoding Visualizer
description: See what pd.get_dummies returns for a categorical column, switch drop_first and dtype, add a new category, and trace each row to the column that marks it.
image: /sims/one-hot-encoding-visualizer/one-hot-encoding-visualizer.png
og:image: /sims/one-hot-encoding-visualizer/one-hot-encoding-visualizer.png
twitter:image: /sims/one-hot-encoding-visualizer/one-hot-encoding-visualizer.png
social:
   cards: false
quality_score: 0
---

# One-Hot Encoding Visualizer

<iframe src="main.html" height="622" width="100%" scrolling="no"></iframe>

[Run the One-Hot Encoding Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The left table is one categorical column of a small DataFrame. The right table is what `pd.get_dummies` makes from it, and the exact call is printed above the tables. There is one new column per category, named `column_value`, and each row is `True` in exactly one of them. pandas sorts the category labels, so the new columns are in alphabetical order whatever order the values first appear in.

The dashed **row sum** column is not part of the DataFrame. It is a check. With every column kept, each row sums to 1, the same as the column of ones a linear model uses for its intercept. One column is therefore redundant and the model's coefficients are not uniquely determined. This is the dummy variable trap.

With `drop_first=True` pandas removes the column of the first label in sorted order (drawn faded). That category becomes the **reference category**. A row that is `False` in every remaining column must belong to it, so no information is lost, and the note shows the formula that rebuilds the dropped column.

Two details differ from the chapter's hand-made table. Current pandas returns `True` and `False`, not 1 and 0, unless you pass `dtype=int`. And the reference category is the first label alphabetically, so for Downtown, Rural, and Suburbs it is Downtown, and the new columns are `neighborhood_Rural` and `neighborhood_Suburbs`.

## How to Use

1. Read the default view. Six rows of `color` become three columns, `color_Blue`, `color_Green`, and `color_Red`. Find the single `True` in each row.
2. Hover over a row in either table. The same row is outlined in both tables and the note says which column it marks. Click to keep a row selected.
3. Check **drop_first=True**. One column fades out and the note names the reference category. Find the rows that are now `False` in every column.
4. Compare the **row sum** column with the box checked and unchecked.
5. Check **dtype=int** to see the same table as 1s and 0s.
6. Press **Add a Category** to append a row whose value has not appeared before, and see where its new column lands. Use the **Column** menu to try `neighborhood` and `fuel`. **Reset** restores the starting rows.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/one-hot-encoding-visualizer/main.html"
        height="622"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
DataFrame columns, the difference between categorical and numerical variables, and the multiple regression equation with an intercept.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain how one-hot encoding turns a categorical column into True/False columns, predict the columns `pd.get_dummies` creates with and without `drop_first=True`, and explain why dropping one column avoids the dummy variable trap.

### Activities

1. **Trace** (4 min): Students hover over each of the six rows of `color` and say which new column is `True` before reading the note. They then predict the column order for `neighborhood` and check it with the menu.
2. **Drop first** (5 min): Students check **drop_first=True** for all three columns and record which category becomes the reference each time. They write down the rule pandas uses to choose it.
3. **Explain the trap** (4 min): Using the row sum column, students explain in two sentences why keeping all k columns is a problem for a linear model with an intercept and why k − 1 columns still identify every category.

### Assessment
Given a column `size` with the values Small, Medium, Large, Medium, students write the column names and the first two rows returned by `pd.get_dummies(df, columns=['size'], drop_first=True)`, name the reference category, and explain why it is Large and not Small.

## References

1. pandas documentation. [pandas.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).
2. scikit-learn documentation. [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
3. Wikipedia. [Dummy variable (statistics)](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)).
