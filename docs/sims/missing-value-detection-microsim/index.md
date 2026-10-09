---
title: Missing Value Detection MicroSim
description: Count the missing values that isnull() will find in a column of a small DataFrame, check your answer, and reveal the hidden missing values that pandas does not detect.
image: /sims/missing-value-detection-microsim/missing-value-detection-microsim.png
og:image: /sims/missing-value-detection-microsim/missing-value-detection-microsim.png
twitter:image: /sims/missing-value-detection-microsim/missing-value-detection-microsim.png
social:
   cards: false
quality_score: 0
---

# Missing Value Detection MicroSim

<iframe src="main.html" height="552" width="100%" scrolling="no"></iframe>

[Run the Missing Value Detection MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

pandas treats `NaN` and `None` as missing, so `df.isnull()` returns `True` for them. An empty string `""` and a placeholder code such as `-999` are ordinary values to pandas. `isnull()` returns `False` for them even though no real value is there. Finding those hidden gaps is the detective work.

The MicroSim shows a DataFrame of 8 rows and 5 columns in five scenarios: NaN only, NaN mixed with None, hidden missing values, missing values that all sit in one column, and sparse data where more than half the cells are missing. The quiz asks how many missing values `isnull().sum()` will count in one column.

- **Show isnull()** tints each cell by what pandas reports: green for present, red for NaN, orange for None. The bottom row gives `df.isnull().sum()` for each column.
- **Find hidden missing** tints empty strings and -999 in yellow and adds their counts.
- The **Detection code** panel shows the total, the percentage of all cells, and the counts of `""` and `-999`.

The first scenario opens with the isnull() mask switched on as a worked example. Choosing another scenario switches both checkboxes off so that you count before you look. `pd.read_csv` already turns an empty field into NaN. Empty strings that reach a DataFrame another way, for example from a database or a JSON file, stay as text.

## How to Use

1. Study the first scenario with **Show isnull()** on. Match the red cells to the counts in the bottom row.
2. Choose another scenario from the menu. The highlights turn off.
3. Count the missing values in the blue column, type the number, and press **Check** (or press Enter).
4. Click a different column name to be asked about that column.
5. Turn on **Show isnull()** and then **Find hidden missing** to see what pandas detected and what it missed.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/missing-value-detection-microsim/main.html"
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
Reading a pandas DataFrame, and knowing that NaN marks a missing value.

### Bloom's Taxonomy Level
Apply (L3)

### Learning Objective
Students will be able to apply isnull() and isnull().sum() to count the missing values in a DataFrame and identify hidden missing values that isnull() does not detect.

### Activities

1. **Worked example** (3 min): With scenario 1 and Show isnull() on, students explain how the bottom row follows from the red cells and compute the percentage missing by hand.
2. **Count and check** (8 min): In scenarios 2 to 5 students answer the quiz for at least two columns each, counting before they reveal the mask.
3. **Find the hidden ones** (4 min): In scenario 3 students compare the isnull() total with the true number of gaps and write the line of code that would convert the hidden values to NaN.

### Assessment
Shown a printed DataFrame that contains NaN, None, an empty string, and -999, students state the output of df.isnull().sum() for each column, give the total and the percentage missing, and name the cells that are missing but not counted.

## References

1. pandas documentation. [Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html).
2. pandas documentation. [pandas.DataFrame.isna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html).
3. VanderPlas, Jake. *Python Data Science Handbook*. O'Reilly Media, 2016. [Handling Missing Data](https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html).
