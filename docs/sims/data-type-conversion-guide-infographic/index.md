---
title: Data Type Conversion Guide Infographic
description: A clickable reference card of nine pandas type conversions, grouped as numeric, datetime, categorical, and string, each with a worked example and a pitfall to remember.
image: /sims/data-type-conversion-guide-infographic/data-type-conversion-guide-infographic.png
og:image: /sims/data-type-conversion-guide-infographic/data-type-conversion-guide-infographic.png
twitter:image: /sims/data-type-conversion-guide-infographic/data-type-conversion-guide-infographic.png
social:
   cards: false
quality_score: 0
---

# Data Type Conversion Guide Infographic

<iframe src="main.html" height="632" width="100%" scrolling="no"></iframe>

[Run the Data Type Conversion Guide Infographic MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Columns often load with the wrong type: numbers stored as text, dates stored as text, repeated labels stored as plain strings. This reference card collects the nine conversions from the chapter in four color-coded groups.

- **Blue, to numeric**: `pd.to_numeric(col)`, `pd.to_numeric(col, errors="coerce")`, `col.astype(float)`
- **Green, to datetime**: `pd.to_datetime(col)`, `pd.to_datetime(col, format="%Y-%m-%d")`
- **Purple, to categorical**: `col.astype("category")`, `pd.Categorical(col, categories=[...], ordered=True)`
- **Orange, to string**: `col.astype(str)`, `col.map("{:.2f}".format)`

Selecting a method opens a panel that says when to use it and what to watch out for, and shows a small column before and after the conversion with its dtype. A "Now you can" line shows one thing the new type makes possible, such as `col.sum()` or `col.dt.day_name()`. The example results are computed by the MicroSim following the pandas rules. Text columns are labeled `object`, as pandas 2 reports them.

Not included: copying a code snippet to the clipboard.

## How to Use

1. Click a method on any card, or use **Next** and **Previous** to go through all nine in order.
2. Read **When to use it** and **Watch out**, then compare the Before and After columns and their dtypes.
3. Turn on **Quiz me**. The result is hidden. Say what the values and the dtype will be, then turn it off to check.
4. Go through all nine methods in Quiz mode until you can predict each result.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/data-type-conversion-guide-infographic/main.html"
        height="632"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Checking column types with df.dtypes and knowing the difference between text, numbers, and dates.

### Bloom's Taxonomy Level
Remember (L1)

### Learning Objective
Students will be able to recall the pandas method that converts a column to a numeric, datetime, categorical, or string type, and one pitfall of each conversion.

### Activities

1. **Tour** (5 min): Students step through the nine methods with Next and read each panel.
2. **Self-test** (6 min): With Quiz me on, students predict the After column and dtype for every method and keep a tally of correct predictions.
3. **Match** (4 min): Given four messy columns described in words (prices with a stray word, dates as text, letter grades, ZIP codes read as numbers), students name the conversion for each and its pitfall.

### Assessment
From memory, students write the conversion call for each of four situations (text to numbers with bad entries, text to dates, strings to an ordered category, numbers to formatted text) and state one thing that can go wrong with each.

## References

1. pandas documentation. [pandas.to_numeric](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html).
2. pandas documentation. [pandas.to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html).
3. pandas documentation. [Categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html).
