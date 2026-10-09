---
title: Duplicate Handling Decision Tree
description: Examine seven small tables that contain duplicates, follow a decision tree to the right strategy, and see what each pandas strategy does to the same rows.
image: /sims/duplicate-handling-decision-tree/duplicate-handling-decision-tree.png
og:image: /sims/duplicate-handling-decision-tree/duplicate-handling-decision-tree.png
twitter:image: /sims/duplicate-handling-decision-tree/duplicate-handling-decision-tree.png
social:
   cards: false
quality_score: 0
---

# Duplicate Handling Decision Tree

<iframe src="main.html" height="627" width="100%" scrolling="no"></iframe>

[Run the Duplicate Handling Decision Tree MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Not every duplicate should be removed, and not every duplicate should be removed the same way. This MicroSim draws the decision tree for the chapter's duplicate-handling strategies and lets you test it on seven cases.

The questions run down the left side of the tree. Is the duplication intentional? Are the rows exactly identical? Which columns identify a record? Do the other columns differ in a way that matters? Which version is correct? The answers lead to seven strategies. Green boxes are safe actions, yellow boxes need investigation, and red boxes call for care.

Clicking a strategy applies it to the case table the way pandas would, so you can compare strategies on the same rows:

- **drop_duplicates()** drops a row only when every column matches an earlier row.
- **subset=keys** and **keep='first'** give the same table, because `keep="first"` is the default.
- **keep='last'** keeps the last row for each key.
- **groupby().agg()** is shown as `df.groupby(key, as_index=False).first()`, which takes the first non-missing value in each column.
- **Manual review** uses `df.duplicated(subset=key, keep=False)` to flag every copy without removing anything.

The feedback says whether the choice fits the case. Case A opens with its path shown as a worked example. The third question is worded more carefully than in the chapter outline. Once rows are known not to be identical, their other columns always differ in some way, so the real question is whether the difference matters.

## How to Use

1. Read case A with **Show the path** on. Follow the green path through the tree and compare the Before and After tables.
2. Choose another case from the menu. The path is hidden again.
3. Read the story and the rows, then answer the questions of the tree from the top.
4. Click the strategy at the end of your path. Read the After table and the feedback.
5. Click other strategies to see what they would do to the same rows, then turn on **Show the path** to check your reasoning.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/duplicate-handling-decision-tree/main.html"
        height="627"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
15-20 minutes

### Prerequisites
Detecting duplicate rows with duplicated() and removing them with drop_duplicates().

### Bloom's Taxonomy Level
Analyze (L4)

### Learning Objective
Students will be able to analyze a table that contains duplicate records and select the handling strategy that fits it, justifying the choice with the decision tree.

### Activities

1. **Worked example** (3 min): Students trace the highlighted path for case A and explain why an exact copy is safe to drop.
2. **Solve the cases** (10 min): Students work through cases B to G. For each one they write down their path through the tree before clicking a strategy.
3. **Compare strategies** (5 min): For case D and case F students apply every strategy and record what each one loses or keeps, then explain why only one is acceptable.

### Assessment
Given a new table with duplicates and a one-line description of where it came from, students name the strategy they would use, write the pandas call, and justify each answer they gave on the way down the tree.

## References

1. pandas documentation. [pandas.DataFrame.drop_duplicates](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html).
2. pandas documentation. [pandas.DataFrame.duplicated](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html).
3. Wikipedia. [Record linkage](https://en.wikipedia.org/wiki/Record_linkage).
