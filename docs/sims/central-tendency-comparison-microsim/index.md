---
title: Central Tendency Comparison MicroSim
description: Drag the dots of a dot plot and add outliers to see how the mean, the median, and the mode each respond to the same change in the data.
image: /sims/central-tendency-comparison-microsim/central-tendency-comparison-microsim.png
og:image: /sims/central-tendency-comparison-microsim/central-tendency-comparison-microsim.png
twitter:image: /sims/central-tendency-comparison-microsim/central-tendency-comparison-microsim.png
social:
   cards: false
quality_score: 0
---

# Central Tendency Comparison MicroSim

<iframe src="main.html" height="542" width="100%" scrolling="no"></iframe>

[Run the Central Tendency Comparison MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Twenty values between 0 and 100 are drawn as a dot plot. Equal values stack, so the plot is also a histogram. Three markers show the center of the data in three ways:

- **Mean** (dashed red line): the sum of the values divided by how many there are, $\bar{x} = \frac{1}{n}\sum x_i$
- **Median** (green line): the middle value of the sorted data, or the average of the two middle values when $n$ is even
- **Mode** (blue band): the value that occurs most often, which is the tallest stack. Data can have more than one mode, or none when no value repeats.

Values snap to multiples of 5 so that stacks form and the mode is easy to see. The table compares each measure at the start with its value now, and the yellow panel puts the comparison into a sentence. The four starting datasets (symmetric, right-skewed, left-skewed, bimodal) come from a seeded generator, so they are the same on every visit.

Moving one value far away changes the mean, because every value enters the sum. It usually leaves the median alone, because the median depends only on which value is in the middle. The mode changes only if the tallest stack changes.

## How to Use

1. Look at the symmetric starting data. The mean, the median, and the mode are all at 50.
2. Press **Add Outlier**. A new value appears at 0 or 100, whichever is farther from the median. Read the Change column of the table to see which measures moved.
3. Drag any dot left or right. Drag one dot slowly across the middle of the data and watch for the moment the median changes.
4. Choose another starting dataset from the menu. For the skewed sets, note the order of the mode, the median, and the mean. For the bimodal set, ask whether 50 is a typical value.
5. Press **Add Point** to add a value at the median, then drag it where you want it. Press **Reset** to return to the starting data.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/central-tendency-comparison-microsim/main.html"
        height="542"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10-15 minutes

### Prerequisites
Adding and dividing to find an average, and sorting a list of numbers.

### Bloom's Taxonomy Level
Understand (L2)

### Learning Objective
Students will be able to explain why an outlier changes the mean but leaves the median and the mode almost unchanged.

### Activities

1. **Predict** (3 min): Before pressing **Add Outlier** on the symmetric data, students predict which of the three measures will move and by roughly how much, then check with the table.
2. **Explore** (6 min): Students drag dots to complete three challenges: make the mean at least 5 above the median, make data with two modes, and make data with no mode.
3. **Explain** (4 min): Students load the right-skewed data and write two sentences that explain why the mean is above the median, using the positions of the dots as evidence.

### Assessment
Given the scores 72, 85, 90, 78, 88 and then the same scores with an added value of 250, students compute the mean and the median of both sets and explain which measure better describes a typical score.

## References

1. Wikipedia. [Central tendency](https://en.wikipedia.org/wiki/Central_tendency).
2. Wikipedia. [Median](https://en.wikipedia.org/wiki/Median).
3. McKinney, Wes. *Python for Data Analysis*, 3rd ed. O'Reilly Media, 2022. Chapter 5, Getting Started with pandas.
