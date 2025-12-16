---
title: Independent vs Dependent Variables
description: An interactive scatter plot MicroSim that lets students experiment with changing an independent variable (study time) and observe how the dependent variable (test score) responds.
image: /sims/independent-dependent-variables/independent-dependent-variables.png
og:image: /sims/independent-dependent-variables/independent-dependent-variables.png
twitter:image: /sims/independent-dependent-variables/independent-dependent-variables.png
social:
   cards: false
---

# Independent vs Dependent Variables

<iframe src="main.html" height="502px" width="100%" scrolling="no"></iframe>

[Run the Independent vs Dependent Variables MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }

## About This MicroSim

This interactive scatter plot visualization helps students understand the fundamental concept of **independent** and **dependent** variables by exploring the relationship between study time and test scores.

### Key Concepts

- **Independent Variable (X-axis)**: Hours Studied - This is what you **control** or **change**
- **Dependent Variable (Y-axis)**: Test Score - This is what you **measure** or **observe**

### Interactive Features

| Control | Description |
|---------|-------------|
| **Add Student** button | Adds a random student data point based on current relationship strength |
| **Generate Class of 30** button | Creates a realistic dataset of 30 students |
| **Clear All** button | Removes all data points |
| **Relationship Strength** slider | Controls how predictable scores are from study time (0-100%) |
| **Show Trend Line** toggle | Display/hide the regression line |
| **Show Prediction** toggle | Show predicted score when hovering over the plot |
| **Click on plot** | Add a student at that exact study time/score location |

### Statistics Panel

The right panel displays:

- **Number of students** in the dataset
- **Average study time** (hours)
- **Average test score** (points)
- **Correlation strength** with emoji indicator (💪 strong, 👍 moderate, 🤷 weak)

### Visual Elements

- 🎓 Regular student data points
- 🌟 Outlier students (those who significantly beat or missed expectations)
- Blue trend line showing the best-fit relationship
- Light blue confidence band around the trend line
- Orange dashed prediction lines when hovering

## Iframe Embedding

You can include this MicroSim on your website using the following `iframe`:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/independent-dependent-variables/main.html"
        height="502px"
        width="100%"
        scrolling="no">
</iframe>
```

## Lesson Plan

### Learning Objectives

By the end of this activity, students will be able to:

1. **Identify** which variable is independent and which is dependent in a scenario
2. **Explain** that the independent variable is what you control, and the dependent variable is what you measure
3. **Observe** how correlation strength affects the scatter of data points
4. **Predict** approximate outcomes based on the trend line
5. **Recognize** that relationships show tendencies, not guarantees (there's always variation)

### Guided Exploration (15-20 minutes)

#### Part 1: Understanding the Relationship (5 min)

1. Start with the default settings (70% relationship strength, 10 students)
2. Ask students: "Which variable does the researcher control?"
3. Ask: "What happens to test scores as study time increases?"
4. Click "Generate Class of 30" to see a more complete picture

#### Part 2: Exploring Correlation Strength (5 min)

1. Set relationship strength to **100%** - observe tight clustering around the line
2. Set relationship strength to **50%** - observe more scatter
3. Set relationship strength to **10%** - observe near-random scatter
4. Discuss: "What does a 'strong relationship' mean in real life?"

#### Part 3: Making Predictions (5 min)

1. Enable "Show Prediction" and hover over the plot
2. Ask: "If someone studies 6 hours, what score might they expect?"
3. Click on the plot to add students at specific locations
4. Discuss outliers: "Why might someone study a lot but still score low?"

### Discussion Questions

1. What are some other examples of independent and dependent variables?
   - Water amount (independent) → Plant growth (dependent)
   - Exercise time (independent) → Fitness level (dependent)
   - Temperature (independent) → Ice cream sales (dependent)

2. Why do we call one variable "dependent"?

3. Can we ever be 100% certain about a prediction? Why or why not?

4. What does the correlation emoji (💪, 👍, 🤷) tell us about the data?

### Extension Activities

- Have students design their own experiment with independent/dependent variables
- Discuss the difference between correlation and causation
- Explore what happens when you add extreme outliers manually

## Technical Details

- **Library**: p5.js 1.11.10
- **Canvas Size**: 700×500px (responsive width)
- **Draw Height**: 400px
- **Control Height**: 100px
- **Bloom's Taxonomy Level**: Apply (L3)
- **Recommended Duration**: 15-20 minutes

## Image Note

Remember to create a screenshot of this MicroSim and save it as `independent-dependent-variables.png` in this directory for social media preview images.
