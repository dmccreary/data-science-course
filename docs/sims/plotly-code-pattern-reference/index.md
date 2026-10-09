---
title: Plotly Code Pattern Reference
description: A clickable Plotly Express reference card covering chart functions, essential parameters, layout settings, saving options, and templates, with a quiz mode for recall practice.
image: /sims/plotly-code-pattern-reference/plotly-code-pattern-reference.png
og:image: /sims/plotly-code-pattern-reference/plotly-code-pattern-reference.png
twitter:image: /sims/plotly-code-pattern-reference/plotly-code-pattern-reference.png
social:
   cards: false
quality_score: 0
---

# Plotly Code Pattern Reference

<iframe src="main.html" height="647" width="100%" scrolling="no"></iframe>

[Run the Plotly Code Pattern Reference MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This reference card collects the Plotly Express patterns used in this chapter in one place. It has four panels and a strip.

- **Common Chart Functions**: the eight `px` functions you will use most, from `px.line()` to `px.violin()`.
- **Essential Parameters**: the arguments that most `px` functions share, such as `color`, `hover_data`, and `facet_col`.
- **Layout Customization**: settings passed to `fig.update_layout()`.
- **Saving Options**: `fig.write_html()`, `fig.write_image()`, and `fig.show()`.
- **Templates**: six built-in themes. Each swatch is drawn with that template's real background, grid, and first two colors.

Clicking an entry opens it in the panel at the bottom, which gives the meaning, one line of example code, and a note about how it behaves. Some of the notes are easy to get wrong from memory: `px.bar()` stacks rows that share an x value, `px.pie()` takes `names` and `values` in place of `x` and `y`, and `fig.write_image()` needs the `kaleido` package.

**Quiz me** hides the meanings so you can test your recall. Not included: copying a snippet to the clipboard. Type the code yourself, which helps you remember it.

## How to Use

1. Scan the four panels and the templates strip to see what the card covers.
2. Click any entry, or press **Next** and **Previous**, to read its meaning, its example code, and the note.
3. Click each template swatch and compare the backgrounds, grid lines, and colors.
4. Tick **Quiz me**. The meanings are replaced by question marks.
5. Pick an entry, say what it does and how you would write it, then press **Reveal** to check.

## Iframe Embed Code

You can add this MicroSim to any web page by adding this to your HTML:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/plotly-code-pattern-reference/main.html"
        height="647"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
High School (11-12) and College Freshman

### Duration
10 minutes

### Prerequisites
Creating a first Plotly Express chart with `px.line()` or `px.scatter()` and a pandas DataFrame.

### Bloom's Taxonomy Level
Remember (L1)

### Learning Objective
Students will be able to recall the common Plotly Express chart functions, their essential parameters, the main layout settings, and the ways to save a figure.

### Activities

1. **Read the card** (3 min): Students click through one entry in each panel and one template, reading the note for each.
2. **Quiz in pairs** (5 min): With Quiz me ticked, one student picks an entry and the partner says what it does before Reveal is pressed. They swap after five entries.
3. **Use it** (4 min): Students write one px call from memory that uses color, title, and labels, then check each argument against the card.

### Assessment
Without the card, students match eight px functions to chart types, name the parameter for each of five tasks (color by a column, extra tooltip columns, small multiples, rename an axis, animate), and write the line that saves a figure as an HTML file.

## References

1. Plotly documentation. [Plotly Express in Python](https://plotly.com/python/plotly-express/).
2. Plotly documentation. [Theming and templates in Python](https://plotly.com/python/templates/).
3. Plotly documentation. [Static image export in Python](https://plotly.com/python/static-image-export/).
