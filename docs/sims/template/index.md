---
title: MicroSim Template
description: A template for creating new MicroSims with standard structure and formatting
quality_score: 65
hide:
  - toc
---
# MicroSim Template

<iframe src="main.html" height="500" scrolling="no"></iframe>

## Embed This MicroSim

Copy this iframe to embed this MicroSim in your website:

```html
<iframe src="https://dmccreary.github.io/data-science-course/sims/template/main.html" height="500px" scrolling="no"></iframe>
```

[Run the MicroSim in Fullscreen](main.html){ .md-button .md-button--primary }

## About This MicroSim

This is a template MicroSim that demonstrates the standard structure and formatting for creating new interactive simulations. Use this as a starting point when developing new MicroSims for the data science course.

## How to Use This Template

1. Copy the entire `template` directory to create a new MicroSim
2. Rename the directory to your MicroSim name (use kebab-case)
3. Update `metadata.json` with your MicroSim's information
4. Replace `main.html` and the JavaScript file with your simulation code
5. Update this `index.md` file with your documentation

## Sample Prompt for Creating a MicroSim

!!! prompt
    Create a single file p5.js sketch.
    Draw a green circle on a 600x400 canvas with a radius of 200.

## References

1. [p5.js Reference](https://p5js.org/reference/) - p5.js Documentation - JavaScript library used to build interactive simulations
2. [Processing Wiki on Positioning Your Canvas](https://github.com/processing/p5.js/wiki/Positioning-your-canvas) - Guide for canvas positioning