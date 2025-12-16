---
title: Data Visualization with Matplotlib and Plotly
description: Transform numbers into compelling visual stories with interactive charts
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

# Data Visualization with Matplotlib and Plotly

## Summary

This chapter teaches students how to create effective data visualizations using matplotlib, Seaborn, and the modern interactive library Plotly. Students will learn visualization architecture (figures and axes), create various plot types (line, scatter, bar, histogram, box, pie), and customize their visualizations with titles, labels, legends, and colors. The chapter emphasizes choosing appropriate visualizations for different data types and creating interactive charts that engage viewers. By the end of this chapter, students will be able to create publication-quality visualizations that effectively communicate insights from data.

## Concepts Covered

This chapter covers the following 25 concepts from the learning graph:

1. Data Visualization
2. Matplotlib Library
3. Figure
4. Axes
5. Plot Function
6. Line Plot
7. Scatter Plot
8. Bar Chart
9. Histogram
10. Box Plot
11. Pie Chart
12. Subplot
13. Figure Size
14. Title
15. Axis Labels
16. Legend
17. Color
18. Markers
19. Line Styles
20. Grid
21. Annotations
22. Save Figure
23. Plot Customization
24. Seaborn Library
25. Statistical Plots

## Prerequisites

This chapter builds on concepts from:

- [Chapter 3: Python Data Structures](../03-python-data-structures/index.md)
- [Chapter 4: Data Cleaning and Preprocessing](../04-data-cleaning/index.md)

---

## Show, Don't Tell: The Visual Superpower

You've loaded data. You've cleaned it. You've wrangled it into perfect shape. But here's the thing—a spreadsheet full of numbers is about as exciting as reading the phone book. Nobody ever changed the world by emailing a CSV file.

**Data visualization** is where data science becomes VISIBLE. It's the superpower that lets you take thousands of numbers and transform them into a single image that tells a story. A well-crafted chart can reveal patterns that would take hours to find in a table. It can convince skeptics, inspire action, and make the invisible visible.

Think about it: every powerful presentation you've ever seen probably had a chart. Every news story about trends shows a graph. Every scientific breakthrough gets communicated through visualization. This is the skill that takes your analysis from "interesting to me" to "interesting to everyone."

In this chapter, you'll learn multiple visualization tools—from the classic **Matplotlib** to the beautiful **Seaborn** to the modern, interactive **Plotly**. By the end, you'll be creating charts that don't just display data—they ENGAGE with it.

#### Diagram: Visualization Library Comparison

<details markdown="1">
    <summary>Python Visualization Library Landscape</summary>
    Type: infographic

    Bloom Taxonomy: Understand (L2)

    Learning Objective: Help students understand when to use different visualization libraries

    Purpose: Compare the major Python visualization libraries and their strengths

    Layout: Three-column comparison card layout

    Column 1: MATPLOTLIB
    - Icon: Classic line graph
    - Tagline: "The Foundation"
    - Color: Blue
    - Strengths:
      - Complete control over every element
      - Publication-quality static images
      - Huge community and documentation
      - Works everywhere
    - Best for:
      - Academic papers
      - Print publications
      - Maximum customization
    - Learning curve: Medium-High
    - Interactivity: Limited (static by default)

    Column 2: SEABORN
    - Icon: Statistical plot with confidence intervals
    - Tagline: "Beautiful Statistics"
    - Color: Teal
    - Strengths:
      - Beautiful defaults
      - Built-in statistical visualizations
      - Works with pandas DataFrames
      - Less code for common plots
    - Best for:
      - Statistical analysis
      - Exploratory data analysis
      - Quick beautiful plots
    - Learning curve: Low-Medium
    - Interactivity: Limited (built on matplotlib)

    Column 3: PLOTLY
    - Icon: Interactive 3D scatter plot with cursor
    - Tagline: "Interactive & Modern"
    - Color: Purple
    - Strengths:
      - Interactive by default (zoom, pan, hover)
      - Web-ready (HTML output)
      - Beautiful modern aesthetics
      - 3D visualizations
      - Dashboards with Dash
    - Best for:
      - Web applications
      - Presentations
      - Data exploration
      - User engagement
    - Learning curve: Low-Medium
    - Interactivity: Full (native)

    Bottom section: Decision flowchart
    - "Need print/PDF?" → Matplotlib
    - "Statistical focus?" → Seaborn
    - "Need interactivity?" → Plotly
    - "Quick exploration?" → Seaborn or Plotly

    Interactive elements:
    - Hover over each library to see code examples
    - Click to see sample output images

    Implementation: HTML/CSS grid with JavaScript hover effects
</details>

## The Classic: Matplotlib Library

Let's start with the grandfather of Python visualization: the **Matplotlib library**. Created in 2003, matplotlib is the foundation that most other Python visualization libraries build upon. It's powerful, flexible, and gives you complete control over every pixel.

```python
import matplotlib.pyplot as plt

# The classic matplotlib import
# pyplot gives you a MATLAB-like interface
```

### Understanding Figures and Axes

Matplotlib has a specific architecture you need to understand. A **figure** is the entire window or page—think of it as your canvas. **Axes** are the actual plots within that figure (yes, the name is confusing—it's not about x-axis and y-axis, but the plot area itself).

```python
import matplotlib.pyplot as plt

# Create a figure and axes
fig, ax = plt.subplots()

# The figure is the container
# The axes (ax) is where you actually draw
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

This figure/axes separation becomes important when you create multiple plots. You can have one figure with many axes (subplots), giving you complete control over complex layouts.

```python
# Create a figure with 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# axes is now a 2x2 array of Axes objects
axes[0, 0].plot([1, 2, 3], [1, 2, 3])  # Top-left
axes[0, 1].bar([1, 2, 3], [3, 1, 2])   # Top-right
axes[1, 0].scatter([1, 2, 3], [2, 3, 1])  # Bottom-left
axes[1, 1].hist([1, 1, 2, 3, 3, 3, 4])    # Bottom-right

plt.tight_layout()  # Prevent overlapping
plt.show()
```

### The Plot Function

The **plot function** is your basic drawing tool. At its simplest, it connects points with lines:

```python
import matplotlib.pyplot as plt

# Basic plot: x values, y values
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]  # y = x²

plt.plot(x, y)
plt.show()
```

But `plot()` can do much more with its many parameters for **line styles**, **markers**, and **colors**.

| Parameter | Example | Description |
|-----------|---------|-------------|
| `color` or `c` | `'red'`, `'#FF5733'`, `'C0'` | Line/marker color |
| `linestyle` or `ls` | `'-'`, `'--'`, `':'`, `'-.'` | Line pattern |
| `linewidth` or `lw` | `2`, `0.5` | Line thickness |
| `marker` | `'o'`, `'s'`, `'^'`, `'*'` | Point markers |
| `markersize` or `ms` | `10`, `5` | Marker size |

```python
# Customized plot
plt.plot(x, y,
         color='purple',
         linestyle='--',
         linewidth=2,
         marker='o',
         markersize=8)
plt.show()
```

!!! tip "The Format String Shortcut"
    Matplotlib has a shortcut: `plt.plot(x, y, 'ro--')` means red (`r`), circles (`o`), dashed line (`--`). It's compact but can be cryptic—use named parameters for clarity in your code.

## Essential Plot Types

Different data calls for different visualizations. Let's master the essential types.

### Line Plot: Trends Over Time

A **line plot** connects data points with lines, perfect for showing how values change over time or across a sequence.

```python
import matplotlib.pyplot as plt
import numpy as np

# Stock price over 30 days
days = np.arange(1, 31)
stock_price = 100 + np.cumsum(np.random.randn(30) * 2)

plt.figure(figsize=(10, 5))
plt.plot(days, stock_price, color='blue', linewidth=2)
plt.title('Stock Price Over 30 Days')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()
```

**When to use:** Time series, trends, continuous data, showing progression.

### Scatter Plot: Relationships Between Variables

A **scatter plot** shows individual data points without connecting them, revealing relationships between two variables.

```python
# Height vs Weight
height = [160, 165, 170, 175, 180, 185, 190]
weight = [55, 62, 68, 75, 82, 88, 95]

plt.figure(figsize=(8, 6))
plt.scatter(height, weight, color='green', s=100, alpha=0.7)
plt.title('Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

**When to use:** Correlation analysis, comparing two numeric variables, finding clusters or outliers.

### Bar Chart: Comparing Categories

A **bar chart** uses rectangular bars to compare values across categories.

```python
# Programming language popularity
languages = ['Python', 'JavaScript', 'Java', 'C++', 'Go']
popularity = [28, 25, 18, 12, 8]

plt.figure(figsize=(10, 6))
plt.bar(languages, popularity, color=['#3776AB', '#F7DF1E', '#ED8B00', '#00599C', '#00ADD8'])
plt.title('Programming Language Popularity')
plt.xlabel('Language')
plt.ylabel('Popularity (%)')
plt.show()
```

**When to use:** Comparing categories, showing rankings, discrete data.

### Histogram: Distribution of Values

A **histogram** shows how values are distributed across ranges (bins). Unlike bar charts, histograms show continuous data grouped into intervals.

```python
import numpy as np

# Test scores
scores = np.random.normal(75, 10, 1000)  # Mean=75, std=10

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=20, color='coral', edgecolor='black', alpha=0.7)
plt.title('Distribution of Test Scores')
plt.xlabel('Score')
plt.ylabel('Number of Students')
plt.axvline(x=75, color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()
```

**When to use:** Understanding distribution shape, finding outliers, comparing to normal distribution.

### Box Plot: Statistical Summary

A **box plot** (or box-and-whisker plot) shows the five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It's excellent for comparing distributions and spotting outliers.

```python
# Compare scores across three classes
class_a = np.random.normal(75, 8, 30)
class_b = np.random.normal(70, 12, 30)
class_c = np.random.normal(80, 5, 30)

plt.figure(figsize=(8, 6))
plt.boxplot([class_a, class_b, class_c], labels=['Class A', 'Class B', 'Class C'])
plt.title('Score Distribution by Class')
plt.ylabel('Score')
plt.show()
```

**When to use:** Comparing distributions, identifying outliers, showing spread and central tendency.

### Pie Chart: Parts of a Whole

A **pie chart** shows proportions of a whole. Use them sparingly—they're often harder to read than bar charts.

```python
# Budget allocation
categories = ['Rent', 'Food', 'Transport', 'Entertainment', 'Savings']
amounts = [1200, 400, 200, 150, 250]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

plt.figure(figsize=(8, 8))
plt.pie(amounts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Monthly Budget Allocation')
plt.show()
```

**When to use:** Showing proportions of a whole (when you have 2-5 categories). Avoid for comparisons.

#### Diagram: Chart Type Selection Guide

<details markdown="1">
    <summary>Which Chart Should I Use?</summary>
    Type: infographic

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Help students choose the appropriate chart type for their data and question

    Purpose: Decision guide for selecting visualization types

    Layout: Flowchart/decision tree with visual examples

    Starting question: "What do you want to show?"

    Branch 1: "Comparison"
    - Few categories → Bar Chart (vertical)
    - Many categories → Bar Chart (horizontal)
    - Over time → Line Chart (multiple lines)
    - Visual: Small example of each

    Branch 2: "Distribution"
    - Single variable → Histogram
    - Compare distributions → Box Plot
    - Density estimate → KDE Plot
    - Visual: Small example of each

    Branch 3: "Relationship"
    - Two variables → Scatter Plot
    - Three variables → Bubble Chart (size = 3rd var)
    - Many variables → Pair Plot
    - Visual: Small example of each

    Branch 4: "Composition"
    - Static → Pie Chart (2-5 parts only!)
    - Over time → Stacked Area Chart
    - Many parts → Treemap
    - Visual: Small example of each

    Branch 5: "Trend"
    - Over time → Line Chart
    - With uncertainty → Line + Confidence Band
    - Multiple series → Multiple Lines + Legend
    - Visual: Small example of each

    Warning callouts:
    - "Pie charts: Only use with 2-5 categories"
    - "3D charts: Avoid! They distort perception"
    - "Dual y-axes: Use carefully, can mislead"

    Interactive elements:
    - Hover over each chart type to see larger example
    - Click to see code snippet

    Visual style: Clean flowchart with colorful chart thumbnails

    Implementation: SVG with interactive JavaScript
</details>

## Customizing Your Visualizations

Raw plots are just the beginning. Professional visualizations need polish. Let's master **plot customization**.

### Title and Axis Labels

Every chart needs a **title** that explains what it shows and **axis labels** that explain the variables:

```python
plt.figure(figsize=(10, 6))
plt.plot(x, y)

# Title with size and style
plt.title('Quadratic Growth: y = x²', fontsize=16, fontweight='bold')

# Axis labels
plt.xlabel('Input Value (x)', fontsize=12)
plt.ylabel('Output Value (y)', fontsize=12)

plt.show()
```

### Legend

A **legend** identifies multiple data series. Position it where it doesn't obscure data:

```python
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = x²')
plt.plot(x, [i**3/10 for i in x], label='y = x³/10')
plt.plot(x, [2**i for i in x], label='y = 2^x')

plt.legend(loc='upper left')  # Options: 'best', 'upper right', 'lower left', etc.
plt.title('Growth Functions Comparison')
plt.show()
```

### Grid and Annotations

A **grid** helps readers estimate values. **Annotations** highlight specific points:

```python
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-o')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add annotation pointing to a specific point
plt.annotate('Maximum value',
             xy=(5, 25),           # Point to annotate
             xytext=(3.5, 20),     # Text position
             fontsize=12,
             arrowprops=dict(arrowstyle='->', color='red'))

plt.title('Annotated Plot')
plt.show()
```

### Figure Size and Saving

Control **figure size** for different outputs and **save** your work:

```python
# Create figure with specific size (width, height in inches)
plt.figure(figsize=(12, 6))
plt.plot(x, y)
plt.title('Wide Format Plot')

# Save to file (before plt.show()!)
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('my_plot.pdf')  # Vector format for publications
plt.savefig('my_plot.svg')  # Vector format for web

plt.show()
```

| Format | Best For | File Size |
|--------|----------|-----------|
| PNG | Web, presentations | Medium |
| PDF | Publications, print | Small |
| SVG | Web (scalable) | Small |
| JPG | Photos (avoid for charts) | Small |

## Subplots: Multiple Views

**Subplots** let you show multiple related visualizations together:

```python
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: Line plot
axes[0, 0].plot(x, np.sin(x), 'b-')
axes[0, 0].set_title('Sine Wave')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('sin(x)')

# Top-right: Scatter plot
axes[0, 1].scatter(np.random.rand(50), np.random.rand(50), c='green', alpha=0.6)
axes[0, 1].set_title('Random Scatter')

# Bottom-left: Bar chart
axes[1, 0].bar(['A', 'B', 'C', 'D'], [23, 45, 56, 78], color='coral')
axes[1, 0].set_title('Category Comparison')

# Bottom-right: Histogram
axes[1, 1].hist(np.random.normal(0, 1, 1000), bins=30, color='purple', alpha=0.7)
axes[1, 1].set_title('Normal Distribution')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
```

!!! tip "Sharing Axes"
    Use `sharex=True` or `sharey=True` in `subplots()` to align axes across plots—essential for fair comparisons.

## Seaborn: Beautiful Statistics

The **Seaborn library** builds on matplotlib to provide beautiful default styles and specialized **statistical plots**. It's perfect for exploratory data analysis.

```python
import seaborn as sns
import pandas as pd

# Seaborn works beautifully with DataFrames
df = pd.DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.normal(0, 1, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Set the style
sns.set_style('whitegrid')

# Create a scatter plot with regression line
sns.lmplot(data=df, x='x', y='y', hue='category', height=6)
plt.title('Scatter Plot with Regression by Category')
plt.show()
```

Seaborn's statistical plots include:

- `sns.histplot()` - Enhanced histograms with KDE
- `sns.boxplot()` - Box plots with category support
- `sns.violinplot()` - Distribution shape visualization
- `sns.heatmap()` - Correlation matrices
- `sns.pairplot()` - All pairwise relationships

```python
# Correlation heatmap
correlation_matrix = df[['x', 'y']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

## Plotly: Interactive Visualization

Now for the exciting part! **Plotly** creates interactive visualizations that users can explore—zoom, pan, hover for details, and more. This is what modern data visualization looks like.

```python
import plotly.express as px
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [100, 120, 140, 135, 160, 180],
    'Profit': [20, 25, 30, 28, 35, 40]
})

# Create interactive line chart - it's this simple!
fig = px.line(df, x='Month', y='Sales',
              title='Monthly Sales Trend',
              markers=True)
fig.show()
```

When you run this, you get a chart where you can:

- **Hover** over points to see exact values
- **Zoom** by clicking and dragging
- **Pan** by holding shift and dragging
- **Download** as PNG with one click
- **Toggle** data series on/off via legend

### Why Plotly Changes Everything

| Feature | Matplotlib | Plotly |
|---------|-----------|--------|
| Default output | Static image | Interactive HTML |
| Hover tooltips | Manual coding | Automatic |
| Zoom/Pan | Not available | Built-in |
| Web embedding | Export as image | Native HTML |
| Learning curve | Medium-High | Low-Medium |
| Customization | Maximum | High |

### Plotly Express: The Fast Lane

`plotly.express` (imported as `px`) provides high-level functions for common chart types:

```python
import plotly.express as px

# Interactive scatter plot
df = px.data.iris()  # Built-in sample dataset
fig = px.scatter(df, x='sepal_width', y='sepal_length',
                 color='species',
                 size='petal_length',
                 hover_data=['petal_width'],
                 title='Iris Dataset: Sepal Dimensions')
fig.show()
```

#### Diagram: Plotly Interactive Features MicroSim

<details markdown="1">
    <summary>Interactive Chart Exploration Playground</summary>
    Type: microsim

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Let students experience and practice using Plotly's interactive features

    Canvas layout (850x600px):
    - Main area (850x450): Interactive Plotly chart
    - Bottom panel (850x150): Feature buttons and instructions

    Visual elements:
    - Sample scatter plot with 50+ data points
    - Multiple colored categories
    - Visible toolbar (zoom, pan, select, download)
    - Hover tooltip showing data values

    Interactive features to demonstrate:
    1. HOVER: Move mouse over points to see tooltips
    2. ZOOM: Click-drag to zoom into a region
    3. PAN: Shift+drag to pan around
    4. BOX SELECT: Draw box to select points
    5. LASSO SELECT: Freeform selection
    6. RESET: Double-click to reset view
    7. DOWNLOAD: Click camera icon to save PNG
    8. LEGEND: Click legend items to toggle series

    Challenge tasks (bottom panel):
    - "Zoom into the cluster in the upper right"
    - "Select all points in category A"
    - "Find the outlier with the highest y-value"
    - "Download the chart as PNG"

    Progress tracker:
    - Checkboxes for each feature used
    - "You've explored X of 8 interactive features!"

    Behavior:
    - Track which features student has used
    - Provide hints for unexplored features
    - Celebrate when all features discovered

    Visual style: Modern dashboard aesthetic

    Implementation: Embedded Plotly.js chart with custom tracking overlay
</details>

### Interactive Line Charts

```python
import plotly.express as px
import pandas as pd
import numpy as np

# Multiple time series
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'Date': dates,
    'Product A': np.cumsum(np.random.randn(100)) + 100,
    'Product B': np.cumsum(np.random.randn(100)) + 100,
    'Product C': np.cumsum(np.random.randn(100)) + 100
})

# Melt for plotly format
df_melted = df.melt(id_vars='Date', var_name='Product', value_name='Sales')

fig = px.line(df_melted, x='Date', y='Sales', color='Product',
              title='Product Sales Over Time')

# Customize interactivity
fig.update_layout(
    hovermode='x unified',  # Show all values at same x position
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()
```

### Interactive Bar Charts

```python
# Animated bar chart
df = px.data.gapminder()
df_2007 = df[df['year'] == 2007].nlargest(10, 'pop')

fig = px.bar(df_2007, x='country', y='pop', color='continent',
             title='Top 10 Countries by Population (2007)',
             labels={'pop': 'Population', 'country': 'Country'},
             hover_data=['gdpPercap', 'lifeExp'])

fig.update_layout(xaxis_tickangle=-45)
fig.show()
```

### Interactive Scatter Plots

```python
# Bubble chart with animation
df = px.data.gapminder()

fig = px.scatter(df, x='gdpPercap', y='lifeExp',
                 animation_frame='year',
                 animation_group='country',
                 size='pop', color='continent',
                 hover_name='country',
                 log_x=True,
                 size_max=60,
                 range_x=[100, 100000],
                 range_y=[25, 90],
                 title='Global Development: GDP vs Life Expectancy')

fig.show()
```

This creates the famous "Gapminder" visualization that Hans Rosling made famous—an animated bubble chart showing how countries develop over time!

### Interactive Histograms and Box Plots

```python
# Interactive histogram with marginal plots
df = px.data.tips()

fig = px.histogram(df, x='total_bill', color='sex',
                   marginal='box',  # Add box plot on margin
                   hover_data=df.columns,
                   title='Distribution of Total Bills')

fig.show()
```

```python
# Interactive box plot
fig = px.box(df, x='day', y='total_bill', color='smoker',
             notched=True,  # Show confidence interval
             title='Bill Distribution by Day and Smoking Status')

fig.show()
```

### Customizing Plotly Charts

Plotly offers extensive customization through `update_layout()` and `update_traces()`:

```python
fig = px.scatter(df, x='total_bill', y='tip', color='day')

# Customize layout
fig.update_layout(
    title=dict(text='Tips vs Total Bill', font=dict(size=24)),
    xaxis_title='Total Bill ($)',
    yaxis_title='Tip ($)',
    legend_title='Day of Week',
    font=dict(family='Arial', size=14),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Customize the data points
fig.update_traces(
    marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)

# Add gridlines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

fig.show()
```

### Subplots in Plotly

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create subplot grid
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('Line', 'Bar', 'Scatter', 'Histogram'))

# Add traces to each subplot
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines'), row=1, col=1)
fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]), row=1, col=2)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 3, 2], mode='markers'), row=2, col=1)
fig.add_trace(go.Histogram(x=np.random.randn(500)), row=2, col=2)

fig.update_layout(height=600, width=800, title_text="Multiple Plot Types")
fig.show()
```

### Saving Plotly Charts

```python
# Save as interactive HTML (can be opened in browser)
fig.write_html('my_interactive_chart.html')

# Save as static image (requires kaleido package)
fig.write_image('my_chart.png', scale=2)  # scale=2 for higher resolution
fig.write_image('my_chart.pdf')
fig.write_image('my_chart.svg')
```

#### Diagram: Plotly Code Pattern Reference

<details markdown="1">
    <summary>Plotly Express Quick Reference Card</summary>
    Type: infographic

    Bloom Taxonomy: Remember (L1)

    Learning Objective: Provide quick reference for common Plotly Express patterns

    Purpose: Cheat sheet for Plotly Express functions and parameters

    Layout: Four-quadrant reference card

    Quadrant 1: "Common Chart Functions"
    ```
    px.line()      - Line charts
    px.scatter()   - Scatter plots
    px.bar()       - Bar charts
    px.histogram() - Histograms
    px.box()       - Box plots
    px.pie()       - Pie charts
    px.area()      - Area charts
    px.violin()    - Violin plots
    ```

    Quadrant 2: "Essential Parameters"
    ```
    x, y           - Data columns
    color          - Color by category
    size           - Size by value
    hover_data     - Extra tooltip info
    title          - Chart title
    labels         - Rename axis labels
    facet_col      - Small multiples (columns)
    facet_row      - Small multiples (rows)
    animation_frame - Animate over values
    ```

    Quadrant 3: "Layout Customization"
    ```python
    fig.update_layout(
        title='My Title',
        xaxis_title='X Label',
        yaxis_title='Y Label',
        legend_title='Legend',
        template='plotly_white',
        height=500,
        width=800
    )
    ```

    Quadrant 4: "Saving Options"
    ```python
    # Interactive HTML
    fig.write_html('chart.html')

    # Static images
    fig.write_image('chart.png')
    fig.write_image('chart.pdf')
    fig.write_image('chart.svg')

    # In Jupyter
    fig.show()
    ```

    Bottom strip: "Templates"
    - plotly, plotly_white, plotly_dark
    - ggplot2, seaborn, simple_white
    - Visual swatches of each

    Color scheme: Purple gradient (Plotly brand color)

    Interactive elements:
    - Hover for expanded code examples
    - Click to copy code snippet

    Implementation: HTML/CSS grid with copy-to-clipboard JavaScript
</details>

## Real-World Visualization Workflow

Let's put it all together with a complete workflow:

```python
import plotly.express as px
import pandas as pd

# Step 1: Load and clean data
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()

# Step 2: Explore with quick visualizations
fig1 = px.histogram(df, x='revenue', title='Revenue Distribution')
fig1.show()

# Step 3: Create main visualization
fig2 = px.scatter(df,
                  x='marketing_spend',
                  y='revenue',
                  color='region',
                  size='units_sold',
                  hover_name='product',
                  trendline='ols',  # Add regression line
                  title='Marketing Spend vs Revenue by Region')

# Step 4: Customize for presentation
fig2.update_layout(
    title=dict(font=dict(size=24)),
    xaxis_title='Marketing Spend ($)',
    yaxis_title='Revenue ($)',
    legend_title='Region',
    template='plotly_white'
)

# Step 5: Save for sharing
fig2.write_html('marketing_analysis.html')  # Interactive version
fig2.write_image('marketing_analysis.png', scale=2)  # For presentations

fig2.show()
```

#### Diagram: Visualization Design MicroSim

<details markdown="1">
    <summary>Chart Design Playground</summary>
    Type: microsim

    Bloom Taxonomy: Create (L6)

    Learning Objective: Let students design and customize their own visualizations interactively

    Canvas layout (900x650px):
    - Left panel (300x650): Controls and options
    - Right panel (600x650): Live chart preview

    Control panel sections:

    Section 1: "Chart Type"
    - Radio buttons: Line, Scatter, Bar, Histogram, Box
    - Visual icon for each type

    Section 2: "Data Selection"
    - Dropdown: X-axis variable
    - Dropdown: Y-axis variable
    - Dropdown: Color by (optional)
    - Dropdown: Size by (optional)

    Section 3: "Customization"
    - Text input: Title
    - Text input: X-axis label
    - Text input: Y-axis label
    - Color picker: Primary color
    - Dropdown: Color palette (categorical)
    - Slider: Marker size (5-50)
    - Slider: Line width (1-5)
    - Toggle: Show grid
    - Toggle: Show legend

    Section 4: "Export"
    - Button: "Copy Code"
    - Button: "Download PNG"
    - Button: "Download HTML"

    Sample dataset:
    - Pre-loaded "tips" style dataset
    - Columns: total_bill, tip, day, time, size, smoker

    Chart preview:
    - Updates in real-time as controls change
    - Fully interactive (zoom, pan, hover)
    - Shows Plotly toolbar

    Code panel (collapsible):
    - Shows Python code that would generate current chart
    - Updates dynamically with changes
    - Syntax highlighted

    Behavior:
    - Every control change immediately updates preview
    - Code panel reflects exact current configuration
    - Copy code button copies to clipboard
    - Download buttons generate files

    Educational features:
    - Tooltips explaining each option
    - "Design tips" suggestions based on data types selected
    - Warnings for bad practices (pie chart with too many categories, etc.)

    Visual style: Modern design tool interface (think Canva/Figma)

    Implementation: p5.js for controls + embedded Plotly.js for preview
</details>

## Choosing the Right Visualization

The most important skill isn't knowing how to make a chart—it's knowing WHICH chart to make. Here's your decision framework:

| Your Question | Best Chart Type | Why |
|---------------|----------------|-----|
| "How does X change over time?" | Line chart | Shows trends and patterns |
| "How are X and Y related?" | Scatter plot | Reveals correlations |
| "How do categories compare?" | Bar chart | Easy comparison |
| "What's the distribution?" | Histogram | Shows shape and spread |
| "How do groups compare statistically?" | Box plot | Shows median, quartiles, outliers |
| "What's the composition?" | Pie chart (2-5 parts) | Shows parts of whole |
| "How do multiple variables relate?" | Pair plot / Scatter matrix | See all relationships |

!!! warning "Visualization Pitfalls to Avoid"
    - **Truncated axes**: Starting y-axis at non-zero exaggerates differences
    - **3D charts**: They look cool but distort perception—avoid them
    - **Too many colors**: Stick to 5-7 distinct colors maximum
    - **Missing labels**: Every chart needs title, axis labels, and legend (if needed)
    - **Pie charts with many slices**: More than 5 categories? Use a bar chart instead

## Best Practices Summary

### The Visualization Checklist

Before sharing any visualization, verify:

- [ ] Clear, descriptive title
- [ ] Labeled axes with units
- [ ] Legend (if multiple series)
- [ ] Appropriate chart type for the data
- [ ] Accessible colors (colorblind-friendly)
- [ ] No unnecessary 3D effects
- [ ] Source cited (if using external data)
- [ ] Interactive features work (for Plotly)

### Code Organization Patterns

```python
# Good: Organized, readable, reusable
def create_sales_chart(df, x_col, y_col, title):
    """Create a customized sales visualization."""
    fig = px.scatter(df, x=x_col, y=y_col,
                     color='region',
                     title=title)

    fig.update_layout(
        template='plotly_white',
        font=dict(family='Arial')
    )

    return fig

# Use the function
chart = create_sales_chart(sales_df, 'spend', 'revenue', 'Marketing ROI')
chart.show()
```

??? question "Chapter 5 Checkpoint: Test Your Understanding"
    **Question:** You have a dataset with columns: `date`, `sales`, `region`, `product_category`. You want to show:
    1. How sales change over time
    2. Sales comparison across regions
    3. The distribution of sales values

    What chart types would you use for each, and would you use matplotlib or Plotly?

    **Click to reveal answer:**

    ```python
    import plotly.express as px

    # 1. Sales over time → Line chart (Plotly for interactivity)
    fig1 = px.line(df, x='date', y='sales', color='region',
                   title='Sales Over Time by Region')

    # 2. Compare regions → Bar chart
    region_totals = df.groupby('region')['sales'].sum().reset_index()
    fig2 = px.bar(region_totals, x='region', y='sales',
                  title='Total Sales by Region')

    # 3. Distribution → Histogram
    fig3 = px.histogram(df, x='sales', nbins=30,
                        title='Distribution of Sales Values')
    ```

    **Why Plotly?** Interactive features let viewers explore the data themselves—hover for details, zoom into interesting regions, and click legend items to focus on specific categories.

!!! success "Achievement Unlocked: Visual Storyteller"
    You can now transform raw numbers into compelling visual narratives. Whether you need static publication graphics (matplotlib), beautiful statistical plots (Seaborn), or interactive web-ready visualizations (Plotly), you have the tools. This is the skill that gets your insights SEEN.

## Key Takeaways

1. **Data visualization** transforms numbers into insights that everyone can understand—it's how data science becomes visible.

2. **Matplotlib** is the foundational library with complete control; understand **figures** (canvas) and **axes** (plot areas).

3. **Seaborn** provides beautiful statistical plots with minimal code—great for exploration.

4. **Plotly** creates interactive visualizations with zoom, pan, hover tooltips—the modern standard for web and presentations.

5. Choose chart types based on your question: **line** for trends, **scatter** for relationships, **bar** for comparisons, **histogram** for distributions, **box** for statistical summaries.

6. **Customize** your plots: meaningful titles, axis labels with units, legends for multiple series, appropriate colors.

7. **Subplots** let you show multiple related views together for comprehensive analysis.

8. **Save** your work: PNG/PDF for static uses, HTML for interactive sharing.

9. **Plotly Express** (`px`) provides high-level functions that create professional interactive charts in one line.

10. The best visualization is one that **answers a question clearly**—not the fanciest chart, but the most appropriate one.

You've now mastered the art of visual communication. In the next chapter, you'll learn the statistical foundations that give your visualizations mathematical backing—the numbers behind the pictures!
