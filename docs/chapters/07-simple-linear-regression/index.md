---
title: Simple Linear Regression
description: Your first predictive model - drawing lines that tell the future
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

# Simple Linear Regression

## Summary

This chapter introduces regression analysis, the foundation of predictive modeling. Students will learn the mathematics behind linear regression, including the least squares method, interpreting coefficients (slope and intercept), and understanding residuals. The chapter covers regression assumptions and teaches students to implement linear regression using scikit-learn. By the end of this chapter, students will be able to build simple linear regression models, interpret their outputs, and make predictions.

## Concepts Covered

This chapter covers the following 25 concepts from the learning graph:

1. Regression Analysis
2. Linear Regression
3. Simple Linear Regression
4. Regression Line
5. Slope
6. Intercept
7. Least Squares Method
8. Residuals
9. Sum of Squared Errors
10. Ordinary Least Squares
11. Regression Coefficients
12. Coefficient Interpretation
13. Prediction
14. Fitted Values
15. Regression Equation
16. Line of Best Fit
17. Assumptions of Regression
18. Linearity Assumption
19. Homoscedasticity
20. Independence Assumption
21. Normality of Residuals
22. Scikit-learn Library
23. LinearRegression Class
24. Fit Method
25. Predict Method

## Prerequisites

This chapter builds on concepts from:

- [Chapter 5: Data Visualization with Matplotlib](../05-data-visualization/index.md)
- [Chapter 6: Statistical Foundations](../06-statistical-foundations/index.md)

---

## From Description to Prediction

Everything you've learned so far has been about understanding data that already exists. Descriptive statistics summarize the past. Visualizations reveal patterns in historical data. Correlation shows relationships between variables.

But here's where data science gets really exciting: **prediction**.

What if, instead of just describing what happened, you could predict what will happen? What if you could look at a student's study hours and predict their exam score? Or see a house's square footage and estimate its price? Or know a car's age and forecast its fuel efficiency?

This is the superpower of **regression analysis**—the ability to draw a line through data that extends into the unknown future. It's the foundation of machine learning, the backbone of forecasting, and your first step into predictive modeling.

In this chapter, you'll learn to build your first predictive model. It's surprisingly simple—just a line—but don't let that fool you. This humble line is one of the most powerful tools in all of data science.

## What is Regression Analysis?

**Regression analysis** is a statistical method for modeling the relationship between variables. It lets you:

1. **Understand** how one variable affects another
2. **Quantify** the strength of that relationship
3. **Predict** values you haven't observed

The term "regression" has a historical origin. In the 1880s, Francis Galton studied the heights of parents and children. He noticed that very tall parents tended to have children shorter than themselves, and very short parents had taller children. Heights "regressed" toward the average. The name stuck, even though modern regression is used for much more than studying heights.

### Linear Regression: The Straight-Line Model

**Linear regression** is the simplest form of regression—it assumes the relationship between variables is a straight line. Despite its simplicity, linear regression is:

- Easy to understand and interpret
- Fast to compute
- Surprisingly effective for many real problems
- The foundation for more complex models

When you have one input variable predicting one output variable, it's called **simple linear regression**. That's what we'll master in this chapter.

```python
import numpy as np
import pandas as pd
import plotly.express as px

# Example: Study hours vs exam score
study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores = [52, 58, 65, 71, 75, 82, 87, 92]

# Create a DataFrame
df = pd.DataFrame({'study_hours': study_hours, 'exam_scores': exam_scores})

# Scatter plot with trendline
fig = px.scatter(df, x='study_hours', y='exam_scores',
                 trendline='ols',
                 title='Study Hours vs Exam Scores',
                 labels={'study_hours': 'Hours Studied', 'exam_scores': 'Exam Score'})
fig.show()
```

See that line Plotly drew through the data? That's a **regression line**—your first predictive model! With it, you can predict the exam score for someone who studied 4.5 hours, even though you don't have that exact data point.

## The Regression Equation

Every straight line can be described by an equation. You probably remember from algebra:

$$y = mx + b$$

In statistics, we write the **regression equation** as:

$$\hat{y} = \beta_0 + \beta_1 x$$

Where:

- $\hat{y}$ (y-hat) = the predicted value
- $x$ = the input variable (predictor, independent variable)
- $\beta_0$ = the **intercept** (where the line crosses the y-axis)
- $\beta_1$ = the **slope** (how much y changes for each unit increase in x)

The $\beta$ values are called **regression coefficients**—they define your model.

### Understanding Slope

The **slope** ($\beta_1$) tells you the rate of change: for every one-unit increase in x, how much does y change?

- **Positive slope**: As x increases, y increases (uphill line)
- **Negative slope**: As x increases, y decreases (downhill line)
- **Zero slope**: x has no effect on y (horizontal line)

```python
# Example interpretation
slope = 5.5  # Our model's slope
print(f"Slope: {slope}")
print(f"Interpretation: For each additional hour of studying,")
print(f"the exam score increases by {slope} points on average.")
```

### Understanding Intercept

The **intercept** ($\beta_0$) is the predicted value when x = 0. It's where the line crosses the y-axis.

```python
intercept = 47.5  # Our model's intercept
print(f"Intercept: {intercept}")
print(f"Interpretation: A student who studies 0 hours")
print(f"would be predicted to score {intercept} points.")
```

!!! warning "Intercept Interpretation Caution"
    The intercept doesn't always have a meaningful interpretation. If x = 0 is outside your data range (like predicting house price for 0 square feet), don't interpret the intercept literally—it's just a mathematical necessity for the line equation.

#### Diagram: Regression Line Anatomy

<details markdown="1">
<summary>Interactive Regression Line Components</summary>
Type: infographic

Bloom Taxonomy: Remember (L1)

Learning Objective: Help students identify and remember the components of a regression line and equation

Purpose: Visual breakdown of regression line with labeled components

Layout: Scatter plot with regression line and labeled callouts

Main visual: Scatter plot (600x400px) showing:
- 10-15 data points with clear linear trend
- Regression line through points
- Y-axis intercept clearly marked
- Rise and run triangle showing slope

Callouts (numbered with leader lines):

1. INTERCEPT (β₀) (pointing to y-axis crossing)
   - "Where line crosses y-axis"
   - "Predicted y when x = 0"
   - "In equation: the constant term"
   - Color: Blue

2. SLOPE (β₁) (pointing to rise/run triangle)
   - "Rise over run"
   - "Change in y per unit change in x"
   - "Positive = uphill, Negative = downhill"
   - Shows: Δy / Δx calculation
   - Color: Red

3. PREDICTED VALUE (ŷ) (pointing to a point on the line)
   - "Value predicted by the model"
   - "Falls exactly on the line"
   - "ŷ = β₀ + β₁x"
   - Color: Green

4. ACTUAL VALUE (y) (pointing to a data point off the line)
   - "Real observed value"
   - "Usually not exactly on line"
   - Color: Orange

5. RESIDUAL (pointing to vertical line between actual and predicted)
   - "Distance from actual to predicted"
   - "Residual = y - ŷ"
   - "What the model got wrong"
   - Color: Purple

Bottom equation display:
ŷ = β₀ + β₁x
With arrows pointing to each component in the equation

Interactive elements:
- Hover over each component for detailed explanation
- Click to highlight related elements
- Toggle to show/hide residuals for all points

Implementation: SVG with JavaScript interactivity
</details>

## Finding the Best Line: Least Squares Method

There are infinite lines you could draw through a scatter plot. So how do we find the best one? We use the **least squares method**.

### Residuals: Measuring Errors

A **residual** is the difference between an actual observed value and the value predicted by the model:

$$\text{Residual} = y - \hat{y} = \text{Actual} - \text{Predicted}$$

```python
# Calculate residuals for our example
actual_scores = [52, 58, 65, 71, 75, 82, 87, 92]
predicted_scores = [47.5 + 5.5*h for h in [1, 2, 3, 4, 5, 6, 7, 8]]

residuals = [actual - predicted for actual, predicted in zip(actual_scores, predicted_scores)]

for i, (actual, pred, resid) in enumerate(zip(actual_scores, predicted_scores, residuals)):
    print(f"Point {i+1}: Actual={actual}, Predicted={pred:.1f}, Residual={resid:.1f}")
```

Residuals tell us how wrong our predictions are:

- **Positive residual**: Model under-predicted (actual > predicted)
- **Negative residual**: Model over-predicted (actual < predicted)
- **Zero residual**: Perfect prediction (actual = predicted)

### Sum of Squared Errors (SSE)

To find the best line, we want to minimize total error. But we can't just add up residuals—positive and negative would cancel out! Instead, we square them first:

$$\text{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \sum_{i=1}^{n}\text{residual}_i^2$$

This is the **sum of squared errors** (also called sum of squared residuals).

```python
# Calculate SSE
sse = sum([r**2 for r in residuals])
print(f"Sum of Squared Errors: {sse:.2f}")
```

### Ordinary Least Squares (OLS)

**Ordinary Least Squares** (OLS) is the method that finds the line minimizing SSE. It's the standard algorithm for linear regression.

The math gives us formulas for the optimal coefficients:

$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

Don't worry about memorizing these—Python will calculate them for you. The important thing is understanding the concept: OLS finds the line that makes the squared prediction errors as small as possible.

#### Diagram: Least Squares MicroSim

<details markdown="1">
<summary>Interactive Least Squares Line Fitting</summary>
Type: microsim

Bloom Taxonomy: Understand (L2)

Learning Objective: Help students understand how the least squares method finds the best-fit line by minimizing squared errors

Canvas layout (900x600px):
- Main area (650x550): Interactive scatter plot with adjustable line
- Right panel (250x550): Controls and error display
- Bottom strip (900x50): SSE meter

Visual elements:
- Scatter plot with 8-12 data points
- Adjustable regression line (can drag slope and intercept)
- Vertical lines from points to line showing residuals
- Squares drawn at each residual (area = squared error)
- Running SSE total displayed prominently

Interactive controls:
- Draggable line: Adjust slope by rotating, intercept by vertical drag
- Slider: Slope (-5 to +5)
- Slider: Intercept (0 to 100)
- Button: "Show Optimal Line" - animates to best fit
- Button: "Reset" - return to initial position
- Toggle: Show/hide residual squares
- Toggle: Show/hide residual values

Display panels:
- Current slope and intercept
- Current SSE
- Optimal SSE (shown after clicking "Show Optimal")
- Percentage improvement from current to optimal

SSE Meter (bottom):
- Visual bar showing current SSE
- Marker showing optimal SSE
- Color gradient: red (high error) → green (low error)

Behavior:
- As line is adjusted, SSE updates in real-time
- Residual squares resize dynamically
- "Show Optimal Line" smoothly animates to least squares solution
- Highlight when current SSE is close to optimal

Educational annotations:
- "Each square's area = squared error for that point"
- "Total area of all squares = SSE"
- "OLS minimizes this total area"

Challenge tasks:
- "Can you get SSE below 50?"
- "Find a line where all residuals are positive"
- "Match the optimal line within 5% SSE"

Visual style: Clean mathematical visualization

Implementation: p5.js with real-time calculations
</details>

## The Line of Best Fit

The **line of best fit** (also called the regression line or trend line) is the line that minimizes SSE. It's the "best" line in the sense that no other straight line would have smaller total squared errors.

```python
import numpy as np
from scipy import stats

# Calculate line of best fit
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
exam_scores = np.array([52, 58, 65, 71, 75, 82, 87, 92])

slope, intercept, r_value, p_value, std_err = stats.linregress(study_hours, exam_scores)

print(f"Line of Best Fit:")
print(f"  Slope (β₁): {slope:.2f}")
print(f"  Intercept (β₀): {intercept:.2f}")
print(f"  Equation: ŷ = {intercept:.2f} + {slope:.2f}x")
```

Properties of the line of best fit:

- It always passes through the point $(\bar{x}, \bar{y})$ (the means)
- The sum of residuals equals zero (positive and negative cancel)
- It minimizes SSE among all possible straight lines

## Fitted Values and Predictions

**Fitted values** are the predictions your model makes for the data points you used to build it. They're the y-values on the regression line at each x in your training data.

```python
# Calculate fitted values
fitted_values = intercept + slope * study_hours

print("Fitted Values (Predictions for Training Data):")
for hours, actual, fitted in zip(study_hours, exam_scores, fitted_values):
    print(f"  {hours} hours: Actual={actual}, Fitted={fitted:.1f}")
```

**Prediction** uses the regression equation to estimate y for new x values—values you haven't observed yet.

```python
# Make predictions for new data
new_hours = [4.5, 9, 10]

for hours in new_hours:
    predicted_score = intercept + slope * hours
    print(f"Predicted score for {hours} hours: {predicted_score:.1f}")
```

!!! tip "Extrapolation Warning"
    Be careful predicting far outside your data range! If your data goes from 1-8 hours, predicting for 20 hours is risky. The linear relationship might not hold for extreme values. This is called **extrapolation** and can lead to unreliable predictions.

## Interpreting Regression Coefficients

**Coefficient interpretation** is crucial—it's how you extract meaning from your model.

### Interpreting the Slope

The slope tells you the **effect size**: how much y changes per unit change in x.

```python
# Our model: score = 47.5 + 5.5 * hours
print(f"Slope Interpretation:")
print(f"  For each additional hour of studying,")
print(f"  exam score increases by {slope:.1f} points on average.")
print(f"")
print(f"  Study 2 more hours → expect {2 * slope:.1f} more points")
print(f"  Study 3 more hours → expect {3 * slope:.1f} more points")
```

The slope also tells you direction:

- **Positive slope (5.5)**: More studying → higher scores (positive relationship)
- If slope were **negative**: More of x → less of y (inverse relationship)

### Interpreting the Intercept

The intercept is the predicted value when x = 0.

```python
print(f"Intercept Interpretation:")
print(f"  A student who studies 0 hours is predicted")
print(f"  to score {intercept:.1f} points.")
```

But context matters! Does x = 0 make sense?

| Scenario | x = 0 Meaningful? | Intercept Interpretation |
|----------|-------------------|-------------------------|
| Study hours → Score | Maybe | Baseline score without studying |
| House sq ft → Price | No | Price of 0 sq ft house? Nonsense! |
| Age → Height (children) | No | Height at age 0? (birth height, maybe) |
| Temperature → Ice cream sales | Maybe | Sales at 0°F (very cold!) |

| Coefficient | Symbol | Interpretation |
|-------------|--------|----------------|
| Slope | β₁ | Change in y per unit change in x |
| Intercept | β₀ | Predicted y when x = 0 |

## Assumptions of Regression

For linear regression to give reliable results, certain **assumptions** should hold. Think of these as the "fine print" of your model.

### 1. Linearity Assumption

The **linearity assumption** requires that the relationship between x and y is actually linear (a straight line fits well).

```python
import plotly.express as px

# Check linearity with scatter plot
fig = px.scatter(df, x='study_hours', y='exam_scores',
                 trendline='ols',
                 title='Checking Linearity: Do Points Follow a Line?')
fig.show()
```

If the relationship is curved, linear regression will give poor predictions. You'd need polynomial regression or other techniques.

### 2. Independence Assumption

The **independence assumption** requires that observations are independent of each other. One data point shouldn't affect another.

Violations occur when:

- Time series data (today's value depends on yesterday's)
- Clustered data (students in same class aren't independent)
- Repeated measurements on same subjects

### 3. Homoscedasticity

**Homoscedasticity** (homo = same, scedasticity = scatter) means the spread of residuals is constant across all x values.

```python
# Check homoscedasticity with residual plot
residuals = exam_scores - fitted_values

fig = px.scatter(x=fitted_values, y=residuals,
                 title='Residual Plot: Checking Homoscedasticity',
                 labels={'x': 'Fitted Values', 'y': 'Residuals'})
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.show()
```

- **Good**: Residuals form a random horizontal band around zero
- **Bad**: Residuals fan out (spread increases with x) = heteroscedasticity

### 4. Normality of Residuals

The **normality of residuals** assumption requires that residuals follow a normal distribution. This matters for confidence intervals and hypothesis tests.

```python
import plotly.figure_factory as ff

# Check normality with histogram
fig = px.histogram(x=residuals, nbins=10,
                   title='Distribution of Residuals',
                   labels={'x': 'Residual', 'y': 'Count'})
fig.show()
```

For small samples, use a Q-Q plot:

```python
from scipy import stats
import plotly.graph_objects as go

# Q-Q plot
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
sorted_residuals = np.sort(residuals)

fig = go.Figure()
fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers'))
fig.add_trace(go.Scatter(x=[-3, 3], y=[-3*np.std(residuals), 3*np.std(residuals)],
                         mode='lines', line=dict(dash='dash')))
fig.update_layout(title='Q-Q Plot: Checking Normality of Residuals',
                  xaxis_title='Theoretical Quantiles',
                  yaxis_title='Sample Quantiles')
fig.show()
```

#### Diagram: Regression Assumptions Checker MicroSim

<details markdown="1">
<summary>Interactive Assumption Diagnostic Tool</summary>
Type: microsim

Bloom Taxonomy: Analyze (L4)

Learning Objective: Help students diagnose regression assumption violations through interactive visualizations

Canvas layout (900x650px):
- Top left (450x300): Original scatter plot with regression line
- Top right (450x300): Residual vs fitted plot
- Bottom left (450x300): Residual histogram
- Bottom right (450x300): Q-Q plot of residuals

Visual elements:
- All four diagnostic plots update together
- Traffic light indicators (green/yellow/red) for each assumption
- Assumption status panel

Interactive controls:
- Dropdown: Dataset selector
  - "Good Data" (all assumptions met)
  - "Non-linear" (curved relationship)
  - "Heteroscedastic" (fan-shaped residuals)
  - "Non-normal residuals" (skewed errors)
  - "Outliers present"
  - "Custom" (add/drag points)
- Button: "Diagnose" - highlights violations
- Toggle: Show/hide assumption guidelines
- Draggable points in custom mode

Assumption indicators:
1. LINEARITY
   - Green: Points follow line well
   - Yellow: Slight curvature
   - Red: Clear non-linear pattern

2. INDEPENDENCE
   - Note: "Cannot diagnose from plot alone"
   - Checkbox: "Data is from independent observations"

3. HOMOSCEDASTICITY
   - Green: Constant spread in residual plot
   - Yellow: Slight fanning
   - Red: Clear funnel shape

4. NORMALITY
   - Green: Histogram bell-shaped, Q-Q on line
   - Yellow: Slight deviation
   - Red: Clear non-normality

Behavior:
- Selecting dataset updates all four plots
- Traffic lights update based on diagnostic rules
- Tooltips explain what each violation means
- "Diagnose" button highlights specific problem areas

Educational annotations:
- "Look for patterns in the residual plot"
- "Points should follow the diagonal in Q-Q plot"
- "Residuals should be roughly bell-shaped"

Visual style: Dashboard layout with coordinated plots

Implementation: p5.js with Plotly.js for statistical plots
</details>

| Assumption | What to Check | Good Sign | Bad Sign |
|------------|---------------|-----------|----------|
| Linearity | Scatter plot | Points follow line | Curved pattern |
| Independence | Study design | Random sampling | Clustered/time data |
| Homoscedasticity | Residual plot | Even spread | Fan/funnel shape |
| Normality | Histogram/Q-Q | Bell curve, diagonal line | Skewed, curved Q-Q |

!!! tip "When Assumptions Are Violated"
    Minor violations often don't matter much. Linear regression is fairly robust. But serious violations require action: transform variables, use robust regression, or try different models. Always check assumptions!

## Implementing Linear Regression with Scikit-learn

Now let's build regression models the professional way using the **scikit-learn library** (also called sklearn). It's the most popular machine learning library in Python.

```python
# Install if needed: pip install scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np
```

### The LinearRegression Class

The **LinearRegression class** is scikit-learn's implementation of ordinary least squares regression.

```python
from sklearn.linear_model import LinearRegression

# Create a LinearRegression object
model = LinearRegression()

print(type(model))  # <class 'sklearn.linear_model._base.LinearRegression'>
```

### The Fit Method

The **fit method** trains the model—it calculates the optimal coefficients from your data.

```python
# Prepare data (sklearn needs 2D array for X)
X = np.array(study_hours).reshape(-1, 1)  # Reshape to column vector
y = np.array(exam_scores)

# Fit the model
model.fit(X, y)

# Access the learned coefficients
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Slope (β₁): {model.coef_[0]:.2f}")
```

The `.fit()` method:

1. Takes X (features) and y (target)
2. Calculates optimal coefficients using OLS
3. Stores them in `model.intercept_` and `model.coef_`
4. Returns the model object (for method chaining)

### The Predict Method

The **predict method** uses the trained model to make predictions.

```python
# Make predictions for training data
y_pred = model.predict(X)
print("Fitted values:", y_pred)

# Make predictions for new data
X_new = np.array([[4.5], [9], [10]])  # Note: 2D array
predictions = model.predict(X_new)

for hours, score in zip([4.5, 9, 10], predictions):
    print(f"Predicted score for {hours} hours: {score:.1f}")
```

### Complete Scikit-learn Workflow

Here's the standard pattern you'll use for all sklearn models:

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Prepare data
df = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'exam_scores': [52, 58, 65, 71, 75, 82, 87, 92]
})

X = df[['study_hours']]  # Features (2D DataFrame or array)
y = df['exam_scores']     # Target (1D)

# 2. Create model
model = LinearRegression()

# 3. Fit model
model.fit(X, y)

# 4. Make predictions
y_pred = model.predict(X)

# 5. Inspect results
print(f"Equation: ŷ = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")

# 6. Visualize
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['study_hours'], y=df['exam_scores'],
                         mode='markers', name='Actual'))
fig.add_trace(go.Scatter(x=df['study_hours'], y=y_pred,
                         mode='lines', name='Predicted'))
fig.update_layout(title='Linear Regression with Scikit-learn',
                  xaxis_title='Study Hours',
                  yaxis_title='Exam Score')
fig.show()
```

#### Diagram: Scikit-learn Workflow

<details markdown="1">
<summary>Machine Learning Pipeline Flowchart</summary>
Type: workflow

Bloom Taxonomy: Apply (L3)

Learning Objective: Help students memorize and apply the standard scikit-learn workflow

Purpose: Visual guide for the fit-predict pattern

Visual style: Horizontal flowchart with code snippets

Steps (left to right):

1. IMPORT
   Icon: Package/box
   Code: `from sklearn.linear_model import LinearRegression`
   Hover text: "Import the model class you need"
   Color: Blue

2. PREPARE DATA
   Icon: Table/spreadsheet
   Code: `X = df[['feature']]` and `y = df['target']`
   Hover text: "X must be 2D, y is 1D"
   Color: Green
   Warning note: "X needs double brackets!"

3. CREATE MODEL
   Icon: Gear/factory
   Code: `model = LinearRegression()`
   Hover text: "Instantiate the model object"
   Color: Orange

4. FIT MODEL
   Icon: Brain/learning
   Code: `model.fit(X, y)`
   Hover text: "Train on your data - learns coefficients"
   Color: Purple
   Output: "model.coef_, model.intercept_"

5. PREDICT
   Icon: Crystal ball
   Code: `y_pred = model.predict(X_new)`
   Hover text: "Generate predictions for any X"
   Color: Red

6. EVALUATE
   Icon: Checkmark/chart
   Code: `model.score(X, y)` or metrics
   Hover text: "Assess model quality"
   Color: Teal

Annotations:
- Arrow from "FIT" to coefficients stored
- Note: "This pattern works for ALL sklearn models!"
- Common errors callout: "Forgot to reshape X?", "Wrong array shape?"

Interactive elements:
- Click each step to see full code example
- Hover for detailed explanation
- Toggle between LinearRegression and other model examples

Implementation: SVG with JavaScript interactivity
</details>

## Putting It All Together: A Complete Example

Let's work through a complete regression analysis from start to finish:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate realistic data: House size vs Price
np.random.seed(42)
n = 50

# True relationship: Price = 50000 + 200 * sqft + noise
sqft = np.random.uniform(800, 2500, n)
price = 50000 + 200 * sqft + np.random.normal(0, 30000, n)

df = pd.DataFrame({'sqft': sqft, 'price': price})

# 1. VISUALIZE THE DATA
fig = px.scatter(df, x='sqft', y='price',
                 title='House Size vs Price',
                 labels={'sqft': 'Square Feet', 'price': 'Price ($)'})
fig.show()

# 2. FIT THE MODEL
X = df[['sqft']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

print("=== MODEL RESULTS ===")
print(f"Intercept: ${model.intercept_:,.0f}")
print(f"Slope: ${model.coef_[0]:.2f} per sqft")
print(f"\nEquation: Price = ${model.intercept_:,.0f} + ${model.coef_[0]:.2f} × sqft")

# 3. INTERPRET COEFFICIENTS
print("\n=== INTERPRETATION ===")
print(f"• Each additional square foot adds ${model.coef_[0]:.2f} to the price")
print(f"• A 100 sqft increase adds ${100 * model.coef_[0]:,.0f}")
print(f"• Base price (theoretical 0 sqft): ${model.intercept_:,.0f}")

# 4. MAKE PREDICTIONS
y_pred = model.predict(X)

# Predict for new houses
new_houses = pd.DataFrame({'sqft': [1000, 1500, 2000, 2500]})
new_predictions = model.predict(new_houses)

print("\n=== PREDICTIONS ===")
for sqft_val, pred in zip(new_houses['sqft'], new_predictions):
    print(f"  {sqft_val} sqft → ${pred:,.0f}")

# 5. CHECK MODEL FIT
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\n=== MODEL QUALITY ===")
print(f"R² Score: {r2:.3f} ({r2*100:.1f}% of variance explained)")
print(f"RMSE: ${rmse:,.0f} (typical prediction error)")

# 6. VISUALIZE RESULTS
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Regression Fit', 'Residual Plot'])

# Scatter with regression line
fig.add_trace(go.Scatter(x=df['sqft'], y=df['price'],
                         mode='markers', name='Actual'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['sqft'], y=y_pred,
                         mode='lines', name='Predicted', line=dict(color='red')),
              row=1, col=1)

# Residual plot
residuals = y - y_pred
fig.add_trace(go.Scatter(x=y_pred, y=residuals,
                         mode='markers', name='Residuals'), row=1, col=2)
fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

fig.update_layout(title='Complete Regression Analysis',
                  height=400, width=900)
fig.show()

# 7. CHECK ASSUMPTIONS
print("\n=== ASSUMPTION CHECKS ===")
print("✓ Linearity: Scatter plot shows linear pattern")
print("✓ Homoscedasticity: Residuals have roughly constant spread")
print("✓ Normality: Check residual histogram (code above)")
```

#### Diagram: Interactive Regression Builder MicroSim

<details markdown="1">
<summary>Build Your Own Regression Model</summary>
Type: microsim

Bloom Taxonomy: Create (L6)

Learning Objective: Let students build, visualize, and interpret their own regression models interactively

Canvas layout (950x700px):
- Left panel (600x700): Main visualization area
  - Top (600x400): Scatter plot with regression line
  - Bottom (600x300): Residual plot
- Right panel (350x700): Controls, coefficients, interpretation

Visual elements:
- Interactive scatter plot
- Regression line (updates with data)
- Residual lines connecting points to line
- Coefficient display
- Equation display
- R² score gauge

Data options:
- Preset datasets:
  - "Study Hours vs Scores" (positive, strong)
  - "House Size vs Price" (positive, moderate)
  - "Car Age vs Value" (negative)
  - "Random Data" (no relationship)
- Custom: Click to add points

Interactive controls:
- Dropdown: Select dataset
- Button: "Add Point" (click on plot to add)
- Button: "Remove Point" (click to remove)
- Button: "Fit Model" - calculates regression
- Button: "Clear All"
- Slider: Noise level (for preset datasets)
- Toggle: Show residuals
- Toggle: Show confidence band

Right panel displays:
- Equation: ŷ = β₀ + β₁x (with actual values)
- Interpretation text:
  - "For each unit increase in X, Y changes by [slope]"
  - "When X = 0, predicted Y = [intercept]"
- Model quality:
  - R² score with visual gauge
  - RMSE value
- Assumption indicators (traffic lights)

Prediction tool:
- Input field: "Enter X value"
- Button: "Predict"
- Output: Predicted Y with confidence interval
- Visual: Point added to plot at prediction

Behavior:
- Adding/removing points triggers model refit
- All statistics update in real-time
- Interpretation text updates with coefficient values
- Warning when extrapolating beyond data range

Educational features:
- "What happens if you add an outlier?"
- "Can you create data with R² > 0.9?"
- "What does negative slope look like?"

Visual style: Professional dashboard with clean aesthetics

Implementation: p5.js with real-time OLS calculations
</details>

## Common Pitfalls and Best Practices

### Pitfall 1: Confusing Correlation with Causation

A strong relationship doesn't mean x CAUSES y. Ice cream sales predict drowning deaths (both increase in summer), but ice cream doesn't cause drowning!

### Pitfall 2: Extrapolating Too Far

Your model is only reliable within the range of your training data. Predicting house prices for 50,000 square feet when your data only goes to 3,000 is dangerous.

### Pitfall 3: Ignoring Assumptions

Always check your assumptions! A model fit on data with severe violations gives misleading results.

### Pitfall 4: Forgetting to Reshape X

Scikit-learn needs X as a 2D array. The most common error:

```python
# WRONG - will cause error
X = df['feature']

# RIGHT - reshape to 2D
X = df[['feature']]  # Double brackets = DataFrame (2D)
# or
X = df['feature'].values.reshape(-1, 1)  # Explicit reshape
```

??? question "Chapter 7 Checkpoint: Test Your Understanding"
    **Question 1:** A model has equation: Price = 25000 + 150 × sqft. Interpret the slope.

    **Question 2:** What's the predicted price for a 1,200 sqft house using this model?

    **Question 3:** In a residual plot, you see residuals fanning out (spreading wider) as fitted values increase. Which assumption is violated?

    **Click to reveal answers:**

    **Answer 1:** For each additional square foot, the predicted price increases by $150. A house that's 100 sqft larger is predicted to cost $15,000 more.

    **Answer 2:** Price = 25000 + 150 × 1200 = 25000 + 180000 = $205,000

    **Answer 3:** Homoscedasticity is violated. The residuals should have constant spread (homoscedastic), but fanning indicates the spread changes with fitted values (heteroscedastic).

!!! success "Achievement Unlocked: Prediction Pioneer"
    You've built your first predictive model! You can now fit lines to data, interpret what those lines mean, make predictions, and check if your model is trustworthy. This is the foundation of all machine learning—everything else builds on these concepts.

## Key Takeaways

1. **Regression analysis** models relationships between variables to make predictions.

2. **Simple linear regression** uses one input (x) to predict one output (y) with a straight line.

3. The **regression equation** is $\hat{y} = \beta_0 + \beta_1 x$, where β₀ is the **intercept** and β₁ is the **slope**.

4. **Slope** tells you how much y changes per unit increase in x. **Intercept** is the predicted y when x = 0.

5. **Residuals** are prediction errors: actual minus predicted. The **least squares method** finds the line minimizing the **sum of squared errors**.

6. **OLS (Ordinary Least Squares)** is the standard algorithm that finds the optimal coefficients.

7. **Fitted values** are predictions for training data; **predictions** can be made for any new x values.

8. **Assumptions**: linearity, independence, homoscedasticity, and normality of residuals. Check them!

9. **Scikit-learn** provides the professional way to do regression: create model → fit(X, y) → predict(X_new).

10. The **LinearRegression class** implements OLS. Use `.fit()` to train and `.predict()` to generate predictions.

You've now mastered the fundamentals of predictive modeling. In the next chapter, you'll learn how to evaluate whether your model is actually good—because fitting a line is easy, but knowing if it's useful is the real skill!
