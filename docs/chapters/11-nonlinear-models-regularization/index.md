# Non-linear Models and Regularization

---
title: Non-linear Models and Regularization
description: Bend the line and tame the beast - mastering curves and preventing overfitting
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

## Summary

This chapter expands modeling capabilities beyond linear relationships. Students will learn polynomial regression for capturing non-linear patterns, various transformation techniques, and the concept of model flexibility. The chapter introduces regularization as a technique for preventing overfitting, covering Ridge regression, Lasso regression, and Elastic Net. By the end of this chapter, students will understand how to balance model complexity with generalization and apply regularization to improve model performance.

## Concepts Covered

This chapter covers the following 15 concepts from the learning graph:

1. Non-linear Regression
2. Polynomial Regression
3. Degree of Polynomial
4. Curve Fitting
5. Transformation
6. Log Transformation
7. Feature Transformation
8. Model Flexibility
9. Regularization
10. Ridge Regression
11. Lasso Regression
12. Elastic Net
13. Regularization Parameter
14. Lambda Parameter
15. Shrinkage

## Prerequisites

This chapter builds on concepts from:

- [Chapter 8: Model Evaluation and Validation](../08-model-evaluation/index.md)
- [Chapter 9: Multiple Linear Regression](../09-multiple-linear-regression/index.md)

---

## Introduction: Beyond the Straight Line

You've mastered linear regression—congratulations! But here's the truth: the real world doesn't always follow straight lines. House prices don't increase linearly with size forever. Learning curves flatten out. Population growth accelerates and then stabilizes. To model these patterns, you need curves.

This chapter gives you two new superpowers:

1. **Bending the line**: Using polynomial regression and transformations to capture curved relationships
2. **Taming the beast**: Using regularization to prevent models from going wild with overfitting

Together, these techniques let you build models that are flexible enough to capture complex patterns yet disciplined enough to generalize to new data. It's a delicate balance—and by the end of this chapter, you'll be a master at finding it.

## Non-linear Regression: When Lines Aren't Enough

**Non-linear regression** refers to any regression approach where the relationship between features and target isn't a simple straight line. Look at this data:

```python
import numpy as np
import plotly.express as px

# Generate curved data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 + 3*x - 0.5*x**2 + 0.05*x**3 + np.random.normal(0, 2, 100)

fig = px.scatter(x=x, y=y, title="This Data Needs a Curve, Not a Line!")
fig.show()
```

If you fit a straight line to this data, you'll miss the curve entirely. The model will systematically underpredict in some regions and overpredict in others. That's not a random error—it's a sign that your model isn't flexible enough.

Non-linear regression captures these curved patterns by:

- Adding polynomial terms (x², x³, etc.)
- Transforming features (log, square root, etc.)
- Using inherently non-linear models (which we'll cover in later chapters)

The key insight: even though the relationship is curved, we can still use linear regression techniques! We just need to transform our features first.

## Polynomial Regression: Curves Through Linear Regression

**Polynomial regression** is a clever trick: we create new features by raising the original feature to different powers, then use regular linear regression on these expanded features.

For a single feature $x$, polynomial regression of degree 3 looks like:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$$

This is still "linear" regression in the sense that it's linear in the *coefficients* (β values). But the resulting curve can bend and twist to fit complex patterns.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Create polynomial regression pipeline
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])

# Reshape x for sklearn
X = x.reshape(-1, 1)

# Fit the model
poly_model.fit(X, y)

# Generate predictions for smooth curve
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred = poly_model.predict(X_plot)

# Visualize
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_pred, mode='lines',
                         name='Polynomial Fit', line=dict(color='red')))
fig.update_layout(title='Polynomial Regression: Curves that Fit!')
fig.show()
```

The magic happens in `PolynomialFeatures`—it takes your original feature and creates new columns for each power up to the specified degree.

## Degree of Polynomial: How Much Flexibility?

The **degree of polynomial** controls how flexible your curve can be:

- **Degree 1**: Straight line (regular linear regression)
- **Degree 2**: Parabola (one bend)
- **Degree 3**: S-curve possible (two bends)
- **Degree 4+**: Increasingly complex curves

Here's the critical trade-off:

| Degree | Flexibility | Risk of Underfitting | Risk of Overfitting |
|--------|-------------|---------------------|---------------------|
| 1 | Low | High | Low |
| 2-3 | Medium | Medium | Medium |
| 5-7 | High | Low | Medium-High |
| 10+ | Very High | Very Low | Very High |

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data', opacity=0.5))

# Fit polynomials of different degrees
degrees = [1, 2, 3, 5, 10]
colors = ['blue', 'green', 'orange', 'red', 'purple']

for degree, color in zip(degrees, colors):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    y_pred = model.predict(X_plot)
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=y_pred, mode='lines',
                             name=f'Degree {degree}', line=dict(color=color)))

fig.update_layout(title='Effect of Polynomial Degree on Fit')
fig.show()
```

Notice how degree 10 goes wild, trying to pass through every data point? That's overfitting in action. The curve fits the training data perfectly but would fail miserably on new data.

#### Diagram: Polynomial Degree Explorer

<details markdown="1">
<summary>Polynomial Degree Explorer</summary>
Type: microsim

Bloom Taxonomy: Apply, Evaluate

Learning Objective: Interactively explore how polynomial degree affects curve flexibility and the bias-variance tradeoff

Canvas Layout (850x550):
- Main area (850x400): Scatter plot with polynomial curve
- Bottom area (850x150): Controls and metrics

Main Visualization:
- Data points (20-50 points) with some noise
- Polynomial curve that updates in real-time
- Shaded confidence region showing uncertainty
- Residual lines from points to curve (optional toggle)

Interactive Controls:
- Slider: Polynomial Degree (1 to 15)
- Dropdown: Dataset type (linear, quadratic, cubic, sine wave, step function)
- Slider: Noise level (0 to high)
- Button: "Generate New Data"
- Checkbox: "Show Train/Test Split"

Metrics Panel:
- Training R²: updates live
- Test R²: updates live (when split enabled)
- Number of coefficients: degree + 1
- Visual warning when overfitting detected (train >> test)

Educational Overlays:
- At degree 1: "Underfitting: Missing the curve"
- At degree 2-4: "Good fit for this data"
- At degree 10+: "Overfitting: Chasing noise!"
- Arrow pointing to where train/test scores diverge

Animation:
- Smooth curve transition when degree changes
- Coefficients displayed with size proportional to magnitude

Implementation: p5.js with polynomial fitting
</details>

## Curve Fitting: The Art and Science

**Curve fitting** is the process of finding the mathematical function that best describes your data. While polynomial regression is one approach, the broader goal is matching the right curve shape to your data's underlying pattern.

Good curve fitting requires:

1. **Visual inspection**: Plot your data first! What shape does it suggest?
2. **Domain knowledge**: Does theory predict a certain relationship?
3. **Validation**: Does the curve generalize to new data?
4. **Parsimony**: Prefer simpler curves when they fit adequately

```python
from sklearn.model_selection import cross_val_score

# Find optimal degree using cross-validation
degrees = range(1, 15)
cv_scores = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

# Plot CV scores vs degree
fig = px.line(x=list(degrees), y=cv_scores, markers=True,
              title='Cross-Validation Score vs Polynomial Degree',
              labels={'x': 'Polynomial Degree', 'y': 'CV R² Score'})
fig.show()

# Find best degree
best_degree = degrees[np.argmax(cv_scores)]
print(f"Optimal degree: {best_degree}")
```

The CV score typically rises, peaks, then falls as degree increases. The peak is your sweet spot—enough flexibility to capture the pattern, not so much that you're fitting noise.

## Transformation: Changing the Shape of Data

**Transformation** is a broader technique for handling non-linear relationships. Instead of adding polynomial terms, we transform the original variables to make the relationship more linear.

Common transformations include:

| Transformation | Formula | Use Case |
|---------------|---------|----------|
| Log | $\log(x)$ | Exponential growth, multiplicative effects |
| Square root | $\sqrt{x}$ | Count data, variance stabilization |
| Reciprocal | $1/x$ | Inverse relationships |
| Power | $x^n$ | Accelerating/decelerating patterns |
| Box-Cox | $(x^\lambda - 1)/\lambda$ | General normalization |

The key insight: if your scatter plot curves, the right transformation can straighten it—making linear regression appropriate again.

## Log Transformation: The Exponential Tamer

The **log transformation** is probably the most useful transformation in data science. It's perfect when:

- Your data spans several orders of magnitude (1 to 1,000,000)
- The relationship looks exponential
- You want to interpret coefficients as percentage changes
- Residuals show increasing variance (heteroscedasticity)

```python
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate exponential-ish data
np.random.seed(42)
x_exp = np.linspace(1, 10, 100)
y_exp = 5 * np.exp(0.4 * x_exp) + np.random.normal(0, 10, 100)

# Create subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Original Scale (Curved)', 'Log-Transformed (Linear!)'])

# Original scale
fig.add_trace(go.Scatter(x=x_exp, y=y_exp, mode='markers', name='Original'),
              row=1, col=1)

# Log-transformed
fig.add_trace(go.Scatter(x=x_exp, y=np.log(y_exp), mode='markers', name='Log(y)'),
              row=1, col=2)

fig.update_layout(height=400, title='The Magic of Log Transformation')
fig.show()
```

Notice how the curved relationship becomes nearly linear after log transformation? Now regular linear regression will work beautifully.

```python
# Log-linear regression
from sklearn.linear_model import LinearRegression

# Transform y
y_log = np.log(y_exp)

# Fit model on log scale
model = LinearRegression()
model.fit(x_exp.reshape(-1, 1), y_log)

print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Interpretation: For each unit increase in x, y increases by {np.exp(model.coef_[0]):.2%}")
```

!!! tip "Interpreting Log-Transformed Coefficients"
    When your target is log-transformed, coefficients represent *multiplicative* effects. A coefficient of 0.4 means each unit of x multiplies y by e^0.4 ≈ 1.49, or a 49% increase.

## Feature Transformation: Engineering Better Inputs

**Feature transformation** is the deliberate modification of input features to improve model performance. This is closely related to the feature engineering we covered earlier, but with a specific focus on mathematical transformations.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Sample data
df = pd.DataFrame({
    'income': [30000, 45000, 55000, 80000, 150000, 500000],
    'age': [22, 28, 35, 42, 55, 65],
    'experience_years': [0, 3, 8, 15, 25, 35]
})

# Log transform skewed income
df['log_income'] = np.log(df['income'])

# Square root of experience (diminishing returns)
df['sqrt_experience'] = np.sqrt(df['experience_years'])

# Age polynomial
df['age_squared'] = df['age'] ** 2

print(df)
```

Scikit-learn's `PowerTransformer` can automatically find good transformations:

```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson transformation (handles zeros and negatives)
pt = PowerTransformer(method='yeo-johnson')
df_transformed = pt.fit_transform(df[['income', 'age', 'experience_years']])

print("Transformation parameters:", pt.lambdas_)
```

#### Diagram: Transformation Gallery

<details markdown="1">
<summary>Transformation Gallery</summary>
Type: infographic

Bloom Taxonomy: Understand, Apply

Learning Objective: Show common transformations side-by-side with their effects on data distribution and relationships

Layout: 2x3 grid of transformation examples

Each Panel Contains:
- Original data histogram/scatter (left mini-plot)
- Transformed data histogram/scatter (right mini-plot)
- Transformation formula
- When to use it

Panels:
1. Log Transformation
   - Before: Right-skewed histogram
   - After: Symmetric histogram
   - Formula: y' = log(y)
   - Use: Exponential relationships, multiplicative effects

2. Square Root
   - Before: Count data with variance proportional to mean
   - After: Stabilized variance
   - Formula: y' = √y
   - Use: Count data, Poisson-like distributions

3. Reciprocal (1/x)
   - Before: Hyperbolic scatter
   - After: Linear scatter
   - Formula: y' = 1/y
   - Use: Inverse relationships

4. Square (x²)
   - Before: Decelerating curve
   - After: Linear relationship
   - Formula: y' = x²
   - Use: Accelerating patterns

5. Box-Cox
   - Before: Arbitrary skewed data
   - After: Approximately normal
   - Formula: y' = (y^λ - 1)/λ
   - Use: General normalization

6. Standardization
   - Before: Different scales
   - After: Mean=0, SD=1
   - Formula: z = (x - μ)/σ
   - Use: Comparing features, regularization

Interactive Elements:
- Click panel to see full-size comparison
- Slider to adjust transformation parameter
- Button: "Try on your data" - upload CSV option

Implementation: HTML/CSS/JavaScript with D3.js visualizations
</details>

## Model Flexibility: The Complexity Dial

**Model flexibility** refers to how adaptable a model is to different patterns in data. A highly flexible model can capture intricate patterns but risks overfitting. A rigid model may miss important patterns but generalizes better.

Think of flexibility as a dial:

- **Low flexibility** (simple models): Few parameters, strong assumptions, high bias, low variance
- **High flexibility** (complex models): Many parameters, weak assumptions, low bias, high variance

The relationship between flexibility and error follows a U-shaped curve:

- **Training error** always decreases with more flexibility
- **Test error** decreases initially, then increases (the overfitting zone)
- The optimal flexibility minimizes test error

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Track errors across flexibility levels
train_errors = []
test_errors = []
degrees = range(1, 20)

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)

    train_errors.append(1 - model.score(X_train, y_train))
    test_errors.append(1 - model.score(X_test, y_test))

# Plot the flexibility curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(degrees), y=train_errors, mode='lines+markers',
                         name='Training Error', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=list(degrees), y=test_errors, mode='lines+markers',
                         name='Test Error', line=dict(color='red')))
fig.update_layout(title='The Bias-Variance Tradeoff in Action',
                  xaxis_title='Model Flexibility (Polynomial Degree)',
                  yaxis_title='Error (1 - R²)')
fig.show()
```

The gap between training and test error is your overfitting indicator. When it's large, your model has learned noise specific to the training data.

## Regularization: Taming Overfitting

Here's the million-dollar question: if more flexibility leads to overfitting, but we need flexibility to capture complex patterns, what do we do?

Enter **regularization**—a technique that adds a penalty for model complexity. Instead of just minimizing prediction error, regularized models minimize:

$$\text{Loss} = \text{Prediction Error} + \lambda \times \text{Complexity Penalty}$$

The complexity penalty discourages large coefficients, effectively simplifying the model. This creates a controlled trade-off between fitting the data and keeping the model simple.

Regularization gives you the best of both worlds:

- Use a flexible model (high-degree polynomial)
- Let regularization automatically "turn off" unnecessary complexity
- Result: captures real patterns, ignores noise

The **regularization parameter** (often called $\lambda$ or `alpha` in scikit-learn) controls this trade-off:

- **λ = 0**: No regularization (standard linear regression)
- **Small λ**: Light penalty, nearly flexible
- **Large λ**: Heavy penalty, nearly rigid
- **λ → ∞**: All coefficients shrink to zero

## Ridge Regression: The L2 Penalty

**Ridge regression** (also called L2 regularization or Tikhonov regularization) adds a penalty proportional to the *squared* coefficients:

$$\text{Loss}_{\text{Ridge}} = \sum(y_i - \hat{y}_i)^2 + \lambda \sum \beta_j^2$$

The squared penalty means:

- All coefficients are shrunk toward zero
- Large coefficients are penalized more heavily
- Coefficients never become exactly zero (just very small)
- Good for multicollinearity—it stabilizes correlated features

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create Ridge regression pipeline with polynomial features
ridge_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),  # High degree
    ('scaler', StandardScaler()),              # Essential for regularization!
    ('ridge', Ridge(alpha=1.0))                # alpha is λ
])

ridge_pipeline.fit(X_train, y_train)

print(f"Train R²: {ridge_pipeline.score(X_train, y_train):.4f}")
print(f"Test R²: {ridge_pipeline.score(X_test, y_test):.4f}")

# Compare to unregularized
unreg_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('linear', LinearRegression())
])
unreg_pipeline.fit(X_train, y_train)

print(f"\nUnregularized Train R²: {unreg_pipeline.score(X_train, y_train):.4f}")
print(f"Unregularized Test R²: {unreg_pipeline.score(X_test, y_test):.4f}")
```

Notice how Ridge maintains good test performance even with degree 10, while unregularized regression overfits!

!!! warning "Scale Your Features for Regularization"
    Regularization penalizes large coefficients. If features are on different scales, the penalty affects them unequally. Always standardize features before applying regularization.

## Lasso Regression: The L1 Penalty

**Lasso regression** (Least Absolute Shrinkage and Selection Operator) uses the *absolute value* of coefficients as the penalty:

$$\text{Loss}_{\text{Lasso}} = \sum(y_i - \hat{y}_i)^2 + \lambda \sum |\beta_j|$$

The L1 penalty has a special property: it can shrink coefficients all the way to exactly zero. This means Lasso performs **automatic feature selection**—useless features get eliminated entirely.

```python
from sklearn.linear_model import Lasso

# Create Lasso pipeline
lasso_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])

lasso_pipeline.fit(X_train, y_train)

# Count non-zero coefficients
coefficients = lasso_pipeline.named_steps['lasso'].coef_
n_features = len(coefficients)
n_nonzero = np.sum(coefficients != 0)

print(f"Total features: {n_features}")
print(f"Non-zero coefficients: {n_nonzero}")
print(f"Features eliminated: {n_features - n_nonzero}")

print(f"\nTrain R²: {lasso_pipeline.score(X_train, y_train):.4f}")
print(f"Test R²: {lasso_pipeline.score(X_test, y_test):.4f}")
```

| Aspect | Ridge (L2) | Lasso (L1) |
|--------|-----------|------------|
| Penalty | Sum of squared coefficients | Sum of absolute coefficients |
| Coefficients | Shrunk toward zero | Can become exactly zero |
| Feature selection | No | Yes (automatic) |
| Multicollinearity | Handles well | Arbitrary selection |
| Best for | Many small effects | Few important features |

#### Diagram: Ridge vs Lasso Comparison

<details markdown="1">
<summary>Ridge vs Lasso Comparison</summary>
Type: microsim

Bloom Taxonomy: Analyze, Evaluate

Learning Objective: Visualize and compare how Ridge and Lasso penalties affect coefficient shrinkage and feature selection

Canvas Layout (900x550):
- Top left (400x250): Ridge coefficient path
- Top right (400x250): Lasso coefficient path
- Bottom (900x250): Side-by-side coefficient comparison and controls

Coefficient Path Plots:
- X-axis: Log(λ) from small to large
- Y-axis: Coefficient values
- Each line represents one coefficient
- Show how coefficients shrink as λ increases
- Ridge: All lines approach zero asymptotically
- Lasso: Lines hit zero and stay there

Comparison Panel:
- Bar chart showing final coefficient values
- Ridge bars (blue): All non-zero, varying heights
- Lasso bars (orange): Some exactly zero
- Highlight eliminated features in gray

Interactive Controls:
- Slider: λ (regularization strength) - both plots update
- Dropdown: Select dataset (housing, synthetic, medical)
- Checkbox: "Show cross-validation optimal λ"
- Toggle: "Show mathematical penalty visualization"

Penalty Visualization (optional):
- 2D contour plot showing loss surface
- Ridge: Circular penalty (L2 ball)
- Lasso: Diamond penalty (L1 ball)
- Optimal point where loss contours meet penalty boundary

Key Insights Displayed:
- "Lasso zeros out X features"
- "Ridge reduces largest coefficient by Y%"
- Optimal λ marked on both paths

Implementation: p5.js with interactive plots
</details>

## Elastic Net: The Best of Both Worlds

**Elastic Net** combines Ridge and Lasso penalties:

$$\text{Loss}_{\text{ElasticNet}} = \sum(y_i - \hat{y}_i)^2 + \lambda_1 \sum |\beta_j| + \lambda_2 \sum \beta_j^2$$

Or equivalently, using a mixing parameter $\rho$ (called `l1_ratio` in scikit-learn):

$$\text{Loss} = \text{MSE} + \alpha \left( \rho \sum |\beta_j| + (1-\rho) \sum \beta_j^2 \right)$$

When $\rho = 1$: Pure Lasso
When $\rho = 0$: Pure Ridge
When $0 < \rho < 1$: Combination

```python
from sklearn.linear_model import ElasticNet

# Elastic Net with equal mix of L1 and L2
elastic_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('scaler', StandardScaler()),
    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5))  # 50% L1, 50% L2
])

elastic_pipeline.fit(X_train, y_train)

coefficients = elastic_pipeline.named_steps['elastic'].coef_
print(f"Non-zero coefficients: {np.sum(coefficients != 0)}")
print(f"Test R²: {elastic_pipeline.score(X_test, y_test):.4f}")
```

Elastic Net is particularly useful when:

- You have groups of correlated features (Lasso arbitrarily picks one; Elastic Net keeps related features together)
- You want some feature selection but not as aggressive as pure Lasso
- You're not sure whether Ridge or Lasso is better (try Elastic Net!)

## Lambda/Alpha Parameter: Finding the Sweet Spot

The **lambda parameter** (called `alpha` in scikit-learn) controls regularization strength. Too small and you overfit; too large and you underfit. Finding the optimal λ is crucial.

Use cross-validation to find the best λ:

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# RidgeCV automatically finds optimal alpha
alphas = np.logspace(-4, 4, 50)  # Range from 0.0001 to 10000

ridge_cv = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=alphas, cv=5))
])

ridge_cv.fit(X_train, y_train)

optimal_alpha = ridge_cv.named_steps['ridge'].alpha_
print(f"Optimal alpha: {optimal_alpha:.4f}")
print(f"Test R²: {ridge_cv.score(X_test, y_test):.4f}")
```

Visualize the regularization path:

```python
from sklearn.linear_model import ridge_regression

# Calculate coefficients for each alpha
coef_path = []
for alpha in alphas:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=5)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(X_train, y_train)
    coef_path.append(model.named_steps['ridge'].coef_)

coef_path = np.array(coef_path)

# Plot coefficient paths
fig = go.Figure()
for i in range(coef_path.shape[1]):
    fig.add_trace(go.Scatter(x=np.log10(alphas), y=coef_path[:, i],
                             mode='lines', name=f'Coef {i}',
                             showlegend=False))

fig.update_layout(title='Ridge Coefficient Path: How λ Affects Coefficients',
                  xaxis_title='log₁₀(λ)',
                  yaxis_title='Coefficient Value')
fig.show()
```

#### Diagram: Lambda Tuning Playground

<details markdown="1">
<summary>Lambda Tuning Playground</summary>
Type: microsim

Bloom Taxonomy: Apply, Evaluate

Learning Objective: Practice finding optimal regularization strength through interactive experimentation

Canvas Layout (850x600):
- Top area (850x350): Data and model fit visualization
- Bottom left (425x250): CV score vs lambda plot
- Bottom right (425x250): Coefficient magnitudes

Top Panel - Model Fit:
- Scatter plot of data
- Polynomial curve showing current fit
- Toggle between Ridge/Lasso/Elastic Net
- Curve updates as lambda changes

Bottom Left - Cross-Validation:
- X-axis: log(λ) scale
- Y-axis: CV Score (R² or MSE)
- Line showing CV performance across λ values
- Vertical marker at current λ
- Optimal λ highlighted with star

Bottom Right - Coefficients:
- Bar chart of coefficient magnitudes
- Updates in real-time as λ changes
- For Lasso: Gray out zero coefficients
- Show total number of non-zero coefficients

Interactive Controls:
- Slider: Lambda value (log scale)
- Dropdown: Regularization type (Ridge, Lasso, Elastic Net)
- Slider: Polynomial degree (2-15)
- Slider: l1_ratio (for Elastic Net, 0-1)
- Button: "Find Optimal λ" - animates search
- Button: "Generate New Data"

Metrics Display:
- Current λ value
- Train R²
- Test R²
- Cross-Validation R²
- Number of non-zero coefficients

Educational Callouts:
- When λ too small: "Overfitting warning!"
- When λ too large: "Underfitting warning!"
- At optimal: "Sweet spot found!"

Implementation: p5.js with real-time model fitting
</details>

## Shrinkage: What Regularization Actually Does

**Shrinkage** is the technical term for what regularization does to coefficients—it pulls them toward zero. But why does shrinking coefficients help prevent overfitting?

Consider what happens when a model overfits:

1. It finds patterns in noise
2. These patterns require extreme coefficients
3. Small changes in data cause large prediction changes
4. High variance = poor generalization

Shrinkage counters this by:

1. Penalizing extreme coefficients
2. Forcing the model to find simpler solutions
3. Reducing sensitivity to noise
4. Lower variance = better generalization

```python
# Demonstrate shrinkage effect
from sklearn.linear_model import Ridge

alphas = [0, 0.01, 0.1, 1, 10, 100]
coefficients_by_alpha = {}

for alpha in alphas:
    if alpha == 0:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=8)),
            ('scaler', StandardScaler()),
            ('linear', LinearRegression())
        ])
    else:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=8)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])

    model.fit(X_train, y_train)

    if alpha == 0:
        coeffs = model.named_steps['linear'].coef_
    else:
        coeffs = model.named_steps['ridge'].coef_

    coefficients_by_alpha[alpha] = coeffs

    # Calculate coefficient magnitude
    coef_magnitude = np.sqrt(np.sum(coeffs**2))
    print(f"α={alpha:6}: Coefficient L2 norm = {coef_magnitude:.2f}, Test R² = {model.score(X_test, y_test):.4f}")
```

As regularization increases:

- Coefficient magnitudes shrink
- Model becomes more stable
- Test performance often improves (up to a point)

## Putting It All Together: A Complete Workflow

Here's a complete workflow for building regularized non-linear models:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
import plotly.express as px

# 1. Load and split data
np.random.seed(42)
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = 3 + 2*X.flatten() - 0.5*X.flatten()**2 + 0.05*X.flatten()**3 + np.random.normal(0, 1, 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create pipeline with polynomial features and regularization
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

# 3. Define hyperparameter grid
param_grid = {
    'poly__degree': [2, 3, 4, 5, 6, 7],
    'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# 4. Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 5. Results
print("Best parameters:", grid_search.best_params_)
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# 6. Visualize the fit
best_model = grid_search.best_estimator_
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
y_plot = best_model.predict(X_plot)

fig = px.scatter(x=X.flatten(), y=y, opacity=0.5, title='Regularized Polynomial Regression')
fig.add_scatter(x=X_plot.flatten(), y=y_plot, mode='lines', name='Best Model',
                line=dict(color='red', width=2))
fig.show()
```

#### Diagram: Regularization Decision Tree

<details markdown="1">
<summary>Regularization Decision Tree</summary>
Type: workflow

Bloom Taxonomy: Evaluate, Apply

Learning Objective: Guide students through choosing the right regularization approach for their problem

Visual Style: Flowchart with decision diamonds and outcome rectangles

Start: "Need to Prevent Overfitting?"

Decision 1: "Linear relationship?"
- Yes → Consider if regularization is needed
- No → Add polynomial features

Decision 2: "How many features vs samples?"
- Many features, few samples → Strong regularization needed
- Balanced → Moderate regularization
- Few features, many samples → Light regularization

Decision 3: "Do you want feature selection?"
- Yes, aggressive → Use Lasso
- Yes, some → Use Elastic Net
- No, keep all features → Use Ridge

Decision 4: "Highly correlated features?"
- Yes → Use Ridge or Elastic Net (Lasso is unstable)
- No → Any method works

Decision 5: "Interpretability important?"
- Yes → Lasso (sparse solution)
- No → Ridge (often better accuracy)

Final Outcomes:
- Ridge: "Many small effects, correlated features"
- Lasso: "Few important features, interpretability"
- Elastic Net: "Best of both, groups of features"

Interactive Elements:
- Click each decision to see explanation
- Hover shows examples of each scenario
- "Take Quiz" mode walks through with your data characteristics

Implementation: HTML/CSS/JavaScript with interactive flowchart
</details>

## Common Pitfalls and Best Practices

**Always Scale Before Regularizing**
Regularization penalizes coefficient magnitude. If features aren't scaled, features with larger values will be unfairly penalized.

```python
# Good: Scale inside pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Bad: Features on different scales penalized unequally
ridge_bad = Ridge()
ridge_bad.fit(X_unscaled, y)
```

**Don't Regularize the Intercept**
Scikit-learn doesn't regularize the intercept by default (which is correct). Be careful if using other implementations.

**Use Cross-Validation for Lambda**
Never set λ by looking at test performance. Use cross-validation to find optimal λ, then evaluate on test data.

**Consider the Problem Type**
- Prediction focus → Ridge often wins
- Interpretation focus → Lasso for sparsity
- Groups of related features → Elastic Net

**Watch for Warning Signs**
- Very large or very small λ optimal → reconsider model specification
- All coefficients near zero → λ too large
- Test performance much worse than CV → something's wrong

## Summary: Your Regularization Toolkit

You now have powerful tools for handling non-linear relationships and overfitting:

- **Polynomial regression** captures curved patterns using powers of features
- **Degree selection** balances flexibility with overfitting risk
- **Transformations** (log, sqrt, etc.) can linearize relationships
- **Model flexibility** is the dial between underfitting and overfitting
- **Regularization** adds complexity penalties to prevent overfitting
- **Ridge (L2)** shrinks all coefficients, handles multicollinearity
- **Lasso (L1)** performs automatic feature selection
- **Elastic Net** combines L1 and L2 penalties
- **Lambda tuning** via cross-validation finds the optimal penalty strength

With these techniques, you can build models that are flexible enough to capture complex real-world patterns while remaining robust enough to generalize to new data.

## Looking Ahead

In the next chapter, we'll explore machine learning more broadly—including classification problems where we predict categories instead of numbers. You'll see how the regularization concepts you learned here apply to new types of models.

---

## Key Takeaways

- Polynomial regression captures non-linear patterns while still using linear regression techniques
- Higher polynomial degrees increase flexibility but also overfitting risk
- Log and other transformations can linearize curved relationships
- Regularization adds a penalty for complexity, balancing fit with generalization
- Ridge (L2) shrinks coefficients; Lasso (L1) can zero them out entirely
- Elastic Net combines both penalties for flexible feature selection
- Always scale features before regularizing and use CV to find optimal λ
- The goal is finding the sweet spot: flexible enough to learn patterns, constrained enough to ignore noise
