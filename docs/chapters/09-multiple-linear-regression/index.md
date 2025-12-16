# Multiple Linear Regression

---
title: Multiple Linear Regression
description: Unlock the power of multiple features to build more accurate predictions
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

## Summary

This chapter extends linear regression to handle multiple predictor variables. Students will learn to build models with multiple features, understand and diagnose multicollinearity, and apply various feature selection methods. The chapter covers handling categorical variables through dummy variables and one-hot encoding, creating interaction terms, and understanding feature importance. By the end of this chapter, students will be able to build and interpret multiple regression models with both numerical and categorical predictors.

## Concepts Covered

This chapter covers the following 15 concepts from the learning graph:

1. Multiple Linear Regression
2. Multiple Predictors
3. Multicollinearity
4. Variance Inflation Factor
5. Feature Selection
6. Forward Selection
7. Backward Elimination
8. Stepwise Selection
9. Categorical Variables
10. Dummy Variables
11. One-Hot Encoding
12. Interaction Terms
13. Polynomial Features
14. Feature Engineering
15. Feature Importance

## Prerequisites

This chapter builds on concepts from:

- [Chapter 7: Simple Linear Regression](../07-simple-linear-regression/index.md)
- [Chapter 8: Model Evaluation and Validation](../08-model-evaluation/index.md)

---

## Introduction: Leveling Up Your Prediction Powers

In the last few chapters, you learned to predict outcomes using a single feature. That's like trying to predict someone's basketball skills by only looking at their height. Sure, height matters, but what about their practice hours, speed, and jumping ability? Real-world predictions almost always depend on *multiple* factors working together.

**Multiple linear regression** is your superpower upgrade. Instead of drawing a line through 2D data, you're now fitting a *hyperplane* through multi-dimensional space. Don't worry if that sounds intimidating—the math is surprisingly similar to what you already know, and scikit-learn handles the heavy lifting. Your job is to understand what the model is doing and how to use it wisely.

By the end of this chapter, you'll be able to build models that consider dozens of features simultaneously, handle both numbers and categories, and identify which features actually matter. That's serious prediction power.

## From One Feature to Many: Multiple Predictors

In simple linear regression, we had one predictor variable $x$ and one target $y$:

$$y = \beta_0 + \beta_1 x$$

With **multiple linear regression**, we have multiple predictors—let's call them $x_1, x_2, x_3$, and so on:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + ... + \beta_p x_p$$

Each $\beta$ coefficient tells you how much $y$ changes when that specific $x$ increases by one unit, *holding all other variables constant*. That last part is crucial—it's what makes multiple regression so powerful. You can isolate the effect of each feature.

Here's a concrete example. Suppose you're predicting house prices with three features:

- $x_1$ = square footage
- $x_2$ = number of bedrooms
- $x_3$ = age of house (years)

Your model might look like:

$$\text{Price} = 50000 + 150 \times \text{SqFt} + 10000 \times \text{Bedrooms} - 1000 \times \text{Age}$$

This tells you:

- Base price is $50,000
- Each square foot adds $150
- Each bedroom adds $10,000
- Each year of age *subtracts* $1,000

The negative coefficient for age makes sense—older houses typically sell for less, all else being equal.

## Building Your First Multiple Regression Model

Let's build a multiple regression model in Python. The process is almost identical to simple regression—scikit-learn handles the complexity behind the scenes.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load housing data
df = pd.read_csv('housing.csv')

# Select multiple features
features = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'lot_size']
X = df[features]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# View the coefficients
print("Intercept:", model.intercept_)
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")
```

The output shows you how each feature contributes to the prediction. Positive coefficients increase the predicted price; negative ones decrease it.

#### Diagram: Multiple Regression Anatomy

<details markdown="1">
<summary>Multiple Regression Anatomy</summary>
Type: infographic

Bloom Taxonomy: Understand

Learning Objective: Help students visualize how multiple features combine to form a single prediction, understanding each coefficient's role

Layout: Central equation with branching explanations for each component

Visual Elements:
- Large central equation: y = β₀ + β₁x₁ + β₂x₂ + β₃x₃
- Each term has an arrow pointing to an explanation box
- β₀ box: "Starting point (intercept) - prediction when all features are zero"
- Each βᵢxᵢ box: shows feature name, coefficient value, and contribution
- Final prediction shown as sum of all contributions with animated addition

Interactive Elements:
- Hover over each term to see its specific contribution
- Slider for each feature value (x₁, x₂, x₃)
- As sliders move, show each term's contribution updating
- Final prediction updates in real-time as sum of all terms
- Color coding: positive contributions in green, negative in red

Example Data:
- House price prediction with square_feet, bedrooms, age
- Show specific numbers: 150 × 1500 sqft = $225,000 contribution

Color Scheme:
- Intercept: Blue
- Positive coefficients: Green gradient
- Negative coefficients: Red gradient
- Final prediction: Gold

Implementation: HTML/CSS/JavaScript with interactive sliders
</details>

## Interpreting Multiple Regression Coefficients

Each coefficient in multiple regression has a specific interpretation: it tells you the expected change in $y$ for a one-unit increase in that feature, **while holding all other features constant**. This "all else being equal" interpretation is what makes multiple regression so valuable.

Let's examine our model's coefficients:

```python
import plotly.express as px

# Create coefficient visualization
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

fig = px.bar(
    coef_df,
    x='Feature',
    y='Coefficient',
    title='Feature Coefficients: Impact on House Price',
    color='Coefficient',
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0
)

fig.update_layout(height=400)
fig.show()
```

A few important caveats about interpreting coefficients:

| Consideration | Why It Matters |
|--------------|----------------|
| Scale differences | A coefficient of 100 for square feet isn't comparable to 10,000 for bedrooms—units differ |
| Correlation between features | If bedrooms and square feet are correlated, their individual effects are harder to isolate |
| Non-linear relationships | Coefficients assume linear effects; reality might be curved |
| Categorical variables | Need special handling (we'll cover this soon) |

!!! tip "Standardizing for Fair Comparison"
    To compare coefficient magnitudes fairly, standardize your features first (subtract mean, divide by standard deviation). Then coefficients represent "effect of one standard deviation change" and are directly comparable.

## The Multicollinearity Problem

Here's a tricky situation: what happens when your predictor variables are highly correlated with each other? This is called **multicollinearity**, and it can cause serious problems for your model.

Imagine predicting house prices with both "square feet" and "number of rooms." These features are strongly related—bigger houses have more rooms. When features are correlated:

- Coefficients become unstable (small data changes cause big coefficient swings)
- Standard errors inflate, making significance tests unreliable
- Individual feature effects become hard to interpret
- The model might still predict well overall, but you can't trust individual coefficients

Think of it like two people trying to push a car together at the exact same angle. You can see the car moved, but you can't tell who pushed harder—their efforts are indistinguishable.

```python
# Check for correlations between features
correlation_matrix = X.corr()

import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    colorscale='RdBu',
    zmid=0
)

fig.update_layout(title='Feature Correlation Matrix', height=500)
fig.show()
```

Look for correlations above 0.7 or below -0.7—these pairs of features might cause multicollinearity issues.

## Variance Inflation Factor: Quantifying Multicollinearity

The **Variance Inflation Factor (VIF)** is a precise way to measure multicollinearity. It tells you how much the variance of a coefficient is inflated due to correlations with other predictors.

- VIF = 1: No correlation with other features (ideal)
- VIF = 1-5: Moderate correlation (usually acceptable)
- VIF > 5: High correlation (concerning)
- VIF > 10: Severe multicollinearity (definitely a problem)

Here's how to calculate VIF for each feature:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data.sort_values('VIF', ascending=False))
```

If you find high VIF values, you have options:

- Remove one of the correlated features
- Combine correlated features into a single composite feature
- Use regularization techniques (covered in a later chapter)
- Accept that individual coefficients may be unreliable, but overall predictions are fine

#### Diagram: Multicollinearity Detector MicroSim

<details markdown="1">
<summary>Multicollinearity Detector MicroSim</summary>
Type: microsim

Bloom Taxonomy: Analyze, Evaluate

Learning Objective: Help students understand how correlated features affect coefficient stability and learn to diagnose multicollinearity using VIF

Canvas Layout (850x500):
- Left panel (400x500): Scatter plot matrix showing feature correlations
- Right panel (450x500): VIF display and coefficient stability visualization

Left Panel Elements:
- 3x3 scatter plot matrix for selected features
- Correlation coefficients displayed on off-diagonal
- Color intensity indicates correlation strength
- Clickable to focus on any pair

Right Panel Elements:
- Bar chart of VIF values for all features
- Color coding: Green (<5), Yellow (5-10), Red (>10)
- Below: Coefficient confidence intervals that widen with higher VIF
- Warning messages for problematic features

Interactive Controls:
- Dropdown: Select dataset (housing, cars, student performance)
- Checkbox: Add highly correlated feature (to demonstrate VIF increase)
- Button: "Simulate 100 data samples" - shows coefficient variation
- Slider: Artificially adjust correlation between two features

Key Demonstrations:
- Watch VIF spike when adding a correlated feature
- See coefficient confidence intervals widen with high VIF
- Observe coefficient values fluctuate wildly when resampling with multicollinearity

Implementation: p5.js with statistical calculations
</details>

## Feature Selection: Choosing the Right Variables

Not every available feature belongs in your model. **Feature selection** is the art and science of choosing which variables to include. Too few features, and you underfit. Too many, and you risk overfitting and multicollinearity.

There are three classic approaches to feature selection:

### Forward Selection

**Forward selection** starts with no features and adds them one at a time. At each step, you add the feature that most improves the model, until no remaining feature provides significant improvement.

The process:

1. Start with an empty model (intercept only)
2. Try adding each remaining feature one at a time
3. Keep the one that gives the biggest improvement (if significant)
4. Repeat until no feature improves the model enough

### Backward Elimination

**Backward elimination** works in reverse. Start with all features and remove the least useful ones:

1. Start with all features in the model
2. Find the feature with the smallest contribution (highest p-value or lowest impact)
3. Remove it if it's below your threshold
4. Repeat until all remaining features are significant

### Stepwise Selection

**Stepwise selection** combines both approaches. At each step, you can either add a feature or remove one, depending on which action most improves the model. This flexibility helps find combinations that neither forward nor backward selection would discover alone.

```python
# Simple implementation of forward selection using cross-validation
from sklearn.model_selection import cross_val_score

def forward_selection(X, y, max_features=None):
    remaining = list(X.columns)
    selected = []
    best_scores = []

    if max_features is None:
        max_features = len(remaining)

    while remaining and len(selected) < max_features:
        best_score = -np.inf
        best_feature = None

        for feature in remaining:
            current_features = selected + [feature]
            X_subset = X[current_features]

            # Use cross-validation to evaluate
            score = cross_val_score(
                LinearRegression(), X_subset, y, cv=5, scoring='r2'
            ).mean()

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature:
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_scores.append(best_score)
            print(f"Added {best_feature}: CV R² = {best_score:.4f}")

    return selected, best_scores

# Run forward selection
selected_features, scores = forward_selection(X_train, y_train)
```

| Method | Starts With | Action | Best For |
|--------|-------------|--------|----------|
| Forward Selection | No features | Adds best one at a time | Many features, few are relevant |
| Backward Elimination | All features | Removes worst one at a time | Fewer features, most are useful |
| Stepwise Selection | Any starting point | Adds or removes each step | Complex relationships |

#### Diagram: Feature Selection Race

<details markdown="1">
<summary>Feature Selection Race</summary>
Type: microsim

Bloom Taxonomy: Apply, Analyze

Learning Objective: Visualize and compare different feature selection strategies, understanding how each method builds or prunes the feature set

Canvas Layout (800x550):
- Top area (800x400): Three parallel "race tracks" for each method
- Bottom area (800x150): Results comparison table

Race Track Elements:
- Each track shows features as checkpoints
- Forward: Start empty, light up features as added
- Backward: Start full, dim features as removed
- Stepwise: Show both add and remove actions
- Current model score displayed at each step

Interactive Controls:
- Button: "Start Race" - animate all three methods simultaneously
- Speed slider: Control animation speed
- Dropdown: Select dataset
- Checkbox: "Show R² at each step"
- Button: "Compare Final Models"

Animation:
- Features light up (added) or dim (removed) as methods progress
- Score counter updates at each step
- Pause at each step to show decision being made
- Highlight which feature is being considered

Results Comparison:
- Table showing: Method, Features Selected, Final R², Time
- Visual indicator of which method "won" (best score)
- Discussion of when each method excels

Implementation: p5.js with step-by-step animation
</details>

## Handling Categorical Variables

So far, we've only used numerical features. But what about **categorical variables** like neighborhood, car brand, or education level? These don't have a natural numeric ordering, so we can't just plug them into the equation.

The solution is to convert categories into numbers using **dummy variables** or **one-hot encoding**.

### Dummy Variables

A **dummy variable** is a binary (0 or 1) variable that represents whether an observation belongs to a category. For a categorical variable with $k$ categories, you create $k-1$ dummy variables.

Why $k-1$ instead of $k$? Because the last category is implied when all dummies are 0. This avoids redundancy and multicollinearity.

Example: For "Neighborhood" with three values (Downtown, Suburbs, Rural):

| Observation | Neighborhood | Is_Downtown | Is_Suburbs |
|-------------|--------------|-------------|------------|
| House 1 | Downtown | 1 | 0 |
| House 2 | Suburbs | 0 | 1 |
| House 3 | Rural | 0 | 0 |
| House 4 | Downtown | 1 | 0 |

Notice that Rural is the "reference category"—it's represented by zeros in both columns.

### One-Hot Encoding

**One-hot encoding** creates $k$ dummy variables (one for each category). While this seems simpler, it creates redundancy that must be handled. Most libraries automatically drop one category to prevent issues.

```python
import pandas as pd

# Method 1: pd.get_dummies (creates one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['neighborhood'], drop_first=True)

# Method 2: Using scikit-learn's OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create preprocessor
categorical_features = ['neighborhood', 'style']
numerical_features = ['square_feet', 'bedrooms', 'age']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Fit and transform
X_processed = preprocessor.fit_transform(X)
```

#### Diagram: One-Hot Encoding Visualizer

<details markdown="1">
<summary>One-Hot Encoding Visualizer</summary>
Type: infographic

Bloom Taxonomy: Understand, Apply

Learning Objective: Demonstrate how categorical variables are transformed into numerical format through one-hot encoding

Layout: Before/After transformation with animated conversion

Visual Elements:
- Left side: Original categorical column with color-coded categories
- Right side: Multiple binary columns (one per category)
- Animated arrows showing the transformation
- Each row clearly shows which column gets the "1"

Example Data:
- Categorical column: Color (Red, Blue, Green, Red, Blue)
- Transforms to: Is_Red, Is_Blue, Is_Green columns
- Shows both "keep all" and "drop first" options

Interactive Elements:
- Dropdown: Select different categorical variables to encode
- Toggle: "Drop first category" vs "Keep all categories"
- Hover: Highlight corresponding cells in original and encoded view
- Button: "Add new category" - shows a new column appears
- Slider: Adjust number of unique categories (2-8) to see encoding grow

Educational Callouts:
- Warning when all categories kept: "This creates multicollinearity!"
- Explanation of reference category concept
- Formula showing how original is reconstructed

Color Scheme:
- Each category has unique color
- Same colors used in binary columns for matching

Implementation: HTML/CSS/JavaScript with smooth animations
</details>

## Interaction Terms: When Features Work Together

Sometimes the effect of one feature depends on the value of another. For example, the value of a swimming pool might depend on whether the house is in a warm or cold climate. A pool adds more value in Arizona than in Alaska!

**Interaction terms** capture these combined effects by multiplying features together:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \times x_2)$$

The interaction term $x_1 \times x_2$ allows the effect of $x_1$ to change depending on the value of $x_2$ (and vice versa).

```python
from sklearn.preprocessing import PolynomialFeatures

# Create interaction terms (degree=2, no squared terms)
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_with_interactions = interaction.fit_transform(X)

# See the new feature names
feature_names = interaction.get_feature_names_out(features)
print("Features with interactions:", feature_names)
```

When to consider interactions:

- Domain knowledge suggests features work together
- Residual plots show patterns when you split by another variable
- Theory indicates multiplicative effects
- You have enough data to estimate additional parameters

!!! warning "Interaction Explosion"
    With many features, the number of possible interactions explodes. Five features have 10 pairwise interactions. Ten features have 45. Only include interactions you have good reason to suspect exist, or use regularization to prevent overfitting.

## Polynomial Features: Capturing Curved Relationships

Remember from simple regression that relationships aren't always linear? **Polynomial features** extend multiple regression to handle curved relationships by including squared, cubed, or higher-order terms.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create polynomial features of degree 2
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])

poly_model.fit(X_train, y_train)

# Evaluate
train_score = poly_model.score(X_train, y_train)
test_score = poly_model.score(X_test, y_test)

print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")
```

With polynomial features of degree 2 on 5 original features, you get:

- 5 original features
- 5 squared terms (x₁², x₂², ...)
- 10 interaction terms (x₁x₂, x₁x₃, ...)
- Total: 20 features!

This is powerful but dangerous. Watch your test score carefully—it's easy to overfit with high-degree polynomials.

## Feature Engineering: The Art of Creating Better Features

**Feature engineering** is the creative process of transforming raw data into features that better represent the underlying problem. This is often where data scientists add the most value—domain knowledge transformed into predictive power.

Common feature engineering techniques:

| Technique | Example | Why It Helps |
|-----------|---------|--------------|
| Log transform | log(income) | Handles skewed distributions |
| Binning | Age groups (20s, 30s, 40s) | Captures non-linear thresholds |
| Date extraction | Day of week from timestamp | Captures cyclical patterns |
| Ratios | Price per square foot | Normalizes for size |
| Aggregations | Average neighborhood price | Incorporates context |
| Domain calculations | BMI from height and weight | Captures known relationships |

```python
# Feature engineering examples
df['price_per_sqft'] = df['price'] / df['square_feet']
df['age_squared'] = df['age'] ** 2
df['log_lot_size'] = np.log(df['lot_size'] + 1)  # +1 to handle zeros
df['rooms_per_bathroom'] = df['bedrooms'] / df['bathrooms']
df['is_new'] = (df['age'] < 5).astype(int)
```

Good feature engineering requires:

- Understanding your domain
- Exploring the data thoroughly
- Creativity and experimentation
- Validation to confirm new features actually help

#### Diagram: Feature Engineering Laboratory

<details markdown="1">
<summary>Feature Engineering Laboratory</summary>
Type: microsim

Bloom Taxonomy: Create, Apply

Learning Objective: Practice creating new features and immediately see their impact on model performance

Canvas Layout (900x550):
- Left panel (350x550): Feature creation interface
- Center panel (350x550): Data preview with new features
- Right panel (200x550): Model performance metrics

Left Panel - Feature Creation:
- Dropdown: Select first variable
- Dropdown: Select operation (+, -, *, /, log, square, bin)
- Dropdown: Select second variable (if applicable)
- Text input: New feature name
- Button: "Create Feature"
- List of created features with delete option

Center Panel - Data Preview:
- Table showing original and engineered features
- First 10 rows of data
- Histogram of new feature distribution
- Correlation of new feature with target

Right Panel - Performance:
- R² score (updates when features change)
- Train vs Test comparison
- Feature importance ranking
- Delta from baseline (how much new features helped)

Interactive Workflow:
1. View baseline model performance
2. Create a new feature
3. See immediate impact on R²
4. Try different transformations
5. Compare which features help most

Preset Examples:
- Button: "Try log transform on skewed feature"
- Button: "Create ratio feature"
- Button: "Add polynomial term"

Implementation: p5.js with real-time model retraining
</details>

## Feature Importance: Understanding What Matters

After building a model with many features, you'll want to know which ones are actually important. **Feature importance** measures how much each feature contributes to predictions.

Several approaches to measure importance:

### Coefficient Magnitude (After Standardization)

When features are standardized, coefficient magnitude indicates relative importance:

```python
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Fit model on standardized data
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y_train)

# Compare coefficient magnitudes
importance_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': np.abs(model_scaled.coef_)
}).sort_values('Coefficient', ascending=False)

print(importance_df)
```

### Permutation Importance

**Permutation importance** measures how much the model's performance drops when you randomly shuffle one feature's values. A big drop means the feature was important:

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

# Visualize with Plotly
fig = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    error_x='Std',
    title='Permutation Feature Importance'
)

fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
fig.show()
```

Permutation importance has advantages:

- Works for any model, not just linear regression
- Captures importance in context of other features
- Accounts for interactions

#### Diagram: Feature Importance Explorer

<details markdown="1">
<summary>Feature Importance Explorer</summary>
Type: microsim

Bloom Taxonomy: Analyze, Evaluate

Learning Objective: Compare different methods of measuring feature importance and understand their trade-offs

Canvas Layout (800x500):
- Left panel (400x500): Importance comparison chart
- Right panel (400x500): Individual feature deep-dive

Left Panel Elements:
- Three parallel horizontal bar charts stacked vertically:
  1. Coefficient magnitude (standardized)
  2. Permutation importance
  3. Drop-column importance (R² drop when feature removed)
- Features aligned across all three charts for easy comparison
- Color coding shows agreement/disagreement between methods

Right Panel - Feature Deep-Dive:
- Select a feature to explore in detail
- Scatter plot: feature vs target
- Partial dependence plot
- Distribution of feature values
- Interaction effects with other top features

Interactive Controls:
- Dropdown: Select which importance method to highlight
- Click on feature bar to see deep-dive in right panel
- Toggle: Show error bars (std across iterations)
- Button: "Run permutation test" (animated shuffling)

Visual Insights:
- Highlight when methods disagree about importance ranking
- Show confidence intervals for permutation importance
- Indicate which features might be redundant (similar importance patterns)

Implementation: p5.js with multiple visualization modes
</details>

## Putting It All Together: A Complete Multiple Regression Workflow

Here's a complete workflow for building a multiple regression model with all the techniques we've learned:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px

# 1. Load and explore data
df = pd.read_csv('housing.csv')

# 2. Identify feature types
numerical_features = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'lot_size']
categorical_features = ['neighborhood', 'style']
target = 'price'

# 3. Create feature engineering
df['price_per_sqft'] = df['price'] / df['square_feet']
df['age_squared'] = df['age'] ** 2

# Update feature list
numerical_features.append('age_squared')

# 4. Split data
X = df[numerical_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# 6. Create full pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 7. Train and evaluate with cross-validation
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 8. Fit final model and evaluate on test set
model_pipeline.fit(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)
print(f"Test R²: {test_score:.4f}")

# 9. Check for multicollinearity in numerical features
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_num = X_train[numerical_features]
vif_data = pd.DataFrame({
    'Feature': numerical_features,
    'VIF': [variance_inflation_factor(X_num.values, i) for i in range(len(numerical_features))]
})
print("\nVIF values:")
print(vif_data)

# 10. Visualize predictions vs actuals
y_pred = model_pipeline.predict(X_test)

fig = px.scatter(
    x=y_test,
    y=y_pred,
    labels={'x': 'Actual Price', 'y': 'Predicted Price'},
    title='Multiple Regression: Actual vs Predicted'
)
fig.add_shape(type='line', x0=y_test.min(), x1=y_test.max(),
              y0=y_test.min(), y1=y_test.max(),
              line=dict(dash='dash', color='red'))
fig.show()
```

#### Diagram: Multiple Regression Pipeline

<details markdown="1">
<summary>Multiple Regression Pipeline</summary>
Type: workflow

Bloom Taxonomy: Apply, Analyze

Learning Objective: Understand the complete workflow for building production-ready multiple regression models

Visual Style: Horizontal flowchart with data transformation stages

Stages:
1. "Raw Data"
   Hover: "Mixed types: numbers, categories, missing values"
   Icon: Database
   Color: Gray

2. "Feature Engineering"
   Hover: "Create new features: ratios, transformations, domain knowledge"
   Icon: Wrench
   Color: Blue
   Sub-items: log transforms, ratios, polynomials

3. "Train/Test Split"
   Hover: "80/20 split before any preprocessing"
   Icon: Scissors
   Color: Purple

4. "Preprocessing"
   Hover: "Scale numerics, encode categoricals"
   Icon: Filter
   Color: Orange
   Sub-items: StandardScaler, OneHotEncoder

5. "Check Multicollinearity"
   Hover: "Calculate VIF, handle correlated features"
   Icon: Warning
   Color: Yellow

6. "Feature Selection"
   Hover: "Forward, backward, or stepwise selection"
   Icon: Checkboxes
   Color: Teal

7. "Model Training"
   Hover: "Fit LinearRegression on training data"
   Icon: Brain
   Color: Green

8. "Cross-Validation"
   Hover: "Get stable performance estimate"
   Icon: Loop
   Color: Blue

9. "Final Evaluation"
   Hover: "Test set performance, residual analysis"
   Icon: Chart
   Color: Red

10. "Feature Importance"
    Hover: "Understand what drives predictions"
    Icon: Bar Chart
    Color: Gold

Data Flow Arrows:
- Show data shape changing at each stage
- Indicate sample counts at train/test split
- Show feature counts growing (engineering) and shrinking (selection)

Interactive Elements:
- Click each stage for expanded view
- Hover shows common pitfalls at each stage
- Toggle to show "what can go wrong" warnings

Implementation: HTML/CSS/JavaScript with click interactions
</details>

## Common Mistakes to Avoid

As you build multiple regression models, watch out for these pitfalls:

**Including Too Many Features**: More features don't always mean better models. Each feature adds complexity and potential for overfitting. Start simple and add features only when they demonstrably help.

**Ignoring Multicollinearity**: High VIF values don't break your model, but they make coefficient interpretation unreliable. If you need to explain what each feature does, address multicollinearity first.

**Forgetting to Encode Categoricals**: Passing string columns directly to scikit-learn causes errors. Always one-hot encode or use a proper preprocessor.

**Data Leakage in Preprocessing**: Fit your scaler and encoder only on training data, then transform both train and test. Using information from test data during preprocessing inflates your performance estimates.

**Overfitting with Interactions and Polynomials**: Each interaction or polynomial term is an additional feature. With the power to add quadratic terms and interactions, it's easy to create dozens of features that overfit your training data.

!!! tip "The Simplicity Principle"
    If a simpler model performs almost as well as a complex one, choose the simpler model. It will be easier to explain, more robust to new data, and less likely to fail in production.

## Summary: Your Multiple Regression Toolkit

You now have a comprehensive toolkit for multiple regression:

- **Multiple predictors** let you model complex, multi-factor relationships
- **Multicollinearity** and **VIF** help you diagnose problematic feature correlations
- **Feature selection** methods (forward, backward, stepwise) find the best feature subsets
- **Dummy variables** and **one-hot encoding** handle categorical features
- **Interaction terms** capture features that work together
- **Polynomial features** model curved relationships
- **Feature engineering** creates new predictive variables from domain knowledge
- **Feature importance** reveals what's driving your predictions

With these tools, you can build models that capture the true complexity of real-world problems while remaining interpretable and reliable.

## Looking Ahead

In the next chapter, we'll explore NumPy in depth—the numerical computing engine that powers all of these calculations. Understanding NumPy will help you work more efficiently with large datasets and understand what's happening under the hood of scikit-learn.

---

## Key Takeaways

- Multiple linear regression extends simple regression to handle any number of features
- Each coefficient represents the effect of that feature while holding others constant
- Multicollinearity occurs when features are correlated; use VIF to detect it
- Feature selection methods help identify the most useful features
- Categorical variables must be encoded as dummy variables before modeling
- Interaction terms capture features that work together in non-additive ways
- Feature engineering often provides more improvement than algorithm choice
- Always validate with cross-validation and check residuals for patterns
