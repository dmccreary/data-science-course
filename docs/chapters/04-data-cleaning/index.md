---
title: Data Cleaning and Preprocessing
description: Transform messy real-world data into pristine, analysis-ready datasets
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

# Data Cleaning and Preprocessing

## Summary

This chapter covers the critical skills of preparing raw data for analysis. Students will learn to identify and handle missing values, detect and remove duplicates, identify outliers, and validate data quality. The chapter also covers data transformation techniques including filtering, type conversion, and feature scaling. By the end of this chapter, students will be able to clean messy real-world datasets and prepare them for visualization and modeling.

## Concepts Covered

This chapter covers the following 20 concepts from the learning graph:

1. Missing Values
2. NaN
3. Null Detection
4. Dropna Method
5. Fillna Method
6. Imputation
7. Data Type Conversion
8. Duplicate Detection
9. Duplicate Removal
10. Outliers
11. Outlier Detection
12. Data Validation
13. String Cleaning
14. Column Renaming
15. Data Filtering
16. Boolean Indexing
17. Query Method
18. Data Transformation
19. Feature Scaling
20. Normalization

## Prerequisites

This chapter builds on concepts from:

- [Chapter 3: Python Data Structures](../03-python-data-structures/index.md)

---

## The Art of Cleaning Up Messes

In the last chapter, you learned to load data and explore it like a detective. You probably felt pretty good about yourself—data was flowing, `head()` was working, and everything seemed under control. Then reality hit: the data was messy.

Welcome to the real world of data science.

Real-world data is never clean. It's full of gaps where values should be, duplicates that snuck in somehow, numbers that shouldn't exist (like someone being -5 years old), and formatting disasters that would make an English teacher weep. This isn't a bug in the data pipeline—it's just how data works in the wild.

Here's the thing: **messy data will destroy your analysis**. Train a machine learning model on dirty data? It learns the wrong patterns. Calculate averages with missing values? Your numbers lie. Build a report with duplicates? Everything is inflated. Garbage in, garbage out—that's the first law of data science.

But here's the good news: cleaning data is a superpower in itself. Most people don't know how to do it well. Master this chapter, and you'll be the hero who transforms chaotic datasets into pristine, analysis-ready gold. Let's get scrubbing!

#### Diagram: Data Cleaning Pipeline Overview

<details markdown="1">
<summary>Data Cleaning Pipeline Overview</summary>
Type: workflow

Bloom Taxonomy: Understand (L2)

Learning Objective: Help students visualize the complete data cleaning workflow from raw data to analysis-ready data

Purpose: Show the sequential steps in a typical data cleaning process

Visual style: Horizontal flowchart with icons for each stage

Steps (left to right):

1. RAW DATA
   Icon: Messy document with question marks
   Color: Red
   Hover text: "Data as received - full of problems"

2. MISSING VALUES
   Icon: Grid with empty cells highlighted
   Color: Orange
   Hover text: "Identify and handle NaN, None, empty strings"

3. DUPLICATES
   Icon: Two identical rows with X on one
   Color: Yellow
   Hover text: "Find and remove duplicate records"

4. OUTLIERS
   Icon: Box plot with point far outside
   Color: Yellow-green
   Hover text: "Detect and decide how to handle extreme values"

5. DATA TYPES
   Icon: Type conversion symbol (A→1)
   Color: Green
   Hover text: "Convert columns to appropriate types"

6. VALIDATION
   Icon: Checkmark in shield
   Color: Blue
   Hover text: "Verify data meets business rules"

7. TRANSFORMATION
   Icon: Gear with arrows
   Color: Purple
   Hover text: "Scale, normalize, and prepare for analysis"

8. CLEAN DATA
   Icon: Sparkly document with checkmark
   Color: Gold
   Hover text: "Analysis-ready dataset!"

Annotations below pipeline:
- "Each step catches different problems"
- "Order matters: missing values before duplicates"
- "Always document your cleaning decisions"

Error feedback loops:
- Dashed arrows from steps 2-6 back to "Log Issues" box
- "Log Issues" connects to "Data Quality Report"

Implementation: SVG with CSS hover effects
</details>

## Missing Values: The Silent Killers

**Missing values** are the most common data quality problem you'll encounter. They're sneaky—sometimes they look like empty cells, sometimes they're the word "NULL," and sometimes they're special numeric codes like -999.

### Understanding NaN and Null

In pandas, missing values are represented as **NaN** (Not a Number). This is a special value from the NumPy library that indicates "something should be here, but isn't."

```python
import pandas as pd
import numpy as np

# Creating data with missing values
df = pd.DataFrame({
    "name": ["Alice", "Bob", None, "Diana"],
    "age": [25, np.nan, 22, 28],
    "score": [85, 92, 78, np.nan]
})
print(df)
```

Output:
```
    name   age  score
0  Alice  25.0   85.0
1    Bob   NaN   92.0
2   None  22.0   78.0
3  Diana  28.0    NaN
```

Notice how `None` and `np.nan` both represent missing data, but they're handled slightly differently. Pandas is smart enough to recognize both as missing values.

| Missing Value Type | Where You'll See It | Pandas Representation |
|-------------------|---------------------|----------------------|
| Empty cell | CSV files, Excel | NaN |
| None | Python code, JSON | NaN (converted) |
| NULL | Databases | NaN (converted) |
| Empty string ("") | Text files | Not automatically NaN! |
| Special codes (-999, 9999) | Legacy systems | Not automatically NaN! |

!!! warning "Hidden Missing Values"
    Be careful! Empty strings and special codes like -999 are NOT automatically treated as missing. You need to identify and convert them manually. Always inspect your data after loading.

### Null Detection: Finding the Gaps

**Null detection** is the process of finding where your missing values hide. Pandas provides several methods for this:

```python
# Check for missing values - returns True/False for each cell
df.isnull()

# Count missing values in each column
df.isnull().sum()

# Percentage of missing values
(df.isnull().sum() / len(df)) * 100

# Find rows with ANY missing values
df[df.isnull().any(axis=1)]

# Find rows with missing values in a specific column
df[df["age"].isnull()]
```

The `isnull()` method is your first line of defense. Run it immediately after loading any dataset—it tells you exactly where the problems are.

```python
# Real-world pattern: Quick missing value assessment
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")
print(f"Percentage missing: {df.isnull().sum().sum() / df.size * 100:.1f}%")
```

#### Diagram: Missing Value Detection MicroSim

<details markdown="1">
<summary>Missing Value Detective MicroSim</summary>
Type: microsim

Bloom Taxonomy: Apply (L3)

Learning Objective: Help students practice identifying and counting missing values in different scenarios

Canvas layout (750x500px):
- Top (750x200): DataFrame display with highlighted missing values
- Bottom left (350x300): Detection code panel
- Bottom right (400x300): Results and quiz area

Visual elements:
- DataFrame grid showing 8 rows × 5 columns
- Missing values highlighted in red when detected
- Color legend showing: Present (green), NaN (red), None (orange), Empty string (yellow)
- Running count of missing values by column

Sample data scenarios (rotate through):
1. Simple NaN values only
2. Mix of NaN and None
3. Hidden missing values (empty strings, -999)
4. Missing values with pattern (all in one column)
5. Sparse data (>50% missing)

Interactive controls:
- Button: "Show isnull()" - highlights all missing
- Button: "Count by Column" - shows bar chart of missing counts
- Button: "Find Hidden Missing" - identifies non-standard missing values
- Dropdown: "Select Scenario" - changes dataset
- Quiz mode: "How many missing in column X?" with input field

Behavior:
- Clicking detection buttons animates the detection process
- Correct quiz answers earn points and unlock harder scenarios
- Hints available after wrong answers
- Progress tracker shows scenarios completed

Visual style: Data detective theme with magnifying glass cursor

Implementation: p5.js with animated highlighting
</details>

### Handling Missing Values: Your Three Options

Once you've found missing values, you have three main strategies:

1. **Drop them** - Remove rows or columns with missing data
2. **Fill them** - Replace missing values with something reasonable
3. **Leave them** - Some algorithms can handle missing values directly

### The dropna Method: Clean Sweep

The **dropna method** removes rows or columns containing missing values:

```python
# Drop any row with missing values
df_clean = df.dropna()

# Drop rows only if ALL values are missing
df_clean = df.dropna(how="all")

# Drop rows with missing values in specific columns
df_clean = df.dropna(subset=["name", "age"])

# Drop columns (instead of rows) with missing values
df_clean = df.dropna(axis=1)
```

When to use `dropna()`:

- Missing values are rare (less than 5% of data)
- Rows with missing data are truly unusable
- You have plenty of data to spare
- Missing values are random (not systematic)

When NOT to use `dropna()`:

- It would remove too much data
- Missing values follow a pattern (systematic missingness)
- The columns with missing data are important
- You suspect the missing data contains signal

### The fillna Method: Filling the Gaps

The **fillna method** replaces missing values with specified values:

```python
# Fill with a constant value
df["age"].fillna(0)

# Fill with the mean of the column
df["age"].fillna(df["age"].mean())

# Fill with the median (more robust to outliers)
df["age"].fillna(df["age"].median())

# Fill with the most common value (mode) for categorical data
df["city"].fillna(df["city"].mode()[0])

# Fill with the previous value (forward fill) - great for time series
df["temperature"].fillna(method="ffill")

# Fill with the next value (backward fill)
df["temperature"].fillna(method="bfill")
```

| Fill Strategy | Best For | Example |
|--------------|----------|---------|
| Mean | Normally distributed numeric data | Age, height |
| Median | Skewed numeric data, outliers present | Income, prices |
| Mode | Categorical data | City, category |
| Forward fill | Time series data | Stock prices, temperatures |
| Constant (0) | When zero has meaning | Transaction amounts |
| Constant ("Unknown") | Categorical placeholders | Status fields |

### Imputation: The Smart Approach

**Imputation** is the fancy term for filling missing values based on patterns in the data. It's more sophisticated than simple mean/median filling.

```python
# Simple imputation: use column statistics
from sklearn.impute import SimpleImputer

# Create imputer that uses mean for numeric columns
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[["age", "score"]]),
    columns=["age", "score"]
)
```

Advanced imputation strategies include:

- **K-Nearest Neighbors (KNN)**: Fill based on similar rows
- **Regression imputation**: Predict missing values from other columns
- **Multiple imputation**: Create multiple filled versions and combine

!!! tip "The Golden Rule of Missing Data"
    Always ask: WHY is the data missing? Random missing values can often be filled. But if data is missing for a reason (like patients who missed follow-up because they got worse), filling it can introduce bias. When in doubt, document your decision and consider sensitivity analysis.

## Duplicate Detection and Removal

Duplicates are copies of data that shouldn't exist. They inflate your counts, skew your statistics, and make your analyses unreliable. They sneak in through:

- Data entry mistakes
- Multiple data imports
- System glitches
- Merging datasets incorrectly

### Duplicate Detection: Finding the Clones

**Duplicate detection** identifies rows that appear more than once:

```python
# Check for exact duplicate rows
df.duplicated()  # Returns True/False for each row

# Count total duplicates
df.duplicated().sum()

# View the duplicate rows
df[df.duplicated()]

# View all copies (including the first occurrence)
df[df.duplicated(keep=False)]

# Check duplicates based on specific columns only
df.duplicated(subset=["name", "email"])
```

The `duplicated()` method marks duplicates starting from the second occurrence. Use `keep=False` to see ALL copies, including the first one.

```python
# Example: Finding partial duplicates
# Maybe the same person appears with slightly different data
df_people = pd.DataFrame({
    "name": ["Alice", "Bob", "Alice", "Charlie"],
    "email": ["a@mail.com", "b@mail.com", "a@mail.com", "c@mail.com"],
    "age": [25, 30, 26, 28]  # Alice's age changed!
})

# Find duplicates by name and email
print(df_people[df_people.duplicated(subset=["name", "email"], keep=False)])
```

### Duplicate Removal: Eliminating the Clones

**Duplicate removal** keeps only unique rows:

```python
# Remove exact duplicates (keep first occurrence)
df_unique = df.drop_duplicates()

# Keep the last occurrence instead
df_unique = df.drop_duplicates(keep="last")

# Remove duplicates based on specific columns
df_unique = df.drop_duplicates(subset=["email"])

# Remove ALL copies of duplicated rows (nuclear option)
df_unique = df.drop_duplicates(keep=False)
```

When removing duplicates, ask yourself:

- Should you keep the first or last occurrence?
- Are you sure the duplicates are truly identical?
- Should you merge duplicate information instead of just dropping?

#### Diagram: Duplicate Handling Decision Tree

<details markdown="1">
<summary>Duplicate Handling Decision Tree</summary>
Type: diagram

Bloom Taxonomy: Analyze (L4)

Learning Objective: Help students decide the appropriate strategy for handling different types of duplicates

Purpose: Guide decision-making process for duplicate handling

Visual style: Decision tree flowchart

Start: "Duplicates Detected"

Decision 1: "Are rows EXACTLY identical?"
- Yes → "Safe to drop_duplicates()"
- No → Decision 2

Decision 2: "Which columns make rows 'the same'?"
- Identify key columns (ID, email, etc.)
- → Decision 3

Decision 3: "Do non-key columns differ?"
- No → "drop_duplicates(subset=[key_cols])"
- Yes → Decision 4

Decision 4: "Which version is correct?"
- First occurrence → "keep='first'"
- Last occurrence → "keep='last'"
- Need to merge → "Use groupby().agg()"
- Can't determine → "Flag for manual review"

Special case branch:
"Is duplication intentional?"
- Yes (e.g., same product bought twice) → "Don't remove!"
- No → Continue with removal

Color coding:
- Green: Safe actions
- Yellow: Need investigation
- Red: Be careful
- Blue: Decision points

Annotations:
- "Always examine duplicates before removing"
- "Document which strategy you used and why"

Implementation: SVG decision tree with interactive highlights
</details>

## Outliers: The Extreme Values

**Outliers** are data points that are unusually far from other observations. They might be:

- **Legitimate extremes**: A billionaire in income data—rare but real
- **Data errors**: Someone's age recorded as 999
- **Measurement errors**: A sensor glitch recording impossible values

Outliers matter because they can dramatically affect your statistics. One outlier can shift your mean, expand your standard deviation, and confuse your machine learning models.

### Outlier Detection: Finding the Extremes

**Outlier detection** identifies values that don't fit the normal pattern. Common methods include:

**1. Visual inspection (always start here!):**
```python
import matplotlib.pyplot as plt

# Box plot - shows outliers as dots beyond the whiskers
df["score"].plot(kind="box")
plt.show()

# Histogram - outliers appear as isolated bars
df["score"].hist(bins=30)
plt.show()
```

**2. Statistical methods:**
```python
# Z-score method: outliers are >3 standard deviations from mean
from scipy import stats
z_scores = stats.zscore(df["score"].dropna())
outliers = df[(abs(z_scores) > 3)]

# IQR method: outliers are beyond 1.5 × IQR from quartiles
Q1 = df["score"].quantile(0.25)
Q3 = df["score"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["score"] < lower_bound) | (df["score"] > upper_bound)]
```

| Method | How It Works | Best For |
|--------|--------------|----------|
| Z-score | Distance from mean in standard deviations | Normal distributions |
| IQR | Distance from quartiles | Any distribution, robust to extremes |
| Domain rules | Based on what's possible | When you know valid ranges |

**3. Domain knowledge rules:**
```python
# Age can't be negative or over 120
invalid_ages = df[(df["age"] < 0) | (df["age"] > 120)]

# Test scores must be 0-100
invalid_scores = df[(df["score"] < 0) | (df["score"] > 100)]
```

#### Diagram: Outlier Detection Methods MicroSim

<details markdown="1">
<summary>Outlier Detection Playground</summary>
Type: microsim

Bloom Taxonomy: Apply (L3)

Learning Objective: Let students experiment with different outlier detection methods and see how they identify different points

Canvas layout (800x550px):
- Left (550x550): Scatter plot / histogram visualization
- Right (250x550): Controls and detected outliers list

Visual elements:
- Data points displayed as circles
- Normal points in blue
- Detected outliers highlighted in red with labels
- Detection threshold lines/zones shown
- Summary statistics displayed

Sample datasets (toggle between):
1. Normal distribution with obvious outliers
2. Skewed distribution (income-like)
3. Multi-modal distribution
4. Real-world messy data with errors

Interactive controls:
- Radio buttons: Detection method (Z-score, IQR, Custom range)
- Slider: Z-score threshold (1.5 to 4.0)
- Slider: IQR multiplier (1.0 to 3.0)
- Number inputs: Custom min/max values
- Dropdown: Dataset selector
- Toggle: Show/hide threshold lines

Display panels:
- Count of outliers detected
- Outlier values and indices
- Percentage of data flagged
- Before/after mean comparison

Behavior:
- Adjusting thresholds immediately updates highlighting
- Hovering over points shows their values
- Clicking outliers adds them to "investigate" list
- Show how different methods catch different outliers

Educational annotations:
- "Z-score catches 3 outliers, IQR catches 5 - which is right?"
- "Lower threshold = more outliers flagged"
- "Some 'outliers' might be valid data!"

Visual style: Clean statistical visualization with gridlines

Implementation: p5.js with dynamic data visualization
</details>

### Handling Outliers: What To Do

Once you've identified outliers, you have options:

```python
# Option 1: Remove them
df_clean = df[(df["score"] >= lower_bound) & (df["score"] <= upper_bound)]

# Option 2: Cap them (winsorization)
df["score_capped"] = df["score"].clip(lower=lower_bound, upper=upper_bound)

# Option 3: Replace with NaN (treat as missing)
df.loc[(df["score"] < lower_bound) | (df["score"] > upper_bound), "score"] = np.nan

# Option 4: Log transform (reduces impact of extremes)
df["score_log"] = np.log1p(df["score"])

# Option 5: Keep them! (if they're legitimate)
```

!!! warning "Don't Just Delete Outliers!"
    Before removing outliers, ask: Are they errors or legitimate extremes? Removing real data points because they're "inconvenient" is bad science. Document every outlier decision.

## Data Validation: Enforcing the Rules

**Data validation** checks that data meets expected criteria. It's the quality control checkpoint that catches problems before they contaminate your analysis.

Common validation checks:

```python
# Check data types are correct
assert df["age"].dtype in ["int64", "float64"], "Age should be numeric"

# Check value ranges
assert df["age"].between(0, 120).all(), "Invalid ages found"

# Check for required values
assert df["customer_id"].notna().all(), "Missing customer IDs"

# Check uniqueness
assert df["email"].is_unique, "Duplicate emails found"

# Check referential integrity
valid_categories = ["A", "B", "C", "D"]
assert df["category"].isin(valid_categories).all(), "Invalid categories"

# Check date logic
assert (df["end_date"] >= df["start_date"]).all(), "End before start"
```

Building a validation function:

```python
def validate_student_data(df):
    """Validate student dataset and return issues found."""
    issues = []

    # Check required columns exist
    required = ["student_id", "name", "age", "gpa"]
    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check data types
    if df["age"].dtype not in ["int64", "float64"]:
        issues.append("Age is not numeric")

    # Check value ranges
    if (df["age"] < 0).any() or (df["age"] > 100).any():
        issues.append(f"Invalid ages: {df[(df['age'] < 0) | (df['age'] > 100)]['age'].tolist()}")

    if (df["gpa"] < 0).any() or (df["gpa"] > 4.0).any():
        issues.append("GPA out of range [0, 4.0]")

    # Check for duplicates
    if df["student_id"].duplicated().any():
        issues.append(f"Duplicate student_ids: {df[df['student_id'].duplicated()]['student_id'].tolist()}")

    if issues:
        print("⚠️ Validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All validations passed!")
        return True

# Use it
validate_student_data(df)
```

## Data Transformation Techniques

Now that your data is clean, it's time to transform it into the shape your analysis needs. **Data transformation** includes filtering, selecting, renaming, type conversion, and more.

### Data Filtering: Selecting Subsets

**Data filtering** extracts rows that meet specific criteria. You learned the basics in Chapter 3—now let's master advanced techniques.

```python
# Basic filtering review
high_scorers = df[df["score"] > 90]

# Multiple conditions (AND)
young_high_scorers = df[(df["age"] < 25) & (df["score"] > 90)]

# Multiple conditions (OR)
special_cases = df[(df["score"] > 95) | (df["age"] > 50)]

# NOT condition
not_from_ny = df[~(df["city"] == "New York")]
```

### Boolean Indexing: The Power Tool

**Boolean indexing** is the technique behind filtering—using True/False arrays to select data:

```python
# Create a boolean mask
mask = df["score"] > 80
print(mask)  # Shows True/False for each row

# Use the mask to filter
high_scorers = df[mask]

# Combine masks
mask1 = df["score"] > 80
mask2 = df["age"] < 30
combined = df[mask1 & mask2]  # Both conditions
either = df[mask1 | mask2]    # Either condition

# Invert a mask
low_scorers = df[~mask1]  # NOT high scorers
```

Understanding boolean indexing makes complex filtering intuitive. You can build masks step-by-step, test them separately, and combine them logically.

### The Query Method: SQL-Like Filtering

The **query method** offers a cleaner syntax for complex filters:

```python
# Instead of this:
result = df[(df["age"] > 25) & (df["city"] == "New York") & (df["score"] >= 80)]

# You can write this:
result = df.query("age > 25 and city == 'New York' and score >= 80")

# Using variables in queries
min_age = 25
target_city = "New York"
result = df.query("age > @min_age and city == @target_city")

# Complex conditions read more naturally
result = df.query("(score > 90) or (age < 20 and score > 80)")
```

| Approach | Syntax | Best For |
|----------|--------|----------|
| Boolean indexing | `df[df["col"] > val]` | Simple conditions, programmatic use |
| Query method | `df.query("col > val")` | Complex conditions, readability |

### String Cleaning: Taming Text Data

**String cleaning** standardizes text data that comes in many messy forms:

```python
# Convert to lowercase
df["name"] = df["name"].str.lower()

# Remove leading/trailing whitespace
df["city"] = df["city"].str.strip()

# Replace characters
df["phone"] = df["phone"].str.replace("-", "")

# Extract patterns with regex
df["zipcode"] = df["address"].str.extract(r"(\d{5})")

# Check for patterns
df["has_email"] = df["contact"].str.contains("@")
```

Common string cleaning operations:

```python
# All-in-one cleaning function
def clean_text(text):
    if pd.isna(text):
        return text
    return (text
            .strip()                      # Remove whitespace
            .lower()                       # Lowercase
            .replace("  ", " ")           # Fix double spaces
           )

df["name"] = df["name"].apply(clean_text)
```

### Column Renaming: Clear Names Matter

**Column renaming** makes your data self-documenting:

```python
# Rename specific columns
df = df.rename(columns={
    "cust_id": "customer_id",
    "amt": "amount",
    "dt": "transaction_date"
})

# Rename all columns at once
df.columns = ["id", "name", "age", "score"]

# Apply a function to all column names
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")

# Chain operations
df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))
```

Good column names are:

- Lowercase with underscores (snake_case)
- Descriptive but not too long
- Free of spaces and special characters
- Consistent across your project

### Data Type Conversion: Getting Types Right

**Data type conversion** ensures columns have appropriate types for analysis:

```python
# Check current types
print(df.dtypes)

# Convert to numeric
df["price"] = pd.to_numeric(df["price"], errors="coerce")  # Invalid → NaN

# Convert to string
df["zipcode"] = df["zipcode"].astype(str)

# Convert to datetime
df["date"] = pd.to_datetime(df["date"])

# Convert to categorical (saves memory, enables ordering)
df["grade"] = pd.Categorical(df["grade"], categories=["F", "D", "C", "B", "A"], ordered=True)

# Convert to integer (handling NaN)
df["age"] = df["age"].astype("Int64")  # Nullable integer type
```

| Original Type | Convert To | Why |
|---------------|-----------|-----|
| String numbers | `float64` or `int64` | Enable math operations |
| Dates as strings | `datetime64` | Enable date arithmetic |
| Numeric codes | `category` | Save memory, show meaning |
| Float IDs | `Int64` | Preserve as integers with NaN support |

#### Diagram: Data Type Conversion Guide Infographic

<details markdown="1">
<summary>Data Type Conversion Reference</summary>
Type: infographic

Bloom Taxonomy: Remember (L1)

Learning Objective: Quick reference for common data type conversions and when to use them

Purpose: Visual guide for choosing the right data type conversion method

Layout: Two-column reference card style

Section 1: "Converting TO Numeric"
- `pd.to_numeric(col)` - Basic conversion
- `pd.to_numeric(col, errors='coerce')` - Invalid → NaN
- `col.astype(float)` - When you're sure it's clean
- Visual: String "42" → Integer 42

Section 2: "Converting TO Datetime"
- `pd.to_datetime(col)` - Smart parsing
- `pd.to_datetime(col, format='%Y-%m-%d')` - Specific format
- Visual: String "2024-03-15" → Datetime object

Section 3: "Converting TO Categorical"
- `col.astype('category')` - Basic categorical
- `pd.Categorical(col, categories=[...], ordered=True)` - Ordered
- Visual: Repeating strings → Category codes

Section 4: "Converting TO String"
- `col.astype(str)` - Simple conversion
- `col.map('{:.2f}'.format)` - With formatting
- Visual: Number 3.14159 → String "3.14"

Common pitfalls callout box:
- "Integer columns with NaN need Int64 (capital I)"
- "datetime parsing can be slow on large datasets"
- "Category type saves memory but changes behavior"

Color coding:
- Blue: Numeric conversions
- Green: Date conversions
- Purple: Categorical conversions
- Orange: String conversions

Interactive elements:
- Hover over conversion methods to see code examples
- Click to copy code snippet

Implementation: HTML/CSS with JavaScript tooltips
</details>

## Feature Scaling and Normalization

When you're preparing data for machine learning, **feature scaling** becomes critical. Different features might have vastly different ranges—age might be 18-80 while income might be 20,000-500,000. Without scaling, the larger numbers dominate the analysis.

### Feature Scaling: Bringing Features to Same Scale

**Feature scaling** transforms features to comparable ranges:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max scaling: transforms to [0, 1] range
scaler = MinMaxScaler()
df[["age_scaled", "income_scaled"]] = scaler.fit_transform(df[["age", "income"]])

# Standard scaling: transforms to mean=0, std=1
scaler = StandardScaler()
df[["age_standard", "income_standard"]] = scaler.fit_transform(df[["age", "income"]])
```

### Normalization: Statistical Standardization

**Normalization** typically refers to scaling to unit norm or standard distribution:

```python
# Z-score normalization (same as StandardScaler)
df["score_normalized"] = (df["score"] - df["score"].mean()) / df["score"].std()

# Min-max normalization (manual)
df["score_minmax"] = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())

# Robust scaling (uses median/IQR, resistant to outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df["income_robust"] = scaler.fit_transform(df[["income"]])
```

| Scaling Method | Formula | When to Use |
|----------------|---------|-------------|
| Min-Max | $(x - min) / (max - min)$ | Neural networks, bounded algorithms |
| Z-score/Standard | $(x - \mu) / \sigma$ | Most ML algorithms, normally distributed data |
| Robust | $(x - median) / IQR$ | Data with outliers |
| Log transform | $\log(x + 1)$ | Right-skewed data (income, counts) |

!!! tip "When to Scale"
    Scale features when using distance-based algorithms (KNN, SVM, K-means) or gradient descent (neural networks). Tree-based algorithms (Random Forest, XGBoost) usually don't need scaling.

#### Diagram: Feature Scaling Comparison MicroSim

<details markdown="1">
<summary>Feature Scaling Visualizer</summary>
Type: microsim

Bloom Taxonomy: Understand (L2)

Learning Objective: Help students visualize how different scaling methods transform data distributions

Canvas layout (800x500px):
- Top (800x200): Original data distribution histogram
- Bottom left (400x300): Scaled distribution histogram
- Bottom right (400x300): Controls and comparison stats

Visual elements:
- Original data histogram with descriptive statistics
- Scaled data histogram (updates with scaling method)
- Before/after comparison statistics table
- Visual axis showing value ranges

Sample datasets:
- Normal distribution (symmetric)
- Right-skewed (income-like)
- With outliers
- Bimodal distribution

Interactive controls:
- Radio buttons: Scaling method
  - None (original)
  - Min-Max [0,1]
  - Standard (Z-score)
  - Robust (median/IQR)
  - Log transform
- Dropdown: Dataset selector
- Checkbox: Show outliers highlighted
- Checkbox: Show before/after overlay

Comparison statistics displayed:
- Min, Max, Range
- Mean, Median
- Std Dev, IQR
- Visual indicator of how outliers are affected

Behavior:
- Switching scaling method animates the transformation
- Hover over bars to see exact values
- Toggle overlay to see original vs scaled superimposed
- Outliers maintain highlighting through transformation

Educational annotations:
- "Notice: Min-Max squishes outliers to 0 or 1"
- "Standard scaling keeps outliers as extreme z-scores"
- "Robust scaling ignores outliers!"
- "Log transform pulls in right tail"

Visual style: Statistical visualization with clean grid

Implementation: p5.js with real-time distribution updates
</details>

## Complete Data Cleaning Workflow

Let's put it all together with a complete data cleaning workflow:

```python
import pandas as pd
import numpy as np

def clean_dataset(df, config=None):
    """
    Complete data cleaning pipeline.

    Parameters:
    -----------
    df : DataFrame
        Raw data to clean
    config : dict, optional
        Cleaning configuration options

    Returns:
    --------
    DataFrame : Cleaned data
    dict : Cleaning report
    """
    report = {"original_shape": df.shape, "issues_found": [], "actions_taken": []}

    # Step 1: Make a copy
    df_clean = df.copy()

    # Step 2: Handle missing values
    missing_counts = df_clean.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        report["issues_found"].append(f"Missing values in {len(cols_with_missing)} columns")

        # Fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                report["actions_taken"].append(f"Filled {col} with median")

        # Fill categorical columns with mode
        cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                report["actions_taken"].append(f"Filled {col} with mode")

    # Step 3: Remove duplicates
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        report["issues_found"].append(f"{n_duplicates} duplicate rows")
        df_clean = df_clean.drop_duplicates()
        report["actions_taken"].append(f"Removed {n_duplicates} duplicates")

    # Step 4: Clean string columns
    string_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].str.strip()
        df_clean[col] = df_clean[col].str.lower()
    report["actions_taken"].append("Cleaned string columns (strip, lowercase)")

    # Step 5: Standardize column names
    df_clean.columns = (df_clean.columns
                        .str.strip()
                        .str.lower()
                        .str.replace(" ", "_")
                        .str.replace("-", "_"))
    report["actions_taken"].append("Standardized column names")

    # Final report
    report["final_shape"] = df_clean.shape
    report["rows_removed"] = report["original_shape"][0] - report["final_shape"][0]

    return df_clean, report

# Use the function
df_raw = pd.read_csv("messy_data.csv")
df_clean, cleaning_report = clean_dataset(df_raw)

print(f"Cleaned {cleaning_report['original_shape'][0]} rows → {cleaning_report['final_shape'][0]} rows")
print("Actions taken:", cleaning_report["actions_taken"])
```

!!! success "Achievement Unlocked: Data Janitor"
    You now have the skills to transform any messy dataset into clean, analysis-ready data. This isn't glamorous work, but it's where real data scientists spend 60-80% of their time. You're now equipped for the real world!

## Common Patterns and Best Practices

### The Cleaning Checklist

Before any analysis, run through this checklist:

- [ ] Load data and check `shape`
- [ ] View `head()` and `tail()` for anomalies
- [ ] Check `dtypes` for correct types
- [ ] Run `isnull().sum()` for missing values
- [ ] Run `duplicated().sum()` for duplicates
- [ ] Check `describe()` for impossible values
- [ ] Validate against business rules
- [ ] Document all cleaning decisions

### Anti-Patterns to Avoid

```python
# DON'T: Modify data without inspection
df = df.dropna()  # How much did you just lose?

# DO: Inspect first, then decide
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Would drop {df.isnull().any(axis=1).sum()} rows")

# DON'T: Hardcode cleaning values
df["status"].fillna("Unknown")  # What if this changes?

# DO: Use data-driven or configurable values
default_status = df["status"].mode()[0]  # Most common value

# DON'T: Silently modify the original
df.dropna(inplace=True)

# DO: Create cleaned versions
df_clean = df.dropna().copy()
```

??? question "Chapter 4 Checkpoint: Test Your Understanding"
    **Question:** You receive a dataset with the following issues:
    - 5% of rows have missing ages
    - 2% of rows have duplicate emails
    - Some scores are recorded as -1 (meaning "not applicable")
    - Income column is stored as strings like "$50,000"

    Write a cleaning plan for this data.

    **Click to reveal answer:**

    ```python
    # 1. Handle missing ages (5% is acceptable to fill)
    df["age"].fillna(df["age"].median(), inplace=True)

    # 2. Handle duplicates based on email
    df = df.drop_duplicates(subset=["email"], keep="last")

    # 3. Handle -1 scores (replace with NaN, not a real value)
    df["score"] = df["score"].replace(-1, np.nan)

    # 4. Clean income column
    df["income"] = (df["income"]
                    .str.replace("$", "")
                    .str.replace(",", "")
                    .astype(float))
    ```

## Key Takeaways

1. **Missing values** (NaN) must be found with `isnull()` and handled with `dropna()`, `fillna()`, or imputation strategies.

2. **Duplicates** inflate your data—detect with `duplicated()` and remove with `drop_duplicates()`.

3. **Outliers** can be errors or legitimate extremes—detect with IQR or z-scores, then decide whether to remove, cap, or keep.

4. **Data validation** enforces business rules—build validation functions to catch problems early.

5. **Boolean indexing** and the **query method** let you filter data with complex conditions.

6. **String cleaning** standardizes text—use `.str` methods for cleaning operations.

7. **Data type conversion** ensures columns have appropriate types for analysis.

8. **Feature scaling** (Min-Max, Standard, Robust) brings features to comparable ranges for machine learning.

9. **Document everything**—cleaning decisions affect your entire analysis, so keep a record.

10. The **cleaning workflow** (missing → duplicates → outliers → types → validation → transformation) should become second nature.

You've mastered the art of data cleaning—arguably the most valuable practical skill in data science. In the next chapter, you'll learn to visualize your clean data, turning numbers into insights that everyone can understand. The glamorous part is coming!
