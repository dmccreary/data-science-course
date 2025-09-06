# YData Profiling Tutorial: Comprehensive Data Analysis

## Overview

YData Profiling (formerly pandas-profiling) is a powerful Python library that generates comprehensive reports for exploratory data analysis. This tutorial demonstrates its capabilities using a synthetic e-commerce customer dataset designed to showcase various data quality issues and patterns that YData Profiling excels at detecting.

## Installation

First, install YData Profiling:

```bash
pip install ydata-profiling
```

For Jupyter notebooks, you might also want:

```bash
pip install ydata-profiling[notebook]
```

## Dataset Overview

Our tutorial dataset contains 503 rows of e-commerce customer data with 16 columns featuring:

- **Missing values** in multiple columns (ages, income, phone numbers, etc.)
- **Data quality issues** (age outliers, inconsistent phone formatting)
- **Various data types** (numeric, categorical, datetime, boolean, text)
- **High cardinality** categorical variables (cities)
- **Correlations** between variables (age and product preferences)
- **Duplicate rows** for detection testing
- **Skewed distributions** (income follows log-normal distribution)

## Basic Usage

### 1. Generate a Simple Report

```python
import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset
df = pd.read_csv('ecommerce_customer_data.csv')

# Generate basic profile report
profile = ProfileReport(df, title="E-commerce Customer Data Analysis")

# Save to HTML file
profile.to_file("basic_report.html")

# Display in Jupyter notebook
# profile.to_notebook_iframe()
```

### 2. Customized Report with Advanced Configuration

```python
# Advanced configuration for more detailed analysis
profile = ProfileReport(
    df,
    title="Advanced E-commerce Data Analysis",
    dataset={
        "description": "Synthetic e-commerce customer dataset for YData Profiling demonstration",
        "creator": "Data Science Tutorial",
        "author": "Tutorial Author"
    },
    variables={
        "descriptions": {
            "customer_id": "Unique identifier for each customer",
            "age": "Customer age in years",
            "annual_income": "Customer's annual income in USD",
            "total_spent": "Total amount spent by customer",
            "satisfaction_score": "Customer satisfaction rating (1-10)"
        }
    },
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": False},  # Skip Kendall for performance
        "phi_k": {"calculate": True},
        "cramers": {"calculate": True},
    },
    missing_diagrams={
        "bar": True,
        "matrix": True,
        "heatmap": True,
        "dendrogram": True,
    },
    duplicates={
        "head": 10,  # Show first 10 duplicate rows
        "key": None  # Check entire row for duplicates
    },
    samples={
        "head": 5,
        "tail": 5,
        "random": 10
    },
    reject_variables=False,  # Don't automatically reject any variables
    infer_dtypes=True,      # Automatically infer better data types
    interactions={
        "continuous": True,   # Analyze continuous variable interactions
        "targets": []        # Specify target variables if doing supervised learning
    }
)

profile.to_file("advanced_report.html")
```

## Key Features Demonstrated

### 1. Data Types and Overview

YData Profiling automatically detects and categorizes:

- **Numerical variables**: age, annual_income, total_spent, satisfaction_score
- **Categorical variables**: gender, education_level, favorite_category
- **DateTime variables**: registration_date
- **Boolean variables**: premium_member
- **Text variables**: last_review
- **High cardinality categorical**: city (60+ unique values)

### 2. Missing Data Analysis

The report provides multiple visualizations for missing data:

- **Bar chart**: Shows missing count per column
- **Matrix plot**: Visualizes missing data patterns
- **Heatmap**: Shows correlations in missingness
- **Dendrogram**: Clusters variables by missing patterns

Our dataset includes strategic missing values:
- 25 missing ages (5%)
- 35 missing incomes (7%)
- 40 missing credit scores (8%)
- 25 missing phone numbers (5%)

### 3. Data Quality Issues Detection

YData Profiling automatically identifies:

**Outliers**: 
- Age outliers (150 and 5 years old) are flagged as extreme values
- Income distribution shows high-value outliers

**Inconsistent Formatting**:
- Phone numbers in multiple formats: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, xxxxxxxxxx
- Gender entries with inconsistent capitalization: Male, M, m, male

**Data Type Issues**:
- Suggests better data types for mixed-format columns

### 4. Distribution Analysis

For each numerical variable, the report shows:

- **Descriptive statistics**: mean, median, std, quartiles
- **Distribution plots**: histograms with optional normal distribution overlay
- **Skewness and kurtosis**: measures of distribution shape

Our dataset demonstrates:
- **Normal distribution**: age (with outliers)
- **Log-normal distribution**: annual_income (right-skewed)
- **Poisson-like distribution**: total_purchases
- **Gamma distribution**: total_spent

### 5. Correlation Analysis

Multiple correlation methods reveal relationships:

- **Pearson**: Linear relationships between continuous variables
- **Spearman**: Monotonic relationships (rank-based)
- **Phi-K**: Correlation for categorical variables
- **Cramér's V**: Association between categorical variables

Expected correlations in our dataset:
- Age and favorite product category
- Income and total spent
- Education level and income

### 6. Categorical Variable Analysis

For categorical variables, the report provides:

- **Frequency tables**: Count and percentage for each category
- **Bar charts**: Visual representation of category distributions
- **Cardinality warnings**: Flags for high-cardinality variables like 'city'

### 7. Duplicate Detection

The report identifies:
- **3 duplicate rows** intentionally added to the dataset
- **Exact matches** across all columns
- **Percentage of duplicates**: Impact on dataset size

### 8. Text Analysis

For text columns like 'last_review':
- **Length distribution**: Character count statistics
- **Sample values**: Examples of text entries
- **Completeness**: Percentage of non-null text entries

## Advanced Configuration Options

### Minimal Configuration for Large Datasets

```python
# For large datasets, use minimal configuration for faster processing
profile = ProfileReport(
    df,
    title="Quick Analysis",
    minimal=True,  # Faster processing, fewer features
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": False},
        "kendall": {"calculate": False},
        "phi_k": {"calculate": False},
        "cramers": {"calculate": False},
    },
    missing_diagrams={
        "bar": True,
        "matrix": False,
        "heatmap": False,
        "dendrogram": False,
    }
)
```

### Sensitive Data Configuration

```python
# For datasets with sensitive information
profile = ProfileReport(
    df,
    title="Sensitive Data Analysis",
    samples={"head": 0, "tail": 0, "random": 0},  # Don't show actual data
    duplicates={"head": 0},  # Don't show duplicate examples
    sensitive=True  # Additional privacy protections
)
```

## Interpreting the Report

### 1. Executive Summary

The report begins with high-level insights:
- Dataset dimensions (503 rows × 16 columns)
- Missing cells percentage
- Duplicate rows count
- Data types distribution

### 2. Variable Analysis

Each variable gets detailed analysis:
- **Distinct count**: Unique values
- **Missing count**: Null values
- **Memory usage**: Storage requirements
- **Type-specific metrics**: Based on variable type

### 3. Warnings and Alerts

YData Profiling automatically flags:
- **High cardinality**: Variables with many unique values
- **High correlation**: Potentially redundant variables
- **Skewed distributions**: Variables needing transformation
- **Constant values**: Variables with no variation
- **Missing values**: Variables with significant missingness

## Best Practices

### 1. Performance Optimization

For large datasets:
- Use `minimal=True` for quick overview
- Disable expensive correlation calculations
- Limit sample sizes
- Use `lazy=False` for immediate computation

### 2. Customization Tips

- Add variable descriptions for better documentation
- Configure correlation methods based on data types
- Customize missing data visualizations
- Set appropriate duplicate detection keys

### 3. Integration Workflow

```python
# Typical data science workflow integration
def analyze_dataset(df, output_path="profile_report.html"):
    """Generate comprehensive data profile report."""
    
    # Basic data info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Generate profile
    profile = ProfileReport(
        df,
        title=f"Data Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
        minimal=False,
        correlations={"phi_k": {"calculate": True}},
        missing_diagrams={"matrix": True, "bar": True}
    )
    
    # Save report
    profile.to_file(output_path)
    print(f"Report saved to: {output_path}")
    
    return profile

# Usage
profile = analyze_dataset(df, "customer_analysis_report.html")
```

## Common Use Cases

### 1. Initial Data Exploration

Perfect for understanding new datasets before analysis:
- Data quality assessment
- Variable relationship discovery
- Missing data patterns
- Distribution characteristics

### 2. Data Quality Monitoring

Regular profiling for ongoing data pipelines:
- Detect data drift
- Monitor missing value trends
- Track distribution changes
- Identify new data quality issues

### 3. Documentation Generation

Automated documentation for datasets:
- Share with stakeholders
- Document data characteristics
- Support reproducible research
- Create data dictionaries

## Conclusion

YData Profiling provides comprehensive automated exploratory data analysis that would take hours to perform manually. This tutorial dataset demonstrates its ability to:

- Automatically detect various data types and quality issues
- Generate publication-ready visualizations
- Identify patterns and relationships
- Provide actionable insights for data cleaning
- Create comprehensive documentation

The tool is invaluable for data scientists, analysts, and anyone working with datasets who needs quick, thorough data understanding.

## Next Steps

After reviewing the YData Profiling report:

1. **Address data quality issues**: Clean outliers, standardize formats
2. **Handle missing values**: Decide on imputation or removal strategies  
3. **Feature engineering**: Use correlation insights for feature selection
4. **Distribution analysis**: Consider transformations for skewed variables
5. **Duplicate handling**: Remove or investigate duplicate records

YData Profiling provides the foundation for informed data preprocessing and analysis decisions.