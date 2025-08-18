## Lab 3: Basic Statistical Analysis with DataFrames

### Objectives
Students will learn to:
- Find minimum and maximum values in a DataFrame
- Calculate basic statistical measures (mean, median, standard deviation)
- Identify specific rows based on conditions
- Use pandas methods for statistical analysis

### Prerequisites
- Completed Lab 2 (Loading and exploring DataFrames)
- Understanding of basic statistical concepts

### Lab Steps

#### Step 1: Load and Prepare the Data
```python
import pandas as pd

# Load the healthcare data
df = pd.read_csv('healthcarepercapita2022.csv')

# Display basic info to remind ourselves of the data structure
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
```

#### Step 2: Find the Country with Lowest Healthcare Expenditure

```python
# Method 1: Using min() and boolean indexing
min_expenditure = df['Health_Exp_PerCapita_2022'].min()
lowest_country = df[df['Health_Exp_PerCapita_2022'] == min_expenditure]

print("Country with LOWEST healthcare expenditure per capita:")
print(f"Country: {lowest_country['Country_Name'].iloc[0]}")
print(f"Expenditure: ${min_expenditure:,.2f}")

# Alternative method using idxmin()
min_index = df['Health_Exp_PerCapita_2022'].idxmin()
print(f"\nAlternative method - Lowest: {df.loc[min_index, 'Country_Name']} (${min_expenditure:,.2f})")
```

#### Step 3: Find the Country with Highest Healthcare Expenditure
```python
# Method 1: Using max() and boolean indexing
max_expenditure = df['Health_Exp_PerCapita_2022'].max()
highest_country = df[df['Health_Exp_PerCapita_2022'] == max_expenditure]

print("\nCountry with HIGHEST healthcare expenditure per capita:")
print(f"Country: {highest_country['Country_Name'].iloc[0]}")
print(f"Expenditure: ${max_expenditure:,.2f}")

# Alternative method using idxmax()
max_index = df['Health_Exp_PerCapita_2022'].idxmax()
print(f"\nAlternative method - Highest: {df.loc[max_index, 'Country_Name']} (${max_expenditure:,.2f})")
```

#### Step 4: Calculate Mean Healthcare Expenditure
```python
# Calculate the average (mean)
mean_expenditure = df['Health_Exp_PerCapita_2022'].mean()

print(f"\nAverage (Mean) healthcare expenditure per capita:")
print(f"${mean_expenditure:,.2f}")

# Round to 2 decimal places for cleaner display
print(f"Rounded: ${round(mean_expenditure, 2):,.2f}")
```

#### Step 5: Calculate Median Healthcare Expenditure
```python
# Calculate the median (middle value)
median_expenditure = df['Health_Exp_PerCapita_2022'].median()

print(f"\nMedian healthcare expenditure per capita:")
print(f"${median_expenditure:,.2f}")

# Compare mean vs median
print(f"\nComparison:")
print(f"Mean:   ${mean_expenditure:,.2f}")
print(f"Median: ${median_expenditure:,.2f}")
if mean_expenditure > median_expenditure:
    print("Mean > Median: Data is likely right-skewed (few very high values)")
elif mean_expenditure < median_expenditure:
    print("Mean < Median: Data is likely left-skewed (few very low values)")
else:
    print("Mean ≈ Median: Data is likely normally distributed")
```

#### Step 6: Calculate Standard Deviation
```python
# Calculate standard deviation
std_expenditure = df['Health_Exp_PerCapita_2022'].std()

print(f"\nStandard deviation of healthcare expenditure:")
print(f"${std_expenditure:,.2f}")

# Interpret the standard deviation
print(f"\nInterpretation:")
print(f"About 68% of countries spend between ${mean_expenditure - std_expenditure:,.2f} and ${mean_expenditure + std_expenditure:,.2f} per capita")
print(f"About 95% of countries spend between ${mean_expenditure - 2*std_expenditure:,.2f} and ${mean_expenditure + 2*std_expenditure:,.2f} per capita")
```

#### Step 7: Summary Statistics (All at Once)
```python
# Get all basic statistics at once
print("\nComplete Statistical Summary:")
print(df['Health_Exp_PerCapita_2022'].describe())

# Create a custom summary
print(f"\n{'='*50}")
print("HEALTHCARE EXPENDITURE ANALYSIS SUMMARY")
print(f"{'='*50}")
print(f"Lowest spending country:  {df.loc[df['Health_Exp_PerCapita_2022'].idxmin(), 'Country_Name']}")
print(f"Highest spending country: {df.loc[df['Health_Exp_PerCapita_2022'].idxmax(), 'Country_Name']}")
print(f"Range: ${min_expenditure:,.2f} - ${max_expenditure:,.2f}")
print(f"Mean: ${mean_expenditure:,.2f}")
print(f"Median: ${median_expenditure:,.2f}")
print(f"Standard Deviation: ${std_expenditure:,.2f}")
print(f"Total countries analyzed: {len(df)}")
```

```
Complete Statistical Summary:
count      238.000000
mean      1930.205882
std       2458.454731
min         39.000000
25%        270.000000
50%        930.500000
75%       2467.500000
max      12434.000000
Name: Health_Exp_PerCapita_2022, dtype: float64

==================================================
HEALTHCARE EXPENDITURE ANALYSIS SUMMARY
==================================================
Lowest spending country:  South Sudan
Highest spending country: United States
Range: $39.00 - $12,434.00
Mean: $1,930.21
Median: $930.50
Standard Deviation: $2,458.45
Total countries analyzed: 238
```

### Discussion Questions

1. **Which country spends the least on healthcare per capita? How much do they spend?**

2. **Which country spends the most on healthcare per capita? How much do they spend?**

3. **What is the difference between the highest and lowest spending countries?**

4. **Is the mean higher or lower than the median? What does this tell us about the distribution of healthcare spending?**

5. **How many countries fall within one standard deviation of the mean?**

6. **If a country spends $2,000 per capita on healthcare, is this above or below average?**

### Extension Activities

**For Advanced Students:**
```python
# Find countries within certain ranges
print("\nCountries spending more than $5,000 per capita:")
high_spenders = df[df['Health_Exp_PerCapita_2022'] > 5000]
print(high_spenders[['Country_Name', 'Health_Exp_PerCapita_2022']].sort_values('Health_Exp_PerCapita_2022', ascending=False))

# Calculate what percentage of countries spend above the mean
above_mean = len(df[df['Health_Exp_PerCapita_2022'] > mean_expenditure])
percentage_above_mean = (above_mean / len(df)) * 100
print(f"\n{above_mean} countries ({percentage_above_mean:.1f}%) spend above the global average")
```

### Key Learning Outcomes
- Understanding the difference between mean and median
- Learning to find extreme values in datasets
- Interpreting standard deviation as a measure of variability
- Using pandas methods for statistical analysis
- Connecting statistical concepts to real-world data

This lab builds naturally from the previous exploration lab and introduces fundamental statistical analysis that students will use throughout their data science journey.