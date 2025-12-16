import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate 500 rows of synthetic e-commerce customer data
n_customers = 500

# Customer demographics
customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

# Age with some outliers and missing values
ages = np.random.normal(35, 12, n_customers).astype(int)
ages = np.clip(ages, 18, 85)
# Add some extreme outliers
ages[10] = 150  # Obvious data entry error
ages[25] = 5    # Another obvious error
# Add missing values
missing_age_indices = np.random.choice(n_customers, 25, replace=False)
ages = ages.astype(float)
ages[missing_age_indices] = np.nan

# Gender with some inconsistent formatting
genders = np.random.choice(['Male', 'Female', 'M', 'F', 'm', 'f', 'male', 'female', 'Other'], 
                          n_customers, p=[0.3, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.04, 0.01])

# Income with skewed distribution and some missing values
incomes = np.random.lognormal(10.5, 0.8, n_customers)
incomes = np.round(incomes, 2)
# Add missing values
missing_income_indices = np.random.choice(n_customers, 35, replace=False)
incomes[missing_income_indices] = np.nan

# Education levels
education_levels = np.random.choice([
    'High School', 'Bachelor\'s Degree', 'Master\'s Degree', 
    'PhD', 'Associate Degree', 'Some College', 'Elementary'
], n_customers, p=[0.25, 0.3, 0.2, 0.05, 0.1, 0.08, 0.02])

# City with many categories (high cardinality)
major_cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
minor_cities = [f"City_{i}" for i in range(1, 51)]
cities = np.random.choice(major_cities + minor_cities, n_customers, 
                         p=[0.05] * 10 + [0.01] * 50)

# Registration dates spanning 3 years
start_date = datetime(2021, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = (end_date - start_date).days
registration_dates = [start_date + timedelta(days=random.randint(0, date_range)) 
                     for _ in range(n_customers)]

# Purchase behavior
total_purchases = np.random.poisson(8, n_customers)
total_spent = np.random.gamma(2, 150, n_customers)  # Gamma distribution for spending

# Product categories with correlations to demographics
product_categories = []
for i in range(n_customers):
    if ages[i] > 50 or pd.isna(ages[i]):
        # Older customers prefer certain categories
        cat = np.random.choice(['Home & Garden', 'Health', 'Books', 'Electronics'], 
                              p=[0.4, 0.3, 0.2, 0.1])
    elif ages[i] < 30:
        # Younger customers prefer different categories
        cat = np.random.choice(['Fashion', 'Electronics', 'Sports', 'Gaming'], 
                              p=[0.3, 0.3, 0.2, 0.2])
    else:
        # Middle-aged customers
        cat = np.random.choice(['Electronics', 'Home & Garden', 'Fashion', 'Automotive'], 
                              p=[0.3, 0.25, 0.25, 0.2])
    product_categories.append(cat)

# Customer satisfaction scores (1-10 scale)
satisfaction_scores = np.random.choice(range(1, 11), n_customers, 
                                     p=[0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.15, 0.12, 0.08])

# Email domain to show patterns
email_domains = []
for i in range(n_customers):
    if np.random.random() < 0.6:
        domain = np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'], 
                                 p=[0.5, 0.2, 0.15, 0.15])
    else:
        domain = np.random.choice(['company.com', 'business.org', 'work.net', 'corp.com'])
    email_domains.append(domain)

# Phone numbers with different formats (data quality issue)
phone_formats = [
    lambda: f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
    lambda: f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
    lambda: f"{random.randint(200, 999)}.{random.randint(200, 999)}.{random.randint(1000, 9999)}",
    lambda: f"{random.randint(2000000000, 9999999999)}",
]

phone_numbers = []
for _ in range(n_customers):
    if np.random.random() < 0.95:  # 95% have phone numbers
        format_func = np.random.choice(phone_formats)
        phone_numbers.append(format_func())
    else:
        phone_numbers.append(np.nan)

# Credit score with some missing values and outliers
credit_scores = np.random.normal(720, 80, n_customers).astype(int)
credit_scores = np.clip(credit_scores, 300, 850)
# Add some missing values
missing_credit_indices = np.random.choice(n_customers, 40, replace=False)
credit_scores = credit_scores.astype(float)
credit_scores[missing_credit_indices] = np.nan

# Boolean column - Premium membership
premium_membership = np.random.choice([True, False], n_customers, p=[0.3, 0.7])

# Text column with varying lengths
reviews = []
review_templates = [
    "Great product, highly recommend!",
    "Excellent service and fast delivery.",
    "Good value for money.",
    "Could be better, but acceptable.",
    "Not satisfied with the quality.",
    "Amazing! Will definitely buy again.",
    "Poor quality, would not recommend.",
    "Fast shipping, product as described.",
    "Outstanding customer service experience.",
    "Product arrived damaged, disappointed."
]

for _ in range(n_customers):
    if np.random.random() < 0.8:  # 80% have reviews
        base_review = np.random.choice(review_templates)
        # Add some variation
        if np.random.random() < 0.3:
            base_review += " " + np.random.choice([
                "Very pleased overall.", "Would shop here again.", 
                "Exceeded expectations.", "Needs improvement."
            ])
        reviews.append(base_review)
    else:
        reviews.append(np.nan)

# Create DataFrame
df = pd.DataFrame({
    'customer_id': customer_ids,
    'age': ages,
    'gender': genders,
    'annual_income': incomes,
    'education_level': education_levels,
    'city': cities,
    'registration_date': registration_dates,
    'total_purchases': total_purchases,
    'total_spent': total_spent,
    'favorite_category': product_categories,
    'satisfaction_score': satisfaction_scores,
    'email_domain': email_domains,
    'phone_number': phone_numbers,
    'credit_score': credit_scores,
    'premium_member': premium_membership,
    'last_review': reviews
})

# Add some duplicate rows to demonstrate duplicate detection
duplicate_indices = [50, 150, 300]
duplicate_rows = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Save to CSV
df.to_csv('ecommerce_customer_data.csv', index=False)

print(f"Dataset created with {len(df)} rows and {len(df.columns)} columns")
print("\nDataset features that will showcase YData Profiling capabilities:")
print("✓ Missing values in multiple columns")
print("✓ Data type variety (numeric, categorical, datetime, boolean, text)")
print("✓ Outliers and data quality issues")
print("✓ High cardinality categorical variables")
print("✓ Correlations between variables")
print("✓ Duplicate rows")
print("✓ Inconsistent data formatting")
print("✓ Skewed distributions")
print("✓ Text data with varying lengths")

# Display basic info about the dataset
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\nMissing values per column:")
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        print(f"  {col}: {missing_count} ({missing_count/len(df)*100:.1f}%)")

# Show first few rows
print(f"\nFirst 5 rows:")
print(df.head())