import pandas as pd

# Step 1: Load the healthcare data
df = pd.read_csv('healthcare-per-capita-2022.csv')

# Display the DataFrame
print("Healthcare Per Capita Data:")
print(df)

# Step 2: Check the shape (rows, columns)
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Get basic information about the DataFrame
print("\nDataFrame Info:")
df.info()

# Step 3: Look at the first and last 5 rows
# Look at first 5 rows
print("First 5 rows:")
print(df.head())

# Look at last 5 rows
print("\nLast 5 rows:")
print(df.tail())

# Step 4: Check for missing values
print("\nRandom sample of 5 rows:")
print(df.sample(5))