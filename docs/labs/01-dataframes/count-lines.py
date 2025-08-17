# Import the pandas library for data manipulation
import pandas as pd

# Load the CSV file into a pandas DataFrame
# This assumes the CSV file is in the same directory as this script
df = pd.read_csv('healthcare-per-capita-2022.csv')

# Count the number of rows (lines) in the DataFrame
line_count = len(df)

# Display the first few rows to verify the data loaded correctly
print("First 5 rows of the data:")
print(df.head())

# Print the total number of lines
print(f"\nTotal number of lines in the CSV file: {line_count}")

# Optional: Display basic information about the DataFrame
print(f"\nDataFrame shape (rows, columns): {df.shape}")
print(f"Column names: {list(df.columns)}")