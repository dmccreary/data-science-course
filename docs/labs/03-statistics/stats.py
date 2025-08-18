import pandas as pd

# Load the healthcare data
df = pd.read_csv('healthcare-per-capita-2022.csv')

min_expenditure = df['Health_Exp_PerCapita_2022'].min()
lowest_country = df[df['Health_Exp_PerCapita_2022'] == min_expenditure]

max_expenditure = df['Health_Exp_PerCapita_2022'].max()
highest_country = df[df['Health_Exp_PerCapita_2022'] == max_expenditure]
mean_expenditure = df['Health_Exp_PerCapita_2022'].mean()
median_expenditure = df['Health_Exp_PerCapita_2022'].median()
std_expenditure = df['Health_Exp_PerCapita_2022'].std()

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