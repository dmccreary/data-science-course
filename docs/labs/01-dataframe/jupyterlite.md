# Lab 1: DataFrames (Browser Version)

This version of Lab 1 runs entirely in your browser using JupyterLite - no installation required!

## What is JupyterLite?

JupyterLite is a Jupyter distribution that runs completely in your web browser using WebAssembly. It includes pandas, numpy, and matplotlib pre-installed, making it perfect for getting started with data science without any setup.

## Launch JupyterLite

Click the button below to open JupyterLite in a new tab:

[Launch JupyterLite :material-rocket-launch:](https://jupyter.org/try-jupyter/lab/){ .md-button .md-button--primary target="_blank" }

## Steps

### Step 1: Create a New Notebook

1. Click the **Launch JupyterLite** button above
2. In JupyterLite, click **File → New → Notebook**
3. Select **Python (Pyodide)** as the kernel

### Step 2: Load the Healthcare Dataset

Copy and paste the following code into the first cell and press **Shift+Enter** to run it:

```python
# Import libraries for JupyterLite/Pyodide
import pandas as pd
from io import StringIO
from pyodide.http import pyfetch

# Fetch the CSV file from the course website
url = "https://dmccreary.github.io/data-science-course/labs/01-dataframe/healthcare-per-capita-2022.csv"
response = await pyfetch(url)
csv_text = await response.string()

# Load into a DataFrame
df = pd.read_csv(StringIO(csv_text))

# Display the first few rows
print("First 5 rows of the data:")
df.head()
```

!!! note "How This Works"
    JupyterLite runs in your browser, which has security restrictions on fetching data from other websites. We use Pyodide's special `pyfetch` function to handle this. The `await` keyword is needed because fetching data over the network is asynchronous.

### Step 3: Explore the DataFrame

Add a new cell and run this code to explore the data:

```python
# Count the number of rows
line_count = len(df)
print(f"Total number of rows: {line_count}")

# Display DataFrame shape and columns
print(f"\nDataFrame shape (rows, columns): {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
```

### Step 4: Basic Statistics

Add another cell to see summary statistics:

```python
# Get summary statistics for the numeric column
print("Summary Statistics for Health Expenditure Per Capita:")
df['Health_Exp_PerCapita_2022'].describe()
```

### Step 5: Find Specific Countries

Let's find some interesting data points:

```python
# Find the country with highest healthcare spending
max_spending = df.loc[df['Health_Exp_PerCapita_2022'].idxmax()]
print(f"Highest spending: {max_spending['Country_Name']} - ${max_spending['Health_Exp_PerCapita_2022']:,}")

# Find the country with lowest healthcare spending
min_spending = df.loc[df['Health_Exp_PerCapita_2022'].idxmin()]
print(f"Lowest spending: {min_spending['Country_Name']} - ${min_spending['Health_Exp_PerCapita_2022']:,}")

# Find the United States
usa = df[df['Country_Code'] == 'USA']
print(f"\nUnited States: ${usa['Health_Exp_PerCapita_2022'].values[0]:,}")
```

## Expected Output

After running all cells, you should see output similar to:

```
First 5 rows of the data:
                  Country_Name Country_Code  Health_Exp_PerCapita_2022
0  Africa Eastern and Southern          AFE                        228
1                  Afghanistan          AFG                        383
2   Africa Western and Central          AFW                        201
3                       Angola          AGO                        217
4                      Albania          ALB                       1186

Total number of rows: 238

DataFrame shape (rows, columns): (238, 3)

Column names: ['Country_Name', 'Country_Code', 'Health_Exp_PerCapita_2022']
```

## Troubleshooting

### "CORS Error" or Data Won't Load

If you encounter a CORS error, you can use this alternative approach with the data embedded directly:

??? example "Click to expand: Embedded Data Alternative"
    ```python
    import pandas as pd
    from io import StringIO

    # First few rows of data embedded directly
    data = """Country_Name,Country_Code,Health_Exp_PerCapita_2022
    United States,USA,12434
    Switzerland,CHE,10668
    Norway,NOR,9927
    Germany,DEU,8454
    Australia,AUS,7072
    Canada,CAN,6991
    France,FRA,6853
    Japan,JPN,5387
    United Kingdom,GBR,6449
    Brazil,BRA,1696
    China,CHN,1136
    India,IND,273
    """

    df = pd.read_csv(StringIO(data))
    df.head()
    ```

### Kernel Won't Start

- Try refreshing the page
- Clear browser cache and try again
- Use a different browser (Chrome or Firefox work best)

## Next Steps

Once you're comfortable with JupyterLite, consider setting up a local Python environment for more advanced work. See the [Desktop Setup Guide](index.md) for instructions.

## Comparison: Browser vs Desktop

| Feature | JupyterLite (Browser) | VS Code (Desktop) |
|---------|----------------------|-------------------|
| Setup required | None | Python + VS Code installation |
| Works offline | No | Yes |
| File access | URLs only | Full filesystem |
| Library support | Core data science libs | Any Python package |
| Performance | Good for small datasets | Better for large datasets |
| Best for | Quick experiments, learning | Real projects, large data |
