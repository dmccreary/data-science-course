# Lab 2: Introduction to DataFrames - Loading and Exploring Data

### Objectives

Students will learn to:
- Load CSV data into a pandas DataFrame
- Explore basic DataFrame properties and methods
- Display and examine data structure

### Lab Steps

#### Step 1: Load the Data

```python
import pandas as pd

# Load the healthcare data
df = pd.read_csv('healthcare-per-capita-2022.csv')

# Display the DataFrame
print("Healthcare Per Capita Data:")
print(df)
```

```
                    Country_Name Country_Code  Health_Exp_PerCapita_2022
0    Africa Eastern and Southern          AFE                        228
1                    Afghanistan          AFG                        383
2     Africa Western and Central          AFW                        201
3                         Angola          AGO                        217
4                        Albania          ALB                       1186
..                           ...          ...                        ...
233                        Samoa          WSM                        396
234                  Yemen, Rep.          YEM                        109
235                 South Africa          ZAF                       1341
236                       Zambia          ZMB                        208
237                     Zimbabwe          ZWE                         96
```

#### Step 2: Explore DataFrame Shape and Info

```python
# Check the shape (rows, columns)
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Get basic information about the DataFrame
print("\nDataFrame Info:")
df.info()
```

#### Step 3: Examine Column Names and Data Types

```python
# Display column names
print("Column names:")
print(df.columns.tolist())

# Check data types
print("\nData types:")
print(df.dtypes)
```

```
Dataset shape: (238, 3)
Number of rows: 238
Number of columns: 3

DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 238 entries, 0 to 237
Data columns (total 3 columns):
 #   Column                     Non-Null Count  Dtype 
---  ------                     --------------  ----- 
 0   Country_Name               238 non-null    object
 1   Country_Code               238 non-null    object
 2   Health_Exp_PerCapita_2022  238 non-null    int64 
dtypes: int64(1), object(2)
memory usage: 5.7+ KB
```

#### Step 4: Preview the Data

```python
# Look at first 5 rows
print("First 5 rows:")
print(df.head())

# Look at last 5 rows
print("\nLast 5 rows:")
print(df.tail())
```

Results:

```
First 5 rows:
                  Country_Name Country_Code  Health_Exp_PerCapita_2022
0  Africa Eastern and Southern          AFE                        228
1                  Afghanistan          AFG                        383
2   Africa Western and Central          AFW                        201
3                       Angola          AGO                        217
4                      Albania          ALB                       1186

Last 5 rows:
     Country_Name Country_Code  Health_Exp_PerCapita_2022
233         Samoa          WSM                        396
234   Yemen, Rep.          YEM                        109
235  South Africa          ZAF                       1341
236        Zambia          ZMB                        208
237      Zimbabwe          ZWE                         96
```

# Step 5: Get basic information about the DataFrame

```python
print("\nDataFrame Info:")
df.info()
```


```
DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 238 entries, 0 to 237
Data columns (total 3 columns):
 #   Column                     Non-Null Count  Dtype 
---  ------                     --------------  ----- 
 0   Country_Name               238 non-null    object
 1   Country_Code               238 non-null    object
 2   Health_Exp_PerCapita_2022  238 non-null    int64 
dtypes: int64(1), object(2)
```

# Look at a random sample of 5 rows

```python
print("\nRandom sample of 5 rows:")
print(df.sample(5))
```

#### Step 5: Basic Data Exploration

```python
# Get basic statistics for numerical columns
print("Basic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Count unique values in each column
print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

### Questions for Students

1. How many countries are included in this dataset?
2. What is the data type of each column?
3. Are there any missing values in the dataset?
4. What country has the highest healthcare expenditure per capita?
5. What is the average healthcare expenditure per capita across all countries?

### Expected Output Discussion

Students should observe:

- The dataset has 238 rows (countries) and 3 columns
- Country_Name and Country_Code are text (object) data types
- Health_Exp_PerCapita_2022 is numerical (integer)
- Whether there are any missing values to handle

### Extension Activities

For advanced students:

- Sort the data by healthcare expenditure
- Find countries with expenditure above/below certain thresholds
- Create simple filtering operations

This lab builds naturally from counting rows to actually working with the data structure, 
introducing essential pandas concepts while keeping the complexity manageable for beginners.