---
title: Python Data Structures
description: Master the containers that hold your data - from Python basics to pandas power
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

# Python Data Structures

## Summary

This chapter introduces the essential data structures used in Python data science workflows. Students will learn Python's native data structures (lists, dictionaries, tuples, arrays) and then progress to pandas, the primary library for data manipulation. The chapter covers DataFrames and Series, data loading from CSV files, and methods for inspecting and selecting data. By the end of this chapter, students will be able to load, explore, and navigate datasets using pandas.

## Concepts Covered

This chapter covers the following 20 concepts from the learning graph:

1. Lists
2. Dictionaries
3. Tuples
4. Arrays
5. Pandas Library
6. DataFrame
7. Series
8. Index
9. Column
10. Row
11. Data Loading
12. CSV Files
13. Read CSV
14. Data Inspection
15. Head Method
16. Tail Method
17. Shape Attribute
18. Info Method
19. Describe Method
20. Data Selection

## Prerequisites

This chapter builds on concepts from:

- [Chapter 1: Introduction to Data Science](../01-intro-to-data-science/index.md)
- [Chapter 2: Python Environment and Setup](../02-python-environment/index.md)

---

## Your Data Needs a Home

In Chapter 1, you learned that data is the raw material of your superpower. In Chapter 2, you built your headquarters and assembled your tools. But here's the thing about data—it doesn't just float around in thin air. Data needs to live somewhere. It needs containers, organization, and structure.

Think about it: even superheroes need ways to organize their stuff. Batman has labeled compartments in his utility belt. Iron Man has JARVIS cataloging his suit designs. The X-Men have Cerebro organizing mutant data. You? You're about to learn the containers that will hold YOUR data—and trust me, these containers are more powerful than they might first appear.

This chapter is where you graduate from "knowing about data science" to actually DOING data science. By the end, you'll be loading real datasets, exploring them like a detective, and selecting exactly the information you need. Let's go!

#### Diagram: Data Structure Hierarchy

<details markdown="1">
    <summary>Python Data Structure Hierarchy</summary>
    Type: diagram

    Bloom Taxonomy: Understand (L2)

    Learning Objective: Help students visualize the relationship between Python's native data structures and pandas structures

    Purpose: Show how basic Python structures lead to more powerful pandas structures

    Layout: Hierarchical tree diagram, top to bottom

    Level 1 (Top): "DATA STRUCTURES"
    - Color: Gold
    - Central hub

    Level 2 (Split into two branches):

    Branch A: "Python Built-in"
    - Color: Blue
    - Children:
      - Lists (ordered, mutable)
      - Dictionaries (key-value pairs)
      - Tuples (ordered, immutable)

    Branch B: "Pandas Structures"
    - Color: Green
    - Children:
      - Series (1D labeled array)
      - DataFrame (2D labeled table)

    Level 3 (Under pandas):
    - DataFrame components: Index, Columns, Rows
    - Series: labeled as "Single column of DataFrame"

    Arrows showing:
    - "List → can become → Series"
    - "Dictionary → can become → DataFrame"
    - "Series + Series → DataFrame"

    Visual style: Modern flowchart with rounded boxes

    Interactive elements:
    - Hover over each structure to see quick definition
    - Click to see code example
    - Color coding: Blue for Python native, Green for pandas

    Implementation: SVG with CSS hover effects
</details>

## Python's Built-in Data Structures

Before we dive into the fancy pandas library, you need to understand Python's native data structures. These are the building blocks—simple containers that Python provides out of the box. Think of them as the basic training before you get the advanced gear.

### Lists: Your Ordered Collection

A **list** is Python's most versatile data structure. It's an ordered collection of items that can hold anything—numbers, strings, even other lists. Lists are mutable, meaning you can change them after creation.

```python
# Creating a list
superheroes = ["Batman", "Iron Man", "Spider-Man", "Wonder Woman"]

# Lists can hold different types
mixed_data = [42, "hello", 3.14, True]

# Lists are ordered - position matters!
print(superheroes[0])  # Output: "Batman" (first item)
print(superheroes[-1]) # Output: "Wonder Woman" (last item)
```

Lists are incredibly flexible:

- Add items with `append()` or `insert()`
- Remove items with `remove()` or `pop()`
- Sort with `sort()`
- Get the length with `len()`

```python
# Modifying lists
superheroes.append("Black Panther")      # Add to end
superheroes.insert(0, "Superman")        # Add at position 0
superheroes.remove("Spider-Man")         # Remove specific item
last_hero = superheroes.pop()            # Remove and return last item
```

!!! tip "When to Use Lists"
    Use lists when you have a collection of items and their ORDER matters. Lists are perfect for sequences, rankings, or any situation where position is important.

### Dictionaries: Your Key-Value Powerhouse

A **dictionary** is a collection of key-value pairs. Instead of accessing items by position (like lists), you access them by a unique key. Dictionaries are like having labeled drawers instead of numbered ones.

```python
# Creating a dictionary
hero_powers = {
    "Batman": "Intelligence and gadgets",
    "Iron Man": "Powered armor suit",
    "Spider-Man": "Spider abilities",
    "Wonder Woman": "Divine powers"
}

# Accessing values by key
print(hero_powers["Batman"])  # Output: "Intelligence and gadgets"

# Adding new key-value pairs
hero_powers["Black Panther"] = "Vibranium suit"

# Checking if key exists
if "Superman" in hero_powers:
    print(hero_powers["Superman"])
else:
    print("Superman not found!")
```

Dictionaries are blazingly fast for lookups. Whether you have 10 items or 10 million, finding a value by its key takes roughly the same amount of time. That's why they're used everywhere in Python.

| Operation | List | Dictionary |
|-----------|------|------------|
| Access by position | ✓ Fast | ✗ Not supported |
| Access by key/name | ✗ Slow (must search) | ✓ Instant |
| Maintains order | ✓ Yes | ✓ Yes (Python 3.7+) |
| Duplicate keys | ✓ Allowed | ✗ Keys must be unique |

### Tuples: Your Immutable Record

A **tuple** is like a list, but immutable—once created, it cannot be changed. This might sound like a limitation, but it's actually a feature. Tuples are perfect for data that shouldn't change.

```python
# Creating a tuple
coordinates = (40.7128, -74.0060)  # NYC latitude, longitude

# Accessing elements (just like lists)
latitude = coordinates[0]
longitude = coordinates[1]

# Tuples can be unpacked
lat, lon = coordinates  # Assigns both at once!

# This would cause an error:
# coordinates[0] = 41.0  # TypeError: tuples are immutable
```

Why use tuples?

- They're slightly faster than lists
- They signal "this data won't change"
- They can be used as dictionary keys (lists cannot)
- They're perfect for returning multiple values from functions

### Arrays: Numeric Power (Preview)

**Arrays** are specialized containers for numeric data. While Python has a built-in `array` module, data scientists typically use NumPy arrays, which we'll explore in depth in Chapter 10.

The key difference: lists can hold any type of data, but arrays are optimized for numbers. This makes mathematical operations on arrays incredibly fast—often 10-100x faster than the same operations on lists.

```python
# Preview: NumPy arrays (covered fully in Chapter 10)
import numpy as np

# Create an array of numbers
scores = np.array([85, 92, 78, 95, 88])

# Mathematical operations are fast and easy
average = scores.mean()
highest = scores.max()
scaled = scores * 1.1  # Multiply all by 1.1
```

For now, just know that arrays exist and are important. We'll come back to them with full superhero treatment later!

#### Diagram: Python Data Structures Comparison MicroSim

<details markdown="1">
    <summary>Data Structure Selection Helper</summary>
    Type: microsim

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Help students choose the right data structure for different scenarios

    Canvas layout (700x500px):
    - Left side (400x500): Scenario display and answer area
    - Right side (300x500): Score tracker and hints

    Visual elements:
    - Scenario card showing a data storage problem
    - Four buttons: List, Dictionary, Tuple, Array
    - Feedback indicator (correct/incorrect with explanation)
    - Progress bar showing scenarios completed
    - Score counter

    Scenarios (10 total):
    1. "Store student names in order of arrival" → List
    2. "Map employee IDs to their salaries" → Dictionary
    3. "Store GPS coordinates that won't change" → Tuple
    4. "Hold 1 million numbers for fast math" → Array
    5. "Keep a to-do list you'll modify" → List
    6. "Store configuration settings by name" → Dictionary
    7. "Return both min and max from a function" → Tuple
    8. "Store pixel values for image processing" → Array
    9. "Track items and their quantities" → Dictionary
    10. "Keep records of (x, y) point pairs" → Tuple or List

    Interactive controls:
    - Click data structure button to answer
    - "Next Scenario" button
    - "Show Hint" button
    - "Reset Quiz" button

    Behavior:
    - Correct answer: Green flash, +10 points, explanation shown
    - Incorrect: Red flash, correct answer revealed with explanation
    - End: Summary showing areas to review

    Visual style: Quiz game aesthetic with achievement badges

    Implementation: p5.js with scenario array and scoring logic
</details>

## Enter Pandas: Your Data Science Superweapon

Now we're getting to the good stuff. **Pandas** is the Python library that transformed data science. Before pandas, working with tabular data in Python was painful. After pandas, it became almost enjoyable.

The **Pandas library** provides two main data structures:

- **Series**: A one-dimensional labeled array
- **DataFrame**: A two-dimensional labeled table (like a spreadsheet)

These aren't just containers—they're containers with superpowers. Filtering, grouping, merging, reshaping... pandas makes operations that would take dozens of lines of code happen in just one.

```python
# The standard way to import pandas
import pandas as pd

# Now you can use pd.DataFrame, pd.Series, pd.read_csv, etc.
```

!!! quote "Why 'pandas'?"
    The name comes from "panel data"—a term from statistics for multi-dimensional data. It has nothing to do with the cute bears, though the library is just as cuddly once you get to know it.

### Series: The One-Dimensional Wonder

A **Series** is like a supercharged list. It's a one-dimensional array with labels (called an index). Think of it as a single column from a spreadsheet, but with its own identity.

```python
import pandas as pd

# Create a Series from a list
scores = pd.Series([85, 92, 78, 95, 88])
print(scores)
```

Output:
```
0    85
1    92
2    78
3    95
4    88
dtype: int64
```

Notice those numbers on the left? That's the **index**—labels for each value. By default, pandas uses 0, 1, 2, etc., but you can use custom labels:

```python
# Series with custom index
scores = pd.Series(
    [85, 92, 78, 95, 88],
    index=["Alice", "Bob", "Charlie", "Diana", "Eve"]
)
print(scores)
```

Output:
```
Alice      85
Bob        92
Charlie    78
Diana      95
Eve        88
dtype: int64
```

Now you can access data by name: `scores["Bob"]` returns `92`. This is incredibly powerful when working with real data.

### DataFrame: The Main Event

The **DataFrame** is pandas' flagship data structure. It's a two-dimensional table with labeled rows and columns—essentially, a super-powered spreadsheet that you can manipulate with code.

| Student | Math | Science | English |
|---------|------|---------|---------|
| Alice | 85 | 90 | 88 |
| Bob | 92 | 88 | 95 |
| Charlie | 78 | 82 | 80 |

In pandas, this becomes:

```python
import pandas as pd

# Create a DataFrame from a dictionary
data = {
    "Math": [85, 92, 78],
    "Science": [90, 88, 82],
    "English": [88, 95, 80]
}
df = pd.DataFrame(data, index=["Alice", "Bob", "Charlie"])
print(df)
```

Output:
```
         Math  Science  English
Alice      85       90       88
Bob        92       88       95
Charlie    78       82       80
```

This is the structure you'll work with 90% of the time in data science. Let's break down its components.

#### Diagram: DataFrame Anatomy

<details markdown="1">
    <summary>DataFrame Anatomy Interactive Diagram</summary>
    Type: infographic

    Bloom Taxonomy: Remember (L1)

    Learning Objective: Help students identify and remember the components of a DataFrame

    Purpose: Visual breakdown of DataFrame structure with labeled components

    Layout: Central DataFrame table with callouts pointing to each component

    Main visual: A 4x3 DataFrame table displaying:
    ```
              Col_A   Col_B   Col_C
    row_0      10      20      30
    row_1      40      50      60
    row_2      70      80      90
    row_3     100     110     120
    ```

    Callouts (numbered, with leader lines):

    1. INDEX (pointing to row labels on left)
       - "Row labels - can be numbers, strings, or dates"
       - "Access with: df.index"
       - Color: Blue

    2. COLUMNS (pointing to column headers)
       - "Column names - usually strings"
       - "Access with: df.columns"
       - Color: Green

    3. ROW (highlighting entire row_1)
       - "One observation/record"
       - "Access with: df.loc['row_1'] or df.iloc[1]"
       - Color: Orange

    4. COLUMN (highlighting entire Col_B)
       - "One variable/feature"
       - "Access with: df['Col_B']"
       - Color: Purple

    5. CELL (highlighting intersection of row_1 and Col_B = 50)
       - "Single value"
       - "Access with: df.loc['row_1', 'Col_B']"
       - Color: Red

    6. VALUES (pointing to all numbers)
       - "The actual data (NumPy array underneath)"
       - "Access with: df.values"
       - Color: Gray

    Interactive elements:
    - Hover over each component to highlight it
    - Click to see code example for accessing that component
    - Toggle "Show Code" to see access patterns

    Visual style: Clean spreadsheet look with color highlights

    Implementation: HTML/CSS with JavaScript interactivity
</details>

### Understanding Index, Columns, and Rows

Let's clarify these fundamental concepts:

**Index**: The row labels of a DataFrame. Think of it as the "name tag" for each row.

```python
print(df.index)  # Index(['Alice', 'Bob', 'Charlie'], dtype='object')
```

**Columns**: The column headers—the names of your variables.

```python
print(df.columns)  # Index(['Math', 'Science', 'English'], dtype='object')
```

**Row**: A single horizontal slice—one observation with values for all columns.

**Column**: A single vertical slice—one variable with values for all observations. When you extract a single column, you get a Series!

```python
# Get a single column (returns a Series)
math_scores = df["Math"]
print(type(math_scores))  # <class 'pandas.core.series.Series'>
```

!!! tip "DataFrame vs Series"
    A DataFrame is essentially a collection of Series that share the same index. Each column is a Series. Understanding this relationship will help you navigate pandas intuitively.

## Loading Real Data: CSV Files

Theory is great, but data science is about DOING. Let's load some real data!

**Data loading** is the process of reading data from external sources into Python. The most common format you'll encounter is CSV.

### What Are CSV Files?

**CSV** stands for Comma-Separated Values. It's a simple text format where:

- Each line is one row of data
- Values in a row are separated by commas
- The first line usually contains column names

Here's what a CSV file looks like inside:

```
name,age,city,score
Alice,25,New York,85
Bob,30,Los Angeles,92
Charlie,22,Chicago,78
Diana,28,Houston,95
```

CSV files are everywhere because they're simple, human-readable, and work with almost any software. Excel, Google Sheets, databases—they all speak CSV.

### The read_csv Method

The **read_csv** function is your gateway to loading CSV data:

```python
import pandas as pd

# Load a CSV file into a DataFrame
df = pd.read_csv("students.csv")

# That's it! df now contains all your data
print(df)
```

Output:
```
      name  age         city  score
0    Alice   25     New York     85
1      Bob   30  Los Angeles     92
2  Charlie   22      Chicago     78
3    Diana   28      Houston     95
```

`read_csv` is incredibly smart. It automatically:

- Detects the delimiter (comma, tab, etc.)
- Identifies the header row
- Infers data types for each column
- Handles missing values

For special situations, you have options:

```python
# Specify a different delimiter
df = pd.read_csv("data.tsv", sep="\t")  # Tab-separated

# Use a different row as header
df = pd.read_csv("data.csv", header=1)  # Second row as header

# No header in file
df = pd.read_csv("data.csv", header=None, names=["A", "B", "C"])

# Set a column as index
df = pd.read_csv("data.csv", index_col="name")
```

!!! warning "File Paths Matter"
    `read_csv("data.csv")` looks for the file in your current working directory. If the file is elsewhere, use the full path: `read_csv("/path/to/data.csv")` or a relative path: `read_csv("../data/data.csv")`.

#### Diagram: Data Loading Workflow

<details markdown="1">
    <summary>CSV Loading Workflow</summary>
    Type: workflow

    Bloom Taxonomy: Understand (L2)

    Learning Objective: Help students understand the complete process from CSV file to usable DataFrame

    Purpose: Visualize the journey of data from file to DataFrame

    Visual style: Horizontal flowchart with icons

    Steps:

    1. CSV FILE ON DISK
       Icon: Document with ".csv" label
       Label: "Raw text file with comma-separated values"
       Color: Gray

    2. pd.read_csv()
       Icon: Pandas logo with arrow
       Label: "Pandas reads and parses the file"
       Color: Orange

    3. PARSING HAPPENS
       Icon: Gears turning
       Sub-steps shown below:
       - "Detect delimiter"
       - "Read header row"
       - "Infer data types"
       - "Handle missing values"
       Color: Blue

    4. DATAFRAME CREATED
       Icon: Table grid
       Label: "Data now in memory as DataFrame"
       Color: Green

    5. READY FOR ANALYSIS
       Icon: Sparkles/magic wand
       Label: "Filter, analyze, visualize!"
       Color: Gold

    Annotations:
    - Below step 1: "Could be 100 rows or 100 million"
    - Below step 4: "Lives in computer memory (RAM)"
    - Below step 5: "This is where the fun begins"

    Error path (branching from step 2):
    - "FileNotFoundError" → "Check your file path!"
    - "ParserError" → "Check file format and delimiter"

    Interactive elements:
    - Hover each step for detailed explanation
    - Click to see common errors at each stage

    Implementation: SVG with CSS animations
</details>

## Data Inspection: Getting to Know Your Data

You've loaded a dataset. Now what? Before doing any analysis, you need to understand what you're working with. This is **data inspection**—exploring your data to understand its structure, size, types, and content.

Think of it like this: you've received a mysterious package. Before using what's inside, you'd want to know how big it is, what's in it, and whether anything's broken. Same with data.

### The head Method: Peek at the Beginning

The **head method** shows you the first few rows of your DataFrame. It's usually the first thing you'll do after loading data.

```python
# Show first 5 rows (default)
df.head()

# Show first 10 rows
df.head(10)
```

Output:
```
      name  age         city  score
0    Alice   25     New York     85
1      Bob   30  Los Angeles     92
2  Charlie   22      Chicago     78
3    Diana   28      Houston     95
4      Eve   26       Boston     88
```

Why is this useful? Real datasets often have thousands or millions of rows. You can't print the whole thing. `head()` gives you a quick preview without overwhelming your screen.

### The tail Method: Check the End

The **tail method** shows the last few rows. It's great for verifying that your data loaded completely and for seeing recent entries in time-series data.

```python
# Show last 5 rows (default)
df.tail()

# Show last 3 rows
df.tail(3)
```

!!! tip "Head and Tail Together"
    Use `head()` AND `tail()` together. Sometimes data has different formatting at the beginning vs. end, or there are footer rows you need to handle. Looking at both ends catches issues early.

### The shape Attribute: Know Your Dimensions

The **shape attribute** tells you the dimensions of your DataFrame—how many rows and columns it has.

```python
print(df.shape)  # Output: (1000, 15)
# This means: 1000 rows, 15 columns
```

Shape is crucial because:

- It tells you how much data you're working with
- It helps identify if data loaded correctly
- It's used for splitting data in machine learning

```python
# Unpack shape into variables
num_rows, num_cols = df.shape
print(f"Dataset has {num_rows} observations and {num_cols} variables")
```

### The info Method: Get the Full Picture

The **info method** provides a complete summary of your DataFrame:

```python
df.info()
```

Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   name    1000 non-null   object
 1   age     998 non-null    int64
 2   city    1000 non-null   object
 3   score   995 non-null    float64
Dtypes: float64(1), int64(1), object(2)
memory usage: 31.4+ KB
```

This tells you:

- Total number of entries (rows)
- Column names and their order
- Non-null counts (how many values aren't missing)
- Data types for each column
- Memory usage

See those "non-null" counts? If they're less than the total rows, you have missing data. The `age` column has 998 non-null out of 1000—meaning 2 values are missing. That's important to know!

### The describe Method: Statistical Summary

The **describe method** gives you summary statistics for all numeric columns:

```python
df.describe()
```

Output:
```
              age        score
count  998.000000   995.000000
mean    27.500000    85.200000
std      5.230000    10.150000
min     18.000000    55.000000
25%     23.000000    78.000000
50%     27.000000    85.000000
75%     32.000000    93.000000
max     45.000000   100.000000
```

In one command, you get:

- **count**: Number of non-null values
- **mean**: Average value
- **std**: Standard deviation (spread)
- **min/max**: Smallest and largest values
- **25%, 50%, 75%**: Quartiles (distribution shape)

This is your first glimpse at what the data looks like statistically. Is the average score 85 or 50? Are ages ranging from 18-25 or 18-80? `describe()` answers these questions instantly.

#### Diagram: Data Inspection Command Center

<details markdown="1">
    <summary>Data Inspection Dashboard MicroSim</summary>
    Type: microsim

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Let students practice using inspection methods and see results immediately

    Canvas layout (800x600px):
    - Top area (800x100): Sample DataFrame display
    - Left panel (300x500): Method buttons
    - Right panel (500x500): Output display area

    Sample DataFrame (displayed at top):
    Small 8x5 DataFrame with realistic data including:
    - name (strings)
    - age (integers with 1 missing)
    - city (strings)
    - score (floats with 1 missing)
    - active (booleans)

    Method buttons (left panel):
    - `.head()` with slider for n (1-8)
    - `.tail()` with slider for n (1-8)
    - `.shape`
    - `.info()`
    - `.describe()`
    - `.columns`
    - `.dtypes`

    Output display (right panel):
    - Shows result of selected method
    - Formatted to look like Jupyter output
    - Syntax highlighting for code

    Interactive controls:
    - Click method button to execute
    - Adjust slider for head/tail n parameter
    - "Show Code" toggle displays the Python code
    - "Reset" button returns to initial state

    Behavior:
    - Clicking method shows its output
    - Output updates immediately when parameters change
    - Code panel shows exact command used
    - Tooltips explain what each method returns

    Educational annotations:
    - For head/tail: "Notice: only shows [n] rows!"
    - For shape: "(rows, columns) - easy to remember!"
    - For info: "Look for non-null counts to find missing data"
    - For describe: "Only numeric columns shown by default"

    Visual style: IDE/notebook aesthetic with dark mode option

    Implementation: p5.js with precomputed outputs
</details>

## Data Selection: Getting What You Need

Now for the really powerful stuff. **Data selection** is how you extract specific pieces from your DataFrame—particular rows, columns, or combinations. This is like having X-ray vision for your data.

### Selecting Columns

The most common selection: grabbing specific columns.

```python
# Select a single column (returns a Series)
ages = df["age"]

# Select multiple columns (returns a DataFrame)
subset = df[["name", "score"]]
```

Notice the double brackets for multiple columns: `df[["col1", "col2"]]`. This is a common source of errors—remember it!

### Selecting Rows by Position: iloc

Use **iloc** (integer location) to select rows by their position number:

```python
# First row
first_row = df.iloc[0]

# First three rows
first_three = df.iloc[0:3]

# Specific rows
some_rows = df.iloc[[0, 2, 5]]

# Rows and columns by position
cell = df.iloc[0, 1]  # First row, second column
```

Think of `iloc` as "**i**nteger **loc**ation"—it works with numbers.

### Selecting Rows by Label: loc

Use **loc** to select rows by their index labels:

```python
# If index is default (0, 1, 2...)
row = df.loc[0]  # Same as iloc[0]

# If index is custom (names)
alice_data = df.loc["Alice"]

# Multiple labels
some_data = df.loc[["Alice", "Charlie"]]

# Rows AND columns by label
score = df.loc["Alice", "score"]  # Alice's score
```

Think of `loc` as "**l**abel **loc**ation"—it works with names.

### Boolean Selection: The Real Superpower

Here's where it gets amazing. You can select rows based on conditions:

```python
# All rows where age > 25
older_students = df[df["age"] > 25]

# All rows where score >= 90
high_scorers = df[df["score"] >= 90]

# Combine conditions with & (and) or | (or)
young_high_scorers = df[(df["age"] < 25) & (df["score"] >= 90)]

# Select specific cities
ny_and_la = df[df["city"].isin(["New York", "Los Angeles"])]
```

This is incredibly powerful. Need all customers who spent over $100 in the last month? One line. All students who failed the exam? One line. All products with low inventory? One line.

| Selection Type | Syntax | Returns |
|----------------|--------|---------|
| Single column | `df["col"]` | Series |
| Multiple columns | `df[["col1", "col2"]]` | DataFrame |
| Rows by position | `df.iloc[0]` or `df.iloc[0:5]` | Series or DataFrame |
| Rows by label | `df.loc["label"]` | Series or DataFrame |
| Boolean filter | `df[df["col"] > value]` | DataFrame |

#### Diagram: Data Selection Playground MicroSim

<details markdown="1">
    <summary>Interactive Data Selection Playground</summary>
    Type: microsim

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Let students practice different selection methods and see results in real-time

    Canvas layout (850x600px):
    - Top area (850x200): Full DataFrame display (10 rows x 5 cols)
    - Bottom left (400x400): Selection builder
    - Bottom right (450x400): Results display

    Sample DataFrame:
    10 rows with columns: name, age, city, score, active
    Mix of data types and some interesting patterns to query

    Selection builder (tabs):

    Tab 1: "Column Selection"
    - Checkboxes for each column
    - Shows: df[["selected", "columns"]]

    Tab 2: "Row Selection (iloc)"
    - Start index input
    - End index input
    - Shows: df.iloc[start:end]

    Tab 3: "Row Selection (loc)"
    - Dropdown of index labels
    - Multi-select for multiple labels
    - Shows: df.loc[[labels]]

    Tab 4: "Boolean Filter"
    - Column dropdown
    - Operator dropdown (>, <, ==, >=, <=, !=)
    - Value input
    - "Add condition" button (for combining)
    - Shows: df[df["col"] > value]

    Results display:
    - Shows selected data as formatted table
    - Shows shape of result
    - Shows code used to generate selection
    - Highlights selected cells in original DataFrame

    Interactive controls:
    - Real-time update as selections change
    - "Copy Code" button
    - "Reset" button
    - Syntax examples shown as hints

    Educational features:
    - Color coding: selected rows/columns highlighted in original
    - Error messages for invalid selections
    - Hints: "Try selecting students with score > 80"

    Challenges (optional):
    - "Select all students from New York"
    - "Find the youngest student"
    - "Get names of students who passed (score >= 70)"

    Implementation: p5.js or React with pandas-like simulation
</details>

## Putting It All Together: A Complete Workflow

Let's walk through a real data exploration workflow, combining everything you've learned:

```python
import pandas as pd

# Step 1: Load the data
df = pd.read_csv("student_data.csv")

# Step 2: Quick peek
print("First few rows:")
print(df.head())

# Step 3: Check dimensions
print(f"\nDataset shape: {df.shape}")

# Step 4: Get detailed info
print("\nDataset info:")
df.info()

# Step 5: Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Step 6: Ask questions with selection
# Who are the high performers?
top_students = df[df["score"] >= 90]
print(f"\nTop students: {len(top_students)}")

# What's the average score by city?
city_scores = df.groupby("city")["score"].mean()
print("\nAverage score by city:")
print(city_scores)
```

This workflow—load, inspect, explore, select—is the foundation of every data science project. Master it, and you're ready for anything.

!!! success "Achievement Unlocked: Data Wrangler"
    You can now load data, inspect its structure, and select exactly what you need. These skills alone put you ahead of most people who just "open files in Excel." You're thinking like a data scientist!

## Common Patterns and Pitfalls

Before we wrap up, here are patterns you'll use constantly and pitfalls to avoid:

### Patterns to Remember

```python
# Load and immediately inspect
df = pd.read_csv("data.csv")
df.head()  # or df.info() or df.shape

# Chain methods for quick exploration
df.head().describe()

# Check for missing values
df.isnull().sum()

# Get unique values in a column
df["column"].unique()
df["column"].nunique()  # count of unique values
df["column"].value_counts()  # frequency of each value
```

### Pitfalls to Avoid

```python
# WRONG: Single brackets for multiple columns
df["col1", "col2"]  # Error!

# RIGHT: Double brackets
df[["col1", "col2"]]

# WRONG: Using iloc with labels
df.iloc["Alice"]  # Error!

# RIGHT: Use loc for labels
df.loc["Alice"]

# WRONG: Forgetting parentheses in boolean logic
df[df["age"] > 25 & df["score"] > 80]  # Error!

# RIGHT: Wrap each condition in parentheses
df[(df["age"] > 25) & (df["score"] > 80)]
```

??? question "Chapter 3 Checkpoint: Test Your Understanding"
    **Question:** You have a DataFrame `df` with columns: student_id, name, major, gpa, credits. Write code to:
    1. Find all students with GPA above 3.5
    2. Select only the name and major columns
    3. Count how many students are in each major

    **Click to reveal answer:**

    ```python
    # 1. High GPA students
    high_gpa = df[df["gpa"] > 3.5]

    # 2. Name and major only
    names_majors = df[["name", "major"]]

    # 3. Count by major
    major_counts = df["major"].value_counts()
    ```

## Key Takeaways

1. **Python's built-in structures** (lists, dictionaries, tuples) are the building blocks, but pandas structures are purpose-built for data science.

2. **Lists** are ordered and mutable—use them when order matters and you need flexibility.

3. **Dictionaries** provide instant key-based lookup—perfect for mapping relationships.

4. **Tuples** are immutable—use them for data that shouldn't change.

5. **Pandas Series** is a 1D labeled array—like a list with an index attached.

6. **Pandas DataFrame** is a 2D labeled table—the workhorse of data science.

7. **CSV files** are the universal data exchange format—`pd.read_csv()` is your friend.

8. **Inspection methods** (`head()`, `tail()`, `shape`, `info()`, `describe()`) help you understand your data before analyzing it.

9. **Data selection** with `loc`, `iloc`, and boolean indexing lets you extract exactly what you need.

10. The **load → inspect → select → analyze** workflow is the foundation of every data science project.

You now have the containers for your data superpower. In the next chapter, you'll learn to clean and prepare that data—because real-world data is messy, and heroes clean up messes. Onward!
