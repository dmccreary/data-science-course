# NumPy and Numerical Computing

---
title: NumPy and Numerical Computing
description: Master the engine that powers all of data science
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

## Summary

This chapter introduces NumPy, the fundamental library for numerical computing in Python. Students will learn to create and manipulate NumPy arrays, understand array shapes and indexing, and leverage broadcasting for efficient operations. The chapter covers vectorized operations, matrix mathematics, and linear algebra concepts essential for machine learning. By the end of this chapter, students will understand why NumPy is critical for computational efficiency and be able to perform fast numerical computations.

## Concepts Covered

This chapter covers the following 15 concepts from the learning graph:

1. NumPy Library
2. NumPy Array
3. Array Creation
4. Array Shape
5. Array Indexing
6. Array Slicing
7. Broadcasting
8. Vectorized Operations
9. Element-wise Operations
10. Matrix Operations
11. Dot Product
12. Matrix Multiplication
13. Transpose
14. Linear Algebra
15. Computational Efficiency

## Prerequisites

This chapter builds on concepts from:

- [Chapter 2: Python Environment and Setup](../02-python-environment/index.md)
- [Chapter 3: Python Data Structures](../03-python-data-structures/index.md)

---

## Introduction: The Speed Superpower

Every superhero has an origin story, and NumPy is the origin story of fast data science. Without NumPy, all those fancy machine learning algorithms you've been using would take hours instead of seconds. NumPy is the invisible engine that makes everything else possible.

Here's the deal: regular Python is fantastic for many things, but it's *slow* at math. Like, embarrassingly slow. When you need to multiply a million numbers, Python's built-in lists make you wait... and wait... and wait. NumPy solves this problem with arrays that are up to 100 times faster than regular Python lists.

Why is NumPy so fast? Three secrets:

1. **Contiguous memory**: NumPy stores numbers in a tight, organized row in your computer's memory, so the CPU can grab them quickly
2. **Compiled C code**: The actual calculations happen in lightning-fast C, not interpreted Python
3. **Vectorization**: Instead of looping through items one by one, NumPy processes entire arrays at once

By the end of this chapter, you'll understand how to wield this speed superpower and why every data science library—pandas, scikit-learn, PyTorch—is built on NumPy's foundation.

## The NumPy Library: Your New Best Friend

The **NumPy library** is imported with the conventional alias `np`. This is so universal that if you see `np` in any data science code, you can be 99.9% certain it means NumPy.

```python
import numpy as np

# Check your version
print(np.__version__)
```

NumPy provides:

- Fast array operations for numerical data
- Mathematical functions (sin, cos, exp, log, etc.)
- Linear algebra operations
- Random number generation
- Tools for reading/writing array data

Essentially, NumPy replaces Python's slow list operations with turbocharged alternatives. Once you start using NumPy, you'll wonder how you ever lived without it.

## NumPy Arrays: The Core Data Structure

The **NumPy array** (technically called `ndarray` for "n-dimensional array") is NumPy's main attraction. Unlike Python lists, NumPy arrays:

- Contain only one data type (all integers, all floats, etc.)
- Have a fixed size when created
- Support fast mathematical operations
- Can have multiple dimensions (1D, 2D, 3D, or more)

Here's your first array:

```python
import numpy as np

# Create a simple 1D array
temperatures = np.array([72, 75, 79, 82, 85, 83, 80])
print(temperatures)
print(type(temperatures))  # <class 'numpy.ndarray'>
```

The difference between a NumPy array and a Python list might seem subtle at first, but watch what happens when we do math:

```python
# Python list: can't do this directly!
python_list = [1, 2, 3, 4, 5]
# python_list * 2  # This gives [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] - not what we want!

# NumPy array: math just works!
numpy_array = np.array([1, 2, 3, 4, 5])
print(numpy_array * 2)  # [2, 4, 6, 8, 10] - exactly what we want!
```

This is the magic of NumPy: mathematical operations work *element by element* automatically.

#### Diagram: NumPy Array vs Python List

<details markdown="1">
<summary>NumPy Array vs Python List</summary>
Type: infographic

Bloom Taxonomy: Understand

Learning Objective: Help students visualize the structural differences between Python lists and NumPy arrays, and why those differences matter for performance

Layout: Side-by-side comparison with memory visualization

Left Side - Python List:
- Show a list [1, 2, 3, 4, 5] as scattered boxes in memory
- Each box contains a pointer to the actual number
- Numbers stored in different memory locations
- Label: "Scattered in memory - slow to access"
- Show Python interpreter stepping through one at a time

Right Side - NumPy Array:
- Show array [1, 2, 3, 4, 5] as contiguous boxes
- Numbers stored directly, side by side
- Label: "Contiguous in memory - fast bulk operations"
- Show CPU processing entire block at once

Performance Comparison:
- Speedometer graphics showing relative speeds
- Python list: "1x speed"
- NumPy array: "50-100x speed"

Interactive Elements:
- Slider: Array size (100 to 1,000,000)
- Button: "Run speed test" - shows actual timing comparison
- Animation: Watch memory access patterns for each type
- Toggle: Show/hide memory addresses

Color Scheme:
- Python list elements: Various colors (scattered)
- NumPy array elements: Uniform blue (organized)
- Memory blocks: Gray background

Implementation: p5.js with animated memory visualization
</details>

## Array Creation: Many Ways to Build Arrays

**Array creation** in NumPy offers many convenient methods beyond `np.array()`. Depending on your needs, you can create arrays filled with specific values, sequences, or random numbers.

### From Python Lists

```python
# 1D array from list
scores = np.array([85, 92, 78, 96, 88])

# 2D array from nested lists
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix)
```

### Arrays of Zeros and Ones

```python
# 5 zeros
zeros = np.zeros(5)
print(zeros)  # [0. 0. 0. 0. 0.]

# 3x4 matrix of ones
ones = np.ones((3, 4))
print(ones)

# 3x3 identity matrix (ones on diagonal)
identity = np.eye(3)
print(identity)
```

### Sequences and Ranges

```python
# Like Python's range, but returns an array
sequence = np.arange(0, 10, 2)  # start, stop, step
print(sequence)  # [0 2 4 6 8]

# Evenly spaced numbers between two values
smooth = np.linspace(0, 1, 5)  # start, stop, num_points
print(smooth)  # [0.   0.25 0.5  0.75 1.  ]
```

### Random Arrays

```python
# Random floats between 0 and 1
random_floats = np.random.random(5)
print(random_floats)

# Random integers in a range
random_ints = np.random.randint(1, 100, size=10)  # 10 random ints from 1-99
print(random_ints)

# Random numbers from normal distribution
normal = np.random.normal(loc=0, scale=1, size=1000)  # mean=0, std=1
print(f"Mean: {normal.mean():.3f}, Std: {normal.std():.3f}")
```

| Creation Method | Use Case | Example |
|----------------|----------|---------|
| `np.array()` | Convert existing data | `np.array([1,2,3])` |
| `np.zeros()` | Initialize placeholders | `np.zeros((3,3))` |
| `np.ones()` | Initialize to ones | `np.ones(5)` |
| `np.arange()` | Integer sequences | `np.arange(0,10,2)` |
| `np.linspace()` | Evenly spaced floats | `np.linspace(0,1,100)` |
| `np.random.random()` | Random floats [0,1) | `np.random.random(10)` |
| `np.eye()` | Identity matrix | `np.eye(4)` |

## Array Shape: Understanding Dimensions

The **array shape** tells you the size of each dimension of your array. This is crucial for understanding how your data is organized and for ensuring operations work correctly.

```python
# 1D array: shape is (n,)
vector = np.array([1, 2, 3, 4, 5])
print(vector.shape)  # (5,)

# 2D array: shape is (rows, columns)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3) - 2 rows, 3 columns

# 3D array: shape is (depth, rows, columns)
cube = np.zeros((2, 3, 4))
print(cube.shape)  # (2, 3, 4)
```

Think of shape as describing the "dimensions" of your data:

- **1D**: A line of numbers (vector)
- **2D**: A table of numbers (matrix)
- **3D**: A stack of tables (tensor)
- **nD**: Higher dimensions follow the same pattern

You can reshape arrays to change their dimensions (as long as the total number of elements stays the same):

```python
# Create 12 numbers
numbers = np.arange(12)
print(numbers)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(numbers.shape)  # (12,)

# Reshape to 3x4 matrix
matrix = numbers.reshape(3, 4)
print(matrix)
print(matrix.shape)  # (3, 4)

# Reshape to 2x2x3 cube
cube = numbers.reshape(2, 2, 3)
print(cube.shape)  # (2, 2, 3)

# Use -1 to let NumPy calculate one dimension
auto_reshaped = numbers.reshape(4, -1)  # 4 rows, auto-calculate columns
print(auto_reshaped.shape)  # (4, 3)
```

!!! tip "The -1 Trick"
    When reshaping, you can use -1 for one dimension, and NumPy will automatically calculate it. This is super handy when you know one dimension but not the other: `array.reshape(-1, 3)` gives you 3 columns with however many rows are needed.

#### Diagram: Array Shape Visualizer

<details markdown="1">
<summary>Array Shape Visualizer</summary>
Type: microsim

Bloom Taxonomy: Understand, Apply

Learning Objective: Help students visualize how array shapes correspond to physical dimensions and how reshaping reorganizes data

Canvas Layout (800x500):
- Left panel (400x500): 3D visualization of array
- Right panel (400x500): Shape controls and data view

Left Panel - Visual Representation:
- 1D: Horizontal row of numbered boxes
- 2D: Grid of numbered boxes (rows × columns)
- 3D: Stack of 2D grids (depth × rows × columns)
- Boxes contain actual values, colored by magnitude
- Axes labeled with dimension sizes

Right Panel - Controls:
- Input fields for each dimension size
- Dropdown: Quick presets (vector, matrix, cube)
- Current shape display: (d1, d2, d3)
- Total elements counter
- Flattened view showing element order

Interactive Elements:
- Click and drag to rotate 3D view
- Slider for each dimension (1-10)
- Button: "Reshape" - animates transition between shapes
- Button: "Flatten" - shows elements laid out in 1D
- Toggle: "Show indices" - displays [i,j,k] for each cell
- Highlight: Click an element to see its index in all views

Reshape Animation:
- Elements smoothly transition from old shape to new
- Color trails show where each element moves
- Error message if total elements don't match

Educational Callouts:
- "Total elements must stay constant when reshaping"
- Show calculation: d1 × d2 × d3 = total

Implementation: p5.js with 3D rendering (WEBGL mode)
</details>

## Array Indexing: Accessing Your Data

**Array indexing** lets you access individual elements or groups of elements. NumPy indexing is similar to Python list indexing but more powerful.

### 1D Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single element (0-indexed)
print(arr[0])   # 10 (first element)
print(arr[2])   # 30 (third element)
print(arr[-1])  # 50 (last element)

# Multiple elements with a list of indices
print(arr[[0, 2, 4]])  # [10 30 50]
```

### 2D Indexing

For 2D arrays, you provide two indices: `[row, column]`.

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Single element
print(matrix[0, 0])  # 1 (top-left)
print(matrix[1, 2])  # 6 (row 1, column 2)
print(matrix[-1, -1])  # 9 (bottom-right)

# Entire row
print(matrix[1])  # [4 5 6]

# Entire column
print(matrix[:, 1])  # [2 5 8]
```

### Boolean Indexing (Filtering)

One of NumPy's most powerful features is boolean indexing—using conditions to select elements:

```python
scores = np.array([85, 92, 78, 96, 88, 73, 95])

# Create boolean mask
passing = scores >= 80
print(passing)  # [ True  True False  True  True False  True]

# Use mask to filter
print(scores[passing])  # [85 92 96 88 95]

# Or do it in one step
print(scores[scores >= 90])  # [92 96 95]

# Combine conditions with & (and) or | (or)
print(scores[(scores >= 80) & (scores < 95)])  # [85 92 88]
```

## Array Slicing: Grabbing Sections

**Array slicing** extracts portions of arrays using the familiar `start:stop:step` notation. With multi-dimensional arrays, you can slice along each dimension.

```python
arr = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]

# Basic slicing
print(arr[2:7])    # [2 3 4 5 6]
print(arr[:5])     # [0 1 2 3 4]
print(arr[5:])     # [5 6 7 8 9]
print(arr[::2])    # [0 2 4 6 8] (every other element)
print(arr[::-1])   # [9 8 7 6 5 4 3 2 1 0] (reversed)
```

### 2D Slicing

```python
matrix = np.arange(20).reshape(4, 5)
print(matrix)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]]

# First two rows, all columns
print(matrix[:2, :])

# All rows, first three columns
print(matrix[:, :3])

# Submatrix: rows 1-2, columns 2-4
print(matrix[1:3, 2:5])
# [[ 7  8  9]
#  [12 13 14]]

# Every other row, every other column
print(matrix[::2, ::2])
# [[ 0  2  4]
#  [10 12 14]]
```

!!! warning "Views vs Copies"
    Slicing in NumPy creates a *view* of the original array, not a copy. Changes to the slice affect the original! Use `.copy()` if you need an independent copy: `my_copy = arr[2:5].copy()`.

#### Diagram: Slicing Playground

<details markdown="1">
<summary>Slicing Playground</summary>
Type: microsim

Bloom Taxonomy: Apply, Analyze

Learning Objective: Practice array slicing with immediate visual feedback, understanding how start:stop:step notation works in multiple dimensions

Canvas Layout (850x500):
- Left panel (500x500): Visual array representation
- Right panel (350x500): Slicing controls and code

Left Panel - Array View:
- 2D grid showing array values (default 6x8)
- Selected elements highlighted in blue
- Unselected elements in gray
- Row and column indices labeled
- Animation when selection changes

Right Panel - Slicing Controls:
- Row slice inputs: start [ ] : stop [ ] : step [ ]
- Column slice inputs: start [ ] : stop [ ] : step [ ]
- Live code preview: `array[0:3, 1:5:2]`
- Result preview showing selected values
- Preset buttons: "First 3 rows", "Last column", "Checkerboard", "Reverse"

Interactive Features:
- Click and drag on grid to visually select region
- Inputs update automatically from visual selection
- Code updates in real-time as inputs change
- "Run" button executes in console and shows result
- Error messages for invalid slices

Quick Challenges:
- "Select the corners" - shows 4 corner elements
- "Select every other element" - checkerboard pattern
- "Reverse the rows" - shows negative step
- Button: "Check Answer" for each challenge

Visual Feedback:
- Green flash when slice is valid
- Red outline for invalid slice notation
- Animation showing element selection order

Implementation: p5.js with interactive grid
</details>

## Vectorized Operations: The Speed Secret

**Vectorized operations** are operations that apply to entire arrays at once, without explicit Python loops. This is NumPy's superpower—it's what makes NumPy fast.

Compare these two approaches to squaring a million numbers:

```python
import time

# Create a million random numbers
data = np.random.random(1_000_000)

# Slow way: Python loop
start = time.time()
result_slow = [x**2 for x in data]
slow_time = time.time() - start
print(f"Python loop: {slow_time:.4f} seconds")

# Fast way: NumPy vectorized
start = time.time()
result_fast = data ** 2
fast_time = time.time() - start
print(f"NumPy vectorized: {fast_time:.4f} seconds")

print(f"NumPy is {slow_time/fast_time:.0f}x faster!")
```

Typical output: NumPy is **50-100x faster** than the Python loop. That's not a typo—NumPy really is that much faster.

### Common Vectorized Operations

```python
arr = np.array([1, 4, 9, 16, 25])

# Arithmetic
print(arr + 10)     # [11 14 19 26 35]
print(arr * 2)      # [ 2  8 18 32 50]
print(arr / 5)      # [0.2 0.8 1.8 3.2 5.0]
print(arr ** 0.5)   # [1. 2. 3. 4. 5.] (square roots)

# Math functions
print(np.sqrt(arr))     # Square root
print(np.log(arr))      # Natural log
print(np.exp(arr))      # e^x
print(np.sin(arr))      # Sine

# Aggregations
print(np.sum(arr))      # 55
print(np.mean(arr))     # 11.0
print(np.std(arr))      # 8.0
print(np.min(arr))      # 1
print(np.max(arr))      # 25
```

The key insight: whenever you're tempted to write a `for` loop over array elements, ask yourself "Is there a NumPy function that does this?" There usually is, and it's almost always faster.

## Element-wise Operations: Array Math

**Element-wise operations** apply the same operation to each corresponding pair of elements in two arrays. The arrays must have compatible shapes (more on this in the Broadcasting section).

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Element-wise arithmetic
print(a + b)   # [11 22 33 44]
print(a - b)   # [-9 -18 -27 -36]
print(a * b)   # [10 40 90 160]
print(a / b)   # [0.1 0.1 0.1 0.1]
print(a ** b)  # Very big numbers!

# Element-wise comparisons
print(a < b)   # [ True  True  True  True]
print(a == 2)  # [False  True False False]
```

This is different from matrix multiplication (which we'll cover soon). Element-wise `*` multiplies position by position; matrix multiplication follows linear algebra rules.

```python
# 2D element-wise operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 20], [30, 40]])

print(A * B)
# [[ 10  40]
#  [ 90 160]]

# Each element multiplied by its counterpart
```

## Broadcasting: The Shape-Matching Magic

**Broadcasting** is NumPy's clever way of handling operations between arrays of different shapes. Instead of requiring identical shapes, NumPy "broadcasts" smaller arrays to match larger ones.

### Simple Broadcasting

```python
# Add a scalar to every element
arr = np.array([1, 2, 3, 4, 5])
print(arr + 10)  # [11 12 13 14 15]

# The scalar 10 is "broadcast" to [10, 10, 10, 10, 10]
```

### Broadcasting with Different Shapes

```python
# 3x3 matrix
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 1D array with 3 elements
row = np.array([10, 20, 30])

# Row is broadcast to each row of matrix
print(matrix + row)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]

# Column vector (shape 3,1)
col = np.array([[100], [200], [300]])

# Column is broadcast to each column
print(matrix + col)
# [[101 102 103]
#  [204 205 206]
#  [307 308 309]]
```

### Broadcasting Rules

Broadcasting works when comparing shapes from right to left:

1. Dimensions are compatible if they're equal OR one of them is 1
2. Missing dimensions are treated as 1

| Shape A | Shape B | Result Shape | Works? |
|---------|---------|--------------|--------|
| (3,) | (3,) | (3,) | Yes - identical |
| (3, 4) | (4,) | (3, 4) | Yes - 4 matches 4 |
| (3, 4) | (3, 1) | (3, 4) | Yes - 1 broadcasts |
| (3, 4) | (2, 4) | Error | No - 3 ≠ 2 |

#### Diagram: Broadcasting Visualizer

<details markdown="1">
<summary>Broadcasting Visualizer</summary>
Type: microsim

Bloom Taxonomy: Understand, Apply

Learning Objective: Visualize how NumPy stretches smaller arrays to match larger ones during broadcasting

Canvas Layout (850x550):
- Top area (850x350): Visual array representations
- Bottom area (850x200): Shape analysis and controls

Top Area - Visual Representation:
- Left: First array (A) with shape label
- Center: Operation symbol (+, *, etc.)
- Right: Second array (B) with shape label
- Below: Result array showing combined operation
- Animation: Smaller array "stretches" to match larger

Broadcasting Animation:
- Show original arrays
- Animate smaller array duplicating to match dimensions
- Show element-wise operation occurring
- Display final result with highlighting

Interactive Controls:
- Dropdown: Select Array A shape (scalar, 1D, 2D options)
- Dropdown: Select Array B shape
- Dropdown: Select operation (+, -, *, /)
- Input: Custom values for arrays
- Button: "Animate Broadcasting"

Shape Analysis Panel:
- Show shapes aligned right-to-left
- Color code: Green = compatible, Red = incompatible
- Explain which dimension broadcasts to which
- Error message for incompatible shapes

Preset Examples:
- "Scalar + Matrix" - simplest broadcast
- "Row + Matrix" - row broadcasts down
- "Column + Matrix" - column broadcasts across
- "Incompatible" - shows error case

Implementation: p5.js with step-by-step animation
</details>

## Matrix Operations: Linear Algebra Essentials

Now we enter the realm of **matrix operations**—the mathematical operations that power machine learning. These are different from element-wise operations and follow the rules of linear algebra.

### The Transpose

The **transpose** of a matrix flips it over its diagonal—rows become columns and columns become rows.

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("Original shape:", A.shape)  # (2, 3)

A_T = A.T  # Transpose
print(A_T)
print("Transposed shape:", A_T.shape)  # (3, 2)

# Original:
# [[1 2 3]
#  [4 5 6]]

# Transposed:
# [[1 4]
#  [2 5]
#  [3 6]]
```

Transpose is used constantly in machine learning, especially when you need to align matrix dimensions for multiplication.

### The Dot Product

The **dot product** of two vectors produces a single number (scalar). It multiplies corresponding elements and sums the results.

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + ... + a_nb_n$$

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot = np.dot(a, b)
print(dot)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

# Alternative syntax
print(a @ b)  # 32 (@ is the matrix multiplication operator)
```

The dot product measures how "aligned" two vectors are. In machine learning, it's used for:

- Computing predictions (features · weights)
- Measuring similarity between vectors
- Computing attention in transformers

### Matrix Multiplication

**Matrix multiplication** extends the dot product to entire matrices. For matrices A (m×n) and B (n×p), the result C is (m×p), where each element is a dot product of a row from A and a column from B.

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])  # 3x2 matrix

B = np.array([
    [7, 8, 9],
    [10, 11, 12]
])  # 2x3 matrix

# Matrix multiplication
C = A @ B  # or np.dot(A, B)
print(C)
print(C.shape)  # (3, 3)

# [[1*7+2*10, 1*8+2*11, 1*9+2*12],    [[27, 30, 33],
#  [3*7+4*10, 3*8+4*11, 3*9+4*12],  =  [61, 68, 75],
#  [5*7+6*10, 5*8+6*11, 5*9+6*12]]     [95, 106, 117]]
```

!!! warning "Shape Requirements for Matrix Multiplication"
    For A @ B to work, the number of columns in A must equal the number of rows in B. Shape (m, **n**) @ (**n**, p) = (m, p). If shapes don't match, you'll get an error.

#### Diagram: Matrix Multiplication Visualizer

<details markdown="1">
<summary>Matrix Multiplication Visualizer</summary>
Type: microsim

Bloom Taxonomy: Understand, Apply

Learning Objective: Visualize how matrix multiplication works by showing the dot products between rows and columns

Canvas Layout (900x550):
- Left area (350x400): Matrix A with row highlighting
- Center area (200x400): Matrix B with column highlighting
- Right area (350x400): Result matrix C with cell highlighting
- Bottom area (900x150): Calculation display

Visual Elements:
- Matrix A displayed as grid (rows emphasized)
- Matrix B displayed as grid (columns emphasized)
- Result C displayed as grid
- Currently computed cell highlighted
- Arrows showing which row and column are being multiplied

Animation Sequence:
- Highlight row i of A in blue
- Highlight column j of B in green
- Show element-wise multiplication along the way
- Sum appears in C[i,j] with flash
- Move to next cell

Interactive Controls:
- Slider: Matrix A rows (1-5)
- Slider: Matrix A columns / B rows (1-5)
- Slider: Matrix B columns (1-5)
- Speed control for animation
- Button: "Step through" - advance one calculation
- Button: "Play all" - animate entire multiplication
- Button: "Reset"

Calculation Panel:
- Shows current calculation: a[i] · b[j] = sum
- Running formula with actual numbers
- Highlight matching elements being multiplied

Shape Validation:
- Green indicator when shapes are compatible
- Red error when inner dimensions don't match
- Show shape calculation: (m,n) @ (n,p) = (m,p)

Implementation: p5.js with step-by-step animation
</details>

## Linear Algebra with NumPy

**Linear algebra** is the mathematical framework for machine learning. NumPy provides essential linear algebra operations through `np.linalg`.

```python
from numpy import linalg as la

# Create a square matrix
A = np.array([
    [4, 2],
    [1, 3]
])

# Determinant
det = la.det(A)
print(f"Determinant: {det}")  # 10.0

# Inverse (A^-1 such that A @ A^-1 = I)
A_inv = la.inv(A)
print("Inverse:")
print(A_inv)

# Verify: A @ A^-1 should equal identity
print("A @ A_inv:")
print(A @ A_inv)  # [[1, 0], [0, 1]] (approximately)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = la.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Solve linear system Ax = b
b = np.array([8, 5])
x = la.solve(A, b)
print(f"Solution to Ax = b: {x}")
print(f"Verification A @ x: {A @ x}")  # Should equal b
```

These operations are foundational for:

- **Inverse**: Solving equations, understanding transformations
- **Determinant**: Checking if matrix is invertible, computing volumes
- **Eigenvalues**: Principal Component Analysis (PCA), understanding matrices
- **Solve**: Linear regression (normal equations), optimization

| Operation | Function | Use Case |
|-----------|----------|----------|
| Inverse | `la.inv(A)` | Solving equations |
| Determinant | `la.det(A)` | Check invertibility |
| Eigendecomposition | `la.eig(A)` | PCA, spectral analysis |
| Solve Ax=b | `la.solve(A, b)` | Linear systems |
| Matrix rank | `la.matrix_rank(A)` | Dimensionality |
| Norm | `la.norm(v)` | Vector/matrix magnitude |

## Computational Efficiency: Why This All Matters

Let's bring it all together and talk about **computational efficiency**—why NumPy's speed matters for real data science work.

### The Numbers Don't Lie

```python
import numpy as np
import time

# Compare approaches for a common operation:
# Normalize a dataset (subtract mean, divide by std)

# Create a million data points
data = np.random.randn(1_000_000)

# Method 1: Pure Python
def normalize_python(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean)**2 for x in data) / len(data)
    std = variance ** 0.5
    return [(x - mean) / std for x in data]

# Method 2: NumPy
def normalize_numpy(data):
    return (data - data.mean()) / data.std()

# Time Python version
start = time.time()
result_python = normalize_python(list(data))
python_time = time.time() - start

# Time NumPy version
start = time.time()
result_numpy = normalize_numpy(data)
numpy_time = time.time() - start

print(f"Python: {python_time:.3f} seconds")
print(f"NumPy: {numpy_time:.4f} seconds")
print(f"Speedup: {python_time/numpy_time:.0f}x")
```

Typical result: NumPy is **100-500x faster** for this operation!

### Memory Efficiency

NumPy arrays also use less memory than Python lists:

```python
import sys

# Python list of 1 million integers
python_list = list(range(1_000_000))
python_size = sys.getsizeof(python_list) + sum(sys.getsizeof(x) for x in python_list)

# NumPy array of 1 million integers
numpy_array = np.arange(1_000_000)
numpy_size = numpy_array.nbytes

print(f"Python list: {python_size / 1e6:.1f} MB")
print(f"NumPy array: {numpy_size / 1e6:.1f} MB")
print(f"Memory savings: {python_size/numpy_size:.1f}x")
```

NumPy typically uses **4-8x less memory** than equivalent Python lists.

### Why Scikit-learn and Pandas Use NumPy

Every major data science library is built on NumPy:

- **Pandas**: DataFrames store data as NumPy arrays internally
- **Scikit-learn**: All models expect NumPy arrays as input
- **PyTorch/TensorFlow**: Tensors are compatible with NumPy arrays
- **Matplotlib/Plotly**: Plotting functions accept NumPy arrays

When you call `df.values` on a pandas DataFrame, you get a NumPy array. When you call `model.fit(X, y)` in scikit-learn, X and y are NumPy arrays. Understanding NumPy means understanding the foundation of modern data science.

#### Diagram: NumPy Ecosystem Map

<details markdown="1">
<summary>NumPy Ecosystem Map</summary>
Type: infographic

Bloom Taxonomy: Understand

Learning Objective: Show how NumPy serves as the foundation for the entire Python data science ecosystem

Layout: Hub-and-spoke diagram with NumPy at center

Center Hub - NumPy:
- Large central circle labeled "NumPy"
- Subtitle: "The Foundation"
- Icon: Array grid symbol

Spokes - Major Libraries:
1. Pandas spoke:
   - "DataFrames use NumPy arrays internally"
   - Icon: Table
   - Arrow showing data flow to/from NumPy

2. Scikit-learn spoke:
   - "All ML models expect NumPy arrays"
   - Icon: Brain/ML
   - Arrow showing fit/predict using arrays

3. Matplotlib/Plotly spoke:
   - "Plotting functions accept arrays"
   - Icon: Chart
   - Arrow showing visualization of arrays

4. SciPy spoke:
   - "Scientific computing extends NumPy"
   - Icon: Integration symbol
   - Arrow showing enhanced operations

5. PyTorch/TensorFlow spoke:
   - "Deep learning tensors interoperate with NumPy"
   - Icon: Neural network
   - Arrow showing array↔tensor conversion

Interactive Elements:
- Hover over each spoke to see code example
- Click to see conversion syntax (e.g., `df.values`, `torch.from_numpy()`)
- Animation: Data flowing from NumPy to each library
- Toggle: Show memory sharing between libraries

Visual Style:
- NumPy in blue (foundation color)
- Each library in its brand color
- Arrows showing bidirectional data flow
- Sizes proportional to library importance

Implementation: HTML/CSS/JavaScript with hover interactions
</details>

## Practical Tips and Best Practices

As you incorporate NumPy into your data science workflow, keep these tips in mind:

**Think in Arrays, Not Loops**
Whenever you write a `for` loop over array elements, stop and ask: "Is there a vectorized way to do this?" There usually is.

```python
# Bad: Loop
result = []
for x in data:
    result.append(x ** 2 + 2 * x + 1)

# Good: Vectorized
result = data ** 2 + 2 * data + 1
```

**Use Broadcasting Intentionally**
Broadcasting is powerful but can be confusing. When shapes don't match as expected, print them:

```python
print(f"A shape: {A.shape}, B shape: {B.shape}")
```

**Be Careful with Views**
Remember that slices create views, not copies. If you need an independent array, use `.copy()`:

```python
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]      # Changes affect original
copy = original[1:4].copy()  # Independent
```

**Check Data Types**
NumPy infers data types, but sometimes you need to be explicit:

```python
# Float array (default for decimals)
floats = np.array([1.0, 2.0, 3.0])
print(floats.dtype)  # float64

# Integer array
ints = np.array([1, 2, 3])
print(ints.dtype)  # int64

# Force a specific type
forced = np.array([1, 2, 3], dtype=np.float32)
print(forced.dtype)  # float32
```

## Summary: Your NumPy Toolkit

You now have a solid foundation in NumPy:

- **NumPy arrays** are faster and more memory-efficient than Python lists
- **Array creation** offers many methods: `zeros`, `ones`, `arange`, `linspace`, `random`
- **Shape** describes array dimensions; `reshape` reorganizes without copying
- **Indexing and slicing** access elements and subarrays powerfully
- **Vectorized operations** apply functions to entire arrays at once
- **Broadcasting** handles operations between different-shaped arrays
- **Matrix operations** (transpose, dot product, matrix multiplication) enable linear algebra
- **Computational efficiency** makes NumPy essential for real-world data science

NumPy is the bedrock of scientific Python. Every time you use pandas, scikit-learn, or PyTorch, NumPy is working behind the scenes. Master NumPy, and you've mastered the foundation.

## Looking Ahead

In the next chapter, we'll explore non-linear models and regularization techniques. You'll see how polynomial features (built with NumPy!) can capture curved relationships, and how regularization prevents overfitting. The matrix operations you learned here will help you understand what's happening inside these more advanced models.

---

## Key Takeaways

- NumPy arrays are 50-100x faster than Python lists for numerical operations
- Arrays have shapes that describe their dimensions; reshape changes organization without copying data
- Indexing with brackets accesses elements; boolean indexing filters based on conditions
- Vectorized operations avoid loops and leverage compiled C code for speed
- Broadcasting stretches smaller arrays to match larger ones automatically
- Matrix multiplication (@) follows linear algebra rules; element-wise (*) operates position by position
- The `np.linalg` module provides essential linear algebra operations
- Every major data science library is built on NumPy—it's the universal foundation
