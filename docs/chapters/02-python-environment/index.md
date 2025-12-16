---
title: Python Environment and Setup
description: Building your data science headquarters - every superhero needs a base of operations
generated_by: chapter-content-generator skill
date: 2025-12-15
version: 0.03
---

# Python Environment and Setup

## Summary

This chapter guides students through setting up a complete Python development environment for data science. Students will install Python, learn about package management with pip and conda, set up virtual environments, and configure their IDE. The chapter also covers Jupyter notebooks in depth, including working with cells, kernels, and importing libraries. By the end of this chapter, students will have a fully functional data science development environment.

## Concepts Covered

This chapter covers the following 15 concepts from the learning graph:

1. Python Installation
2. Package Management
3. Pip
4. Conda Environment
5. Virtual Environment
6. IDE Setup
7. VS Code
8. Jupyter Notebooks
9. Notebook Cells
10. Code Cell
11. Markdown Cell
12. Cell Execution
13. Kernel
14. Import Statement
15. Python Libraries

## Prerequisites

This chapter builds on concepts from:

- [Chapter 1: Introduction to Data Science](../01-intro-to-data-science/index.md)

---

## Building Your Data Science Headquarters

Every superhero needs a headquarters. Batman has the Batcave. Iron Man has Stark Tower. The Avengers have their compound. You? You're about to build something just as cool—your personal data science command center.

In Chapter 1, you discovered your new superpower: the ability to extract insights from data. But a superpower without the right tools is like having super strength but nowhere to punch. This chapter is where you assemble your utility belt, stock your armory, and set up the base of operations where all your data science magic will happen.

By the end of this chapter, you'll have:

- Python installed and ready to go
- A professional code editor configured for data science
- Jupyter notebooks for interactive exploration
- Package management skills to add new tools whenever you need them
- A clean, organized environment that won't cause headaches later

Let's build your headquarters!

#### Diagram: Data Science Environment Architecture

<details markdown="1">
    <summary>Data Science Environment Architecture</summary>
    Type: diagram

    Bloom Taxonomy: Understand (L2)

    Learning Objective: Help students visualize how all components of their data science environment fit together

    Purpose: Show the layered architecture of a data science setup

    Layout: Vertical stack diagram showing layers from bottom to top

    Layers (bottom to top):
    1. OPERATING SYSTEM (base layer)
       - Windows, macOS, or Linux
       - Color: Dark gray
       - Icon: Computer

    2. PYTHON INSTALLATION
       - Python interpreter (the engine)
       - Color: Blue (Python blue)
       - Icon: Python logo

    3. PACKAGE MANAGER
       - pip or conda (the supply chain)
       - Arrows showing packages flowing in
       - Color: Orange
       - Icon: Package box

    4. VIRTUAL ENVIRONMENT
       - Isolated workspace (the clean room)
       - Shows boundary separating from other environments
       - Color: Green
       - Icon: Bubble/container

    5. PYTHON LIBRARIES
       - pandas, numpy, matplotlib (the tools)
       - Multiple small icons representing different libraries
       - Color: Various colors for each library

    6. IDE / JUPYTER (top layer)
       - VS Code or Jupyter Notebook (the cockpit)
       - Color: Purple
       - Icon: Code editor window

    Side annotations:
    - Arrow from user to IDE: "You work here"
    - Arrow from libraries to IDE: "Tools you use"
    - Bracket around virtual environment: "Keeps projects separate"

    Interactive elements:
    - Hover over each layer to see description and purpose
    - Click to see common problems at each layer

    Visual style: Modern, clean boxes with rounded corners, superhero HQ aesthetic

    Implementation: SVG with CSS hover effects
</details>

## Python Installation: Powering Up Your System

**Python installation** is the first and most critical step. Python is the programming language that powers your data science work—it's the engine of your entire operation. Without it, nothing else works.

Think of Python like electricity in your headquarters. You can have the fanciest equipment in the world, but without power, it's all just expensive furniture. Installing Python gives your computer the ability to understand and execute Python code.

### Choosing Your Python Distribution

Here's where it gets interesting: there are actually several ways to install Python. The two main approaches are:

| Approach | Best For | Includes |
|----------|----------|----------|
| Python.org (Official) | Minimalists, learning basics | Just Python, nothing extra |
| Anaconda Distribution | Data scientists (that's you!) | Python + 250+ data science packages |

For this course, we **strongly recommend Anaconda**. Why? Because it comes pre-loaded with almost everything you'll need—pandas, NumPy, matplotlib, Jupyter, and hundreds of other tools. It's like buying a fully furnished headquarters instead of an empty warehouse.

!!! tip "Superhero Shortcut"
    Installing Anaconda is like getting a starter kit with all the gadgets already assembled. You could build everything from scratch, but why? Batman didn't forge his own Batarangs (okay, maybe he did, but you get the point).

### Installation Steps

The installation process is straightforward:

1. Visit [anaconda.com](https://www.anaconda.com/download) and download the installer for your operating system
2. Run the installer and accept the default options
3. Wait for installation to complete (it might take a few minutes—grab a snack)
4. Open a terminal or Anaconda Prompt to verify it worked

To verify your installation, open a terminal and type:

```bash
python --version
```

You should see something like `Python 3.11.5` (the exact version may differ). If you see an error, don't panic—check the troubleshooting section at the end of this chapter.

## Package Management: Your Supply Chain

**Package management** is how you add new tools and capabilities to your Python installation. In the superhero world, this is like having access to a warehouse full of gadgets you can requisition whenever you need them.

Python's real power comes from its ecosystem of **Python libraries**—pre-written code packages that handle specific tasks. Need to work with data tables? There's a library for that (pandas). Need to create visualizations? There's a library for that (matplotlib). Need to do machine learning? Yep, library for that too (scikit-learn).

But how do you get these libraries? That's where package managers come in.

### Pip: The Original Package Manager

**Pip** stands for "Pip Installs Packages" (yes, it's a recursive acronym—programmers think they're funny). It's the original Python package manager and comes built into Python.

Using pip is simple. Open a terminal and type:

```bash
pip install pandas
```

That's it! Pip will download pandas and all its dependencies, install them, and you're ready to go. Need to install multiple packages? Just list them:

```bash
pip install pandas numpy matplotlib seaborn
```

Common pip commands you'll use:

| Command | What It Does |
|---------|--------------|
| `pip install package_name` | Install a package |
| `pip uninstall package_name` | Remove a package |
| `pip list` | Show all installed packages |
| `pip show package_name` | Show details about a package |
| `pip install --upgrade package_name` | Update to latest version |

### Conda: The Data Scientist's Choice

While pip is great, data scientists often prefer **conda**—the package manager that comes with Anaconda. Conda does everything pip does, plus it manages non-Python dependencies and creates isolated environments (more on that soon).

```bash
conda install pandas
```

The syntax is almost identical to pip. So why use conda? Because some data science packages have complicated dependencies involving C libraries, Fortran code, or other system-level components. Conda handles all of that automatically, while pip sometimes struggles.

!!! warning "Pick One (Mostly)"
    Using both pip and conda in the same environment can sometimes cause conflicts—like having two quarterbacks calling different plays. In general, prefer conda for data science packages. Use pip only when a package isn't available through conda.

#### Diagram: Package Manager Workflow

<details markdown="1">
    <summary>How Package Managers Work</summary>
    Type: workflow

    Bloom Taxonomy: Understand (L2)

    Learning Objective: Help students understand the flow of installing and using packages

    Purpose: Visualize the package installation process from command to usage

    Visual style: Horizontal flowchart with icons

    Steps:

    1. USER TYPES COMMAND
       Icon: Keyboard
       Example: "pip install pandas"
       Color: Blue

    2. PACKAGE MANAGER SEARCHES
       Icon: Magnifying glass
       Label: "Searches PyPI (pip) or Anaconda Cloud (conda)"
       Color: Orange

    3. DOWNLOADS PACKAGE
       Icon: Download arrow
       Label: "Downloads package + all dependencies"
       Color: Green

    4. INSTALLS TO ENVIRONMENT
       Icon: Folder with checkmark
       Label: "Saves files to your Python environment"
       Color: Purple

    5. READY TO IMPORT
       Icon: Python logo with sparkles
       Label: "import pandas as pd"
       Color: Gold

    Annotations:
    - Between steps 2-3: "Internet connection required"
    - Below step 3: "May download multiple packages (dependencies)"

    Interactive elements:
    - Hover each step to see common errors and solutions
    - Animation: Package icon traveling through pipeline

    Implementation: SVG with CSS animations
</details>

## Virtual Environments: Your Clean Room

Here's a scenario that trips up many beginners: You're working on two different projects. Project A needs pandas version 1.5, but Project B needs pandas version 2.0. If you install version 2.0, Project A breaks. Install version 1.5, and Project B breaks. What do you do?

The answer: **virtual environments**.

A **virtual environment** is an isolated Python installation where you can install packages without affecting your main system or other projects. Think of it as having multiple separate headquarters, each set up exactly how one specific mission requires.

### Why Virtual Environments Matter

Without virtual environments:

- Package conflicts are inevitable
- Upgrading one project can break another
- Your system Python gets cluttered with random packages
- Reproducing your work on another computer becomes a nightmare

With virtual environments:

- Each project has exactly the packages it needs
- No conflicts between projects
- Easy to share your exact setup with others
- Clean, organized, professional workflow

### Conda Environments: Your Mission-Specific Loadout

Since you're using Anaconda, you'll create environments using conda. Here's how:

```bash
# Create a new environment named "datascience" with Python 3.11
conda create --name datascience python=3.11

# Activate the environment (start using it)
conda activate datascience

# Install packages into this environment
conda install pandas numpy matplotlib jupyter

# When you're done, deactivate
conda deactivate
```

Once activated, anything you install goes into that environment only. Your other environments stay clean and untouched.

!!! example "Real-World Analogy"
    Imagine you're a chef who cooks Italian food, Japanese food, and Mexican food. You COULD keep all your ingredients in one giant pantry. But it's much easier to have three separate stations, each stocked with exactly what that cuisine needs. Virtual environments are your separate cooking stations.

#### Diagram: Virtual Environment Isolation MicroSim

<details markdown="1">
    <summary>Virtual Environment Isolation Simulator</summary>
    Type: microsim

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Let students experiment with creating environments and installing packages to see how isolation works

    Canvas layout (750x500px):
    - Left side (500x500): Visual representation of environments
    - Right side (250x500): Controls and terminal simulation

    Visual elements:
    - Base system shown as gray platform at bottom
    - Virtual environments as colored bubbles floating above
    - Packages shown as small icons inside environments
    - Conflict indicators (red X) when same package different versions

    Interactive controls:
    - Button: "Create Environment" - Adds new bubble with name input
    - Dropdown: "Select Environment" - Choose which env to work in
    - Button: "Install Package" - Shows package picker
    - Package picker: pandas, numpy, matplotlib with version selector
    - Button: "Delete Environment" - Removes selected environment
    - Toggle: "Show Conflicts" - Highlights version conflicts

    Terminal simulation (right panel):
    - Shows conda commands being "typed"
    - Displays output messages
    - Command history

    Default state:
    - Base system with Python
    - One environment "project-a" with pandas 1.5
    - One environment "project-b" with pandas 2.0
    - No conflicts (isolated!)

    Behavior:
    - Creating environment adds new bubble
    - Installing package adds icon to current environment bubble
    - Installing conflicting versions in same env shows warning
    - Different versions in different envs shows green checkmarks
    - Hover over package shows version and description

    Educational messages:
    - "Notice: Each environment is completely separate!"
    - "Try installing different pandas versions in different environments"
    - "See? No conflicts when properly isolated!"

    Implementation: p5.js with interactive elements
</details>

## IDE Setup: Your Command Center

An **IDE** (Integrated Development Environment) is your primary workspace—the command center where you'll write code, run experiments, and analyze results. While you could technically write Python in Notepad, that's like trying to fight crime with a flashlight instead of the Batcomputer.

A good IDE provides:

- Syntax highlighting (code is color-coded for readability)
- Auto-completion (suggests code as you type)
- Error detection (catches mistakes before you run)
- Integrated terminal (run commands without switching windows)
- Debugging tools (find and fix problems)
- Extension ecosystem (add new features)

### VS Code: The Modern Hero's Choice

**VS Code** (Visual Studio Code) is our recommended IDE. It's free, fast, incredibly powerful, and loved by millions of developers worldwide. Microsoft makes it, but don't hold that against it—it's genuinely excellent.

Why VS Code for data science?

| Feature | Benefit |
|---------|---------|
| Python Extension | First-class Python support with IntelliSense |
| Jupyter Integration | Run notebooks directly in VS Code |
| Git Integration | Version control built right in |
| Extensions Marketplace | Thousands of add-ons available |
| Remote Development | Code on servers, containers, WSL |
| Free Forever | No subscriptions, no premium tiers |

### Setting Up VS Code for Data Science

After installing VS Code, you'll want to add some extensions. Think of extensions as upgrades to your equipment—each one adds new capabilities.

Essential extensions for data science:

1. **Python** (by Microsoft) - Core Python support
2. **Jupyter** (by Microsoft) - Notebook support in VS Code
3. **Pylance** - Advanced Python language features
4. **Python Indent** - Fixes indentation automatically
5. **Rainbow CSV** - Makes CSV files readable

To install an extension:

1. Click the Extensions icon in the left sidebar (looks like four squares)
2. Search for the extension name
3. Click "Install"
4. That's it—no restart required!

!!! tip "Keyboard Shortcut Superpower"
    Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac) to open the Command Palette—your gateway to every VS Code feature. Start typing what you want to do, and VS Code will find it.

#### Diagram: VS Code Interface Tour

<details markdown="1">
    <summary>VS Code Interface Guided Tour</summary>
    Type: infographic

    Bloom Taxonomy: Remember (L1)

    Learning Objective: Help students identify and remember the key parts of the VS Code interface

    Purpose: Interactive labeled diagram of VS Code interface

    Layout: Screenshot-style representation of VS Code with numbered callouts

    Main areas:

    1. ACTIVITY BAR (far left vertical strip)
       - Explorer, Search, Git, Debug, Extensions icons
       - Label: "Quick access to major features"
       - Color highlight: Blue

    2. SIDE BAR (left panel)
       - File explorer showing project structure
       - Label: "Your project files and folders"
       - Color highlight: Green

    3. EDITOR AREA (center, main area)
       - Code with syntax highlighting
       - Label: "Where you write code"
       - Color highlight: Purple

    4. TABS (top of editor)
       - Multiple file tabs
       - Label: "Switch between open files"
       - Color highlight: Orange

    5. MINIMAP (right edge of editor)
       - Zoomed-out code preview
       - Label: "Navigate large files quickly"
       - Color highlight: Teal

    6. TERMINAL (bottom panel)
       - Integrated command line
       - Label: "Run commands without leaving VS Code"
       - Color highlight: Red

    7. STATUS BAR (bottom strip)
       - Python version, line number, encoding
       - Label: "Current file info and settings"
       - Color highlight: Gray

    Interactive elements:
    - Hover over each numbered area to see detailed description
    - Click to see common tasks performed in that area
    - "Hide Labels" toggle to test recall

    Visual style: Clean, modern, matching VS Code dark theme

    Implementation: HTML/CSS with image map and tooltips
</details>

## Jupyter Notebooks: Your Interactive Lab

While VS Code is great for writing Python scripts, data scientists have a special tool that's become essential to the craft: **Jupyter Notebooks**.

A Jupyter Notebook is an interactive document that combines:

- Live code you can run piece by piece
- Rich text explanations with formatting
- Visualizations and charts
- Output from your code, displayed inline

It's like having a lab notebook that can actually DO the experiments, not just record them. You write some code, run it, see the results immediately, then write more code based on what you learned. It's perfect for exploration and experimentation.

### Why Notebooks Are Perfect for Data Science

Traditional programming is linear: write all the code, then run it all at once. But data science is iterative: load data, look at it, clean something, look again, try an analysis, adjust, repeat.

Notebooks support this workflow beautifully:

- **Immediate feedback**: See results instantly after each step
- **Documentation built-in**: Explain your thinking as you go
- **Easy sharing**: Send a notebook, and others see your code AND results
- **Visual output**: Charts appear right where you create them
- **Reproducibility**: Anyone can re-run your analysis step by step

!!! quote "A Data Scientist's Best Friend"
    "I never understood data until I started using Jupyter. Being able to see my data, tweak my code, and immediately see what changed—that's when everything clicked." — Every data scientist, basically

### Starting Jupyter

If you installed Anaconda, you already have Jupyter. To launch it:

```bash
# Make sure your environment is activated
conda activate datascience

# Launch Jupyter Notebook
jupyter notebook
```

This opens a browser window with the Jupyter interface. From there, you can create new notebooks, open existing ones, and organize your files.

Alternatively, in VS Code with the Jupyter extension, you can create and run notebooks directly without the browser interface.

## Notebook Cells: Building Blocks of Discovery

**Notebook cells** are the fundamental units of a Jupyter notebook. Think of cells as individual building blocks—each one contains either code or text, and you can rearrange, add, or delete them as needed.

There are two main types of cells:

### Code Cells

A **code cell** contains Python code that you can execute. When you run a code cell, Python processes the code and displays any output directly below the cell.

```python
# This is a code cell
x = 5
y = 10
print(f"The sum is: {x + y}")
```

Output:
```
The sum is: 15
```

Code cells have some special features:

- The last expression in a cell is automatically displayed (no `print()` needed)
- Variables created in one cell are available in all other cells
- You can run cells in any order (though running in order is usually best)

### Markdown Cells

A **markdown cell** contains formatted text using Markdown syntax. Use these to:

- Explain what your code does
- Document your analysis process
- Add headers and structure
- Include images or links
- Write conclusions and insights

Here's what Markdown looks like:

```markdown
# This is a Header

This is regular text with **bold** and *italic* words.

- Bullet point one
- Bullet point two

> This is a quote block
```

The beauty of Markdown is that it's readable even before rendering. But when you "run" a Markdown cell, Jupyter converts it to beautifully formatted text.

| Cell Type | Contains | Run Behavior |
|-----------|----------|--------------|
| Code Cell | Python code | Executes code, shows output |
| Markdown Cell | Formatted text | Renders as HTML |

#### Diagram: Notebook Cell Types Interactive Demo

<details markdown="1">
    <summary>Notebook Cell Types Interactive Demo</summary>
    Type: microsim

    Bloom Taxonomy: Apply (L3)

    Learning Objective: Let students practice creating, editing, and running different cell types

    Canvas layout (700x550px):
    - Main area (700x450): Notebook simulation
    - Bottom panel (700x100): Controls and instructions

    Visual elements:
    - Simulated notebook interface with cells
    - Cell type indicator (Code/Markdown) on left side
    - Run button for each cell
    - Add cell buttons between cells
    - Cell highlight when selected

    Starting cells:
    1. Markdown cell: "# My First Notebook\nWelcome to data science!"
    2. Code cell: "x = 42\nprint(f'The answer is {x}')"
    3. Empty code cell (ready for input)

    Interactive controls:
    - Click cell to select
    - Button: "Run Cell" (or Shift+Enter simulation)
    - Button: "Add Code Cell"
    - Button: "Add Markdown Cell"
    - Button: "Delete Cell"
    - Button: "Change Cell Type"
    - Text input area for editing selected cell

    Behavior:
    - Running code cell shows output below
    - Running markdown cell renders formatted text
    - Variables persist between cells (running cell 2 sets x=42 for later use)
    - Error messages shown if code has bugs
    - Cell execution order numbers appear [1], [2], etc.

    Instructions panel:
    - "Click a cell to select it"
    - "Press Run Cell to execute"
    - "Try changing the value of x and re-running!"

    Implementation: p5.js with text rendering and simple Python interpreter simulation
</details>

## Cell Execution: Bringing Your Code to Life

**Cell execution** is the process of running a cell and getting results. This is where the magic happens—where your ideas become reality.

### Running Cells

There are several ways to run a cell:

- **Shift + Enter**: Run current cell and move to next cell
- **Ctrl + Enter**: Run current cell and stay on it
- **Run button**: Click the play icon next to the cell
- **Run All**: Execute all cells in order (from menu)

### Execution Order Matters

Here's something crucial to understand: cells can be run in any order, but the ORDER you run them in determines the result. Watch this:

```python
# Cell 1
x = 5

# Cell 2
x = x + 10

# Cell 3
print(x)
```

If you run cells 1, 2, 3 in order: output is `15`
If you run cells 1, 3, 2, 3 in order: output is `15` then `25`
If you run cell 3 first: ERROR! (x doesn't exist yet)

!!! warning "The Restart Trap"
    A common mistake: you run cells out of order, get confused, then can't reproduce your results. Solution? Use **Kernel > Restart & Run All** regularly to verify your notebook runs correctly from top to bottom.

### Execution Numbers

Notice those numbers in brackets next to code cells? Like `[1]`, `[2]`, `[3]`? Those tell you:

1. Which cells have been run
2. What order they were run in

If you see `[5]` followed by `[3]` followed by `[7]`, that's a red flag that cells were run out of order. In a clean, reproducible notebook, numbers should be sequential: `[1]`, `[2]`, `[3]`, etc.

## The Kernel: Your Python Brain

The **kernel** is the computational engine behind your notebook. It's a running Python process that:

- Executes your code cells
- Keeps track of variables and their values
- Remembers function definitions
- Maintains the state of your session

Think of the kernel as Python's brain. When you run a cell, you're asking the brain to process that code and remember the results. All cells share the same brain, which is why a variable defined in one cell is available in all others.

### Kernel Operations

Sometimes you need to control the kernel directly:

| Operation | What It Does | When to Use |
|-----------|--------------|-------------|
| Restart | Clear all variables, fresh start | When things get confusing |
| Restart & Clear Output | Restart + clear all outputs | Clean slate for sharing |
| Restart & Run All | Fresh start, then run everything | Verify reproducibility |
| Interrupt | Stop a running cell | When code takes too long |

The most common kernel operation: **Restart & Run All**. This verifies that your notebook works from scratch—essential before sharing your work or submitting an assignment.

!!! tip "Kernel Health Check"
    See a circular icon in the top right of Jupyter? That's the kernel status indicator. Empty circle = idle (ready). Filled circle = busy (running code). If it's stuck on busy forever, you might need to interrupt or restart.

#### Diagram: Kernel State Visualization

<details markdown="1">
    <summary>How the Kernel Remembers Variables</summary>
    Type: infographic

    Bloom Taxonomy: Understand (L2)

    Learning Objective: Help students understand that the kernel maintains state across cell executions

    Purpose: Visualize the kernel as a memory bank that persists between cell runs

    Layout: Split view - notebook cells on left, kernel memory on right

    Left side (Notebook cells):
    Cell 1: `name = "Alice"`
    Cell 2: `age = 25`
    Cell 3: `greeting = f"Hello {name}, you are {age}"`
    Cell 4: `print(greeting)`

    Right side (Kernel Memory Bank):
    Visual representation of memory slots:
    - After Cell 1: name → "Alice"
    - After Cell 2: name → "Alice", age → 25
    - After Cell 3: name → "Alice", age → 25, greeting → "Hello Alice, you are 25"

    Animation flow:
    - Running each cell shows variable flowing into memory bank
    - Memory bank glows briefly when accessed
    - Clear visual that all cells share the same memory

    Bottom section:
    "Restart Kernel" button → Memory bank empties → Variables gone
    Message: "After restart, you must re-run cells to recreate variables"

    Interactive elements:
    - Step through button to simulate running each cell
    - Restart button to clear memory visualization
    - Hover over variable in memory to see when it was created

    Visual style: Clean, bright, "mind palace" aesthetic

    Implementation: HTML/CSS/JavaScript with animation
</details>

## Import Statements: Summoning Your Tools

An **import statement** tells Python to load a library so you can use its features. Without imports, you're limited to basic Python. With imports, you have access to the entire data science arsenal.

### Basic Import Syntax

There are several ways to import libraries:

```python
# Import the entire library
import pandas

# Now use it with the full name
data = pandas.read_csv("file.csv")
```

```python
# Import with a nickname (alias) - MOST COMMON
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Now use the shorter names
data = pd.read_csv("file.csv")
```

```python
# Import specific functions only
from math import sqrt, pi

# Now use them directly
result = sqrt(16)  # No "math." prefix needed
```

### Standard Data Science Imports

The data science community has agreed on standard aliases. Using these makes your code readable to others:

| Library | Standard Import | What It's For |
|---------|-----------------|---------------|
| pandas | `import pandas as pd` | Data manipulation |
| numpy | `import numpy as np` | Numerical computing |
| matplotlib | `import matplotlib.pyplot as plt` | Basic plotting |
| seaborn | `import seaborn as sns` | Statistical visualization |
| scikit-learn | `from sklearn import ...` | Machine learning |

!!! warning "Import Etiquette"
    Always put imports at the TOP of your notebook or script, not scattered throughout. This makes it easy to see what libraries your code requires. It's like listing ingredients at the start of a recipe—polite and helpful.

### What Happens During Import?

When you write `import pandas as pd`, Python:

1. Searches for the pandas library (in your environment)
2. Loads the library's code into memory
3. Creates a reference called `pd` that points to it
4. Makes all pandas functions available as `pd.something()`

If you get an error like `ModuleNotFoundError: No module named 'pandas'`, it means the library isn't installed in your current environment. Solution: `conda install pandas` or `pip install pandas`.

## Python Libraries: Your Superpower Extensions

**Python libraries** (also called packages or modules) are collections of pre-written code that extend Python's capabilities. They're the reason Python dominates data science—thousands of brilliant people have written code you can use for free.

The data science ecosystem includes hundreds of libraries, but you'll start with a core set:

### The Essential Five

| Library | Superpower | Example Use |
|---------|------------|-------------|
| **pandas** | Data manipulation | Load CSVs, filter rows, calculate statistics |
| **numpy** | Fast math | Array operations, linear algebra |
| **matplotlib** | Visualization | Line plots, bar charts, histograms |
| **seaborn** | Beautiful stats plots | Distribution plots, heatmaps |
| **scikit-learn** | Machine learning | Classification, regression, clustering |

### The Broader Ecosystem

Beyond the essential five, you'll encounter:

- **jupyter** - The notebook system itself
- **scipy** - Scientific computing
- **statsmodels** - Statistical modeling
- **plotly** - Interactive visualizations
- **pytorch / tensorflow** - Deep learning
- **requests** - Web data fetching
- **beautifulsoup** - Web scraping

The beauty of package management: when you need a new capability, you can probably find a library for it. Someone else has already solved your problem—you just need to `pip install` their solution.

#### Diagram: Python Data Science Ecosystem Map

<details markdown="1">
    <summary>Python Data Science Library Ecosystem</summary>
    Type: graph-model

    Bloom Taxonomy: Remember (L1)

    Learning Objective: Help students understand the landscape of Python data science libraries and how they relate

    Purpose: Show relationships between major libraries and their purposes

    Node types:
    1. Core (large gold hexagons)
       - Python, NumPy

    2. Data (blue rectangles)
       - pandas, SQL connectors

    3. Visualization (green circles)
       - matplotlib, seaborn, plotly

    4. Machine Learning (purple diamonds)
       - scikit-learn, XGBoost

    5. Deep Learning (red stars)
       - PyTorch, TensorFlow

    6. Utilities (gray rounded rectangles)
       - Jupyter, requests, BeautifulSoup

    Relationships (edges):
    - NumPy → pandas (built on)
    - NumPy → matplotlib (uses arrays)
    - pandas → seaborn (data source)
    - matplotlib → seaborn (built on)
    - NumPy → scikit-learn (data format)
    - scikit-learn → XGBoost (similar API)
    - NumPy → PyTorch (similar arrays)
    - NumPy → TensorFlow (similar arrays)

    Layout: Hierarchical with Python/NumPy at center

    Interactive features:
    - Hover node: See library description and common use cases
    - Click node: See example import statement
    - Filter by category (checkboxes)

    Visual styling:
    - Node size indicates popularity/importance
    - Edge thickness shows strength of dependency
    - Cluster by function area

    Implementation: vis-network JavaScript
    Canvas size: 800x600px
</details>

## Putting It All Together: Your First Complete Setup

Let's walk through setting up a complete data science environment from scratch. Follow along on your own computer!

### Step 1: Install Anaconda

1. Download Anaconda from [anaconda.com/download](https://www.anaconda.com/download)
2. Run the installer (accept defaults)
3. Open Anaconda Prompt (Windows) or Terminal (Mac/Linux)

### Step 2: Create Your Environment

```bash
# Create a new environment for this course
conda create --name datascience python=3.11

# Activate it
conda activate datascience

# Install essential packages
conda install pandas numpy matplotlib seaborn jupyter scikit-learn
```

### Step 3: Install VS Code

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com)
2. Run the installer
3. Open VS Code
4. Install the Python and Jupyter extensions

### Step 4: Create Your First Notebook

1. In VS Code, press `Ctrl+Shift+P` and type "Create New Jupyter Notebook"
2. Select your `datascience` environment as the kernel
3. Add a markdown cell with a title
4. Add a code cell with your first imports:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Setup complete! Your data science headquarters is ready.")
```

5. Run the cell. If you see the message, congratulations—you're ready!

!!! success "Achievement Unlocked: Data Science HQ Online!"
    You now have a professional-grade data science environment. The same tools used at Google, Netflix, and NASA are now at your fingertips. The only difference between you and a professional data scientist? Practice. Let's get started!

## Troubleshooting Common Issues

Even superheroes face setbacks. Here are solutions to common setup problems:

### "Python not found"

**Cause:** Python isn't in your system PATH.
**Solution:** Reinstall Anaconda and check "Add to PATH" option, OR always use Anaconda Prompt.

### "Module not found"

**Cause:** Package not installed in current environment.
**Solution:** Activate your environment (`conda activate datascience`) then install the package.

### "Kernel died" in Jupyter

**Cause:** Usually a memory issue or package conflict.
**Solution:** Restart the kernel. If persistent, restart your computer or recreate the environment.

### VS Code doesn't see my environment

**Cause:** VS Code hasn't refreshed its environment list.
**Solution:** Press `Ctrl+Shift+P`, type "Python: Select Interpreter", and choose your environment manually.

### Everything is slow

**Cause:** Possibly too many packages or old hardware.
**Solution:** Make sure you're using a clean environment with only the packages you need.

??? question "Chapter 2 Checkpoint: Test Your Understanding"
    **Question:** You're starting a new project and need pandas version 2.0, but your existing project uses pandas 1.5. How do you handle this without breaking either project?

    **Click to reveal answer:**

    Create a separate virtual environment for each project!

    ```bash
    # For the new project
    conda create --name new_project python=3.11
    conda activate new_project
    conda install pandas=2.0

    # For the old project (already exists)
    conda activate old_project
    # pandas 1.5 is already there, unchanged
    ```

    Virtual environments keep projects isolated so different versions don't conflict.

## Key Takeaways

1. **Python installation** (via Anaconda) is your foundation—everything else builds on it.

2. **Package managers** (pip and conda) let you install new tools whenever you need them.

3. **Virtual environments** keep your projects isolated and conflict-free—use them for every project.

4. **VS Code** is your command center—customize it with extensions for data science.

5. **Jupyter Notebooks** combine code, text, and visualizations for interactive exploration.

6. **Cells** (code and markdown) are the building blocks of notebooks; run them with Shift+Enter.

7. The **kernel** is Python's brain—restart it when things get confusing.

8. **Import statements** load libraries; use standard aliases like `pd`, `np`, `plt`.

9. **Python libraries** are your superpower extensions—pandas, numpy, matplotlib are essential.

Your headquarters is built. Your tools are ready. In the next chapter, you'll learn to wield them by working with Python's most important data structures. The real adventure begins now!
