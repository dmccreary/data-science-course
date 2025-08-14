# Setting up Python, Visual Studio Code and Jupyter Notebooks

In this course will will use, Python, Visual Studio Code and Jupyter Notebooks to do our classroom work.
We selected these tools because they are the most popular development environments
in data science today.  You are free to use other tools in this course, however
you may need to use generative AI to translate our code into other systems.
In addition, we strongly recommend each student create their own personal
GitHub repository to store their work.  Because GitHub is backed up the
students only need to check their code into GitHub using either a command line
or Source Control section of Visual Studio Code.

We will be using the following tools for our desktop setup:

1. [Visual Studio Code](../../glossary.md#visual-studio-code) - integrated development environment
2. Conda - virtual environment
3. Pip - Python installer package

## Conda

Create a "Data Science" conda environment.

```sh
conda create -n "ds" python=3
conda activate ds
```

## Installing Jupyter Notebooks

Jupyter Notebooks are essential for this course as they provide an interactive environment for data science work. There are several ways to install and use Jupyter Notebooks:

### Method 1: Install via Conda (Recommended)

After activating your conda environment, install Jupyter:

```sh
conda activate ds
conda install jupyter
```

### Method 2: Install via Pip

Alternatively, you can install using pip:

```sh
pip install jupyter
```

### Method 3: Install JupyterLab (Enhanced Interface)

For a more modern interface, you can install JupyterLab:

```sh
pip install jupyterlab
```

## Running Jupyter Notebooks

### Starting Jupyter Notebook Server

To start the Jupyter Notebook server:

```sh
# Activate your environment first
conda activate ds

# Start Jupyter Notebook
jupyter notebook
```

This will:
1. Start a local server (usually at `http://localhost:8888`)
2. Automatically open your web browser to the Jupyter interface
3. Display your file system where you can create new notebooks or open existing ones

### Starting JupyterLab

If you installed JupyterLab:

```sh
jupyter lab
```

## Using Jupyter Notebooks in Visual Studio Code

Visual Studio Code provides excellent built-in support for Jupyter Notebooks, offering a seamless development experience.

### Installing the Jupyter Extension

1. Open Visual Studio Code
2. Go to the Extensions view (`Ctrl+Shift+X` or `Cmd+Shift+X`)
3. Search for "Jupyter" 
4. Install the official "Jupyter" extension by Microsoft

### Working with Notebooks in VS Code

**Creating a New Notebook:**
1. Open VS Code
2. Create a new file with `.ipynb` extension (e.g., `my_analysis.ipynb`)
3. VS Code will automatically recognize it as a Jupyter notebook

**Selecting Python Interpreter:**
1. Open a notebook file
2. Click on "Select Kernel" in the top-right corner
3. Choose your conda environment (`ds`) as the Python interpreter

**Running Code Cells:**
- **Run Single Cell**: Click the play button next to a cell or press `Shift+Enter`
- **Run All Cells**: Use `Ctrl+Shift+P` (or `Cmd+Shift+P`) and search "Jupyter: Run All Cells"
- **Add New Cell**: Click the `+ Code` or `+ Markdown` buttons between cells

### Advantages of Using Jupyter in VS Code

- **Integrated Development**: Code editing, debugging, and notebook execution in one interface
- **Git Integration**: Easy version control for notebook files
- **IntelliSense**: Code completion and syntax highlighting
- **Variable Explorer**: View variables and data structures in real-time
- **Debugging Support**: Set breakpoints and debug notebook code
- **Extensions**: Access to VS Code's rich extension ecosystem

### Installing Essential Data Science Packages

Install the core packages you'll need for this course:

```sh
# Activate your environment
conda activate ds

# Install essential data science packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Optional: Install additional useful packages
pip install plotly ipywidgets
```

## Transcript

```
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 23.5.2
  latest version: 24.9.2

Please update conda by running

    $ conda update -n base -c defaults conda

Or to minimize the number of packages updated during conda update use

     conda install conda=24.9.2


## Package Plan ##

  environment location: /Users/danmccreary/miniconda3/envs/ds

  added / updated specs:
    - python=3


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ca-certificates-2024.9.24  |       hca03da5_0         131 KB
    expat-2.6.3                |       h313beb8_0         154 KB
    libmpdec-4.0.0             |       h80987f9_0          69 KB
    openssl-3.0.15             |       h80987f9_0         4.3 MB
    pip-24.2                   |  py313hca03da5_0         2.4 MB
    python-3.13.0              |h4862095_100_cp313        13.7 MB
    python_abi-3.13            |          0_cp313           7 KB
    setuptools-75.1.0          |  py313hca03da5_0         1.9 MB
    tzdata-2024b               |       h04d1e81_0         115 KB
    wheel-0.44.0               |  py313hca03da5_0         139 KB
    ------------------------------------------------------------
                                           Total:        23.0 MB

The following NEW packages will be INSTALLED:

  bzip2              pkgs/main/osx-arm64::bzip2-1.0.8-h80987f9_6 
  ca-certificates    pkgs/main/osx-arm64::ca-certificates-2024.9.24-hca03da5_0 
  expat              pkgs/main/osx-arm64::expat-2.6.3-h313beb8_0 
  libcxx             pkgs/main/osx-arm64::libcxx-14.0.6-h848a8c0_0 
  libffi             pkgs/main/osx-arm64::libffi-3.4.4-hca03da5_1 
  libmpdec           pkgs/main/osx-arm64::libmpdec-4.0.0-h80987f9_0 
  ncurses            pkgs/main/osx-arm64::ncurses-6.4-h313beb8_0 
  openssl            pkgs/main/osx-arm64::openssl-3.0.15-h80987f9_0 
  pip                pkgs/main/osx-arm64::pip-24.2-py313hca03da5_0 
  python             pkgs/main/osx-arm64::python-3.13.0-h4862095_100_cp313 
  python_abi         pkgs/main/osx-arm64::python_abi-3.13-0_cp313 
  readline           pkgs/main/osx-arm64::readline-8.2-h1a28f6b_0 
  setuptools         pkgs/main/osx-arm64::setuptools-75.1.0-py313hca03da5_0 
  sqlite             pkgs/main/osx-arm64::sqlite-3.45.3-h80987f9_0 
  tk                 pkgs/main/osx-arm64::tk-8.6.14-h6ba3021_0 
  tzdata             pkgs/main/noarch::tzdata-2024b-h04d1e81_0 
  wheel              pkgs/main/osx-arm64::wheel-0.44.0-py313hca03da5_0 
  xz                 pkgs/main/osx-arm64::xz-5.4.6-h80987f9_1 
  zlib               pkgs/main/osx-arm64::zlib-1.2.13-h18a0788_1 


Proceed ([y]/n)? y


Downloading and Extracting Packages
                                                                                                     
Preparing transaction: done                                                                          
Verifying transaction: done                                                                          
Executing transaction: done                                                                        
#                                                                                                    
# To activate this environment, use                                                                  
#                                                                                                    
#     $ conda activate ds                                                                            
#                                                                                                    
# To deactivate an active environment, use                                                           
#
#     $ conda deactivate
```

## Pip Install Transcript

(ds) src/line-plot $ pip install matplotlib

```
Collecting matplotlib
  Downloading matplotlib-3.9.2-cp313-cp313-macosx_11_0_arm64.whl.metadata (11 kB)
Collecting contourpy>=1.0.1 (from matplotlib)
  Downloading contourpy-1.3.1-cp313-cp313-macosx_11_0_arm64.whl.metadata (5.4 kB)
Collecting cycler>=0.10 (from matplotlib)
  Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib)
  Downloading fonttools-4.55.0-cp313-cp313-macosx_10_13_universal2.whl.metadata (164 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib)
  Downloading kiwisolver-1.4.7-cp313-cp313-macosx_11_0_arm64.whl.metadata (6.3 kB)
Collecting numpy>=1.23 (from matplotlib)
  Downloading numpy-2.1.3-cp313-cp313-macosx_14_0_arm64.whl.metadata (62 kB)
Collecting packaging>=20.0 (from matplotlib)
  Downloading packaging-24.2-py3-none-any.whl.metadata (3.2 kB)
Collecting pillow>=8 (from matplotlib)
  Downloading pillow-11.0.0-cp313-cp313-macosx_11_0_arm64.whl.metadata (9.1 kB)
Collecting pyparsing>=2.3.1 (from matplotlib)
  Downloading pyparsing-3.2.0-py3-none-any.whl.metadata (5.0 kB)
Collecting python-dateutil>=2.7 (from matplotlib)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib)
  Using cached six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
Downloading matplotlib-3.9.2-cp313-cp313-macosx_11_0_arm64.whl (7.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 35.2 MB/s eta 0:00:00
Downloading contourpy-1.3.1-cp313-cp313-macosx_11_0_arm64.whl (255 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.55.0-cp313-cp313-macosx_10_13_universal2.whl (2.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.8/2.8 MB 19.8 MB/s eta 0:00:00
Downloading kiwisolver-1.4.7-cp313-cp313-macosx_11_0_arm64.whl (63 kB)
Downloading numpy-2.1.3-cp313-cp313-macosx_14_0_arm64.whl (5.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 MB 33.8 MB/s eta 0:00:00
Downloading packaging-24.2-py3-none-any.whl (65 kB)
Downloading pillow-11.0.0-cp313-cp313-macosx_11_0_arm64.whl (3.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 28.6 MB/s eta 0:00:00
Downloading pyparsing-3.2.0-py3-none-any.whl (106 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
Installing collected packages: six, pyparsing, pillow, packaging, numpy, kiwisolver, fonttools, cycler, python-dateutil, contourpy, matplotlib
Successfully installed contourpy-1.3.1 cycler-0.12.1 fonttools-4.55.0 kiwisolver-1.4.7 matplotlib-3.9.2 numpy-2.1.3 packaging-24.2 pillow-11.0.0 pyparsing-3.2.0 python-dateutil-2.9.0.post0 six-1.16.0
(ds) src/line-plot $ python line-plot.py   
2024-11-15 07:14:21.701 python[53332:2841291] +[IMKClient subclass]: chose IMKClient_Legacy
2024-11-15 07:14:21.701 python[53332:2841291] +[IMKInputSession subclass]: chose IMKInputSession_Legacy
```

