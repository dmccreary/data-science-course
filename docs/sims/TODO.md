# MicroSim TODO List

This file tracks all `#### Diagram` placeholders found in chapter content that still need MicroSims generated.

**Last updated**: 2026-02-22
**Total diagrams found**: 54
**Already implemented**: 8 (all in Chapter 1)
**Remaining to build**: 46

---

## Chapter 1: Introduction to Data Science — Complete ✓

All 8 diagrams have existing MicroSims:
- [x] Data Science Superpower Concept Map → `sims/data-science-superpower/`
- [x] Variable Types Decision Tree → `sims/variable-types-decision-tree/`
- [x] Measurement Scales Pyramid → `sims/measurement-scales-pyramid/`
- [x] Variable Type Sorter → `sims/variable-type-sorter/`
- [x] Independent vs Dependent Variable MicroSim → `sims/independent-dependent-variables/`
- [x] Data Science Workflow Hero's Journey → `sims/data-science-heros-journey/`
- [x] Data Science Language Trends → `sims/ds-prog-lang/`
- [x] Chapter 1 Concept Mind Map → `sims/chapter-1-concept-map/`

---

## Chapter 2: Python Environment and Setup — 0/7 done

- [ ] **Data Science Environment Architecture** — Vertical stack diagram showing layered architecture from OS through Python installation, package manager, virtual environment, libraries, to IDE/Jupyter. Static infographic with hover tooltips on each layer.
- [ ] **Package Manager Workflow** — Horizontal flowchart with 5 steps from user command through package manager search, download, installation, to ready-to-import; includes error paths for common issues.
- [ ] **Virtual Environment Isolation MicroSim** — Visual representation with colored bubbles representing environments; includes conflict indicators; interactive controls for creating/deleting environments.
- [ ] **VS Code Interface Tour** — Labeled diagram of VS Code with numbered callouts: Activity Bar, Side Bar, Editor Area, Tabs, Minimap, Terminal, Status Bar.
- [ ] **Notebook Cell Types Interactive Demo** — Notebook simulation with 3 starting cells; interactive controls for adding/deleting cells, changing types (Code/Markdown), with real-time output.
- [ ] **Kernel State Visualization** — Split view showing notebook cells on left and kernel memory bank on right; demonstrates variable persistence across cells with animation.
- [ ] **Python Data Science Ecosystem Map** — Hub-and-spoke diagram with NumPy at center connecting to pandas, scikit-learn, matplotlib/plotly, SciPy, PyTorch/TensorFlow.

---

## Chapter 3: Python Data Structures — 0/4 done

- [ ] **Data Structure Hierarchy** — Hierarchical tree showing relationship between Python native structures (Lists, Dictionaries, Tuples) and pandas structures (Series, DataFrame).
- [ ] **Python Data Structures Comparison MicroSim** — Quiz game with 10 scenarios showing when to use List, Dictionary, Tuple, or Array; includes scoring and achievement badges.
- [ ] **DataFrame Anatomy** — Central DataFrame table with callouts pointing to: Index, Columns, Row, Column, Cell, Values; interactive hover and code examples.
- [ ] **Data Loading Workflow** — Horizontal flowchart of 5 steps from CSV file on disk through parsing to ready-for-analysis DataFrame.

---

## Chapter 4: Data Cleaning and Preprocessing — 0/6 done

- [ ] **Data Cleaning Pipeline Overview** — Horizontal flowchart showing 8 sequential steps: Raw Data → Missing Values → Duplicates → Outliers → Data Types → Validation → Transformation → Clean Data; includes feedback loops for logging issues.
- [ ] **Missing Value Detection MicroSim** — MicroSim with DataFrame display, detection code panel, and results area; includes 5 rotating scenarios of missing data patterns; quiz mode for counting missing values.
- [ ] **Duplicate Handling Decision Tree** — Decision tree flowchart starting with "Duplicates Detected"; includes decisions about exact matches, key columns, and merge strategies; color-coded risk levels.
- [ ] **Outlier Detection Methods MicroSim** — Left: scatter plot/histogram visualization; Right: controls for Z-score, IQR, custom range methods; shows detected outliers highlighted in red; compares different method results.
- [ ] **Data Type Conversion Guide Infographic** — Four-quadrant reference card: Converting to Numeric, Datetime, Categorical, String with code examples.
- [ ] **Feature Scaling Comparison MicroSim** — Top: original data histogram; Bottom split: scaled histogram and before/after comparison stats; shows Min-Max, Standard, Robust, Log transform effects.

---

## Chapter 5: Data Visualization with Matplotlib and Plotly — 0/5 done

- [ ] **Visualization Library Comparison** — Three-column comparison: Matplotlib, Seaborn, Plotly with strengths, best uses, learning curve; includes decision flowchart.
- [ ] **Chart Type Selection Guide** — Decision tree starting with "What do you want to show?" with 5 branches (Comparison, Distribution, Relationship, Composition, Trend) and specific chart recommendations.
- [ ] **Plotly Interactive Features MicroSim** — Main interactive Plotly chart with challenge tasks; demonstrates 8 features: hover, zoom, pan, box/lasso select, reset, download, legend toggles.
- [ ] **Plotly Code Pattern Reference** — Four-quadrant quick reference card with Common Chart Functions, Essential Parameters, Layout Customization, Saving Options.
- [ ] **Visualization Design MicroSim** — Left: controls for chart type, data selection, customization options; Right: live chart preview with code panel; features "Design tips" suggestions.

---

## Chapter 6: Statistical Foundations — 0/6 done

- [ ] **Central Tendency Comparison MicroSim** — Top: histogram with draggable data points and mean/median/mode lines; Bottom: statistics display and controls; challenge tasks to manipulate distribution.
- [ ] **Box Plot Anatomy** — Central horizontal box plot with 7 labeled callouts: minimum, Q1, median, Q3, maximum, IQR, outliers; interactive hover and code examples.
- [ ] **Normal Distribution Explorer MicroSim** — Main: interactive normal distribution curve; Controls: sliders for mean and standard deviation; shows 68-95-99.7 regions; includes preset examples.
- [ ] **Central Limit Theorem Simulator MicroSim** — Left: original population distribution; Right: distribution of sample means; shows how sample means become normal regardless of population shape; dropdown for 5 distribution types.
- [ ] **Hypothesis Testing Workflow** — Vertical flowchart with decision diamonds and process rectangles from research question through hypothesis statement, data collection, p-value calculation, decision, and reporting.
- [ ] **Correlation Visualizer MicroSim** — Left: scatter plot with adjustable correlation; Right: Pearson r, Spearman r, p-value, R², sample size displays; challenge tasks for correlation patterns.

---

## Chapter 7: Simple Linear Regression — 0/4 done

- [ ] **Regression Line Anatomy** — Scatter plot with regression line and 5 labeled callouts: intercept (β₀), slope (β₁), predicted value (ŷ), actual value (y), residual.
- [ ] **Least Squares MicroSim** — Left: adjustable regression line through data; Right: SSE display and comparison; shows residual squares resizing with line adjustments; "Show Optimal" button with animation. (Note: `sims/least-squares/` may exist — verify coverage.)
- [ ] **Scikit-learn Workflow** — Horizontal flowchart of 6 steps: Import → Prepare Data → Create Model → Fit Model → Predict → Evaluate; includes common errors callout.
- [ ] **Interactive Regression Builder MicroSim** — Left: scatter plot with regression line and residuals; Right: coefficient display, equation, R² gauge, interpretation text; prediction tool with confidence intervals.

---

## Chapter 8: Model Evaluation and Validation — 0/8 done

- [ ] **Train-Test Split Visualization** — Horizontal bar showing data divided into 80/20 split; interactive slider (50–90%); warning states; visual eye/blindfold icons.
- [ ] **Metrics Comparison MicroSim** — Left: scatter plot with draggable points; Right: real-time R², MSE, RMSE, MAE metrics; shows how different error metrics respond to outliers.
- [ ] **Residual Pattern Detective** — 2×2 grid of residual plot examples: Healthy Residuals, Curved Pattern, Funnel Shape, Clustered Groups; each with diagnostic advice.
- [ ] **Bias-Variance Dartboard** — Four dartboard panels showing combinations of bias/variance; interactive complexity slider; real-time bias/variance/total error indicators.
- [ ] **Complexity Curve Explorer** — Top: data points with polynomial curve; Bottom: error vs complexity chart with train and test error lines; shaded zones for under/over-fitting.
- [ ] **K-Fold Cross-Validation Animator** — Visual representation with 50 colored squares divided into K folds; animation shows rotation through test folds; results table with mean/std/min/max.
- [ ] **Model Selection Dashboard** — Left: model configuration and training interface; Right: leaderboard table and visualization; workflow to train multiple models and select winner.
- [ ] **Model Evaluation Workflow** — Vertical flowchart with 10 steps through model development lifecycle; swimlanes for Full Dataset, Training, Validation, Test; point of no return barrier before final evaluation.

---

## Chapter 9: Multiple Linear Regression — 0/7 done

- [ ] **Multiple Regression Anatomy** — Central equation with branching explanations for intercept and each coefficient term; interactive sliders for feature values; real-time contribution visualization.
- [ ] **Multicollinearity Detector MicroSim** — Left: scatter plot matrix; Right: VIF bar chart with color coding; simulates effect of adding correlated features on VIF and coefficient confidence intervals.
- [ ] **Feature Selection Race** — Three parallel "race tracks" for Forward/Backward/Stepwise selection methods; features light up as added/dimmed as removed; results comparison table.
- [ ] **One-Hot Encoding Visualizer** — Before/after transformation showing categorical column splitting into binary columns; animated arrows; toggle "drop first" option; supports 2–8 categories.
- [ ] **Feature Engineering Laboratory** — Left: feature creation interface; Center: data preview with new features; Right: model performance metrics; preset transformation examples.
- [ ] **Feature Importance Explorer** — Three stacked bar charts comparing methods; interactive feature deep-dive with scatter, partial dependence, and distribution plots.
- [ ] **Multiple Regression Pipeline** — Horizontal flowchart of 10 stages: Raw Data → Feature Engineering → Train/Test Split → Preprocessing → Multicollinearity Check → Feature Selection → Training → Cross-Validation → Final Evaluation → Feature Importance.

---

## Chapter 10: NumPy and Numerical Computing — 0/6 done

- [ ] **NumPy Array vs Python List** — Side-by-side comparison showing scattered vs contiguous memory; speedometer graphics showing 50–100× performance advantage; includes speed test button.
- [ ] **Array Shape Visualizer** — Left: 3D visualization of arrays; Right: shape controls with dimension sliders; shows 1D/2D/3D representations; includes flatten and reshape animations.
- [ ] **Slicing Playground** — Left: 2D grid with highlighted selection; Right: slicing controls with start/stop/step inputs; preset buttons; error messages for invalid slices.
- [ ] **Broadcasting Visualizer** — Top: array representations; Bottom: shape analysis aligned right-to-left; preset examples for scalar+matrix, row+matrix, column+matrix scenarios.
- [ ] **Matrix Multiplication Visualizer** — Left: Matrix A with row highlighting; Center: Matrix B with column highlighting; Right: Result C; animation shows dot product calculation steps.
- [ ] **NumPy Ecosystem Map** — Hub-and-spoke with NumPy at center connecting to: Pandas, Scikit-learn, Matplotlib/Plotly, SciPy, PyTorch/TensorFlow; hover shows code examples.

---

## Chapter 11: Non-linear Models and Regularization — 0/5 done

- [ ] **Polynomial Degree Explorer** — Scatter plot with polynomial curve; slider for degree (1–15); displays train/test R² metrics; warns about overfitting; animated transitions.
- [ ] **Transformation Gallery** — 2×3 grid of transformations: Log, Square Root, Reciprocal, Square, Box-Cox, Standardization; before/after histograms for each.
- [ ] **Ridge vs Lasso Comparison** — Left panels: coefficient paths for Ridge and Lasso; Right: bar chart comparison; penalty visualization showing L2 ball vs L1 diamond.
- [ ] **Lambda Tuning Playground** — Top: model fit with polynomial curve; Bottom left: CV score vs lambda plot; Bottom right: coefficient magnitudes; interactive lambda slider.
- [ ] **Regularization Decision Tree** — Flowchart from "Need to Prevent Overfitting?" through 5 decision points to outcomes: Ridge, Lasso, or Elastic Net.

---

## Chapter 12: Introduction to Machine Learning — 0/5 done

- [ ] **Supervised vs Unsupervised Learning** — Side-by-side comparison panels; Left shows labeled training data → model → predictions; Right shows unlabeled data → model → discovered groups; includes quiz mode.
- [ ] **Training Process Animator** — Main: scatter plot with evolving regression line; Bottom: controls for step/play/pause; shows MSE decreasing; metrics display; convergence messages.
- [ ] **Error Types Visualizer** — Left: training data and fit; Right: test data and fit; Bottom: error bar chart comparing train vs test; diagnosis text for overfitting/underfitting.
- [ ] **Gradient Descent Visualizer** — 3D surface or 2D contour of cost function; animated ball rolling downhill; learning rate effects visualization; shows gradient arrows.
- [ ] **Optimization Landscape Explorer** — 2D function plot with multiple minima; optimizer starts at clickable position; animation shows ball rolling; demonstrates local vs global minima; momentum toggle.

---

## Chapter 13: Neural Networks and PyTorch — 0/3 done

- [ ] **Activation Function Explorer** — Graph showing multiple activation functions (Linear, Step, Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax); toggles for derivatives; vanishing gradient highlighting.
- [ ] **Neural Network Architecture Builder** — Left: network visualization with neurons and connections; Right: architecture controls for hidden layers and neuron counts; parameter counter; forward pass animation.
- [ ] **Training Loop Visualizer** — Left: five training step cards with code; Right: loss curve and weight visualization; step-by-step animation; metrics display; learning rate control.
