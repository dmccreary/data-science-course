# Foundations of Data Science

## Introduction to Data Science and Its Applications

Data science is the discipline of extracting meaningful insights from data by combining **statistics**, **programming**, and **domain expertise**. It powers many of the services and tools we use every day—from recommendation engines on streaming platforms to real-time fraud detection in banking. Governments, businesses, and non-profits alike depend on data science to make **evidence-based decisions** and improve efficiency.

Applications of data science span nearly every field:

* **Healthcare:** Predicting disease risks, optimizing treatment effectiveness, and analyzing healthcare costs.
* **Finance:** Credit scoring, algorithmic trading, and fraud detection.
* **Retail:** Personalized recommendations and demand forecasting.
* **Transportation:** Route optimization and autonomous vehicle navigation.
* **Environmental Science:** Climate modeling and resource management.

### First Lab: Exploring Sample Datasets

To begin, we will explore real datasets. A simple but practical task is to generate a **CSV file** of **per-capita annual healthcare costs for 2023** across the 100 largest countries. You will then use Python to:

1. Load the data into a Pandas **DataFrame**.
2. Compute summary statistics such as mean, median, and standard deviation.
3. Create visualizations such as bar charts and scatter plots.
4. Ask ChatGPT (or another LLM) to interpret the results and suggest insights.

Here is a sample of what this data looks like:

```csv
Country Name,Country Code,health_exp_pc_ppp_2022
Africa Eastern and Southern,AFE,228
Afghanistan,AFG,383
Africa Western and Central,AFW,201
Angola,AGO,217
Albania,ALB,1186
Andorra,AND,5136
Arab World,ARB,776
United Arab Emirates,ARE,3814
Argentina,ARG,2664
Armenia,ARM,1824
Antigua and Barbuda,ATG,1436
```

Note that this list from the World Bank.

This exercise introduces **data cleaning**, **exploration**, and **visualization**, which form the foundation of every data science project.

## Data Science Workflow

<iframe src="../../sims/data-science-workflow/main.html" height="300px" scrolling="no"></iframe>

Above is an interactive infographic that allows you to explore the six steps in a typical data
science workflow.  For each step, hover over the step and view the text description below the
step.

## Why Python for Data Science?

Python is the most widely used programming language for data science. It is popular because of:

* A rich **ecosystem of libraries** (NumPy, Pandas, scikit-learn, Matplotlib, PyTorch).
* Readable, beginner-friendly syntax.
* Strong community support and open-source resources.

Over the past 15 years, Python has steadily risen to become the dominant language in data science. Other languages such as R, Java, and Julia are used in specific contexts, but Python’s versatility has made it the industry standard.

*Activity:* Show students a graph of **programming language popularity** in data science from 2008–2023. Highlight Python’s exponential growth alongside the plateau of other languages.

---

## Basic Data Types and Structures in Python

Before analyzing data, students must understand how Python stores it. Core **data types** include:

* **Integers** (whole numbers, e.g., `42`)
* **Floats** (decimal numbers, e.g., `3.14`)
* **Strings** (text, e.g., `"data science"`)
* **Booleans** (`True` or `False`)

Core **data structures** include:

* **Lists** – ordered, mutable collections (e.g., `[1,2,3]`)
* **Tuples** – ordered, immutable collections (e.g., `(1,2,3)`)
* **Dictionaries** – key-value pairs (e.g., `{"name": "Alice", "age": 20}`)
* **Sets** – unordered, unique elements (e.g., `{1,2,3}`)

Later in the course, we will rely heavily on **NumPy arrays** and **Pandas DataFrames**, which are optimized for data manipulation.

---

## Understanding the Data Science Workflow

Every data science project follows a structured workflow:

1. **Define the problem** – Clarify what question is being answered.
2. **Collect data** – Gather raw data from reliable sources.
3. **Clean and preprocess data** – Handle missing values, errors, and inconsistencies.
4. **Explore and visualize** – Use plots and descriptive statistics to understand patterns.
5. **Modeling** – Build predictive or explanatory models.
6. **Evaluate** – Use metrics to test accuracy and generalizability.
7. **Deploy and communicate results** – Share findings with stakeholders.

This workflow is iterative. A failed model often sends us back to collect new data or engineer better features.

### MicroSim – Data Science Workflow Infographic

Students can explore an **interactive infographic** where clicking each stage of the workflow reveals its purpose, key tools, and example questions.

---

## Data Science Roles and Career Paths

Data science is a team effort, involving many specialized roles:

* **Data Scientist:** Builds models, interprets results, and communicates insights.
* **Data Engineer:** Designs pipelines and storage systems for reliable data access.
* **Machine Learning Engineer:** Deploys and optimizes models in production systems.
* **Business Analyst:** Translates data insights into actionable strategies.
* **Ethics & Compliance Specialist:** Ensures fairness, transparency, and privacy in projects.

These roles often overlap, and many entry-level positions expect a blend of programming, statistics, and communication skills.

---

## Ethics and Best Practices in Data Science

Data science has great potential, but also significant risks. Poorly designed or biased models can reinforce inequalities or cause harm. To practice **ethical data science**, we must:

* **Protect privacy**: Respect data ownership and confidentiality.
* **Avoid bias**: Check datasets and models for fairness across subgroups.
* **Be transparent**: Document methods and assumptions clearly.
* **Ensure reproducibility**: Use version control and pipelines so results can be verified.
* **Balance efficiency and responsibility**: Consider environmental and social impacts.

*Suggested MicroSim:* **Ethical Impact Assessment Tool** (students adjust fairness, privacy, and transparency sliders to see overall model “ethical score”).

---

✅ This completes the **Foundations of Data Science** chapter, preparing students for Week 1 of the course.

---

Would you like me to also generate the **first CSV dataset** (per-capita healthcare costs for 100 countries in 2023) so students can immediately use it in the “First Lab” section?
