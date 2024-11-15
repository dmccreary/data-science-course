# Introduction to Data Science FAQs

#### What is data science, and why is it important?

Data science is an interdisciplinary field that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data. It is important because it enables organizations to make informed decisions, predict trends, and solve complex problems by analyzing large datasets.

#### How is Python used in data science?

Python is widely used in data science due to its simplicity and versatility. It offers a vast ecosystem of libraries and frameworks like NumPy, Pandas, Matplotlib, and scikit-learn, which facilitate data manipulation, analysis, visualization, and machine learning tasks.

#### What are the key Python libraries for data analysis?

The key Python libraries for data analysis include:

-   **NumPy**: For numerical computing and array operations.
-   **Pandas**: For data manipulation and analysis using data structures like DataFrames.
-   **Matplotlib**: For creating static, animated, and interactive visualizations.
-   **Seaborn**: For statistical data visualization built on top of Matplotlib.
-   **scikit-learn**: For machine learning algorithms and predictive data analysis.

#### How do you import a CSV file into a Pandas DataFrame?

You can import a CSV file using the `read_csv()` function from Pandas:

```python
import pandas as pd

df = pd.read_csv('file_name.csv')
```

Replace `'file_name.csv'` with the path to your CSV file.

#### What is a DataFrame in Pandas?

A DataFrame is a two-dimensional, size-mutable, and heterogeneous tabular data structure with labeled axes (rows and columns). It is similar to a spreadsheet or SQL table and is the primary data structure used in Pandas for data manipulation.

#### How do you handle missing data in a dataset?

Missing data can be handled by:

-   **Removing missing values**: Using `dropna()` to remove rows or columns with missing values.
-   **Imputing missing values**: Using `fillna()` to replace missing values with a specific value, mean, median, or mode.
-   **Interpolate missing values**: Using `interpolate()` to estimate missing values based on other data points.

#### What is the difference between NumPy arrays and Python lists?

NumPy arrays are fixed-size, homogeneous collections of elements (all of the same data type) optimized for numerical computations. Python lists are dynamic, heterogeneous collections that can contain elements of different data types. NumPy arrays offer better performance for mathematical operations.

#### How do you select a subset of data from a DataFrame?

You can select subsets using two methods:

1. Label-based indexing
2. Integer-based indexing

-  **Label-based indexing with `.loc`**:

```pytho
df_subset = df.loc[row_labels, column_labels]
```

-   **Integer-based indexing with `.iloc`**:

```python
df_subset = df.iloc[row_indices, column_indices]
```

#### What is data visualization, and why is it important?

Data visualization is the graphical representation of data to communicate information clearly and efficiently. It is important because it helps identify patterns, trends, and outliers in data, making complex data more accessible and understandable.

#### How do you create a simple line plot using Matplotlib?

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Simple Line Plot')
plt.show()
```

#### What is the purpose of the `groupby()` function in Pandas?

The `groupby()` function is used to split data into groups based on some criteria, perform operations on each group independently, and then combine the results. It is useful for aggregation, transformation, and filtration of data.

#### How do you merge two DataFrames in Pandas?

You can merge two DataFrames using the `merge()` function:

```python
merged_df = pd.merge(df1, df2, on='common_column')
```

Replace `'common_column'` with the column name that is common to both DataFrames.

#### What is the difference between `merge()` and `concat()` in Pandas?

-   **`merge()`**: Combines two DataFrames based on the values of common columns (similar to SQL joins).
-   **`concat()`**: Concatenates DataFrames either vertically or horizontally, stacking them along an axis.

#### How do you calculate basic statistical measures like mean and median in Pandas?

You can use built-in functions:

- **Mean**:

```python
mean_value = df['column_name'].mean()
```

- **Median**:

```python
median_value = df['column_name'].median()
```

#### What is the purpose of the `apply()` function in Pandas?

The `apply()` function allows you to apply a function along an axis of the DataFrame (either rows or columns). It is useful for performing complex operations on DataFrame elements.

#### How do you create a pivot table in Pandas?

You can create a pivot table using the `pivot_table()` function:

```python
pivot = pd.pivot_table(df, values='value_column', index='index_column', columns='columns_column', aggfunc='mean')
```

#### What is the difference between supervised and unsupervised learning?

-   **Supervised Learning**: Involves training a model on labeled data, where the target outcome is known. Examples include regression and classification.
-   **Unsupervised Learning**: Involves finding patterns in unlabeled data without predefined outcomes. Examples include clustering and dimensionality reduction.

#### How do you perform linear regression using scikit-learn?

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### What is overfitting in machine learning?

Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor generalization to new, unseen data. It performs well on training data but poorly on test data.

#### How can you prevent overfitting?

Overfitting can be prevented by:

-   **Cross-validation**: Splitting data into training and validation sets.
-   **Regularization**: Adding penalties for complex models (e.g., Lasso, Ridge).
-   **Simplifying the model**: Reducing the number of features.
-   **Early stopping**: Halting training when performance on a validation set starts to degrade.

#### What is cross-validation?

Cross-validation is a technique for assessing how a model will generalize to an independent dataset. It involves partitioning the data into subsets, training the model on some subsets, and validating it on the remaining ones.

#### How do you evaluate the performance of a regression model?

Common metrics include:

-   **Mean Absolute Error (MAE)**
-   **Mean Squared Error (MSE)**
-   **Root Mean Squared Error (RMSE)**
-   **R-squared (Coefficient of Determination)**

#### What is the purpose of feature scaling?

Feature scaling standardizes the range of independent variables, improving the performance and convergence speed of some machine learning algorithms that are sensitive to the scale of data, such as gradient descent optimization.

#### How do you perform feature scaling in Python?

Using scikit-learn's `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

#### What is one-hot encoding?

One-hot encoding is a process of converting categorical variables into a binary (0 or 1) representation. Each category becomes a new column, and a value of 1 indicates the presence of that category.

#### How do you perform one-hot encoding in Pandas?

Using the `get_dummies()` function:

```python
encoded_df = pd.get_dummies(df, columns=['categorical_column'])
```

#### What is a confusion matrix?

A confusion matrix is a table used to evaluate the performance of a classification model. It displays the true positives, true negatives, false positives, and false negatives, providing insight into the types of errors made by the model.

#### How do you calculate accuracy, precision, and recall from a confusion matrix?

-   **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
-   **Precision**: TP / (TP + FP)
-   **Recall**: TP / (TP + FN)

where:

* TP=True Positives
* TN=True Negatives
* FP=False Positives
* FN=False Negatives.

#### What is the purpose of the `train_test_split` function?

The `train_test_split` function splits a dataset into training and testing sets, allowing you to train the model on one subset and evaluate its performance on another to prevent overfitting.

#### How do you split data into training and testing sets in scikit-learn?

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

#### What is the difference between classification and regression?

-   **Classification**: Predicts categorical labels or classes.
-   **Regression**: Predicts continuous numerical values.

#### How do you handle categorical variables in machine learning models?

Categorical variables can be handled by:

-   **Label Encoding**: Assigning a unique integer to each category.
-   **One-Hot Encoding**: Creating binary columns for each category.

#### What is k-means clustering?

K-means clustering is an unsupervised learning algorithm that partitions data into **k** clusters, where each data point belongs to the cluster with the nearest mean. It aims to minimize the within-cluster sum of squares.

#### How do you determine the optimal number of clusters in k-means?

Common methods include:

-   **Elbow Method**: Plotting the explained variance as a function of the number of clusters and looking for an "elbow" point.
-   **Silhouette Score**: Measuring how similar a data point is to its own cluster compared to other clusters.

#### What is principal component analysis (PCA)?

PCA is a dimensionality reduction technique that transforms a large set of variables into a smaller one that still contains most of the information by identifying the principal components (directions of maximum variance).

#### How do you perform PCA in scikit-learn?

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)
```

#### What is the bias-variance tradeoff?

The bias-variance tradeoff is the balance between a model's ability to generalize to new data (low variance) and its accuracy on training data (low bias). High bias can lead to underfitting, while high variance can lead to overfitting.

#### What is regularization in machine learning?

Regularization involves adding a penalty term to the loss function to prevent overfitting by discouraging complex models. Common regularization techniques include Lasso (L1) and Ridge (L2) regression.

#### How do you implement Ridge regression in scikit-learn?

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

#### What is logistic regression?

Logistic regression is a classification algorithm used to predict binary outcomes (0 or 1) by modeling the probability of a certain class using a logistic function.

#### How do you evaluate the performance of a classification model?

Common metrics include:

-   **Accuracy**
-   **Precision**
-   **Recall**
-   **F1 Score**
-   **ROC AUC Score**

#### What is the Receiver Operating Characteristic (ROC) curve?

The ROC curve plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) at various threshold settings. It illustrates the diagnostic ability of a binary classifier.

#### How do you handle imbalanced datasets?

Techniques include:

-   **Resampling Methods**: Oversampling the minority class or undersampling the majority class.
-   **Synthetic Data Generation**: Using methods like SMOTE to generate synthetic examples.
-   **Using Appropriate Evaluation Metrics**: Focusing on precision, recall, or F1 score instead of accuracy.

#### What is time series analysis?

Time series analysis involves analyzing data points collected or recorded at specific time intervals to identify trends, cycles, and seasonal variations for forecasting and other purposes.

#### How do you deal with date and time data in Pandas?

Pandas provides the `to_datetime()` function to convert strings to datetime objects, and you can use datetime properties and methods to manipulate date and time data.

```python
df['date_column'] = pd.to_datetime(df['date_column'])
```

#### What is autocorrelation in time series data?

Autocorrelation is the correlation of a signal with a delayed copy of itself. In time series data, it measures the relationship between a variable's current value and its past values.

#### How do you perform forecasting using ARIMA models?

Using the `statsmodels` library:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(time_series_data, order=(p, d, q))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
```

#### What is natural language processing (NLP)?

NLP is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language, enabling machines to understand, interpret, and generate human language.

#### How do you perform text preprocessing in NLP?

Common steps include:

-   **Tokenization**: Splitting text into words or sentences.
-   **Stop Word Removal**: Removing common words that add little meaning.
-   **Stemming/Lemmatization**: Reducing words to their base or root form.
-   **Encoding**: Converting text to numerical representation using methods like TF-IDF or word embeddings.

#### What is TF-IDF?

Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic that reflects how important a word is to a document in a collection. It balances the frequency of a word in a document with how common the word is across all documents.

#### How do you handle large datasets that don't fit into memory?

Techniques include:

-   **Data Sampling**: Working with a subset of the data.
-   **Batch Processing**: Processing data in chunks.
-   **Distributed Computing**: Using tools like Apache Spark.
-   **Out-of-core Learning**: Using algorithms that can learn from data incrementally.

#### What is a pipeline in scikit-learn?

A pipeline is a sequence of data processing steps assembled into a single object. It ensures that all steps are applied consistently during training and testing, simplifying the workflow.

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
```

#### How do you save and load trained models in scikit-learn?

Using the `joblib` library:

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')
```

#### What is gradient descent?

Gradient descent is an optimization algorithm used to minimize the cost function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient.

#### What is the difference between batch, stochastic, and mini-batch gradient descent?

-   **Batch Gradient Descent**: Uses the entire dataset to compute gradients.
-   **Stochastic Gradient Descent (SGD)**: Uses one sample at a time.
-   **Mini-Batch Gradient Descent**: Uses a small subset (batch) of the data.

#### How do you handle multicollinearity in regression analysis?

Techniques include:

-   **Removing correlated features**.
-   **Principal Component Analysis (PCA)** to reduce dimensionality.
-   **Regularization methods** like Ridge regression.

#### What is the Central Limit Theorem?

The Central Limit Theorem states that the sampling distribution of the sample means approaches a normal distribution as the sample size becomes large, regardless of the population's distribution.

#### What is hypothesis testing?

Hypothesis testing is a statistical method used to decide whether there is enough evidence to reject a null hypothesis in favor of an alternative hypothesis based on sample data.

#### What is p-value?

A p-value is the probability of observing results at least as extreme as those measured when the null hypothesis is true. A low p-value indicates that the observed data is unlikely under the null hypothesis.

#### How do you perform a t-test in Python?

Using `scipy.stats`:

```python
from scipy import stats

t_statistic, p_value = stats.ttest_ind(sample1, sample2)
```

#### What is the difference between Type I and Type II errors?

-   **Type I Error**: Rejecting a true null hypothesis (false positive).
-   **Type II Error**: Failing to reject a false null hypothesis (false negative).

#### What is an ANOVA test?

Analysis of Variance (ANOVA) is a statistical method used to compare means across three or more groups to see if at least one mean is different from the others.

#### How do you perform an ANOVA test in Python?

Using `scipy.stats`:

```python
from scipy import stats

f_statistic, p_value = stats.f_oneway(group1, group2, group3)
```

#### What is bootstrapping in statistics?

Bootstrapping is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement. It allows estimation of the sampling distribution of almost any statistic.

#### What is the law of large numbers?

The law of large numbers states that as the number of trials increases, the sample mean will converge to the expected value (population mean).

#### What is a probability distribution?

A probability distribution describes how the values of a random variable are distributed. It defines the probabilities of different outcomes.

#### What are common probability distributions used in data science?

-   **Normal Distribution**
-   **Binomial Distribution**
-   **Poisson Distribution**
-   **Exponential Distribution**

#### How do you generate random numbers following a normal distribution in NumPy?

```python
import numpy as np

random_numbers = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
```

#### What is the curse of dimensionality?

The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces, often leading to challenges like overfitting and increased computational cost.

#### How do you reduce dimensionality?

Techniques include:

-   **Feature Selection**: Choosing a subset of relevant features.
-   **Feature Extraction**: Transforming data into a lower-dimensional space (e.g., PCA).

#### What is the difference between bagging and boosting?

-   **Bagging**: Combines multiple models (usually of the same type) trained on different subsets of the data to reduce variance.
-   **Boosting**: Sequentially trains models, where each new model focuses on correcting errors made by previous ones, reducing bias.

#### What is a decision tree?

A decision tree is a flowchart-like structure used for classification and regression that splits data into branches based on feature values to make predictions.

#### How do you prevent a decision tree from overfitting?

By:

-   **Pruning**: Removing branches that have little power in predicting target variables.
-   **Setting a maximum depth**: Limiting the depth of the tree.
-   **Setting a minimum number of samples per leaf**.

#### What is random forest?

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.

#### How do you implement a random forest classifier in scikit-learn?

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

#### What is ensemble learning?

Ensemble learning combines predictions from multiple machine learning algorithms to produce a more accurate prediction than any individual model.

#### What is a neural network?

A neural network is a computational model inspired by the human brain, consisting of layers of interconnected nodes (neurons) that can learn complex patterns through training.

#### What is the difference between deep learning and machine learning?

-   **Machine Learning**: Involves algorithms that parse data, learn from it, and make decisions.
-   **Deep Learning**: A subset of machine learning using neural networks with multiple layers to model complex patterns.

#### How do you handle class imbalance in classification problems?

By:

-   **Resampling the dataset**: Oversampling the minority class or undersampling the majority class.
-   **Using appropriate evaluation metrics**: Such as ROC AUC, precision-recall curve.
-   **Using algorithms that handle imbalance**: Like XGBoost.

#### What is the purpose of the `map()` function in Pandas?

The `map()` function is used to map values of a Series according to an input mapping or function, useful for substituting values or applying a function element-wise.

#### How do you sort a DataFrame in Pandas?

Using the `sort_values()` function:

```python
sorted_df = df.sort_values(by='column_name', ascending=True)
```

#### What is the difference between `apply()` and `applymap()` in Pandas?

-   **`apply()`**: Applies a function along an axis of the DataFrame (rows or columns).
-   **`applymap()`**: Applies a function element-wise to the entire DataFrame.

#### How do you remove duplicates from a DataFrame?

Using the `drop_duplicates()` function:

```python
df_unique = df.drop_duplicates()
```

#### What is an outlier, and how do you detect them?

An outlier is a data point significantly different from others. Detection methods include:

-   **Statistical methods**: Using Z-scores or IQR.
-   **Visualization**: Box plots or scatter plots.

#### How do you handle outliers in data?

By:

-   **Removing them**: If they are errors.
-   **Transforming data**: Using log or square root transformations.
-   **Using robust algorithms**: That are less sensitive to outliers.

#### What is data normalization?

Data normalization scales numerical data into a specific range, typically \[0,1\], ensuring that each feature contributes equally to the analysis.

#### How do you perform data normalization in scikit-learn?

Using `MinMaxScaler`:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
```

#### What is a heatmap, and when do you use it?

A heatmap is a graphical representation of data where individual values are represented as colors. It is used to visualize correlation matrices or to display patterns in data.

#### How do you create a heatmap in Seaborn?

```python
import seaborn as sns

sns.heatmap(data, annot=True)
```

#### What is a pairplot in Seaborn?

A pairplot creates a grid of Axes such that each variable in the data is shared across the y-axes across a single row and the x-axes across a single column, visualizing pairwise relationships.

```python
sns.pairplot(df)
```

#### How do you interpret a correlation coefficient?

A correlation coefficient measures the strength and direction of a linear relationship between two variables. Values range from -1 to 1:

-   **\-1**: Perfect negative correlation.
-   **0**: No correlation.
-   **1**: Perfect positive correlation.

#### What is the purpose of the `describe()` function in Pandas?

The `describe()` function generates descriptive statistics, including count, mean, standard deviation, min, max, and quartile values for numerical columns.

#### How do you handle date/time features for machine learning models?

By extracting meaningful components:

-   **Year, Month, Day**
-   **Weekday**
-   **Hour, Minute, Second**
-   **Time since a specific date**

#### What is the difference between `.loc` and `.iloc` in Pandas?

-   **`.loc`**: Label-based indexing to select data by row and column labels.
-   **`.iloc`**: Integer-based indexing to select data by row and column positions.

#### How do you rename columns in a DataFrame?

Using the `rename()` function:

```python
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

#### What is the purpose of the `astype()` function in Pandas?

The `astype()` function is used to change the data type of a Series or DataFrame columns.

```python
df['column_name'] = df['column_name'].astype('float')
```

#### How do you detect missing values in a DataFrame?

Using `isnull()` or `isna()` functions:

```python
missing_values = df.isnull().sum()
```

#### What is an ensemble method in machine learning?

An ensemble method combines predictions from multiple machine learning models to improve performance over a single model. Examples include Random Forest, Gradient Boosting.

#### How do you implement Gradient Boosting in scikit-learn?

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
```

#### What is XGBoost?

XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting library designed to be highly efficient, flexible, and portable, widely used for its performance and speed.

#### How do you install and use XGBoost in Python?

Install using pip:

```
bash
pip install xgboost
```

Use in code:

```python
import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
```

#### What is cross-entropy loss?

Cross-entropy loss measures the performance of a classification model whose output is a probability between 0 and 1. It increases as the predicted probability diverges from the actual label.

#### How do you calculate the learning rate in gradient descent?

The learning rate is a hyperparameter that you set manually. It determines the step size at each iteration while moving toward a minimum of a loss function.

#### What is the difference between epochs, batches, and iterations?

-   **Epoch**: One complete pass through the entire training dataset.
-   **Batch**: A subset of the training data used in one iteration.
-   **Iteration**: One update of the model's parameters.

#### How do you perform hyperparameter tuning?

By:

-   **Grid Search**: Exhaustively searching through a specified subset of hyperparameters.
-   **Random Search**: Randomly sampling hyperparameter combinations.
-   **Bayesian Optimization**: Using probabilistic models to select hyperparameters.

#### What is the purpose of the `pipeline` module in scikit-learn?

It allows you to chain preprocessing steps and estimators in a sequential manner, ensuring consistent application of transformations during training and testing.

#### How do you evaluate a clustering algorithm?

Using metrics like:

-   **Silhouette Score**
-   **Calinski-Harabasz Index**
-   **Davies-Bouldin Index**

#### What is a dummy variable trap?

The dummy variable trap occurs when multicollinearity is introduced in a regression model due to the inclusion of dummy variables that are linearly dependent. It can be avoided by dropping one dummy variable.

#### How do you create a correlation matrix in Pandas?

```python
corr_matrix = df.corr()
```

#### What is an ROC curve, and how do you plot it?

An ROC (Receiver Operating Characteristic) curve plots the true positive rate against the false positive rate at various threshold settings. You can plot it using scikit-learn:

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
plt.plot(fpr, tpr)
```

#### What is a type I error?

A type I error occurs when the null hypothesis is true, but we incorrectly reject it (false positive).

#### What is a type II error?

A type II error occurs when the null hypothesis is false, but we fail to reject it (false negative).

#### How do you calculate the p-value in a hypothesis test?

Using statistical tests from libraries like `scipy.stats`, which return the p-value as part of the output.

#### What is the difference between parametric and non-parametric tests?

-   **Parametric Tests**: Assume underlying statistical distributions (e.g., t-test).
-   **Non-Parametric Tests**: Do not assume any specific distribution (e.g., Mann-Whitney U test).

#### How do you perform a chi-squared test in Python?

Using `scipy.stats`:

```python
from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(observed_values)
```

#### What is the purpose of the `seaborn` library?

Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.

#### How do you create a box plot in Seaborn?

```python
import seaborn as sns

sns.boxplot(x='categorical_column', y='numerical_column', data=df)
```

#### What is a violin plot?

A violin plot combines a box plot and a kernel density plot to provide a richer depiction of the data distribution.

#### How do you handle multivariate time series data?

By:

-   **Using models designed for multivariate data**: Like VAR (Vector Autoregression).
-   **Feature engineering**: Creating lag features for each variable.

#### What is an A/B test?

An A/B test is an experiment comparing two variants (A and B) to determine which one performs better regarding a specific metric.

#### How do you analyze A/B test results?

By:

-   **Calculating the difference in metrics between groups**.
-   **Performing statistical tests**: Like t-tests or chi-squared tests.
-   **Checking for statistical significance**: Using p-values and confidence intervals.

#### What is the Bonferroni correction?

A method to adjust p-values when multiple comparisons are made to reduce the chances of obtaining false-positive results (Type I errors).

#### What is survivorship bias?

Survivorship bias occurs when analyses are conducted only on surviving subjects, leading to skewed results due to the overlooking of those that did not survive or were not included.

#### What is data leakage?

Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates that won't generalize to new data.

#### How do you prevent data leakage?

By:

-   **Separating training and testing data properly**.
-   **Performing data preprocessing steps within cross-validation**.
-   **Avoiding using future data in model training**.

#### What is a hash table?

A hash table is a data structure that implements an associative array, mapping keys to values using a hash function to compute an index into an array of buckets.

#### What is memoization?

Memoization is an optimization technique used primarily to speed up computer programs by storing the results of expensive function calls and returning the cached result when the same inputs occur again.

#### How do you reverse a linked list?

By iterating through the list and reversing the pointers of each node to point to the previous node.

#### What is the time complexity of quicksort?

The average time complexity is **O(n log n)**, while the worst-case is **O(n^2)**.

#### What is a generator in Python?

A generator is a special type of function that returns an iterator object which can iterate over a sequence of values. It uses the `yield` keyword.

```python
def my_generator():
    yield value
```

#### How do you handle exceptions in Python?

Using try-except blocks:

```python
try:
    # Code that may raise an exception
except ExceptionType as e:
    # Code to handle the exception
```

#### What is a decorator in Python?

A decorator is a function that modifies the behavior of another function or method. It allows for the addition of functionality to existing code in a modular way.

```python
def decorator_function(func):
    def wrapper():
        # Code before function call
        func()
        # Code after function call
    return wrapper
```

#### How do you read and write JSON files in Python?

Using the `json` module:

```python
import json

# Read JSON
with open('file.json', 'r') as f:
    data = json.load(f)

# Write JSON
with open('file.json', 'w') as f:
    json.dump(data, f)
```

#### What is multithreading, and how do you implement it in Python?

Multithreading allows concurrent execution of threads (lightweight processes) to improve performance. In Python, you can use the `threading` module:

```python
import threading

def function_to_run():
    pass

thread = threading.Thread(target=function_to_run)
thread.start()
```

#### What is the Global Interpreter Lock (GIL) in Python?

The GIL is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes at once. It simplifies memory management but can limit performance in CPU-bound multi-threaded programs.

#### How do you handle file operations in Python?

Using `open()`:

```python
with open('file.txt', 'r') as file:
    content = file.read()
```

#### What are lambda functions in Python?

Lambda functions are anonymous functions defined using the `lambda` keyword, useful for short, simple functions.

```python
add = lambda x, y: x + y
```

#### How do you work with databases in Python?

By using database connectors and ORMs like:

-   **SQLite**: Using `sqlite3` module.
-   **MySQL**: Using `mysql-connector-python`.
-   **ORMs**: Using libraries like SQLAlchemy.

#### What is the purpose of virtual environments in Python?

Virtual environments allow you to create isolated Python environments with specific packages and dependencies, preventing conflicts between projects.

```
bash
python -m venv myenv
```

#### How do you install packages in Python?

Using `pip`:

```
bash
pip install package_name
```

#### What are the common data types in Python?

-   **Numeric Types**: int, float, complex
-   **Sequence Types**: list, tuple, range
-   **Text Type**: str
-   **Mapping Type**: dict
-   **Set Types**: set, frozenset
-   **Boolean Type**: bool

#### How do you create a class in Python?

```python
class MyClass:
    def __init__(self, attribute):
        self.attribute = attribute
```

#### What is inheritance in Python?

Inheritance allows a class (child) to inherit attributes and methods from another class (parent), promoting code reusability.

```python
class ChildClass(ParentClass):
    pass
```

#### What is polymorphism in Python?

Polymorphism allows methods to have the same name but behave differently in different classes. It enables methods to be used interchangeably.

o1