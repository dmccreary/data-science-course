# List of 100 Top MicroSims

This is a list of the 100 most important MicroSims for a Data Science course according to GPT-5.
Each MicroSim is described in a level-2 header with the description, learning goals and input controls
in level 3 headers.  We will have agents generate these MicroSims one at a time.

Note that if students don't have a strong statistics background, they should review them.

## 1. Exploring Data Points

### Description

Students click to add or remove points on a 2D scatter plot, instantly seeing the effect on the overall distribution.

### Learning Goals

-   Recognize individual observations in a dataset

-   Understand x--y coordinate representation

-   See how adding/removing points changes data shape

### Input Controls

1.  **Add Point** (click on canvas)

2.  **Remove Point** (click existing point)

3.  **Clear All Points** (button)

## 2. Histogram Builder


### Description

Students adjust bin sizes to see how histograms change, revealing over-smoothing and under-smoothing effects.

### Learning Goals

-   Understand bins and frequencies

-   Relate bin size to data detail retention

### Input Controls

1.  **Bin Size Slider**

2.  **Dataset Selector** (dropdown: normal, uniform, skewed)

3.  **Toggle Grid Lines** (checkbox)

## 3. Mean and Median Explorer

### Description

Drag points along a number line to see how mean and median shift differently.

### Learning Goals

-   Differentiate mean vs. median

-   Observe robustness of median to outliers

### Input Controls

1.  **Drag Points** (mouse)

2.  **Add Outlier** (button)

3.  **Reset Points** (button)

## 4. Correlation Playground

### Description

Students drag clusters of points to adjust correlation, watching the correlation coefficient update in real-time.

### Learning Goals

-   Visualize correlation strength and direction

-   Understand positive, negative, and zero correlation

### Input Controls

1.  **Drag Cluster** (mouse)

2.  **Add Noise** (slider)

3.  **Show Best Fit Line** (toggle)

## 5. Least Squares Line Fitter

### Description

Adjust slope and intercept manually to minimize sum of squared errors, with real-time residual visualization.

### Learning Goals

-   Understand slope, intercept, and residuals

-   Experience trial-and-error fitting

### Input Controls

1.  **Slope Slider**

2.  **Intercept Slider**

3.  **Toggle Residual Squares** (checkbox)

## 6. R² Intuition Builder

### Description

Manipulate data spread around a fitted line to see how R² changes.

### Learning Goals

-   Understand coefficient of determination

-   Relate R² to model fit quality

### Input Controls

1.  **Noise Level Slider**

2.  **Number of Points Slider**

3.  **Reset Dataset** (button)

## 7. Train-Test Split Visualizer

### Description

Randomly split a dataset and see how train/test points differ in model performance.

### Learning Goals

-   Understand importance of splitting data

-   See overfitting risk when test set is too small

### Input Controls

1.  **Train/Test Ratio Slider**

2.  **Resample Dataset** (button)

3.  **Model Complexity Slider**


## 8. Cross-Validation Simulator

### Description

Animate k-fold cross-validation, showing shifting train/test subsets and aggregated scores.

### Learning Goals

-   Understand cross-validation mechanics

-   See benefits over single train-test split

### Input Controls

1.  **Number of Folds Slider**

2.  **Dataset Size Slider**

3.  **Play/Pause Animation** (button)

## 9. Overfitting vs. Underfitting Explorer

### Description

Adjust polynomial degree to see bias--variance trade-off on train vs. test errors.

### Learning Goals

-   Recognize overfitting and underfitting patterns

-   Connect complexity to generalization

### Input Controls

1.  **Polynomial Degree Slider**

2.  **Noise Level Slider**

3.  **Toggle Error Curves** (checkbox)


## 10. $2Multiple Regression Plane

### Description

Manipulate two independent variables in 3D space to see a regression plane fit to data.

### Learning Goals

-   Visualize multivariate linear regression

-   See plane adjustment with variable changes

### Input Controls

## 1.  **Rotate View** (mouse drag)

2.  **Noise Level Slider**

3.  **Add/Remove Points** (click)



## 11. Residuals Heatmap Viewer

### Description

Color-code residuals on a scatter plot to identify patterns and non-linearity.

### Learning Goals

-   Understand residual analysis

-   Detect systematic errors in model predictions

### Input Controls

1.  **Model Complexity Slider**

2.  **Toggle Residual Colors**

3.  **Noise Level Slider**



## 12. Distribution Shape Explorer

### Description

Morph between uniform, normal, skewed, and bimodal distributions.

### Learning Goals

-   Identify common data distributions

-   Understand skewness and kurtosis visually

### Input Controls

1.  **Distribution Type Selector**

2.  **Skewness Slider**

3.  **Kurtosis Slider**


## 13. Box Plot Anatomy

### Description

Interactively adjust dataset values to see quartiles, whiskers, and outliers update in real time.

### Learning Goals

-   Interpret box plot components

-   Relate box plot features to dataset properties

### Input Controls

1.  **Drag Data Points**

2.  **Add Outlier** (button)

3.  **Reset Data** (button)



## 14. Central Limit Theorem Animator

### Description

Sample repeatedly from various population distributions and watch sampling distribution approach normality.

### Learning Goals

-   Visualize the CLT in action

-   Understand why normality emerges

### Input Controls

1.  **Population Distribution Selector**

2.  **Sample Size Slider**

3.  **Number of Samples Slider**



## 15. Sampling Bias Demonstrator

### Description

Draw samples from skewed or representative datasets to see effect on mean/median estimates.

### Learning Goals

-   Recognize sampling bias

-   Relate bias to poor generalization

### Input Controls

1.  **Sampling Method Selector** (random, biased)

2.  **Sample Size Slider**

3.  **Reset Data** (button)


## 16. Hypothesis Testing Visualizer

### Description

Adjust population mean and see how p-values change for given sample statistics.

### Learning Goals

-   Understand null/alternative hypotheses

-   Interpret p-values visually

### Input Controls

1.  **Population Mean Slider**

2.  **Sample Mean Slider**

3.  **Sample Size Slider**



## 17. Confidence Interval Explorer

### Description

Show multiple sample means with confidence intervals and see coverage percentage.

### Learning Goals

-   Understand confidence interval interpretation

-   See how sample size affects interval width

### Input Controls

1.  **Confidence Level Slider**

2.  **Sample Size Slider**

3.  **Number of Samples Slider**



## 18. t-Test Simulator

### Description

Compare means of two groups with adjustable overlap and see t-statistic and p-value.

### Learning Goals

-   Perform and interpret t-tests

-   Relate group separation to statistical significance

### Input Controls

1.  **Mean Difference Slider**

2.  **Sample Size Slider**

3.  **Variance Slider**


## 19. Correlation vs. Causation Scenario Builder

### Description

Toggle between linked and independent variables with visual storytelling elements.

### Learning Goals

-   Distinguish correlation from causation

-   Recognize spurious correlations

### Input Controls

1.  **Relationship Type Selector**

2.  **Add Confounder Variable** (button)

3.  **Noise Level Slider**


## 20. Data Cleaning Sandbox

### Description

Interactively identify and fix missing values, duplicates, and inconsistencies in a small dataset.

### Learning Goals

-   Practice data cleaning operations

-   Recognize data quality issues

### Input Controls

1.  **Highlight Missing Values** (checkbox)

2.  **Fill Missing Values Method Selector**

3.  **Remove Duplicates** (button)



## 21. Missing Data Imputation Lab

### Description

Students choose different strategies to fill in missing values and compare how summaries change.

### Learning Goals

-   Explore mean, median, mode, and model-based imputation

-   See effects of imputation on dataset statistics

### Input Controls

1.  **Imputation Method Selector**

2.  **Preview Changes** (toggle)

3.  **Apply Changes** (button)



## 22. One-Hot Encoding Demonstrator

### Description

Convert categorical variables into binary columns and see the dataset shape change.

### Learning Goals

-   Understand one-hot encoding

-   Recognize dataset expansion with categorical variables

### Input Controls

1.  **Category Count Slider**

2.  **Toggle Encoding** (checkbox)

3.  **Reset Categories** (button)



## 23. Feature Scaling Visualizer

### Description

Scale features using min-max, standardization, or robust scaling, and compare results.

### Learning Goals

-   Understand scaling methods

-   Recognize scaling's impact on model training

### Input Controls

1.  **Scaling Method Selector**

2.  **Dataset Selector**

3.  **Apply Scaling** (button)



## 24. Scatter Plot Matrix Explorer

### Description

Select variables to display in an interactive scatter plot matrix.

### Learning Goals

-   Visualize pairwise relationships

-   Identify potential multicollinearity

### Input Controls

1.  **Variable Selector** (multi-select)

2.  **Highlight Correlated Pairs** (toggle)

3.  **Reset Matrix** (button)



## 25. Multicollinearity Detector

### Description

Add or remove features and see the correlation heatmap update in real time.

### Learning Goals

-   Recognize multicollinearity

-   Learn its impact on regression models

### Input Controls

1.  **Add Feature** (dropdown)

2.  **Remove Feature** (click)

3.  **Threshold Slider** for correlation warning



## 26. Gradient Descent Animation

### Description

Visualize gradient descent steps on a 3D loss surface.

### Learning Goals

-   Understand optimization paths

-   See effects of learning rate changes

### Input Controls

1.  **Learning Rate Slider**

2.  **Start Position Selector**

3.  **Play/Pause Steps** (button)



## 27. Loss Function Comparator

### Description

Compare MSE, MAE, and Huber loss on the same dataset.

### Learning Goals

-   Understand different loss functions

-   Recognize how they respond to outliers

### Input Controls

1.  **Loss Function Selector**

2.  **Add Outlier** (button)

3.  **Reset Data** (button)



## 28. Logistic Regression Probability Curve

### Description

Adjust slope and intercept to see how the logistic curve shifts and steepens.

### Learning Goals

-   Understand logistic regression shape

-   Relate slope to classification threshold sharpness

### Input Controls

1.  **Slope Slider**

2.  **Intercept Slider**

3.  **Show Decision Boundary** (toggle)



## 29. Confusion Matrix Builder

### Description

Manually adjust predictions to see confusion matrix cells update and metrics recalculate.

### Learning Goals

-   Interpret precision, recall, F1-score

-   See trade-offs in prediction thresholds

### Input Controls

1.  **Threshold Slider**

2.  **Toggle Misclassification Highlight** (checkbox)

3.  **Reset Predictions** (button)



## 30. ROC Curve Interactive Plotter

### Description

Drag threshold point along the curve to see corresponding confusion matrix metrics.

### Learning Goals

-   Understand ROC curves

-   Relate AUC to model performance

### Input Controls

1.  **Move Threshold Point** (mouse drag)

2.  **Toggle AUC Display** (checkbox)

3.  **Dataset Selector**



## 31. Precision-Recall Trade-off Tool

### Description

Visualize precision and recall lines as threshold changes, highlighting the F1-score peak.

### Learning Goals

-   Recognize trade-offs between precision and recall

-   Identify optimal balance point

### Input Controls

1.  **Threshold Slider**

2.  **Show F1 Peak** (toggle)

3.  **Reset Chart** (button)



## 32. Decision Tree Split Explorer

### Description

Select split features and thresholds to see how data partitions change.

### Learning Goals

-   Understand feature-based splitting

-   Recognize overfitting in deep trees

### Input Controls

1.  **Feature Selector**

2.  **Threshold Slider**

3.  **Add Split** (button)



## 33. Random Forest Voting Visualizer

### Description

Show predictions from individual trees and how majority vote determines the final prediction.

### Learning Goals

-   Understand ensemble voting

-   See stability from multiple models

### Input Controls

1.  **Number of Trees Slider**

2.  **Tree Depth Slider**

3.  **Noise Level Slider**



## 34. Bagging vs. Boosting Simulator


### Description

Switch between bagging and boosting to compare error reduction over iterations.

### Learning Goals

-   Contrast two ensemble methods

-   Understand impact on bias and variance

### Input Controls

1.  **Method Selector**

2.  **Number of Estimators Slider**

3.  **Learning Rate Slider** (for boosting)



## 35. k-Means Clustering Playground

### Description

Move cluster centers and see point assignments change instantly.

### Learning Goals

-   Understand k-means mechanics

-   Recognize sensitivity to initialization

### Input Controls

1.  **Number of Clusters Slider**

2.  **Drag Cluster Centers**

3.  **Reset Clusters** (button)



## 36. Elbow Method Visualizer

### Description

Generate k-means cost curve to find optimal k.

### Learning Goals

-   Apply elbow method for cluster selection

-   Interpret inertia curve

### Input Controls

1.  **Max k Slider**

2.  **Dataset Selector**

3.  **Recalculate Curve** (button)



## 37. Hierarchical Clustering Dendrogram

### Description

Cut dendrogram at different heights to form clusters.

### Learning Goals

-   Interpret dendrograms

-   Relate cut height to cluster count

### Input Controls

1.  **Cut Height Slider**

2.  **Dataset Selector**

3.  **Toggle Leaf Labels** (checkbox)



## 38. PCA Variance Explorer

### Description

Adjust number of principal components and see variance explained update.

### Learning Goals

-   Understand dimensionality reduction

-   Relate components to variance retention

### Input Controls

1.  **Number of Components Slider**

2.  **Dataset Selector**

3.  **Show Projection** (toggle)



## 39. PCA Projection Visualizer

### Description

Project high-dimensional data into 2D and explore structure.

### Learning Goals

-   Visualize principal component projections

-   Detect patterns in reduced space

### Input Controls

1.  **Rotate Projection** (mouse drag)

2.  **Highlight Class Labels** (toggle)

3.  **Reset View** (button)



## 40. Feature Importance Bar Chart

### Description

Interactively remove features and see model accuracy update.

### Learning Goals

-   Rank feature contributions

-   Recognize redundancy in predictors

### Input Controls

1.  **Remove Feature** (click bar)

2.  **Recalculate Accuracy** (button)

3.  **Reset Features** (button)


## 41. Time Series Trend Explorer

### Description

Students add or remove long-term upward or downward trends to see their effect on time series plots.

### Learning Goals

-   Recognize trends in time series data

-   Separate trend from noise visually

### Input Controls

1.  **Trend Slope Slider**

2.  **Noise Level Slider**

3.  **Reset Series** (button)


## 42. Seasonality Animator

### Description

Add seasonal patterns to time series and adjust amplitude/frequency.

### Learning Goals

-   Understand seasonality components

-   Differentiate seasonal effects from trends

### Input Controls

1.  **Amplitude Slider**

2.  **Frequency Slider**

3.  **Toggle Seasonal Component** (checkbox)



## 43. Autocorrelation Plot Builder

### Description

Interactively generate autocorrelation plots for different time series patterns.

### Learning Goals

-   Recognize autocorrelation signatures

-   Link patterns to time lags

### Input Controls

1.  **Pattern Selector** (trend, seasonal, white noise)

2.  **Series Length Slider**

3.  **Recalculate Plot** (button)



## 44. Moving Average Filter

### Description

Smooth noisy time series using different window sizes.

### Learning Goals

-   Apply moving average smoothing

-   Understand trade-off between smoothing and responsiveness

### Input Controls

1.  **Window Size Slider**

2.  **Toggle Original Series** (checkbox)

3.  **Reset Filter** (button)



## 45. Exponential Smoothing Explorer

### Description

Adjust smoothing factor to see effect on responsiveness to new data.

### Learning Goals

-   Understand exponential smoothing

-   Compare to simple moving average

### Input Controls

1.  **Smoothing Factor Slider** (0--1)

2.  **Toggle Forecast Values** (checkbox)

3.  **Reset Data** (button)



## 46. ARIMA Model Simulator

### Description

Experiment with AR, I, and MA parameters to fit simple time series.

### Learning Goals

-   Recognize ARIMA components

-   See parameter effects on forecast shape

### Input Controls

1.  **AR Order Slider**

2.  **I Order Slider**

3.  **MA Order Slider**



## 47. Train-Test Split for Time Series

### Description

Split data chronologically and compare forecasting performance.

### Learning Goals

-   Understand why random splits don't work in time series

-   Practice chronological evaluation

### Input Controls

1.  **Split Point Slider**

2.  **Model Selector**

3.  **Show Forecast Horizon** (toggle)



## 48. Outlier Impact on Time Series

### Description

Insert outliers into time series and see how forecasts change.

### Learning Goals

-   Recognize sensitivity to anomalies

-   Understand need for preprocessing

### Input Controls

1.  **Insert Outlier** (click point)

2.  **Outlier Magnitude Slider**

3.  **Remove Outliers** (button)



## 49. TF-IDF Text Weighting Tool

### Description

Type or paste text and see term frequencies and TF-IDF scores update live.

### Learning Goals

-   Understand term frequency weighting

-   Recognize the role of inverse document frequency

### Input Controls

1.  **Text Input Box**

2.  **Toggle Stopword Removal** (checkbox)

3.  **Recalculate Scores** (button)



## 50. Tokenization Visualizer

### Description

See text split into tokens using different tokenization rules.

### Learning Goals

-   Understand tokenization

-   Compare word vs. subword methods

### Input Controls

1.  **Text Input Box**

2.  **Tokenizer Type Selector**

3.  **Show Token IDs** (toggle)



## 51. Word Embedding Explorer

### Description

Plot word embeddings in 2D space and explore semantic similarity.

### Learning Goals

-   Understand word vector representations

-   See clusters of related words

### Input Controls

1.  **Select Word to Highlight** (dropdown)

2.  **Show Similar Words** (toggle)

3.  **Reset Embeddings** (button)



## 52. Sentiment Classification Threshold Tool

### Description

Adjust sentiment score threshold to see how classifications change.

### Learning Goals

-   Understand sentiment score distributions

-   See trade-offs in precision and recall for sentiment tasks

### Input Controls

1.  **Threshold Slider**

2.  **Show Confusion Matrix** (checkbox)

3.  **Dataset Selector**



## 53. Bag-of-Words vs. Embeddings

### Description

Switch between BoW and embedding-based representations to compare classification accuracy.

### Learning Goals

-   Contrast sparse vs. dense text features

-   Recognize embedding advantages

### Input Controls

1.  **Representation Selector**

2.  **Dataset Selector**

3.  **Recalculate Accuracy** (button)



## 54. Neural Network Layer Visualizer

### Description

Show how inputs propagate through fully connected layers with activation functions.

### Learning Goals

-   Understand forward propagation

-   Visualize activation transformations

### Input Controls

1.  **Number of Layers Slider**

2.  **Activation Function Selector**

3.  **Reset Network** (button)



## 55. Activation Function Explorer

### Description

Compare sigmoid, ReLU, and tanh shapes and outputs for input ranges.

### Learning Goals

-   Recognize activation function behaviors

-   See saturation and dead neuron effects

### Input Controls

1.  **Function Selector**

2.  **Input Range Slider**

3.  **Toggle Derivative Curve** (checkbox)



## 56. Weight Initialization Impact

### Description

Initialize neural network weights differently and observe training convergence.

### Learning Goals

-   Understand initialization strategies

-   See effect on loss curve and accuracy

### Input Controls

1.  **Initialization Method Selector**

2.  **Learning Rate Slider**

3.  **Reset Training** (button)



## 57. Learning Rate Finder


### Description

Gradually increase learning rate to see where loss diverges or minimizes fastest.

### Learning Goals

-   Tune learning rate

-   Recognize underfitting and instability from wrong rates

### Input Controls

1.  **Start Rate Slider**

2.  **End Rate Slider**

3.  **Run LR Finder** (button)



## 58. Convolution Filter Visualizer

### Description

Apply filters to images and see resulting feature maps.

### Learning Goals

-   Understand convolution in CNNs

-   Recognize edge and texture detection

### Input Controls

1.  **Filter Type Selector**

2.  **Kernel Size Slider**

3.  **Toggle Original Image** (checkbox)



## 59. Pooling Layer Explorer

### Description

Compare max pooling and average pooling effects on feature maps.

### Learning Goals

-   Understand pooling

-   See dimensionality reduction effects

### Input Controls

1.  **Pooling Type Selector**

2.  **Pool Size Slider**

3.  **Toggle Feature Map Overlay** (checkbox)



## 60. Overfitting in Deep Networks

### Description

Increase network capacity and watch training vs. validation loss diverge.

### Learning Goals

-   Recognize overfitting in neural nets

-   See regularization benefits

### Input Controls

1.  **Number of Neurons Slider**

2.  **Dropout Rate Slider**

3.  **Toggle Validation Curve** (checkbox)


## 61. L1 vs. L2 Regularization Visualizer

### Description

Toggle between L1 and L2 regularization and see coefficient shrinkage effects.

### Learning Goals

-   Understand Lasso vs. Ridge regression

-   See how regularization affects weights

### Input Controls

1.  **Regularization Type Selector**

2.  **Penalty Strength Slider**

3.  **Reset Coefficients** (button)



## 62. Dropout Effect Simulator

### Description

Adjust dropout rates and watch neuron activations disappear during training.

### Learning Goals

-   Understand dropout regularization

-   Recognize its role in preventing overfitting

### Input Controls

1.  **Dropout Rate Slider**

2.  **Toggle Training/Inference View** (checkbox)

3.  **Reset Network** (button)



## 63. Early Stopping Demonstrator

### Description

Visualize training and validation loss to determine optimal stop point.

### Learning Goals

-   Understand early stopping criteria

-   Avoid overtraining a model

### Input Controls

1.  **Patience Slider**

2.  **Max Epochs Slider**

3.  **Toggle Loss Curves** (checkbox)



## 64. SHAP Value Explorer


### Description

Show feature contributions to individual predictions using SHAP values.

### Learning Goals

-   Interpret model predictions

-   Recognize key contributing features

### Input Controls

1.  **Select Data Point** (dropdown)

2.  **Show Positive/Negative Contributions** (toggle)

3.  **Reset View** (button)



## 65. Partial Dependence Plot Tool

### Description

Adjust a single feature and see average prediction change while holding others constant.

### Learning Goals

-   Interpret partial dependence

-   Detect feature impact trends

### Input Controls

1.  **Feature Selector**

2.  **Value Slider**

3.  **Toggle Confidence Interval** (checkbox)



## 66. Counterfactual Example Generator

### Description

Change features to flip a prediction outcome.

### Learning Goals

-   Understand counterfactual reasoning

-   Identify decision boundaries

### Input Controls

1.  **Feature Sliders**

2.  **Toggle Prediction Probability** (checkbox)

3.  **Reset Features** (button)



## 67. Bias Detection Dashboard

### Description

Compare model accuracy across demographic subgroups.

### Learning Goals

-   Detect model bias

-   Understand fairness metrics

### Input Controls

1.  **Group Selector**

2.  **Metric Selector**

3.  **Show Disparity Alert** (checkbox)



## 68. Fairness Metric Comparator

### Description

Compare demographic parity, equalized odds, and other fairness metrics.

### Learning Goals

-   Interpret multiple fairness definitions

-   Recognize trade-offs between them

### Input Controls

1.  **Metric Selector**

2.  **Group Selector**

3.  **Highlight Best Metric** (checkbox)



## 69. Adversarial Example Creator

### Description

Add small perturbations to input data and see if predictions change.

### Learning Goals

-   Understand adversarial vulnerability

-   Recognize security risks in ML

### Input Controls

1.  **Perturbation Magnitude Slider**

2.  **Noise Pattern Selector**

3.  **Reset Data** (button)



## 70. Model Drift Monitor


### Description

Compare live data predictions to historical model performance.

### Learning Goals

-   Detect concept and data drift

-   Understand retraining triggers

### Input Controls

1.  **Time Window Selector**

2.  **Drift Metric Selector**

3.  **Refresh Data** (button)



## 71. Hyperparameter Search Playground

### Description

Run grid/random searches and compare performance heatmaps.

### Learning Goals

-   Understand hyperparameter optimization

-   Interpret search results

### Input Controls

1.  **Search Type Selector**

2.  **Parameter Range Sliders**

3.  **Run Search** (button)



## 72. Model Stacking Visualizer


### Description

Show predictions from multiple base models and meta-learner output.

### Learning Goals

-   Understand stacking ensembles

-   See diversity benefits

### Input Controls

1.  **Base Model Selector**

2.  **Meta-Learner Selector**

3.  **Toggle Base Predictions** (checkbox)



## 73. Pipeline Builder


### Description

Chain preprocessing and modeling steps interactively.

### Learning Goals

-   Understand ML pipelines

-   Ensure reproducible workflows

### Input Controls

1.  **Add Step** (dropdown)

2.  **Remove Step** (click)

3.  **Run Pipeline** (button)



## 74. Model Export and Import Simulator

### Description

Save and reload trained models to demonstrate persistence.

### Learning Goals

-   Understand model serialization

-   Practice deployment readiness

### Input Controls

1.  **Save Model** (button)

2.  **Load Model** (button)

3.  **Reset Session** (button)



## 75. API Endpoint Tester

### Description

Send requests to a mock ML API and view JSON responses.

### Learning Goals

-   Understand model serving endpoints

-   Practice request formatting

### Input Controls

1.  **Input Data Field**

2.  **Send Request** (button)

3.  **View Raw Response** (toggle)



## 76. Batch vs. Real-Time Prediction Tool

### Description

Switch between batch file processing and live API predictions.

### Learning Goals

-   Understand latency differences

-   Recognize trade-offs in deployment modes

### Input Controls

1.  **Mode Selector**

2.  **Upload Dataset** (file input)

3.  **Simulate Real-Time** (button)



## 77. Model Version Comparator

### Description

Load two model versions and compare accuracy and latency.

### Learning Goals

-   Track performance over versions

-   Make informed upgrade decisions

### Input Controls

1.  **Version Selector A**

2.  **Version Selector B**

3.  **Compare Now** (button)



## 78. A/B Testing Simulator

### Description

Split traffic between two models and track conversions.

### Learning Goals

-   Understand online experimentation

-   Interpret statistical significance

### Input Controls

1.  **Traffic Split Slider**

2.  **Run Experiment** (button)

3.  **View p-Value** (toggle)



## 79. Cost of Prediction Calculator

### Description

Estimate compute cost for different model sizes and usage levels.

### Learning Goals

-   Relate model complexity to cost

-   Make cost-aware deployment decisions

### Input Controls

1.  **Model Size Slider**

2.  **Requests per Minute Slider**

3.  **Region Selector**


## 80. Energy Efficiency Meter

### Description

Track power consumption estimates during model inference.

### Learning Goals

-   Recognize environmental impact of ML

-   Optimize for efficiency

### Input Controls

1.  **Model Type Selector**

2.  **Batch Size Slider**

3.  **Toggle Energy Display** (checkbox)


## 81. Model Interpretability Dashboard

### Description

Combine SHAP, partial dependence, and counterfactuals in one view for a selected prediction.

### Learning Goals

-   Integrate multiple interpretability methods

-   Develop storytelling skills for predictions

### Input Controls

1.  **Data Point Selector**

2.  **Interpretation Method Toggle**

3.  **Export Dashboard** (button)



## 82. LIME Explainer Tool

### Description

Generate local linear approximations for individual predictions.

### Learning Goals

-   Understand local interpretability

-   Compare to global feature importance

### Input Controls

1.  **Data Point Selector**

2.  **Number of Samples Slider**

3.  **Toggle Highlighted Features** (checkbox)



## 83. What-If Analysis Playground

### Description

Change feature values and watch prediction changes in real time.

### Learning Goals

-   Explore "what-if" scenarios

-   Understand sensitivity of predictions

### Input Controls

1.  **Feature Sliders**

2.  **Reset to Original** (button)

3.  **Show Probability Curve** (toggle)



## 84. Bias Mitigation Simulator


### Description

Apply pre-processing or in-processing bias mitigation and measure impact.

### Learning Goals

-   Evaluate fairness interventions

-   Compare accuracy before and after

### Input Controls

1.  **Mitigation Method Selector**

2.  **Target Group Selector**

3.  **Recalculate Metrics** (button)



## 85. Model Robustness Tester

### Description

Add noise, missing values, or feature shifts to test model stability.

### Learning Goals

-   Assess robustness under real-world conditions

-   Identify fragile models

### Input Controls

1.  **Noise Level Slider**

2.  **Missing Value Percentage Slider**

3.  **Feature Shift Toggle**



## 86. Ensemble Diversity Visualizer

### Description

Plot decision boundaries of ensemble members to show diversity benefits.

### Learning Goals

-   Understand why diversity improves ensembles

-   Detect overcorrelated base models

### Input Controls

1.  **Number of Models Slider**

2.  **Model Type Selector**

3.  **Toggle Overlay Boundaries** (checkbox)



## 87. Transfer Learning Feature Explorer

### Description

Load pre-trained model features and visualize them for a custom dataset.

### Learning Goals

-   Understand feature reuse

-   See adaptation benefits

### Input Controls

1.  **Pre-Trained Model Selector**

2.  **Layer Output Selector**

3.  **Toggle Feature Map Display**



## 88. Fine-Tuning Tracker

### Description

Compare base and fine-tuned model accuracy/loss curves.

### Learning Goals

-   Understand fine-tuning process

-   Evaluate improvements over baseline

### Input Controls

1.  **Learning Rate Slider**

2.  **Epoch Count Slider**

3.  **Toggle Base Model Curve** (checkbox)



## 89. Multi-Task Learning Visualizer

### Description

Train on two tasks simultaneously and track performance for each.

### Learning Goals

-   Understand shared representations

-   Recognize trade-offs in multi-task setups

### Input Controls

1.  **Task Weight Sliders**

2.  **Epoch Count Slider**

3.  **Toggle Shared Layers** (checkbox)


## 90. Attention Mechanism Explorer

### Description

Visualize attention weights for sequence-to-sequence models.

### Learning Goals

-   Understand how models focus on parts of input

-   Interpret attention heatmaps

### Input Controls

1.  **Input Sequence Field**

2.  **Highlight Attention Matrix** (checkbox)

3.  **Reset Example** (button)


## 91. Transformer Architecture Flow

### Description

Step through encoder and decoder layers with visual activations.

### Learning Goals

-   See data flow in transformer models

-   Recognize role of each sublayer

### Input Controls

1.  **Layer Stepper** (next/prev)

2.  **Toggle Positional Encoding View**

3.  **Reset Sequence** (button)



## 92. Hyperparameter Sensitivity Map

### Description

Generate heatmaps showing accuracy changes across parameter ranges.

### Learning Goals

-   Identify sensitive parameters

-   Focus tuning efforts effectively

### Input Controls

1.  **Parameter Range Sliders**

2.  **Run Grid Search** (button)

3.  **Toggle Best Point Marker** (checkbox)


## 93. Model Compression Simulator

### Description

Prune weights and quantize parameters, tracking accuracy drop.

### Learning Goals

-   Understand trade-offs between size and performance

-   Recognize deployment benefits of smaller models

### Input Controls

1.  **Pruning Percentage Slider**

2.  **Quantization Level Selector**

3.  **Apply Compression** (button)


## 94. Edge Deployment Emulator

### Description

Simulate running a model on constrained hardware.

### Learning Goals

-   Understand latency and memory constraints

-   Optimize for edge environments

### Input Controls

1.  **Hardware Profile Selector**

2.  **Batch Size Slider**

3.  **Toggle Latency Display** (checkbox)


## 95. Streaming Data Dashboard

### Description

Stream incoming data and update predictions in real time.

### Learning Goals

-   Handle continuous inputs

-   Recognize challenges in online learning

### Input Controls

1.  **Stream Speed Slider**

2.  **Pause/Resume Stream** (button)

3.  **Reset Dashboard** (button)


## 96. Online Learning Visualizer

### Description

Update model incrementally with new data and track evolving accuracy.

### Learning Goals

-   Understand incremental training

-   Monitor performance drift

### Input Controls

1.  **Learning Rate Slider**

2.  **Batch Size Slider**

3.  **Toggle History Chart** (checkbox)


## 97. Capstone Project Data Selector

### Description

Choose dataset for final project from curated sources and preview statistics.

### Learning Goals

-   Practice dataset selection skills

-   Evaluate dataset suitability

### Input Controls

1.  **Dataset Selector**

2.  **Preview Stats** (button)

3.  **Download Data** (button)



## 98. Model Comparison Dashboard

### Description

Compare multiple models across accuracy, latency, and fairness metrics.

### Learning Goals

-   Perform multi-metric evaluation

-   Select best model for deployment

### Input Controls

1.  **Model Selector (multi-select)**

2.  **Metric Selector**

3.  **Toggle Best Model Highlight**



## 99. End-to-End Workflow Builder

### Description

Drag-and-drop stages to build a full ML pipeline from data to deployment.

### Learning Goals

-   Integrate all learned concepts

-   Visualize project workflow

### Input Controls

1.  **Stage Palette** (drag items)

2.  **Connect Stages** (mouse drag)

3.  **Run Workflow** (button)


100. Ethical Impact Assessment Tool

### Description

Rate model across transparency, fairness, privacy, and societal impact dimensions.

### Learning Goals

-   Incorporate ethics into data science projects

-   Balance technical and social factors

### Input Controls

1.  **Impact Category Sliders**

2.  **Generate Report** (button)

3.  **Reset Assessment** (button)