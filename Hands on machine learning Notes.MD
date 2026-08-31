# Hands on machine learning notes

## Chapter 2

### Frame the problem

Building a model is probably not the end goal, you must know how the company expect to use and benefit from this model.

### Pipelines

A sequence of data processing components. It is very common to use it as there is a lot of data to manipulate and many data transformation to apply.  

Components typically run asynchronously. Each component pulls in a large amount of data, processes it, and spits out the result in another data store. Then, sometime later, the next component in the pipeline pulls in this data and spits out its own output. Each component is fairly self-contained: the interface between components is simply the data store.  

The pipeline is a list of steps: each step is a name + estimator pair.
Only the last step (here model) can be a predictor (like a classifier or regressor).  
The earlier steps must be transformers (objects with .fit() and .transform() methods).  

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

```

Pipelines are used to make the code cleaner, avoid repeating preprocessing, no data leakage in cross-validation and it is easier to debug and main.

We will need to determine the kind of the supervision the model need: supervised, unsupervised, semi-supervised, self-supervised, or reinforcement learning task? And is it a classification task, a regression task, or something else? Should you use batch learning or online learning techniques?
Note:  if data were huge, you could either split your batch learning wok across multiple servers using MapReduce technique or use an online learning technique.

### Data leakage

Data leakage occurs when information from outside the training dataset is inadvertently used to train the model, leading to unrealistically high performance (over fitting) that won’t hold in real-world scenarios

#### Types of data leakage

1. Train – Test contamination:  
    You accidentally allow test data to influence training

    ```python
    #  BAD PRACTICE:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scales the ENTIRE dataset

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    ```

    Scaling before splitting is classic and dangerous example of Train – Test contamination.

    - The StandardScaler() calculates mean and standard deviation using the entire dataset including the test data.
    - That means information from the test set influences the training process.
    - Now the model’s evaluation on x-test is not truly “unseen” data anymore.  

2. Leaky features:
    Include features that are too predictive because they indirectly or directly contain the target (or future info).

Example: predicting someone will default on loan using:

- “Loan paid off” clearly a leaky feature
- “Next payment amount” in dataset where you are predicting default status

| Issue                         | Consequence                         |
| ----------------------------- | ----------------------------------- |
| Model sees future information | Unrealistically high accuracy       |
| Test data indirectly seen     | Model will fail in real-world use   |
| Trust in model is lost        | Business or clinical decisions fail |

#### How to prevent data leakage

1) Use train_test_split() early: never transform or peek at full data before splitting.
2) Use pipelines: They fit only on training and transform test correctly.
3) Avoid leaky features: Don’t include columns that contain info about the target.
4) Time-aware spitting (if needed): Use time-based validation if your data has a timeline.
5) Use cross-validation safely: Ensure each fold only sees data from its fold’s training set.

Simple rule to remember:

- "Train on training data. Only test on truly unseen data."
- If the model seems too good to be true it probably has leakage

### Select a performance measure

Using the next step is to select a performance measure. For regression problems is the root mean square error (RMSE).  
If there are many outliers you may consider using the mean absolute error (MAE).  
Both the RMSE and the MAE are ways to measure the distance between two vectors, the vector of predictions and the vector of target values.  

Computing the root of sum of squares (RMSE) is corresponds to the Euclidean norm.  
Computing the root of sum of absolutes (MAE) corresponds to the Manhattan norm.  

### Check the assumptions

It is good practice to list and verify the assumptions that have been made so far this can help you catch serious issues early on.  

### Scikit-Learn Design

Scikit learns API is well designed and these are the main design principles

#### Consistency

All objects share a consistent and simple interface:

**Estimators:** Any object that can estimate some parameters based on a dataset is called an estimator (ex `SimpleImputer`). The estimation is performed by the `fit()` method and it takes a dataset as a parameter or two in supervised learning algorithms where the second dataset contains the labels.  
The parameters that needed to guide the estimation process is considered a hyperparameter (ex `SimpleImputer’s strategy`) and it must set as an instance variable

**Transformers:** Some estimators (ex `SimpleImputer`) can also transform a dataset these are called transformers. The transformation also performed by the `transform()` method and it returns the transformed dataset. All transformers also have a convenance method called `fit_transform()` which is the equivalent to calling fit then transform but sometimes `fit_transform()` is optimized and runs much faster

**Predictors:** Some estimators are capable of making predictions; they called predictors (ex `LinearRegression` model). A predictor has a `predict()` method that take dataset of new instances and returns a dataset of corresponding predictions.

#### Inspection

All the estimator’s hyperparameters are accessible directly via public instance variables (ex: `imputer.strategy`), and all estimator’s  learned parameters are accessible via public via public instance variables with an underscore suffix (ex `imputer.strategy_`).

#### Nonproliferation of classes

Datasets are represented as NumPy or SciPy sparse matrices, instead of homemade classes.  
Hyperparameters meters are just regular python strings or numbers

#### Composition

Existing building blocks are reused as much as possible. For example, it is easy to create a pipeline estimator from an arbitrary sequence of transformers followed by a final estimator.

#### Sensible defaults

Scikit-Learn provides reasonable default values for most parameters, making it easy to quickly create a baseline working system.

### stages of ML model development

#### Get the data

Getting data can be done from public datasets sites like:

- Kaggle.
- UCI machine learning repository.
- Hugging face datasets.
- Government open-data portals.
- Google Dataset search.
- Dataset repositories from universities/research organizations.

APIs:  
Instead of downloading a static dataset, you can collect data programmatically from an API>

For example:

- Weather API → weather prediction dataset.
- Financial API → market data.
- Reddit/API → text classification data.
- Government API → demographic/economic data.

Web scraping:  
You collect publicly available information from websites and turn it into structured data.  
Typical workflow: `Website → scraper → raw data → cleaning → dataset`

#### Explore the data

Data Exploration, also known as Exploratory Data Analysis (EDA), is the process of examining your dataset before applying any machine learning algorithms. The goal is to understand the data, detect patterns, spot anomalies, check assumptions, and summarize the dataset’s main characteristics using statistical and visualization techniques.

Steps:

1) Loading the Data.

2) Understanding the Structure.

3) Descriptive Statistics.

4) Detecting Missing Values.

5) Detecting Outliers.

6) Data Visualization.

7) Analyzing Relationships Between Variables.

8) Data Quality Checks.

#### Prepare the data

Data preparation is oftentimes the most time-consuming step of the modeling process. It is also one of the most important with model accuracy often contingent on the quality of data inserted. To this end, we'll be applying the following transformations to this data.

Note:
samples: rows of data 
features: columns of data.

**Handling Missing Values:**  
When some features have missing data, you have 3 options:

- Drop the rows with missing values.
- Drop the entire feature (column) if too many values are missing.
- Fill (impute) the missing values (Imputation): Inserting values where they might be missing to have a complete dataset:
  - Using a specific value (like 0, or the mean/median/mode of the column).
  - Using more complex methods (later, with models).

In the book, it is recommended to fill missing values using the median because it is robust to outliers.

Since the median can only be calculated on numerical attributes so you need to create a copy of the data with only the numerical attributes.

One of the complex methods is iterative imputer which trains a regression model per feature to predict the missing values based again on the updated data, and repeats the process several times, improving the models and the replacement values as each iteration.

**Fixing Text and Categorical Attributes:**  
Some features are text (categorical) — you can't feed them directly to a machine learning model.

Need to encode them:

- Ordinal encoding: If the categories have an order (e.g., small < medium < large → 0, 1, 2).
- One-hot encoding: If the categories have no natural order (e.g., "cat", "dog", "rabbit" → binary columns).
- Label encoding: assigning an integer to each category.

Label encoder usually for target labels only (y-values) while ordinal encoding and one-hot encoding are used for input features(X-values)

It’s better to use label encoding when you have 2 classes and then one hot encoding when you have more than that

The book uses `OneHotEncoder` from Scikit-Learn.

You need to use `fit_transform()` the first time you apply the encoder to learn the structure of the data (i.e., the categories), not just transform it.

**Feature Scaling:**  
After cleaning missing and categorical data, numerical features often need to be scaled:

- Standardization: shift data to have mean = 0 and std = 1.
- Min-Max scaling: rescale data to a fixed range (often [0, 1]).

The book recommends using StandardScaler (standardization) most of the time, unless a specific reason exists to use min-max scaling  
i.e.  
We apply scaling using `scaler.fit()` then scaler.transform() or through `scaler.fit_transform()`.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = hot_encoded_data.copy()
scaled_data_temp = scaled_data.iloc[:, -1]#take the last column into a temporary variable
scaled_data = scaled_data.iloc[:, :-1]#take all columns except the last one
scaled_data = scaler.fit_transform(scaled_data)
```

`fit()`: Learns the parameters (mean and standard deviation) from the data X. But does NOT transform the data.

`transform()`: standardize each feature (column) of your data.  
It also will standardize each feature (column) of your data by:  

$$
xscaled   = (x – μ) / (σ x)
$$

where: 

- x is a value from the original dataset
- μ is the mean of the column (learned during `fit()`)
- σ is the standard deviation of the column (also learned)
- `fit_transform` Learns from the data (same as `fit()`), and immediately applies the transformation.

i.e.  
`fit()`: learns the mapping (categories → columns) we use it on training data only.

`transform()`: uses the learned mapping to convert new data we use it on validation or test data.

`fit_transform()`: Combines both: learn + transform on training data only

Note: we apply `fit()` once per project.

**Feature Engineering:**  
Creating new features (columns) out of our raw data using domain knowledge.  
For example: from `total_rooms` and `households`, you could create `rooms_per_household`.  

**Stratified Sampling for Fair Splitting:**  
When a feature (like income) has a skewed distribution, it's important to split the data so that train/test sets reflect this distribution.

Stratified sampling ensures each subset has similar proportions of each category.

```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
```

#### Select and train model

After data gathering and exploring and create preprocessing pipeline to automate the cleanup and preparing the data now it is time for select and train the machine learning model.

First, we choose a model ex linear regression model.

```python
from sklearn.linear_model import LinearRegression
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
```

One of evaluating ways is to take prediction results done by the model then comparing them to the labels using RMSE (Root mean square error)

```python
housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2) # -2 = rounded to the nearest hundred
#output: array([243700., 372400., 128800., 94400., 328300.])

housing_labels.iloc[:5].values
#output: array([458300., 483800., 101700., 96100., 361800.])
```

If the evaluation score is not good enough, we can change the model into better one like decision tree regression.

```python
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
```

We can apply evaluation on the model on the training data set

```python
housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions,
     squared=False)
tree_rmse
```

The result of the evaluation is no error which mean the model is perfect (which is not true) or the model has badly overfit the data, an RMSE of 0.0 suggests the model has overfit — it simply memorized the training data and cannot generalize to new/unseen data.

To get a true estimate of how your model performs on unseen data Cross validation is used.

```python
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(
    tree_reg,
    housing,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
```

We use - because Scikit-learn uses negative scores for loss metrics by convention.

```python
import pandas as pd
pd.Series(tree_rmses).describe()

```

#### Fine-tune the system

After having a shortlist of promising models, you start fine-tune them in order to achieve the best accuracy and performance.  

One option is to fiddle with the hyperparameters manually until finding a great combination of hyperparameter values. But this takes a lot of work and time

**Grid search:**  
Instead, we can use grid search class from scikit-learn (`GridSearchCV`), which takes list of hyperparameters values and will use cross-validation to evaluate all the possible combinations of the values and output the best combinations.

**Randomized search:**  
`RandomizedSearchCV` is more preferable than grid search especially when the hyperparameter search space is large.

It can be used in much the same way of the `GridSearchCV` but instead of trying all possible combination it evaluates a fixed number of combinations, selecting a random value of each hyperparameter at every iteration.

**Ensemble methods**  
Another way to fine-tune the system is to combine the models that perform best, the group / ensemble will often perform better than the best individual model ex random forests perform better than the individual decision trees.

#### Present and launch your solution

To get the final solution ready production you need to polish the code, write documentation (test, and so on) then deploy the model to the production environment.  
This done by save the best trained model, transfer the file to the production  environment, and load it.

The model can be saved using `joblib` library

```python
import joblib
joblib.dump(final_model, "my_california_housing_model.pkl")
```

**Lunch the model:**  
Once the model is transferred to production it can be load and used

The production model can be

- Exposed as an API (e.g., with Flask or FastAPI).
- Integrated it into a larger system (web app, backend, etc.).
- Running it in the cloud (AWS, Azure, GCP).

**Monitor the model:**  
After deploying the model, it must be monitored regularly as the real-world data changes (users change behavior, economic or health conditions shift or sensor and data sources evolve)

This causes data drift or concept drift where the input distribution (features) changes and the relationship between the inputs and labels changes

Creating backup for every version of the model and every dataset is very helpful

**Maintain and update the model:**  
A model in production needs maintenance as any software

The model needs to be retrained regularly and use fresh data to update and improve accuracy

The model can have automated lifecycle using ML pipelines (like MLflow, Kubreflow) to collect new date, retrain the model, evaluate the new version and deploy automatically if it improves performance.  
This called MLOps (machine learning operations)
