
# Small Machine Learning Projects

[![krasvachev](https://img.shields.io/badge/GitHub-krasvachev-181717.svg?style=flat&logo=github)](https://github.com/krasvachev)
![Static Badge](https://img.shields.io/badge/machine-learning-red)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pandas)](https://www.python.org/downloads/release/python-31212/)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/) 


Welcome to the Small Machine Learning Repo. Here, small Machine Learning (ML) models are implemented. Linear Regression, Random Forest, XGBoost and K-Means are just a portion of them. The models solve various tasks in a wide variety of areas. A Medical Cost Dataset is tackled with the goal of estimating the annual medical expenditures of every new customer. A rider's taxi fare is predicted in New York City via a Gradient Boosting Regressor. And more...

The Repo is a good starting point for every beginner who takes their first steps in the ML field. In addition to the ML models implementation, the most common practices for data analysis and pre-processing are overviewed. The inspiration for the projects is the [Jovian ML with Python and Scikit-learn Beginner Course](https://www.youtube.com/watch?v=hDKCxebp88A). There, you could find a detailed explanation for some of the implemented models. Most of the datasets in the projects are taken from the Kaggle platform. It's worth trying to enrol in one of the competitions while you practice the ML models.



## Tools I Used


* **Python** - The backbone of the projects. The analyses, the data preparation and pre-processing, the model implementation, the visualisations and insights, all of them are done with Python Libraries. The most important of them are:
  * **Pandas** - Analyse and Prepare the Data
  * **Numpy** - Process the Data
  * **Scikit-learn** - Pre-process the Data and Develop ML Models 
  * **Matplotlib** - Visualise the Data
  * **Seaborn** - Visualise the Data
  * **Plotly** - Visualise the Data via Interactive Plots
  * **Xgboost** - Implement Gradient Boosting Algorithm
* **Jupyter Notebook** - The platform where the code was written
## The Notebooks

>The notebook for each ML model contains five main sections:
>
>1. Import and Process the Data
>
>2. Analyze the Dataset
>
>3. Train and Test Split
>
>4. Pre-processing
>
>5. ML model

The notebook for each ML model contains five main sections:

1. [Import and Process the Data](#Import%and%Process%the%Data)

2. Analyze the Dataset

3. Train and Test Split

4. Pre-processing

5. ML model

There are also bonus sections for some of the models. Two of them should be mentioned. The first bonus section is the Feature Engineering. Feature engineering was implemented to improve the model's performance. The second section is the Base models. Sometimes, base models were developed as a starting point for the analysis.

Below, a brief overview will be given for the five main sections.

### Import and Process the Data

At the beginning of each ML project, the key libraries are imported. Afterwards, the file with the data is read and the data is prepared for further analysis. A code snippet from **Logistic Regression Notebook** can be seen below:

```python
import os
import pandas as pd

data_dir = "/content/weather-dataset-rattle-package"
train_csv = data_dir + "/weatherAUS.csv"

raw_df = pd.read_csv(train_csv)
```

### Analyze and Visualize the Data

The next main section is the analysis of the data. In this section, the data features and the relation between them are explored. Different figures are created to explore the relations between the features of the data. On the basis of the data insights captured from the figures, a strategy is developed on how to tackle the ML problem. Let's take two examples from the notebooks. 

**Figure 1**

In the first figure, the relation between the rainy and non-rainy days is depicted. The data is taken from the Rain in Australia dataset and the **Logistic Regression Notebook Link it**. It shows the distribution of the positive and negative classes.

**Figure 2**

In the second figure, a `scatter plot` is given with the targets of the ML model (sales) and the number of customers. 

### Train and Test Split
A crucial step in every ML project is to divide the data into a train and test set. Most of the time, the train set is further split into a train and a validation set. For example, the Rain in Australia dataset is divided into three sets before it is passed to the Logistic Regression model:

```python
from sklearn.model_selection import train_test_split

temp_df, test_df = train_test_split(raw_df, test_size = 0.2, random_state = 42)
train_df, val_df = train_test_split(temp_df, test_size = 0.25, random_state = 42)

train_df.shape, val_df.shape, test_df.shape
```

### Pre-processing
There is another important step in the ML model development - pre-processing the data. The data is manipulated in order to make it understandable for the ML algorithm. In most of the projects in the repo, imputing, scaling and encoding are applied. 

Imputation is the process of adding missing values to the data. The following code illustrates implementation for the Decision Tree Model:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "mean").fit(raw_df[numeric_cols])
```
```python
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])
```

The goal of the scaling process is to normalise the values of the data into a certain numerical range. An example of scaling the data for the Decision Tree is provided in the code:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(raw_df[numeric_cols])
```
```python
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
```

Encoding the data gives us the opportunity to transform the categorical data into numerical:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output = False, handle_unknown = "ignore")
encoder.fit(raw_df[categorical_cols])
```

### ML Model
As mentioned above, dozens of ML models were implemented. In `scikit-learn`, the model is assigned to an object. Then it is trained with the `fit()` method. In the end, the model makes predictions via `predict()` and it is evaluated with various metrics. The code that shows this process for the Linear Regression is:

```python
from sklearn.linear_model import LinearRegression

# Create a model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
test_preds = model.predict(X_test)

# Evaluation of the model
loss = loss_rmse(y_test, test_preds)
print(f"Test Loss: {loss}")
```
## Contributing

Contributions are always welcome!




## License

[MIT](https://choosealicense.com/licenses/mit/)

