# Health-Insurance-Charges-Analysis
Analysis and prediction of health insurance charges using exploratory data analysis, linear regression, and random forest models. Includes data preprocessing, feature engineering, and performance evaluation.

## Project Overview
This project aims to analyze a health insurance dataset to predict insurance charges based on various factors such as age, sex, BMI, number of children, smoking status, and region. We perform exploratory data analysis (EDA) to understand the data distribution and relationships between the variables. We also build predictive models using linear regression and random forest algorithms to estimate insurance charges.

## Dataset Description
The dataset contains the following columns:

* age: Age of the primary beneficiary
  
* sex: Gender of the primary beneficiary (male/female)

* bmi: Body mass index, a measure of body fat based on height and weight

* children: Number of children covered by the insurance

* smoker: Smoking status of the primary beneficiary (yes/no)

* region: Residential area in the US (northeast, northwest, southeast, southwest)

* charges: Individual medical costs billed by health insurance


## Project Structure

The project is structured as follows:

* data/: Contains the dataset used for analysis.

* notebooks/: Contains the Jupyter Notebook with the EDA and modeling.

* README.md: Project overview and instructions.


## Installation and Setup

To run this project, you'll need to have Python installed along with the following libraries:

* pandas

* numpy

* matplotlib

* seaborn

* scikit-learn

You can install these libraries using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn

```
## Usage
1. Clone the repository:
```
git clone https://github.com/vikramnigam/Health-Insurance-Charges-Analysis/blob/main/Medical_Project.ipynb
cd health-insurance-charges-analysis

```

2. Open the Jupyter Notebook:
```
jupyter notebook notebooks/health_insurance_charges_analysis.ipynb
```
3. Follow the steps in the notebook to perform EDA and build predictive models.

   
## Exploratory Data Analysis (EDA)

We perform EDA to understand the data distribution and relationships between the variables. This includes:

* Descriptive statistics

* Data visualization using histograms, scatter plots, box plots, and correlation matrices
* Data Preprocessing
* To handle skewness in the charges column, we apply a log transformation:

```
import numpy as np

# Applying log transformation to the 'charges' column
data['log_charges'] = np.log(data['charges'])
```

# Predictive Modeling

## Linear Regression

We build and evaluate a linear regression model to predict insurance charges:

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Splitting the data
X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['log_charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = linear_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
```
# Random Forest

We build and evaluate a random forest model to predict insurance charges:

```
from sklearn.ensemble import RandomForestRegressor

# Training the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicting and evaluating
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f'Mean Absolute Error: {mae_rf}')
```

# Results
## Linear Regression Model: MAE = 0.11
## Random Forest Model: MAE = 0.09

The random forest model outperforms the linear regression model, achieving a lower mean absolute error.

# Conclusion
Through EDA and predictive modeling, we gained valuable insights into the factors affecting health insurance charges and developed effective models to predict these charges. The random forest model provides more accurate predictions compared to the linear regression model. Future work could involve exploring additional features or advanced modeling techniques to further improve prediction accuracy.
