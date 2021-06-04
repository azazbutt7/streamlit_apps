import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import r2_score

# read the data and load it into memory
uk_data = pd.read_csv('energy_dataset.csv', low_memory=False)

# print(uk_data['time']).dtype

# uk_data.shape

# viewing the features
# uk_data.columns

# viewing the first 3 rows of data
# uk_data.head(3)

# uk_data.info()

"""This data information tells is that:
* We have 25 floating point values, 3 integet type and 1 string value
* We have null and NaN values in our data
* Total of 35064 entries, and 29 columns
"""

# loading the data again, this time with parsing data and time column
uk_data = pd.read_csv('energy_dataset.csv',
                      parse_dates=['time'], low_memory=False)

# vieweing the datetim column
# uk_data['time'].head(3)

# sorting the time format with ascending order
uk_data.sort_values(by=['time'], inplace=True, ascending=True)
# uk_data['time'].head(3)

# float_dtypes = [i for i in uk_data.columns if uk_data[i].dtype == float]
# float_dtypes

MPE = np.mean(np.abs((uk_data['total load actual'] - uk_data['total load forecast']) / uk_data['total load actual']))
MPE = MPE * 100
# print('Total Mean Absolute Percentage Error: {:.2f}'.format(MPE))

# making a copy of original dataframe
data_copy = uk_data.copy()
# data_copy.shape

# checking null values present in each column
data_copy.isnull().sum().to_numpy()

# data_copy['time'][:4]

data_copy = data_copy.drop(['generation hydro pumped storage aggregated',
                            'forecast wind offshore eday ahead'], axis=1)
# data_copy.shape

# making a new feature timestamp by converting the time feature to datetime
data_copy['timestamp'] = pd.to_datetime(data_copy['time'], utc=True)
data_copy = data_copy.drop('time', axis=1)

# data_copy.head(3)

# sorting values of datetime ascedingly
data_copy.sort_values(by=['timestamp'], inplace=True, ascending=True)
# data_copy['timestamp'].head(3)


# (data_copy['timestamp']).dtype
time = data_copy['timestamp'][:1].dt.day
# time

# creating new features in our data
data_copy['Year'] = data_copy['timestamp'].dt.year
data_copy['Month'] = data_copy['timestamp'].dt.month
data_copy['Day'] = data_copy['timestamp'].dt.day
data_copy['DayOfWeek'] = data_copy['timestamp'].dt.dayofweek
data_copy['DayOfYear'] = data_copy['timestamp'].dt.dayofyear
data_copy['HourofDay'] = data_copy['timestamp'].dt.hour

data_copy.head(3)

# dropping the timestamp column
data_copy.drop("timestamp", axis=1, inplace=True)

# data_copy.columns


# finding each column with a string in it
'''
for label, content in data_copy.items():
    if pd.api.types.is_string_dtype(content):
        print(label)
'''

# again checking the null values
data_copy.isnull().sum().to_numpy()

# dividing the sum of all null values in our data to th total number of columns
data_copy.isnull().sum() / len(data_copy)

# saving the cleaned data to a new file
data_copy.to_csv('cleaned_data.csv')

# loading the cleaned data in our memory
data_cleaned = pd.read_csv('cleaned_data.csv')
# data_cleaned.shape

"""We now want to see if there is some numerical values in our data. The code below does this for us:"""

'''
# printing just numerical values in our data
for label, content in data_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)
'''

'''
# checking for which column in our data, the numeric value is missing
for label, content in data_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
'''

# filling the missing numerical rows with median of numbers
for label, content in data_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Assign a binary column which will tell us if a column has null values
            data_copy[label + "_is_missing:"] = pd.isnull(content)
            # fill the missing numerical values with median
            data_copy[label] = content.fillna(content.median())

"""We have filled all missing values in a column with median of that column"""

'''
# checking for which column in our data, the numeric value is missing
for label, content in data_copy.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

# checking for missing categorical values
for label, contents in data_copy.items():
    if not pd.api.types.is_numeric_dtype(contents):
        print(label)
'''

'''
# turn categorical values into numbers
for label, content in data_copy.items():
    if not pd.api.types.is_numeric_dtype(content):
        # add a binary indicator to check for missing values
        data_copy[label + "_is_missing"] = pd.isnull(content)
        # turn categorical things into numbers
        data_copy[label] = pd.Categorical(content).codes + 1
'''

# data_copy.info()

# data_copy.isna().sum()[:100]

# dropping columns with null values
# data_copy = data_copy.drop(['generation hydro pumped storage aggregated'], axis=1)

# data_copy.isna().sum()[:100]

data_copy = data_copy.drop(['generation biomass_is_missing:',
                            'generation fossil brown coal/lignite_is_missing:',
                            'generation fossil coal-derived gas_is_missing:',
                            'generation fossil gas_is_missing:',
                            'generation fossil hard coal_is_missing:',
                            'generation fossil oil_is_missing:',
                            'generation fossil oil shale_is_missing:',
                            'generation fossil peat_is_missing:',
                            'generation hydro run-of-river and poundage_is_missing:',
                            'generation hydro water reservoir_is_missing:',
                            'generation marine_is_missing:', 'generation nuclear_is_missing:',
                            'generation other_is_missing:',
                            'generation other renewable_is_missing:',
                            'generation solar_is_missing:', 'generation waste_is_missing:',
                            'generation wind offshore_is_missing:',
                            'generation wind onshore_is_missing:'], axis=1)

# data_copy.head(3)

# data_copy.isna().sum()[:100]

"""Till this point, all data we have is cleaned, and ready for modelling.

## Data Cleaned!
At this point, we have cleaned our data. This means that we have done following things:
* Removed all missing values
* Removed all null values
* Features have been engineered
* Redundant Features have been removed
"""

# saving the final cleaned data to a .csv file
data_cleaned.to_csv('final_cleaned_data.csv')

"""## Checking the Model on Full Data
Now, we will check the performance of our model on full data, of how well does it perform. Let's see how does it work
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Instantiate the ML Model
model = RandomForestRegressor(n_jobs=-1, random_state=42)
# 
# fit your model
model.fit(data_copy.drop("total load actual", axis=1), data_copy["total load actual"])

# checking the accuracy of model on overall data
model.score(data_copy.drop("total load actual", axis=1), data_copy["total load actual"])

"""We have got **99%** accuracy on our overall data using **Random Forest Regression**. But we  will have to split our data into training & testing sets to see if our model works better on unseen data.

## Selecting X and Y variables
"""

X = data_copy.drop('total load actual', axis=1)
y = data_copy['total load actual']
# X, y


# data split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# checking the shape of splits
# X_train.shape, X_val.shape, y_train.shape, y_val.shape

def rmsle(y_test, y_pred):
    '''
  A evaluation function, calculated root mean squared log error between predictions 
  and actual values
  '''
    return np.sqrt(MSLE(y_test, y_pred))


# create a custom function to evaluate model on various levels
def show_score(model):
    train_predictions = model.predict(X_train)
    val_preds = model.predict(X_val)

    scores = {"Training MAE": MAE(y_train, train_predictions),
              "Validation MAE": MAE(y_val, val_preds),
              "Training RMSLE": rmsle(y_train, train_predictions),
              "Validation RMSLE": rmsle(y_val, val_preds),
              "Training R^2": r2_score(y_train, train_predictions),
              "Validation R^2": r2_score(y_val, val_preds),
              "Training accuracy": model.score(X_train, y_train),
              "Validation accuracy": model.score(X_val, y_val)}
    return scores


# len(X_train), len(y_train)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
acc_rr_test = model.score(X_test, y_test)
acc_rr_train = model.score(X_train, y_train)
# print('Accuracy of Random Forest on Train:', (acc_rr_train))
# print('Accuracy of Random Forest on Test: {:.2f}'.format(acc_rr_test))

# making predictions
print('Here are the predictions for the total cost for power generation:')
print('\n')

predictions = pd.DataFrame({'Actual Cost (USD)': y_test, 'Predicted Cost (USD)': y_pred})
print(predictions.head(10))
predictions.to_csv('test predictions.csv')

# viweing random forest predictions (power consumption has been predicted)
predictions.head(5)

"""### b) Support Vector Machines (SVM)"""

'''
# instantiate SVM
model_svm = SVR()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # fit on training data and make predictions
model_svm.fit(X_train, y_train)
y_pred_svm = model.predict(X_test)

# using a pre-defined function to calculate model scores for SVM
show_score(model_svm)

# making predictions
# y_pred_svm = model.predict(X_test)
acc_svm_test = model_svm.score(X_test, y_test)
acc_svm_train = model_svm.score(X_train, y_train)
print('Accuracy of SVM on Training Data:', acc_svm_train)
print('Accuracy of SVM on Test Data:', model_svm.score(X_test, y_test))

"""## Evaluation Visualization
Now, we will visualizw the results of SVM and Random Forest using a Bar Plot
"""

fig, axes = plt.subplots()
models = ['SVM', 'Random Forest Regression']
scores_test = [acc_svm_test, acc_rr_test]
scores_train = [acc_svm_train, acc_rr_train]
plt.title("Testing Accuracy of SVM and Random Forest")
axes.bar(models, scores_test)

plt.title("Training Accuracy of SVM and Random Forest")
plt.bar(models, scores_train)
'''
"""## **Conclusion**
After fitting both models and calculating results, we can deduct following things:
* **Random Forest Regression** is best at predicting unbalanced, huge and multivariate datastes
* Since random forest uses lots of **Decision Tree Algorithms** (100 by default) and combines them to make a result, this algorithm is better
* **Support Vector Regression** did not perform well on data, so the recommendation for making a big regression and time series models is Random Forest
"""
