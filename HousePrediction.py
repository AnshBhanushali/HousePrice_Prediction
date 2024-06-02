import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("housing.csv")

# Display the first few rows of the dataset
print(data.head())

# Display the information about the dataset
print(data.info())

# Drop rows with missing values
data.dropna(inplace=True)

# Display the information after dropping missing values
data.info()

from sklearn.model_selection import train_test_split

# Separate features and target variable
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Combine training features and target variable for analysis
train_data = x_train.join(y_train)

# Display the combined training data
print(train_data.head())

# Plot histograms for the training data
train_data.hist(figsize=(15, 8))

# Identify non-numeric columns
non_numeric_columns = train_data.select_dtypes(include=['object', 'category']).columns

# Drop non-numeric columns
train_data = train_data.drop(columns=non_numeric_columns)

# Display the correlation matrix
train_data_corr = train_data.corr()
print(train_data_corr)

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(15, 8))
sns.heatmap(train_data_corr, annot=True, cmap='YlGnBu')

# Apply logarithmic transformation to certain columns
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# Plot histograms after transformation
train_data.hist(figsize=(15, 8))

# Create additional features
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

# Plot a heatmap with the new features
plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')

# Encode categorical variables
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# Ensure both train and test sets have the same dummy variables
x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Prepare the test data similarly to train data
test_data = x_test.join(y_test)
test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)
test_data = test_data.join(pd.get_dummies(test_data.households)).drop(['median_house_value'], axis=1)

# Create additional features for the test data
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

# Initialize and train the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(x_train_scaled, y_train)

# Evaluate the model on the test set
print(forest.score(x_test_scaled, y_test))

from sklearn.model_selection import GridSearchCV

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    "n_estimators": [3, 10, 30],
    "max_features": [2, 4, 6, 8]
}

# Initialize the Random Forest Regressor
forest = RandomForestRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(forest, param_grid, cv=5, 
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

# Fit the grid search
grid_search.fit(x_train_scaled, y_train)
