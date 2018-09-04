import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Salary_Data.csv")
df.head(5)
df.shape

df.describe()

# Scatter plot of all the data points included in the dataset
plt.scatter(df['YearsExperience'], df['Salary'], color="red")
plt.title("Salary based on Years of Experience")
plt.xlabel("Experience(Years)")
plt.ylabel("Salary")
plt.show()
# calculate the correlation between the feature and the target
df.corr()

# Separating the feature and the target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  #20% pour test, 80% pour taining

# Fitting simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualize the data points in teh tarining set and the fitting line
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary based on Years of Experience")
plt.xlabel("Experience(Years)")
plt.ylabel("Salary")
plt.show()

# Prediction of the values in the test set
y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color="red")
plt.scatter(X_test, y_pred, color="green")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary based on Years of Experience")
plt.xlabel("Experience(Years)")
plt.ylabel("Salary")
plt.show()

