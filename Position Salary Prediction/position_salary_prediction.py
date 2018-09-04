import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("Position_Salaries.csv")
print df.head(5)
print df.shape

print df.describe()

# Scatter plot of all the data points included in the dataset
plt.scatter(df['Level'], df['Salary'], color="red")
plt.title("Salary based on Position Level")
plt.xlabel("Level)")
plt.ylabel("Salary")
#plt.show()

# calculate the correlation between the feature and the target
print df.corr()

# Separating the feature and the target
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Polynomial Regression
# fit Linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)
print "linear reg score:",linear_reg.score(X, y)


# Visualization of linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_reg.predict(X), color='blue')
plt.title("Truth or bluff (Linear reg)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Fit polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_poly, y)
print "polynomial reg score:",linear_reg_2.score(X_poly, y)
print "coef", linear_reg_2.coef_

# Visualization of linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_reg.predict(X), color='blue')
plt.title("Truth or bluff (Linear reg)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
