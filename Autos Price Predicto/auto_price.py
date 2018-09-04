import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn


from mypack.datawrangling import attribute_to_list

# Getting data
path = "autos.csv"
file_headers = "attributes.csv"
df = pd.read_csv(path, names=attribute_to_list(file_headers, False))
df.replace("?", np.nan, inplace=True)


# Force type as numerical , not object
df['normalized-losses'] = df['normalized-losses'].astype('float64', errors="ignore")
df['bore'] = df['bore'].astype('float64', errors="ignore")
df['stroke'] = df['stroke'].astype('float64', errors="ignore")
df['horsepower'] = df['horsepower'].astype('float64', errors="ignore")
df['peak-rpm'] = df['peak-rpm'].astype('float64', errors="ignore")
df['price'] = df['price'].astype('float64', errors="ignore")

df.to_csv("auto_new.csv", index=None)
# Exploration : data description
print "\nData Type :: \n", df.dtypes
print "\nFirts rows :: \n", df.head(4)
print "\nLast rows :: \n",  df.tail(4)
print "\nData Infos :: \n", df.info()
print "\nDescriptive analysis :: \n", df.describe(include="all")
print "\nNumber of missing values :: \n", df.apply(lambda x: sum(x.isnull()), axis=0)

# Visu missing values
#sbn.heatmap(df.isnull(), yticklabels=False,cbar=False, linecolor="blue" ,linewidths=.01, cmap='cubehelix') #linewidths=.2,
#plt.show()



# Missing values
df['num-of-doors'].fillna(df['num-of-doors'].mode()[0], inplace=True)
df['normalized-losses'].fillna(df['normalized-losses'].mean(), inplace=True)
dfg = df.groupby('fuel-type')['normalized-losses'].mean()
print "dfg:: \n", dfg
print df['normalized-losses'].mean()
df['bore'].fillna(df['bore'].mean(), inplace=True)
df['stroke'].fillna(df['stroke'].mean(), inplace=True)
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
df['peak-rpm'].fillna(df['peak-rpm'].mean(), inplace=True)
df['price'].fillna(df['price'].mean(), inplace=True)


print "\nDescriptive analysis :: \n", df.describe(include="all")
print "\nNumber of missing values :: \n", df.apply(lambda x: sum(x.isnull()), axis=0)

# Visu missing values
#sbn.heatmap(df.isnull(), yticklabels=False,cbar=False, linecolor="blue" ,linewidths=.01, cmap='cubehelix') #linewidths=.2,
#plt.show()

print df.corr()

# Pair plot
#sbn.pairplot(df)
#plt.show()





# ------- Model -----------
#Define X and Y

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
dependent_var = list(df.columns.values)[:-1]


print "Dependant variables: ", dependent_var

#Categoriacl Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
categorical_labels = list()
for index in range(2,9):
    X[:, index] = labelencoder_X.fit_transform(X[:, index])
    print "labels:", list(labelencoder_X.classes_)#list(labelencoder_X.transform(list(labelencoder_X.classes_)))
    if len(list(labelencoder_X.classes_))>2:
        categorical_labels.append({'pos':index, 'liste_categories': list(labelencoder_X.classes_)})


for index in 14, 15, 17:
    X[:, index] = labelencoder_X.fit_transform(X[:, index])
    print "labels:", list(labelencoder_X.classes_), list(labelencoder_X.transform(list(labelencoder_X.classes_)))
    if len(list(labelencoder_X.classes_))>2:
        categorical_labels.append({'pos':index, 'liste_categories': list(labelencoder_X.classes_)})
#print "test labels:", list(labelencoder_X.classes_), list(labelencoder_X.transform(['mfi']))

print "labels hot encoder :", categorical_labels


count = 0
for var in categorical_labels:
    index = var['pos']
    nb_col = len(X[0, :])
    onehotencoder = OneHotEncoder(categorical_features=[index + count])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    count = count + len(X[0, :]) - nb_col
    dependent_var.pop(index)
    dependent_var = var['liste_categories'][1:] + dependent_var
    print "new dependant var:: ", dependent_var

#track nom colonnes
dependent_var_ld = list()
for var in dependent_var:
    dependent_var_ld.append({'pos':dependent_var.index(var), 'variable':var})
print "Dependant variables dict: ", dependent_var_ld

# Training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# verif values
print X_train[0:5, :]
print y_train[0:5]
print len(X_train),  len(y_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaling_X = StandardScaler()
X_train = scaling_X.fit_transform(X_train)
X_test = scaling_X.transform(X_test)

from mypack.valueshandling import array1dto2d
scaling_y = StandardScaler()
y_train = scaling_X.fit_transform(array1dto2d(y_train))
y_test = scaling_X.transform(array1dto2d(y_test))


# Fitting - MLR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction
y_pred = regressor.predict(X_test)
print "Test vs Pred"
for index in range(len(y_pred)):
    print y_test[index], "vs", y_pred[index], " -- Residuals: ", y_test[index][0]- y_pred[index][0]

import matplotlib.pyplot as plt

#plt.plot(X, regressor.predict((X)), color='blue')


#Backward elimination
import statsmodels.formula.api as stats
X = np.append(arr=np.ones([len(X), 1]).astype(int), values=X, axis=1)

"""
from mypack.valueshandling import genlistnum

liste_index = genlistnum(0, len(X[0, :])-1)
X_opt = X[:, liste_index]
print type(X_opt)
slevel = 0.05
regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()
print regressor_ols.summary()
print regressor_ols.pvalues
"""
slevel = 0.05

print "nb var avant", len(dependent_var_ld)
from mypack.valueshandling import backwardelimination
liste_index = backwardelimination(X, y, slevel)
print "nb var:", len(liste_index)

for index in liste_index:
    print (item for item in dependent_var_ld if item['pos'] == index).next()

#New prediction
regressor.fit(X_train[:, liste_index], y_train)
print "score linear model",regressor.score(X_train[:, liste_index], y_train)
#print liste_index
#print len(X_test[0,:]), X_test[0,0:6]
y_pred = regressor.predict(X_test[:,liste_index])
residuals = y_test - y_pred
print "residuals::",residuals[1:3,], "\ny pred::", y_pred[1:3,]
y_pred_liste,residuals_liste = list(), list()
for id in range(len(y_pred)):
    y_pred_liste.append(y_pred[id,0])
    residuals_liste.append(residuals[id,0])

print "length::", len(y_pred_liste), len(residuals_liste)
print y_pred_liste[0:4] ,residuals_liste[0:4]
#plt.scatter( y_pred_liste[0:4] ,residuals_liste[0:4], c="red")
print "min", min(y_pred_liste), "max", max(y_pred_liste),

print "min", min(residuals_liste), "max", max(residuals_liste),
plt.scatter([0,2,4,6,8] ,[0,2,4,6,8], c="red")
plt.title("Residuals")
plt.show()


