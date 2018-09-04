import pandas as pd
import numpy as np

# Get Data
data = pd.read_csv('Iris_data.csv', header=None)
header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data.columns = header

# data exploration
print data.head(5)
print data.corr()



# splitting feature and label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# splitting train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print "Train X",  len(X_train)
print "Test X",  len(X_test)
print "Train y",  len(y_train)
print "Test y ", len(y_test)

#classification :logistic
from sklearn.svm import  SVC
classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

for i in range(0, 29):
    print y_test[i], y_pred[i]

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print "confusion matrix"
print cm

# Accuracy
print "accuracy on training set:", classifier.score(X_train, y_train)
print "accuracy on test set:    ", classifier.score(X_test, y_test)


print list(labelencoder.classes_)
print list(labelencoder.transform(list(labelencoder.classes_)))


#Visualisation

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])
X_set, y_set = X_train, y_train

x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


# Plot training points
plt.scatter(X_set[:, 0], X_set[:, 1], s=50, c=y, cmap=cmap_bold, edgecolor = 'black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

patch0 = mpatches.Patch(color='#FF0000', label='apple')
patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
patch2 = mpatches.Patch(color='#0000FF', label='orange')
patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
plt.legend(handles=[patch0, patch1, patch2, patch3])

plt.xlabel('height (cm)')
plt.ylabel('width (cm)')
plt.title("Logistic regression ")
plt.show()


"""
#Visualisation
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.figure()
plt.pcolormesh(X1, X2, classifier.predict(np.array([X_train, y_train]).T).reshape(X1.shape),
              cmap=ListedColormap(('red', 'green', "blue")))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)


plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""