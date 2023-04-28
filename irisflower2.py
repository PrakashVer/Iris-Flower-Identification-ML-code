# Training a logistic regression classifier to predict whether a flower is iris virginica or not :
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
print(list(iris.keys()))
# print(iris['data'].shape) # matlab order of matrix
# print(iris['target'])
# print(iris['DESCR'])
# print(iris["data"])

# Now lets store our data and target in variables :
X = iris["data"][:, 3:]  # @ slicing is used to get 3rd column only ie sepal width
# print(X)

y = (iris["target"] == 2).astype(np.int64)  # is line ka matlab ki agar flower virginica ho to true k rup me 1 or 0 as false 
# print(y)

# Train a logistic regression classifier
clf = LogisticRegression() # isko lana prega from sklearn.linear_model import LogisticRegression
clf.fit(X,y)
example = clf.predict(([[2.6]]))
print(example)

# Using matplotlib to plot the visualization :
X_new = np.linspace(0,3,1000).reshape(-1,1)  # linspace 0 aur 3 k beech 1000 points de dega aur shape change kr dega like an 1D array matlab petal width  ki 1000 values leli hai 0-3 k beech ki
# y probability ki value 
y_prob = clf.predict_proba(X_new)  
print(y_prob)
plt.plot(X_new, y_prob[:,1], "g-", label="virginica")
plt.show()





