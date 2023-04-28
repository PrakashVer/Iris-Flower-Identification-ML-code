# A machine learning model to predict whether a flower is  0  - Iris-Setosa ,1 - Iris-Versicolour, 2 - Iris-Virginica
# on basis of  sepal length in cm ,sepal width in cm, petal length in cm, petal width in cm

# loading requires modules 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris  = datasets.load_iris() #Loading Dataset
            

#Printing features and labels
features = iris.data
labels = iris.target

print(iris.DESCR)  # Describes whole data set

print(features[0],labels[0])  

#Training the classifier

clf = KNeighborsClassifier()   # cammand- which type of classifier we are using ie KNeighborsClassifier()
#  me jo bhi set of rules likhe hue hai un rules k hisab se isne ek model me usko train kiya jiska naam hai clf
clf.fit(features, labels)   # classifier fits into data @trainig 

preds = clf.predict([[9.1, 9.5, 6.4, 0.2]])  # predict which flower it is in the form of 2d array
print(preds)

        # - sepal length in cm
        # - sepal width in cm
        # - petal length in cm
        # - petal width in cm
        # - class:
            #    0  - Iris-Setosa
            #    1 - Iris-Versicolour
            #    2 - Iris-Virginica


