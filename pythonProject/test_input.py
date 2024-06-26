import numpy as np
import sklearn
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv",sep=";")
# print(data.head())
data = data[["G1","G2","G3"]]
# print(data.head())
predict = "G3"
x = np.array(data.drop([predict], axis=1))
# x = np.array([[5,  6,  2,  0,  6]]) #[[ 5  6  2  0  6],[ 5  5  2  0  4],[ 7  8  2  3 10]],...,[10  8  1  3  3],[11 12  1  0  0],[ 8  9  1  0  5]]
# print(x)
y = np.array(data[predict])
# y = np.array([5]) #[ 6  6 10 15 10 15 11  6 19 15  9 12 14 11 16 14 14 10  5 10 15 15 16 12]
# print(y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
# best = 0
# for count in range(100):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     if accuracy > best:
#         print("Accuracy: ",accuracy)
#         best = accuracy
#         with open("studentinputmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studentinputmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("coefficients:",linear.coef_)
print("Intercepts: ",linear.intercept_)
x_test = [[15, 21]]
y_test = [18]
predictions = linear.predict(x_test)
print("Prediction: ",predictions)

for x in range(len(predictions)):
    print(predictions[x],x_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()