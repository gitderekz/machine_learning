import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv('sample_data/student-mat.csv')
# print(data.head())
data = data[['G1','G2','G3','studytime','failures','absences']]
# print(f'{data.head()}\nEND OF PRINTING TREAMED DATASET')

predict = 'G3' #label
x = np.array(data.drop([predict],axis=1)) #features/attributes (training data)
y = np.array(data[predict]) #label
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

best = 0.94
for _ in range(30):
  x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

  # #training model
  # linear = linear_model.LinearRegression()
  # linear.fit(x_train,y_train)
  # accuracy = linear.score(x_test,y_test)
  # print(f'ACCURACY: {accuracy}')
  #
  # if accuracy > best:
  #   best = accuracy
  #   print(f'BEST: {best}')
  #   #storing model
  #   with open('studentmodel.pickle','wb') as f:
  #     pickle.dump(linear,f)
  #     print('IMEFANIKIWA')


#opening stored model
pickle_in = open('studentmodel.pickle','rb')
linear = pickle.load(pickle_in)
#
# print(f'COEFFICIENTS: {linear.coef_}')
# print(f'INTERCEPT: {linear.intercept_}\n')
# #demonstrating prediction, attributes and label
# predictions = linear.predict(x_test)
# for x in range(len(predictions)):
#   print(predictions[x],x_test[x],y_test[x])

#PLOTING
feature = 'absences' #ONE OF THE ATTRIBUTE/FEATURE
style.use('ggplot')
pyplot.scatter(data[feature],data[predict])
pyplot.xlabel(feature)
pyplot.ylabel('Final Grade')
pyplot.show()

print(f'\n')