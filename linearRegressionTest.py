from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import Sets as s
import DataHandler as dh

fmnist_set, class_name = dh.load_data() #load the fashion mnist dataset
linear_model = LinearRegression() #declare a linear model 
linear_model.fit(fmnist_set.train.x, fmnist_set.train.y) #fit the training set (X = n_samples ,Y = n_outputs)
y_prediction = linear_model.predict(fmnist_set.test.x) #predict Y from the test set 

# The coefficients
print('Coefficients: \n', linear_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(fmnist_set.test.x, y_prediction))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(fmnist_set.test.y, y_prediction))

# Plot outputs
plt.scatter(fmnist_set.test.x, fmnist_set.test.y,  color='black')
plt.plot(fmnist_set.test.x, y_prediction, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

