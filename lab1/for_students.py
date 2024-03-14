import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]
y_new = y_train.reshape(-1)
x_new = np.c_[np.ones((len(x_train),1)),x_train]
x_new_transposed = x_new.T
x_inverted = np.linalg.inv(np.matmul(x_new_transposed,x_new))
x_multiply_2 = np.matmul(x_new_transposed,y_new)
theta = np.matmul(x_inverted,x_multiply_2)
theta = theta.reshape(-1)
theta_best[0] = theta[0]
theta_best[1] = theta[1]
# TODO: calculate error

def calculate_error(x_value, y_value):
    m = len(x_value)
    y_exp = theta[0] + theta[1]*x_value
    undersum = (y_exp - y_value)**2
    mse = (1/m)*np.sum(undersum)
    return mse

print("\nClosed-forms solution") 
print("MSE dla y_train: " + str(calculate_error(x_train, y_train)))
print("MSE dla y_test: " + str(calculate_error(x_test, y_test)))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)
y_test_mean = np.mean(y_test)
y_test_std = np.std(y_test)

x_train_standardized = (x_train - x_train_mean) / x_train_std
y_train_standardized = (y_train - y_train_mean) / y_train_std


# TODO: calculate theta using Batch Gradient Descent
learning_rate = 0.01  
iterations = 1000  

def gradient(theta,x,y):
    m = len(x)
    x_made = np.c_[np.ones((len(x),1)),x]
    x_made_transposed = x_made.T
    gradient = ((2/m)*np.matmul(x_made_transposed,((np.matmul(x_made,theta))-y)))
    print(gradient)
    return gradient

m = len(x_train_standardized)
gradientTheta = [np.random.random(), np.random.random()]

for _ in range(iterations):
    gradientTheta = gradientTheta - learning_rate*gradient(gradientTheta,x_train_standardized,y_train_standardized)

theta_best = gradientTheta




# TODO: calculate error
x_test_standardized = (x_test - x_train_mean) / x_train_std
y_test_standardized = (y_test - y_train_mean) / y_train_std

y_predicted_standartized = theta_best[0]+x_test_standardized*theta_best[1]
y_test_predicted =[]
y_test_predicted = y_predicted_standartized*y_train_std+y_train_mean

mse = (1/len(x_test)) * np.sum((y_test_predicted-y_test)**2)

print("\nBatch Gradient Descent:") 
print(mse)

# plot the regression line
x = np.linspace(min(x_test_standardized), max(x_test_standardized), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test_standardized, y_test_standardized)
plt.xlabel('Weight (scaled)')
plt.ylabel('MPG (scaled)')
plt.show()