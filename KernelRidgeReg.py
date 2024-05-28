import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

hw4reg = pickle.load(open('hw4reg.pkl', 'rb'))
x_train = hw4reg['data']
y_train = hw4reg['labels']
x_val = hw4reg['valdata']
y_val = hw4reg['vallabels']
x_test = hw4reg['testdata']
y_test = hw4reg['testlabels']

x_train = x_train[:, np.newaxis]
x_val = x_val[:, np.newaxis]
x_test = x_test[:, np.newaxis]

def kernel_function(x, z):
    return np.minimum(x, z)

lambda_values = [2**i for i in range(-20, 21)]
val_errors = {}

for lambda_val in lambda_values:
    model = KernelRidge(kernel=kernel_function, alpha=lambda_val)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)
    val_errors[mse] = lambda_val

lowest_mse = min(val_errors.keys())
best_lambda = val_errors[lowest_mse]

print(best_lambda)

def predict(x_data):
    model = KernelRidge(kernel=kernel_function, alpha=best_lambda)
    model.fit(x_train, y_train)
    return model.predict(x_data)

test_pred = predict(x_test)
test_mse = mean_squared_error(y_test, test_pred)
print(test_mse)


pred = predict(hw4reg['grid'][:, np.newaxis])
plt.figure()
plt.plot(hw4reg['data'], hw4reg['labels'], '.')
plt.plot(hw4reg['grid'], pred, linestyle='-')
plt.legend(['training data', '$f(x)$'], loc='lower left')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('hw4reg.pdf', bbox_inches='tight')