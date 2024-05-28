import pickle
import numpy as np
click = pickle.load(open('hw5click.pkl', 'rb'))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def negative_log_likelihood(w, b, x, y):
    z = np.dot(x, w) + b
    return -np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))

def gradient(w, b, x, y):
    z = np.dot(x, w) + b
    error = sigmoid(z) - y
    grad_w = -np.dot(x.T, error)
    grad_b = -np.sum(error)
    return grad_w, grad_b

x_train, y_train = click['data'], click['labels']
w_mle, b_mle = click['w_mle'], click['b_mle']

print(w_mle)
print(b_mle)

J_mle = negative_log_likelihood(w_mle, b_mle, x_train, y_train)
grad_w_mle, grad_b_mle = gradient(w_mle, b_mle, x_train, y_train)
norm_grad_mle = np.linalg.norm(np.concatenate([grad_w_mle, [grad_b_mle]]))
print(f"Euclidean Norm: {norm_grad_mle}")
x_test, y_test = click['testdata'], click['testlabels']

def logistic_regression_classifier(x, w, b):
    return np.where(np.dot(x, w) + b > 0, 1, 0)

train_predictions = logistic_regression_classifier(x_train, w_mle, b_mle)
test_predictions = logistic_regression_classifier(x_test, w_mle, b_mle)

train_error_rate = np.mean(train_predictions != y_train)
test_error_rate = np.mean(test_predictions != y_test)

print(f"Training Error Rate: {train_error_rate}")
print(f"Test Error Rate: {test_error_rate}")

training_click_rate = np.mean(y_train==1)
test_click_rate = np.mean(y_test==1)
print(f"Training Click Rate: {training_click_rate}")
print(f"Test Click Rate: {test_click_rate}")