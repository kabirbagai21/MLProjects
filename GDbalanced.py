import numpy as np

from p6 import logistic_regression_classifier, x_train, y_train, w_mle, b_mle, x_test, y_test


def gradient_Jbal(w, b, wmle, bmle, lambda_val, data_S0, data_S1):
    n0 = len(data_S0)
    n1 = len(data_S1)

    reg_gradient_w = lambda_val * (w - wmle)
    reg_gradient_b = lambda_val * (b - bmle)

    logits_S0 = np.dot(data_S0, w)+b
    logits_S1 = np.dot(data_S1, w)+b

    logistic_S0 = 1/(1+np.exp(logits_S0))
    logistic_S1 = 1/(1+np.exp(-logits_S1))

    loss_gradient_S0 = np.dot(data_S0.T,(logistic_S0))/np.sum(logistic_S0)
    loss_gradient_S1 = np.dot(data_S1.T,(logistic_S1))/np.sum(logistic_S1)

    gradient_w = reg_gradient_w - (1 / (2 * n0)) * loss_gradient_S0 - (1 / (2 * n1)) * loss_gradient_S1
    gradient_b = reg_gradient_b - (1 / (2 * n0)) * (1-np.sum(logistic_S0) / np.sum(logistic_S0)) - (1 / (2 * n1)) *  (np.sum(logistic_S1) / np.sum(logistic_S1))
    return gradient_w, gradient_b

def balanced_logistic_gradient_descent(learning_rate,T,lambda_val,x,y,wmle,bmle):
  data_S0 = x[y==0]
  data_S1 = x[y==1]

  w = np.zeros(len(wmle))
  b = 0

  for t in range(T):
    grad_w,grad_b = gradient_Jbal(w,b,wmle,bmle,lambda_val,data_S0,data_S1)
    w = w-learning_rate*grad_w
    b = b-learning_rate*grad_b

  return w,b

eta = 0.001
T = 50000
lambda_val = 0.01
w_bal, b_bal = balanced_logistic_gradient_descent(eta, T, lambda_val, x_train, y_train, w_mle, b_mle)
w_bal,b_bal

predictions_train=logistic_regression_classifier(x_train,w_bal,b_bal)
training_error_0 = len(predictions_train[(predictions_train==1) & (y_train==0)])/len(y_train[y_train==0])
training_error_1 = len(predictions_train[(predictions_train==0) & (y_train==1)])/len(y_train[y_train==1])
training_error_bal = 0.5*training_error_0+0.5*training_error_1
print(f"Training Error: {training_error_bal}")

predictions_test=logistic_regression_classifier(x_test,w_bal,b_bal)
test_error_0 = len(predictions_test[(predictions_test==1) & (y_test==0)])/len(y_test[y_test==0])
test_error_1 = len(predictions_test[(predictions_test==0) & (y_test==1)])/len(y_test[y_test==1])
test_error_bal = 0.5*test_error_0+0.5*test_error_1
print(f"Test Error: {test_error_bal}")