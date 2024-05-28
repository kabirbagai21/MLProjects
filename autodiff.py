import numpy as np
import torch
from p1 import predict, x_test, y_test, x, y


def autodiff(X, W, y):
  v1 = torch.matmul(X, W)
  v2 = torch.softmax(v1, dim=1)
  v3 = v2[torch.arange(len(y)), y]
  v4 = -torch.log(v3)
  v5 = torch.sum(v4)
  return v5

def logreg_nll_gd_ad(K, eta, T, X, y):
  X = torch.tensor(X, dtype=torch.double)
  y = torch.tensor(y, dtype=torch.int)
  W = torch.zeros(X.shape[1], K, dtype=torch.double)
  W.requires_grad = True
  for _ in range(T):
    J = autodiff(X, W, y)
    J.backward()
    with torch.no_grad():
      W -= eta * W.grad
      W.grad.zero_()

  return W.detach()

params = logreg_nll_gd_ad(3, eta, T, x, y)
training_prediction_2 = predict(params.numpy(), x)
training_error_rate_2 = 1 - np.mean(training_prediction_2 == y)
test_prediction_2 = predict(params.numpy(), x_test)
test_error_rate_2 = 1 - np.mean(test_prediction_2 == y_test)
print(training_error_rate_2)
print(test_error_rate_2)

diff = W - params.numpy()
frob_norm = np.linalg.norm(diff, ord = 'fro')
print(f"The Frobenius norm is {frob_norm}")