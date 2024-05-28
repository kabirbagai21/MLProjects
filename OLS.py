import pickle
import numpy as np
freedman = pickle.load(open('freedman.pkl', 'rb'))

n,d = freedman['data'].shape
p_hat = np.mean(freedman['data']*freedman['labels'][:,np.newaxis],axis=0)
j_hat = [j for j in range(d) if abs(p_hat[j]) > 2/np.sqrt(n)]
print(f"n = {len(j_hat)}")

X = freedman['data'][:,j_hat]

XTX_inv = np.linalg.inv(X.T @ X)
w_hat = XTX_inv @ X.T @ freedman['labels']

weights = np.zeros(freedman['data'].shape[1])
weights[j_hat] = w_hat

y_pred = freedman['data']@weights
sq_error = (y_pred-freedman['labels'])**2
empirical_risk=np.mean(sq_error)
print(f"Empirical Risk = {empirical_risk}")

y_pred_test = freedman['testdata']@weights
sq_error_test = (y_pred_test-freedman['testlabels'])**2
test_risk = np.mean(sq_error_test)
print(f"Test Risk = {test_risk}")

X_2 = freedman['data2'][:,j_hat]

XTX_inv_2 = np.linalg.inv(X.T @ X)
w_hat_2 = XTX_inv_2 @ X_2.T @ freedman['labels2']

weights_2 = np.zeros(freedman['data2'].shape[1])
weights_2[j_hat] = w_hat_2

y_pred_2 = freedman['data2']@weights_2
sq_error_2 = (y_pred_2-freedman['labels2'])**2
empirical_risk_2=np.mean(sq_error_2)
print(empirical_risk_2)