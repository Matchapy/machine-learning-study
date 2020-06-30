#%% Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels
import sklearn.gaussian_process as sk_gauss
#%% Prepare true function
def true_func(x):
    '''
    Returns true function
    :param np.array x:
    :return y = f(x)
    :rtyoe: np.array
    '''
    y = np.sin(2*x*np.pi)
    return y

#%% setup train data

# Training data is 20 points
train_x = np.random.normal(0, 1., 20)
# True function is sin(2*pi*x) with gaussian noise
train_y = true_func(train_x) + np.random.normal(loc=0, scale=0.2, size=len(train_x))

# Plot true graph
xx = np.linspace(-3, 3, 200)
plt.scatter(train_x, train_y, label='Data')
plt.plot(xx, true_func(xx), '--', color='C0', label='True function')
plt.grid()
plt.legend()
plt.title('Training Data')
plt.savefig('GPR_training_data.png', dpi=500)

#%% Make model with Scikit-learn
kernel = kernels.RBF(1.0, (1e-3, 1e3)) + kernels.ConstantKernel(1.0, (1e-3, 1e3)) + kernels.WhiteKernel()
clf = sk_gauss.GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-10,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=20,
    normalize_y=True
)

#%% Fit the model
# X's shape must be changed to (n_samples, n_features)
clf.fit(train_x.reshape(-1, 1), train_y)

print(clf.kernel_)

#%% plot result
test_x = np.linspace(-3, 3, 200).reshape(-1, 1)
pred_mean, pred_std = clf.predict(test_x, return_std = True)

def plot_result(test_x, mean, std):
    plt.plot(test_x[:, 0], mean, color='C0', label='predict mean')
    plt.fill_between(test_x[:, 0], mean + std, mean - std, color='C0', alpha=0.3, label='1 sigma confidence')
    plt.plot(train_x, train_y, 'o', label='training data')
    plt.grid()

plot_result(test_x, pred_mean, pred_std)
plt.title('Prediction by Scikit-learn')
plt.legend()
plt.savefig('sklearn_predict.png', dpi=500)
