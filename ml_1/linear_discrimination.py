import numpy as np
import matplotlib.pyplot as plt


def mk_class(n_points, mu, cov_mat):
    return np.random.multivariate_normal(mu, cov_mat, n_points)


def mk_covmat(d):
    A = np.random.uniform(-5, 5, (d, d))
    return np.dot(A, A.transpose())

def compute_cov(c1, c2):
    mu1 = np.mean(c1, axis=0)
    mu2 = np.mean(c2, axis=0)

    delta1 = lambda c: mu1 - c
    delta2 = lambda c: mu2 - c

    cov = sum(np.outer(delta1(c), delta2(c).transpose()) for c in c1) / c1.shape[0] + sum(
        np.outer(delta2(c), delta2(c).transpose()) for c in c2) / c2.shape[0]

    return mu1, mu2, cov


def compute_linear_sep(c1, c2):
    mu1, mu2, cov = compute_cov(c1, c2)

    inv_conv = np.linalg.inv(cov)

    h = np.dot((mu1 - mu2).transpose(), inv_conv)

    P1 = - np.dot(np.dot(mu1.transpose(), inv_conv), mu1)
    P2 = np.dot(np.dot(mu2.transpose(), inv_conv), mu2)
    b = (1 / 2) * (P1 + P2) + 0

    def linear_sep(x):
        return np.dot(h, x) + b

    return linear_sep, mu1, mu2, cov, h, b

S = mk_covmat(2)
X_1 = mk_class(100, [10,10], S)
X_2 = mk_class(100, [-10,-10], S)

lin_sep, mu1, mu2, cov, h, b = compute_linear_sep(X_1, X_2)

plt.scatter(X_1[:, 0], X_1[:, 1], marker="+", c="red" )
plt.scatter(X_2[:, 0], X_2[:, 1], marker="o", c="blue")

plt.show()

