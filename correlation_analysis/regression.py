from sklearn.linear_model import LinearRegression
import numpy as np

def linreg(
    X: np.ndarray,
    property: np.ndarray
    ):
    '''
    Perform linear regression of property vs X

    Parameters
    ----------
    X: np.ndarray
        The free variables, shape n_samples x n_dimensions

    property: np.ndarray
        The bound variable, shape n_samples x 1

    Returns
    --------
    coefs: np.ndarray
        The gradient of property w.r.t. X

    intercept: float
        The y-axis intercept of property
    '''

    assert len(X) == len(property)

    regressor = LinearRegression()

    regressor.fit(X, property)

    return regressor.coef_, regressor.intercept_