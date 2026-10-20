import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # # ***************************************************
    # # INSERT YOUR CODE HERE
    # # TODO: compute loss by MSE
    # # ***************************************************
    # raise NotImplementedError

    return np.sum(np.pow(y - tx @ w, 2))/(2*y.shape[0])


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    # ***************************************************
    # raise NotImplementedError

    xT = tx.T

    # print(f'X^T: {xT.shape}')
    # print(f'w: {w.shape}')
    # print(f'y: {y.shape}')

    e = y - tx@w

    # print(f'e: {e.shape}')

    g = -(xT@e)/y.shape[0]
    
    return g

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape=(N,). The labels.
        tx: shape=(N, D). The feature matrix.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: int. The number of gradient descent steps to run.
        gamma: float. The step size.

    Returns:
        w: shape=(D,). The weight vector after the last iteration.
        loss: float. The MSE loss corresponding to ``w``.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        # raise NotImplementedError
        
        loss = compute_loss(y,tx,w)
        g = compute_gradient(y,tx,w)
        
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        # raise NotImplementedError
        w = w - gamma*g

        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print(
        #     "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )

    losses.append(compute_loss(y,tx,w))

    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent, with mini-batch size 1.

    Args:
        y: shape=(N,). The labels.
        tx: shape=(N, D). The feature matrix.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: int. The number of SGD steps to run.
        gamma: float. The step size.

    Returns:
        w: shape=(D,). The weight vector after the last iteration.
        loss: float. The MSE loss over the *full* dataset, corresponding to ``w``.
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        # raise NotImplementedError

        n = np.random.randint(len(y))
        y_n = y[n:n+1]
        tx_n = tx[n:n+1]
        loss = compute_loss(y_n,tx_n,w)

        # idx = np.random.permutation(len(y))
        # y_shuf, tx_shuf = y[idx], tx[idx]
        
         
        # for minibatch_y, minibatch_tx in zip(y_shuf, tx_shuf):
    
        g = compute_gradient(y_n,tx_n,w)
        w = w - gamma*g
        
        ws.append(w)
        
        # print(
        #     "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )
        losses.append(loss)

    losses.append(compute_loss(y,tx,w))

    return ws[-1], losses[-1] 


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solves ``(tx^T tx) w = tx^T y``.

    Args:
        y: shape=(N,). The labels.
        tx: shape=(N, D). The feature matrix.

    Returns:
        w: shape=(D,). The optimal weight vector.
        loss: float. The MSE loss corresponding to ``w``.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    
    # Computes the "exact" solution, `x`, of the well-determined, i.e., full
        # rank, linear matrix equation `ax = b`.


    w = np.linalg.solve(tx.T@tx,tx.T@y)

    return w, compute_loss(y,tx,w).item()


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Solves ``(tx^T tx + 2 N lambda_ I) w = tx^T y``.

    Args:
        y: shape=(N,). The labels.
        tx: shape=(N, D). The feature matrix.
        lambda_: float. The regularization parameter.

    Returns:
        w: shape=(D,). The optimal weight vector.
        loss: float. The MSE loss corresponding to ``w``, *without* the penalty term.
    """
    # raise NotImplementedError

    # mse = compute_loss(y,tx, w)
    # loss = mse + lambda_ * np.sum(np.pow(w,2))
    
    # L' = MSE' + 2 lambda w = -1/N X^T(y-Xw) + 2 lambda w 

    #L' = 0 <=> -1/N X^T(y-Xw) + 2 lambda w  = X^T(y-Xw) = 2N lambda w
    # <=> X^Ty - X^T Xw = 2N lambda w <=> X^Ty = X^T Xw + 2N lambda w <=> X^Ty = (X^T X + 2N lambda I)w

    # Computes the "exact" solution, `x`, of the well-determined, i.e., full
            # rank, linear matrix equation `ax = b`.

            #a = (X^T X + 2N lambda I)
            #b = X^Ty
    w = np.linalg.solve(tx.T @ tx + 2*y.shape[0] * lambda_ * np.eye(tx.shape[1]),tx.T@y)

    return w, compute_loss(y, tx, w).item()


def compute_logistic_loss(y, tx, w):
    """Computes the negative log-likelihood for logistic regression.

    Args:
        y: shape=(N, ). The labels, in {0, 1}.
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    # L = -1/N sum y*log(sigma(z)) + (1-y)*log(1-sigma(z))   with z = xw
    #   = 1/N sum log(1 + e^z) - y*z

    z = tx @ w

    # logaddexp(0, z) is log(1 + e^z) computed without overflowing for large z,
    # which the naive log(1 + exp(z)) would do as soon as a point saturates.
    return np.mean(np.logaddexp(0, z) - y*z)


def compute_logistic_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    # L' = 1/m sum (1/(1 + e^(-wx-b)) - y)x
    
    # g = np.sum(1/(1+np.exp(-w@tx))-y)*tx/y.shape[0]
    g = (1/(1+np.exp(-tx@w))-y) @ tx/y.shape[0]
    
    return g


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: shape=(N,). The labels, in {0, 1}.
        tx: shape=(N, D). The feature matrix.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: int. The number of gradient descent steps to run.
        gamma: float. The step size.

    Returns:
        w: shape=(D,). The weight vector after the last iteration.
        loss: float. The negative log-likelihood corresponding to ``w``.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        # raise NotImplementedError
        
        loss = compute_logistic_loss(y,tx,w)
        g = compute_logistic_gradient(y,tx,w)
        
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        # raise NotImplementedError
        w = w - gamma*g

        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print(
        #     "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )

    losses.append(compute_logistic_loss(y,tx,w))

    return ws[-1], losses[-1]
    


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    The objective minimized is the negative log-likelihood plus the penalty
    ``lambda_ * ||w||^2``.

    Args:
        y: shape=(N,). The labels, in {0, 1}.
        tx: shape=(N, D). The feature matrix.
        lambda_: float. The regularization parameter.
        initial_w: shape=(D,). The initial weight vector.
        max_iters: int. The number of gradient descent steps to run.
        gamma: float. The step size.

    Returns:
        w: shape=(D,). The weight vector after the last iteration.
        loss: float. The negative log-likelihood corresponding to ``w``,
            *without* the penalty term.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        # raise NotImplementedError
        
        loss = compute_logistic_loss(y,tx,w)
        g = compute_logistic_gradient(y,tx,w)

        # Same as linear regression (ridge regression)
        g  = g + lambda_ * 2*w
        
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        # raise NotImplementedError
        w = w - gamma*g

        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print(
        #     "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )

    losses.append(compute_logistic_loss(y,tx,w))

    return ws[-1], losses[-1]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, lambda_, get_trained_weights, loss_func):
    """return the loss of a model for a fold corresponding to k_indices

    Args:
        y:                   shape=(N,)
        x:                   shape=(N, D)
        k_indices:           2D array returned by build_k_indices()
        k:                   scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:             scalar, the regularization parameter passed to get_trained_weights
        get_trained_weights: function (y, x, lambda_) -> w, trains the model and returns its weights
        loss_func:           function (y, x, w) -> scalar, the loss (or score, e.g. F1) to evaluate

    Returns:
        train and test values, as computed by loss_func
    """
    
    train_indices = np.delete(k_indices, k, axis=0).flatten()
    test_indices = k_indices[k]

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    w = get_trained_weights(y_train, x_train, lambda_)

    loss_tr = loss_func(y_train, x_train, w)
    loss_te = loss_func(y_test, x_test, w)

    return loss_tr, loss_te


def f1_score(y, y_pred):
    """F1 score with label 1 as the positive class."""
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def compute_logistic_f1s(y, tx, w, thresholds):
    """F1 score of a logistic model for each cutoff in thresholds, predicting 1 when sigma(tx @ w) >= cutoff."""
    p = 1 / (1 + np.exp(-tx @ w))
    return np.array([f1_score(y, (p >= t).astype(int)) for t in thresholds])



def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters):
    
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        p = 1 / (1 + np.exp(-tx @ w))
        # Hessian: 1/N X^T diag(p(1-p)) X + 2 lambda_ I
        h = (tx * (p * (1 - p))[:, None]).T @ tx / y.shape[0] + 2 * lambda_ * np.eye(tx.shape[1])
        w = w - np.linalg.solve(h, g)
    return w, compute_logistic_loss(y, tx, w)



def cross_fold_reg_logistic_regression_newton(y, tx, initial_w, k_fold, lambdas, thresholds, max_iters, seed=1):
    """Picks lambda and the decision cutoff jointly by maximizing the mean held-out F1 over k folds,
    then retrains on all of (y, tx) with the chosen lambda.

    Returns:
        w:         shape=(D,). The weights trained with the chosen lambda.
        loss:      float. The negative log-likelihood corresponding to ``w``.
        threshold: float. The chosen cutoff on sigma(tx @ w).
    """

    k_indices = build_k_indices(y, k_fold, seed)

    def get_trained_weights(y_, tx_, lambda_):
        return reg_logistic_regression_newton(y_, tx_, lambda_, initial_w, max_iters)[0]

    def f1s(y_, tx_, w):
        return compute_logistic_f1s(y_, tx_, w, thresholds)

    # f1_tes[i, j]: mean held-out F1 for lambdas[i] with cutoff thresholds[j]
    f1_tes = np.zeros((len(lambdas), len(thresholds)))
    for i, lambda_ in enumerate(lambdas):
        for k in range(k_fold):
            _, f1_te = cross_validation(y, tx, k_indices, k, lambda_, get_trained_weights, f1s)
            f1_tes[i] += f1_te / k_fold

    i, j = np.unravel_index(np.argmax(f1_tes), f1_tes.shape)

    w, loss = reg_logistic_regression_newton(y, tx, lambdas[i], initial_w, max_iters)
    return w, loss, thresholds[j]