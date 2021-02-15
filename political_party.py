"""
Predicts the political party a district will vote for given data on median
income, demographics, unemployment rate, and percent with Bachelor's degree for
400 Democrat districts and 400 Republican districts from 2016.
Information used in this program is credited to:
USA 2016 Presidential Election by County, from
    https://public.opendatasoft.com/explore/dataset/usa-2016-presidential-election-by-county/
U.S. Census Bureau QuickFacts: United States, from
    https://www.census.gov/quickfacts/
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import sys
from tqdm import tqdm


def load_data_file(filename):
    """
    Load the data of m voting districts with n features into a NumPy array.

    Argument:
        filename: the name of the file containing the data

    Return value:
        an m x n NumPy array
    """
    try:
        with open(filename) as f:
            data = []
            for line in f:
                nums = []
                line_lst = line.split()
                for num in line_lst:
                    nums.append(float(num.strip()))
                data.append(nums)
        return np.array(data)
    except FileNotFoundError:
        print(f"{filename} does not exist in this directory.", file=sys.stderr)


def load_results_file(filename):
    """
    Load the result of m voting districts into a NumPy array.

    Argument:
        filename: the name of the file containing the data

    Return value:
        an m x 1 NumPy array
    """
    try:
        results = np.array([])
        with open(filename) as f:
            for line in f:
                results = np.append(results, int(line.strip()))
        return results
    except FileNotFoundError:
        print(f"{filename} does not exist in this directory.", file=sys.stderr)


def sigmoid(X):
    """
    Calculate the sigmoid of an input.

    Argument:
        X: an integer or a NumPy array

    Return value:
        The sigmoid of the input, which can be an integer or a NumPy array
    """
    try:
        return 1 / (1 + np.e ** (-X))
    except TypeError:
        error = "Only numbers and NumPy arrays allowed as input."
        print(error, file=sys.stderr)


def compute_cost(theta, X, y, reg_const):
    """
    Compute the cost of the parameters with regularization.

    Arguments:
        theta: an m x 1 vector of parameters to fit the data
        X: an n x m matrix of floats of the test data
        y: an n x 1 vector of 1's (Democrat) and 0's (Republican)
        reg_const: the regularization constant

    Return value:
        the cost of the parameters
    """
    try:
        n = len(y)
        h = sigmoid(np.matmul(X, theta))
        x1 = np.matmul(np.transpose(y), np.log(h))
        x2 = np.matmul(np.transpose(1 - y), np.log(1 - h))
        c_reg = reg_const * sum(theta[1:] ** 2) / (2 * n)
        cost = - (x1 + x2) / n + c_reg
        return cost
    except ValueError:
        print("Dimension mismatch.", file=sys.stderr)


def compute_grad(theta, X, y, reg_const):
    """
    Compute the gradient of the parameters with regularization.

    Arguments:
        theta: an m x 1 vector of parameters to fit the data
        X: an n x m matrix of floats of the test data
        y: an n x 1 vector of 1's (Democrat) and 0's (Republican)
        reg_const: the regularization constant

    Return value:
        the gradient of the parameters
    """
    try:
        n = len(y)
        h = sigmoid(np.matmul(X, theta))
        g_theta = np.copy(theta)
        g_theta[0] = 0
        g_reg = reg_const * g_theta / n
        grad = np.matmul(np.transpose(X), h - y) / n + g_reg
        return grad
    except ValueError:
        print("Dimension mismatch.", file=sys.stderr)


def gradient_descent(X, y, alpha, num_iter):
    """
    Iterate gradient descent on the parameters to minimize cost as well as
    gathering a cost_vec vector to be used for plotting the cost as a function
    of iterations.

    Arguments:
        X: an n x m matrix of floats of the test data
        y: an n x 1 vector of 1's (Democrat) and 0's (Republican)
        alpha: the learning rate
        num_iter: the number of times to iterate gradient descent

    Return value:
        a (theta, cost_vec) tuple of the parameters after gradient descent and
        the vector of the cost after each iteration
    """
    try:
        m = X.shape[1]
        theta = np.zeros(m, dtype=float)
        reg_const = 0.05
        cost_vec = np.array([])
        convergence = False
        improvement = 1e-9
        for _ in tqdm(range(num_iter)):
            old_error = compute_cost(theta, X, y, reg_const)
            cost_vec = np.append(cost_vec, old_error)
            theta -= alpha * compute_grad(theta, X, y, reg_const)
            new_error = compute_cost(theta, X, y, reg_const)
            convergence = old_error - new_error < improvement
            if old_error < new_error:
                raise ValueError("Learning rate is too large.")
            if convergence:
                break
        return (theta, cost_vec)
    except ValueError as e:
        print(e, file=sys.stderr)


def run_gradient_descent():
    """
    Run gradient descent on the specified text files.

    Arguments:
        none

    Return value:
        a (theta, cost_vec) tuple of the parameters after gradient descent and
        the vector of the cost after each iteration
    """
    X = load_data_file("training_district_info.txt")
    # note that the figures in the text file has been rescaled so that all
    # values are between 0 and 10 for faster convergence
    y = load_results_file("training_party_affiliation.txt")
    a, n = 0.1, 40000
    return gradient_descent(X, y, a, n)


def plot_cost():
    """
    Plot the cost of the parameters as a function of the number of iterations
    done by gradient descent.

    Arguments:
        none

    Return value:
        none
    """
    plt.plot(cost_vec)
    plt.title("Gradient descent")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()


def run_bfgs():
    """
    Run BFGS on the specified text files to find the optimal parameters.

    Arguments:
        none

    Return value:
        the optimal parameters after running BFGS
    """
    X = load_data_file("training_district_info.txt")
    # note that the figures in the text file has been rescaled so that all
    # values are between 0 and 10 for faster convergence
    y = load_results_file("training_party_affiliation.txt")
    t0 = np.zeros(X.shape[1], dtype=float)
    arguments = (X, y, 0.05)
    theta = optimize.minimize(compute_cost, t0, method="bfgs",
                              jac=compute_grad, args=(X, y, 0.05))
    return theta.x


def accuracy():
    """
    Test the accuracy of the parameters obtained from gradient descent and BFGS
    on training and test data data.

    Arguments:
        none

    Return value:
        a (gd_train_acc, bfgs_train_acc, gd_test_acc, bfgs_test_acc) tuple
    """
    X_train = load_data_file("training_district_info.txt")
    X_test = load_data_file("test_district_info.txt")
    y_train = load_results_file("training_party_affiliation.txt")
    y_test = load_results_file("test_party_affiliation.txt")
    h_train1 = sigmoid(np.matmul(X_train, theta_gd))
    h_train2 = sigmoid(np.matmul(X_train, theta_bfgs))
    h_test1 = sigmoid(np.matmul(X_test, theta_gd))
    h_test2 = sigmoid(np.matmul(X_test, theta_bfgs))
    gd_train_acc = sum(np.round(h_train1) == y_train) / len(y_train) * 100
    bfgs_train_acc = sum(np.round(h_train2) == y_train) / len(y_train) * 100
    gd_test_acc = sum(np.round(h_test1) == y_test) / len(y_test) * 100
    bfgs_test_acc = sum(np.round(h_test2) == y_test) / len(y_test) * 100
    return (gd_train_acc, bfgs_train_acc, gd_test_acc, bfgs_test_acc)


def interactive_predict():
    """
    Predict which party a district will vote for by asking for user input.

    Arguments:
        none

    Return value:
        none
    """
    try:
        name = input("County or state name: ").capitalize()
        inflation = 1.152  # adjust from 2018 to 2010 US dollar
        median_income = float(input("Median income: ")) / 10000 / inflation
        demographics = float(input("Percent white: ")) / 100
        unemployment_rate = float(input("Unemployment rate: ")) / 100
        bachelors = float(input("Percent with Bachelor's degree: ")) / 100
        # rescale above values to match the training data
        info = [1, median_income, demographics, unemployment_rate, bachelors]
        party = sigmoid(np.dot(theta_gd, info))
        if round(party):
            chance = party * 100
            print(f"{name}: {chance:.2f}% probability of voting Democrat.")
        else:
            chance = (1 - party) * 100
            print(f"{name}: {chance:.2f}% probability of voting Republican.")
    except ValueError:
        print("All values entered should be numbers.", file=sys.stderr)


def file_predict(filename):
    """
    Predict which party a district will vote for using information loaded from
    a text file.

    Arguments:
        filename: the name of the text file containing the information

    Return value:
        none
    """
    try:
        with open(filename) as f:
            for line in f:
                line_lst = line.split()
                inflation = 1.152  # adjust from 2018 to 2010 US dollar
                name = line_lst[0]
                x1 = float(line_lst[1]) / 10000 / inflation
                x2 = float(line_lst[2]) / 100
                x3 = float(line_lst[3]) / 100
                x4 = float(line_lst[4]) / 100
                info = [1, x1, x2, x3, x4]
                h = sigmoid(np.dot(theta_gd, info))
                if round(h):
                    print(f"{name}: Democrat")
                else:
                    print(f"{name}: Republican")
    except FileNotFoundError:
        print(f"{filename} does not exist in this directory.", file=sys.stderr)
    except ValueError or IndexError:
        print("The file is not in the correct format.", file=sys.stderr)


if __name__ == "__main__":
    theta_gd, cost_vec = run_gradient_descent()
    theta_bfgs = run_bfgs()
    param1 = np.around(theta_gd, decimals=4)
    param2 = np.around(theta_bfgs, decimals=4)
    gd_train_acc, bfgs_train_acc, gd_test_acc, bfgs_test_acc = accuracy()
    print(f"With gradient descent, the parameters are {param1}\n"
          f"with a training set accuracy of {gd_train_acc:.2f}% "
          f"and a test set accuracy of {gd_test_acc:.2f}%.")
    print(f"With BFGS, the parameters are {param2}\n"
          f"with a training set accuracy of {bfgs_train_acc:.2f}% "
          f"and a test set accuracy of {bfgs_test_acc:.2f}%.")
    # plot_cost()
    # interactive_predict()
