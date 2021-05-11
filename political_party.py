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
        print(f'{filename} does not exist in this directory.')


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
        print(f'{filename} does not exist in this directory.')


def sigmoid(X):
    """
    Calculate the sigmoid of an input.

    Argument:
        X: an integer or a NumPy array

    Return value:
        The sigmoid of the input, which can be an integer or a NumPy array
    """
    return 1 / (1 + np.e ** (-X))


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
    n = len(y)
    h = sigmoid(np.matmul(X, theta))
    x1 = np.matmul(np.transpose(y), np.log(h))
    x2 = np.matmul(np.transpose(1 - y), np.log(1 - h))
    c_reg = reg_const * sum(theta[1:] ** 2) / (2 * n)
    cost = - (x1 + x2) / n + c_reg
    return cost


def compute_grad(theta, X, y, reg_const=0.0):
    """
    Compute the gradient of the parameters with regularization.

    Arguments:
        theta: an m x 1 vector of parameters to fit the data
        X: an n x m matrix of floats of the test data
        y: an n x 1 vector of 1's (Democrat) and 0's (Republican)
        reg_const: the regularization constant (default value is 0.0)

    Return value:
        the gradient of the parameters
    """
    n = len(y)
    h = sigmoid(np.matmul(X, theta))
    g_theta = np.copy(theta)
    g_theta[0] = 0
    g_reg = reg_const * g_theta / n
    grad = np.matmul(np.transpose(X), h - y) / n + g_reg
    return grad


def gradient_descent(X, y, lr, num_iter):
    """
    Iterate gradient descent on the parameters to minimize cost as well as
    gathering a cost_vec vector to be used for plotting the cost as a function
    of iterations.

    Arguments:
        X: an n x m matrix of floats of the test data
        y: an n x 1 vector of 1's (Democrat) and 0's (Republican)
        lr: the learning rate
        num_iter: the number of times to iterate gradient descent

    Return value:
        a (theta, train_cost, test_cost) tuple of the parameters after gradient
        descent and the vector of the training and test dat cost after each
        iteration
    """
    m = X.shape[1]
    theta = np.zeros(m, dtype=float)
    reg_const = 0.05
    train_cost, test_cost = np.array([]), np.array([])
    for _ in tqdm(range(num_iter)):
        old_error = compute_cost(theta, X, y, reg_const)
        train_cost = np.append(train_cost, old_error)
        test_cost = np.append(test_cost, test_accuracy(theta)[0])
        theta -= lr * compute_grad(theta, X, y, reg_const=reg_const)
        new_error = compute_cost(theta, X, y, reg_const)
        if old_error - new_error < 1e-9:
            break
    return theta, train_cost, test_cost


def plot_cost(vec1, vec2):
    """
    Plot the cost of the parameters as a function of the number of iterations
    done by gradient descent.

    Argument:
        vec1: the vector of in-sample cost to plot
        vec2: the vector of out-of-sample cost to plot
    """
    plt.plot(vec1, label='In-Sample Cost')
    plt.plot(vec2, label='Out-of-Sample Cost')
    plt.legend()
    plt.title('Gradient Descent vs. Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


def test_accuracy(theta):
    """
    Calculates the cost and test the accuracy of the parameters obtained from
    gradient descent on test data.

    Argument:
        theta: the parameters of the model

    Return value:
        a (cost, correct) tuple of the cost and the percentage of correct
        predictions associated with the given theta
    """
    X_test = load_data_file('test_district_info.txt')
    y_test = load_results_file('test_party_affiliation.txt')
    h_test = sigmoid(np.matmul(X_test, theta))
    accuracy = sum(np.round(h_test) == y_test) / len(y_test) * 100
    return compute_cost(theta, X_test, y_test, 0.05), accuracy


def user_predict(theta):
    """
    Predict which party a district will vote for by asking for user input.

    Argument:
        theta: the parameters of the model
    """
    try:
        name = input('County or state name: ').capitalize()
        inflation = 1.152  # adjust from 2018 to 2010 US dollar
        median_income = float(input('Median income: ')) / 10000 / inflation
        demographics = float(input('Percent white: ')) / 100
        unemployment_rate = float(input('Unemployment rate: ')) / 100
        bachelors = float(input('Percent with Bachelor\'s degree: ')) / 100
        # rescale above values to match the training data
        info = [1, median_income, demographics, unemployment_rate, bachelors]
        party = sigmoid(np.dot(theta, info))
        if round(party):
            chance = party * 100
            print(f'{name}: {chance:.2f}% probability of voting Democrat.')
        else:
            chance = (1 - party) * 100
            print(f'{name}: {chance:.2f}% probability of voting Republican.')
    except ValueError:
        print('All values entered should be numbers.')


def main():
    # note that the figures in the text file has been rescaled so that all
    # values are between 0 and 10 for faster convergence
    X_train = load_data_file('training_district_info.txt')
    y_train = load_results_file('training_party_affiliation.txt')
    lr, num_iter = 0.1, 100000
    print('Training...')
    theta, train_cost, test_cost = gradient_descent(X_train, y_train, lr,
                                                    num_iter)
    print('Finished')
    accuracy = test_accuracy(theta)[1]
    theta = np.around(theta, decimals=4)
    print(f'Using gradient descent, the parameters are {theta}'
          f' with a test set accuracy of {accuracy:.2f}%.')
    plot_cost(train_cost, test_cost)
    # user_predict(theta)


if __name__ == '__main__':
    main()
