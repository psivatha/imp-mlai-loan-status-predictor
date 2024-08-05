import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(
        train_file_name='data/loan_sanction_train.csv',
        test_file_name='data/loan_sanction_test.csv',
        standardise=False):
    train_df = pd.read_csv(train_file_name)
    # The original test file doesn't contain the Loan_Status field
    # Nevertheless loading it to construct a test set for another algorithm
    test_df = pd.read_csv(test_file_name)

    for df in [train_df, test_df]:
        # Convert categorical variables into numeric
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
        df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
        df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
        df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
        df['Dependents'] = df['Dependents'].replace('3+', 3)

        # Fill missing values. Do it after converting categorical values into numeric
        df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

        # Create extra features that can be useful and meaningful
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['LoanIncomeRatio'] = df['LoanAmount'] / df['TotalIncome']

    # Convert the output variable into numeric
    train_df['Loan_Status'] = train_df['Loan_Status'].map({'Y': 1, 'N': 0})
    X = train_df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = train_df['Loan_Status']

    if standardise:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    common_ids = set(train_df['Loan_ID']).intersection(set(test_df['Loan_ID']))
    common_ids_list = list(common_ids)
    assert not common_ids_list

    # Carry out train/test split from the given training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, train_df, test_df


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(-1, 1)

def bayesian_optimisation(n_iters, sample_loss, bounds, x0, y0, gp_params):
    """
    Bayesian optimization for decision trees
    :param n_iters: No. of bayesian optimization iterations
    :param sample_loss: The function that trains and evaluates the decision trees. See `optimise_decision_tree`
    :param bounds: The bounds of each hyperparameter of the decision tree
    :param x0: Initial sample points
    :param y0: Initial evaluation points
    :param gp_params: GPR parameters (GPR is initialised with Matern kernel)
    :return: All sampled hyperparamters combinations, evaluations of them, final GPR
    """
    X_sample = x0
    Y_sample = y0

    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), **gp_params)

    for i in range(n_iters):
        gpr.fit(X_sample, Y_sample)

        X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)
        params = X_next.flatten()
        Y_next = sample_loss(*params)

        X_sample = np.vstack((X_sample, X_next.T))
        Y_sample = np.vstack((Y_sample, Y_next))

    return X_sample, Y_sample, gpr


def write_new_data_file(model, X_train, y_train, test_df, target_filename):
    model.fit(X_train, y_train)
    X_test_final = test_df.drop(columns=['Loan_ID'])
    y_test_pred = model.predict(X_test_final)
    test_df['Loan_Status'] = y_test_pred
    # Save the file
    test_df.to_csv(target_filename, index=False)
    print(f"Predictions have been saved to {target_filename}.")


