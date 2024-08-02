import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(standardise=False):
    train_df = pd.read_csv('data/loan_sanction_train.csv')
    # The original test file doesn't contain the Loan_Status field
    # Nevertheless loading it to construct a test set for another algorithm
    test_df = pd.read_csv('data/loan_sanction_test.csv')

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

#%%