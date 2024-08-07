# Loan Status Predictor

To predict loan approvals, I tested several models: Decision Tree, Convolutional Neural Network (CNN),
K-Nearest Neighbours (KNN), Logistic Regression (LR), and LeNet5 (a neural network).
The parameters used in each model were optimised using another technique called Bayesian Optimisation.
I also split our initial data to train and test these models. Logistic Regression and LeNet5 showed the best accuracy
in our initial tests.

To further validate, I used predictions from these models as new test data for each other.
Remarkably, LR and LeNet5 predicted each other's test data with perfect accuracy.
This consistency suggests both models are highly reliable, but I chose LeNet5 for its slightly better performance
in overall tests.

The chosen LeNet5 model analyses various factors such as income and credit history to predict whether a
loan will be approved, helping to streamline and improve decision-making in loan processing.

The chosen LeNet5 model, originally designed for image recognition, was adapted to process tabular data. It uses
multiple layers to learn complex patterns in the data. Specifically, it analyses various factors such as
income, loan amount, credit history, and other personal details to predict whether a loan will be approved.
By doing so, LeNet5 helps streamline and improve decision-making in loan processing, providing a robust tool
for accurately assessing loan applications while the data is being entered.

The following is the matrix of each model's performance against the validation data constructed using the other models.
The rows correspond to the performance of the model and the columns represent the validation data produced by each
model.

|       | CNN  | DT   | KNN  | LeNet5 | LR   |
|-------|------|------|------|--------|------|
| **CNN**   | N/A  | 0.92 | 0.45 | 0.95   | 0.95 |
| **DT**    | 0.21 | N/A  | 0.49 | 0.95   | 0.95 |
| **KNN**   | 0.17 | 0.95 | N/A  | 0.99   | 0.99 |
| **LeNet5**| 0.16 | 0.95 | 0.47 | N/A    | 1    |
| **LR**    | 0.16 | 0.95 | 0.47 | 1      | N/A  |



# How to set up the environment:
- Run the following set of commands to set the environment from the root directory of the project:
    ```
    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip -r requirements.text
    ```
- Use the python in this environment as the interpreter for the Jupyter notebook.
