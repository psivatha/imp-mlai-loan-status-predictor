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
