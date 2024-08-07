# Model Card

This is a simple LeNet5 model to predict the loan approval status inline while a home loan application form is being
filled by an applicant.

## Author:
Para Sivatharman

## Model Description

**Input:**
Inputs include information such as Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History,
Property Area, Gender, Marital Status, Education, Self-Employment Status, and Number of Dependents.

**Output:**
The model outputs a binary prediction indicating whether a loan will be approved (Y) or not (N).

**Model Architecture:**
The model uses an adapted `LeNet5` Convolutional Neural Network (CNN) architecture, originally designed for image
recognition, to process tabular data. The architecture includes:

- Convolutional layers to extract features
- Flatten layer to convert the feature maps to a 1D vector
- Dropout layer to prevent overfitting
- Dense layer for the final binary classification

## Performance

**Accuracy:**
The model achieved an accuracy of 78.86% on the training data. However, it performed exceptionally well against a
construction of unknown inputs with predictions from other models such as Linear Regression and K-Nearest Neighbours.

**Validation:** During custom cross-validation, where the predictions of one model was used as validation data for
the other models, the Logistic Regression (LR) and LeNet5 models showed a consistent
high performance on the test data produced by each other.

**Data:** The model was trained on a dataset containing loan application information with features such as income, loan
amount, credit history, etc. The data was preprocessed to handle missing values, encode categorical variables,
and standardize the inputs. Although, there was a test data file available, it could not be used as validation data
because it was missing the approval status, which is the output of the model. It was therefore decided to use those
input values from the test_data file to make predictions by various models and store them as potential validation data
for the other models developed here.

## Limitations

**Generalization:** The model's performance may vary with different datasets. It should be evaluated
on new data to ensure it generalises well.

**Feature Dependency:** The model's predictions are only as good as the input features.
Inaccurate or biased input data can lead to incorrect predictions.

**Bias:** Potential biases in the dataset, such as those related to gender, marital status, or property area, may
affect the model's fairness.

## Trade-offs

**Fairness vs. Accuracy:** Ensuring fairness in the model might lead to a trade-off
with accuracy, as adjustments to mitigate bias may affect the model's predictive performance.

**Complexity vs. Interpretability:** The use of a deep learning model like LeNet5 improves accuracy but
reduces interpretability compared to simpler models like Logistic Regression or Decision Trees.

**Training Time:** The model requires more time to train compared to simpler models, especially with any future
large datasets that might be large. This is due to its complex architecture.
