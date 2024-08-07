# Datasheet Template

Dream Housing Finance company deals in all home loans. They have a presence across all urban, semi-urban and rural
areas. The customer first applies for a home loan after that company validates the customer's eligibility for a loan.
The company wants to automate the loan eligibility process (real-time) based on customer detail provided while
filling out the online application form.

## Motivation

- The company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling
  out the online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income,
  Loan Amount, Credit History and others.
- Dream Housing Finance company created the dataset and funded its creation.


## Composition

- The dataset comprises various information of applicants in a CSV format.
- The train dataset contained 614 rows and the provided test data contained 367 records but without the dependent
  variable information, which is the loan status.
- There were quite a few missing data, which have been filled with the mean of the existing data. Loan amount,
  loan amount term, credit history, marital status, dependents, and employment information were the features that
  contained missing data.
- Credit history of individuals could be considered confidential information.

## Collection process

- There was not enough information to determine how the data was collected.
- The data is not a sample of a larger dataset.
- The information about the period in which the data was collected is not available

## Preprocessing/cleaning/labelling

- None of preprocessing/cleaning/labeling of the data done as part of this project. i.e.
  None of discretisation or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of
  instances, processing of missing values were done.

## Uses

- The data could be used for potentially determining other relationship between family size and loan amount or loan
  amount and total income etc.
- There doesn't seem to be any limitation for this data to be used on any future projects due to how it is pre-processed
  cleaned or labelled.
- It's not obvious if there's any tasks that the dataset should not be used to.

## Distribution

- The data is distributed as csv files on kaggle.com.
- The data is publicly available and has no copyright restrictions.

## Maintenance

- There's no clear indication as to who or which company will maintain this dataset.
