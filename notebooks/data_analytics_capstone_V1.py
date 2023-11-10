import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

# Data Dictionary
df_dict = pd.read_csv("../data/raw/data_dictionary.csv")
df_dict

# Load Data
df = pd.read_csv("../data/raw/lending_club_2007_2011_6_states.csv")
df.head()

# Check the info and missing value of the data
df.info(verbose=True)
df.isnull().sum()

# Summary Statistics
df.describe()

# EDA
# Loan term distribution
df["term"].value_counts()
sns.countplot(x="term", data=df)
plt.show()

# Interest rate distribution
df["int_rate"].describe()
sns.distplot(df["int_rate"])
plt.show()

# Average interest rate by loan term
df.groupby("term")["int_rate"].mean()
sns.barplot(x="term", y="int_rate", data=df)
plt.show()

# Loan grade distribution (sorted by grade)
df["grade"].value_counts()
sns.countplot(x="grade", data=df, order=sorted(df.grade.unique()))
plt.show()

# Loan grade and interest rate
df.groupby("grade")["int_rate"].mean()
sns.barplot(x="grade", y="int_rate", data=df, order=sorted(df.grade.unique()))
plt.show()


# Loan by state (sorted by state)
df["addr_state"].value_counts()
sns.countplot(x="addr_state", data=df, order=sorted(df.addr_state.unique()))
plt.xticks()
plt.show()

# Borrower Annual Income distribution
df["annual_inc"].describe()
sns.distplot(df["annual_inc"])
plt.show()

# Borrower Annual Income less than 250,000
df[df["annual_inc"] < 250000]["annual_inc"].describe()
sns.distplot(df[df["annual_inc"] < 250000]["annual_inc"])
plt.show()

# Borrower median annual income by state
df.groupby("addr_state")["annual_inc"].median()
sns.barplot(x="addr_state", y="annual_inc", data=df)
plt.xticks()
plt.show() 

# Annual income by interest rate
df.groupby("int_rate")["annual_inc"].mean()
sns.scatterplot(x="int_rate", y="annual_inc", data=df)
plt.show()

# Annual income less than 300,000 by interest rate
df[df["annual_inc"] < 300000].groupby("int_rate")["annual_inc"].mean()
sns.scatterplot(x="int_rate", y="annual_inc", data=df[df["annual_inc"] < 300000])
plt.show()

# Convert the issue_d column to datetime
df["issue_d"] = pd.to_datetime(df["issue_d"])
df["issue_d"].head()

# Create a new column for year
df["issue_y"] = df["issue_d"].dt.year

# Loan issued by year
df["issue_y"].value_counts()
sns.countplot(x="issue_y", data=df)
plt.show()

# Interest rate change in each state over the years
# Create pivot table, index is issue_y, columns is addr_state, values is int_rate, aggfunc is median
df_pivot = df.pivot_table(index="issue_y", columns="addr_state", values="int_rate", aggfunc=np.median)
df_pivot
# Plot the pivot table
df_pivot.plot(kind="bar", figsize=(15, 10))
plt.show()

# Loan status distribution
df["loan_status"].value_counts()
sns.countplot(x="loan_status", data=df)
plt.xticks(rotation=90)
plt.show()
# Paid off rate (Fully Paid / (Fully paid + Charged Off))
df["loan_status"].value_counts()["Fully Paid"] / (df["loan_status"].value_counts()["Fully Paid"] + df["loan_status"].value_counts()["Charged Off"])
print("Paid off rate: {:.2f}%".format(df["loan_status"].value_counts()["Fully Paid"] / (df["loan_status"].value_counts()["Fully Paid"] + df["loan_status"].value_counts()["Charged Off"]) * 100))

# Create pivot table pt_term out of df, set index as term, columns as loan_status, values as int_rate, aggfunc as count
pt_term = df.pivot_table(index="term", columns="loan_status", values="int_rate", aggfunc="count")
pt_term
# Calculate the paid off rate of each loan term
pt_term["Paid off rate"] = pt_term["Fully Paid"] / (pt_term["Fully Paid"] + pt_term["Charged Off"])
pt_term
# Plot the paid off rate of each loan term
pt_term["Paid off rate"].plot(kind="bar")
plt.show()
# Plot the pivot table
pt_term.plot(kind="bar")
plt.show()

# Loan grade and loan status
# Create pivot table pt_grade out of df, set index as grade, columns as loan_status, values as int_rate, aggfunc as count
pt_grade = df.pivot_table(index="grade", columns="loan_status", values="int_rate", aggfunc="count")
pt_grade
# Calculate the paid off rate of each loan grade
pt_grade["Paid off rate"] = pt_grade["Fully Paid"] / (pt_grade["Fully Paid"] + pt_grade["Charged Off"])
pt_grade
# Plot the paid off rate of each loan grade with title and show the prentage
pt_grade["Paid off rate"].plot(kind="bar", title="Paid off rate by loan grade")
plt.show()
# Plot the pivot table with title and ignore the paid off rate
pt_grade.drop("Paid off rate", axis=1).plot(kind="bar", title="Loan grade by loan status")
plt.show()

# Check the info and missing value of the data
df.info(verbose=True)
df.isnull().sum()



# Create a 'repaid' column in loan_df by encoding loan status and map Charged Off to 0 and Fully Paid to 1
df['repaid'] = df['loan_status'].apply(lambda x: 1 if x == 'Fully Paid' else 0)
df['repaid'].value_counts()

# Create a new column loan_term_year in loan_df to encode: '36 months' to 3 and '60 months' to 5
df['loan_term_year'] = df['term'].apply(lambda x: int(x.split()[0]) / 12)
df['loan_term_year'].value_counts()

df.head()
df.info()

# Initialize lists for different types of features
numeric_features = []
categorical_features = []

# Numeric features: int64, float64 (excluding 'repaid' which is the target)
numeric_features = [
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 
    'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 
    'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 
    'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 
    'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med', 
    'policy_code', 'acc_now_delinq', 'chargeoff_within_12_mths', 'delinq_amnt', 
    'pub_rec_bankruptcies', 'tax_liens', 'loan_term_year'
]

# Categorical features: object (excluding 'term' which is now represented by 'loan_term_year' and text features)
categorical_features = [
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'pymnt_plan', 'purpose', 'addr_state', 'initial_list_status', 'application_type', 
    'hardship_flag', 'disbursement_method', 'debt_settlement_flag', 'issue_y'
]

# Columns to drop because they have no non-null data
# columns_to_drop = ['next_pymnt_d', 'mths_since_last_major_derog']

# Dropping the columns with no non-null data
# df.drop(columns=columns_to_drop, inplace=True)

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95))
])

pipeline
df.info()

# Split the data
X = df.drop(columns=['term', 'issue_d', 'loan_status', 'repaid', 'next_pymnt_d', 'mths_since_last_major_derog', 'emp_title', 'title', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'debt_settlement_flag_date'])
X.info()
y = df['repaid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the data using the pipeline
pipeline.fit(X_train, y_train)
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Now you can use X_train_transformed and X_test_transformed for modeling
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

# Define a function to compare model performances
def compare_model_performances(X_train, y_train, model_list):
    for model in model_list:
        model_name = model.__class__.__name__
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{model_name}: Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")

# List of models to train and compare
models = [
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100),
    LogisticRegression(max_iter=1000),
    SVC()
]

# Use the function with the training data
compare_model_performances(X_train_transformed, y_train, models)

# You would then select the best performing model based on the cross-validation scores
# For example, if RandomForestClassifier was the best:
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_transformed, y_train)
# Now 'best_model' can be used to make predictions on new data

