import pandas as pd
import random

# Loading the dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Randomly sample 10% of the data
data = data.sample(frac=0.1, random_state=42)

# Explore the Data
print(data.head())
print(data.info())
print(data.describe())

# Handling missing values (if any)
data.dropna(inplace=True)

# Feature Engineering
data['Churn'] = data['Churn'].map({'Yes' : 1, 'No' : 0})


# # Encode categorical variables using one-hot encoding
# data = pd.get_dummies(data, columns=['gender', 'Contract', 'PaymentMethod'], drop_first=True)

# Explicitly setting data type of categorical columns to CategoricalDtype
categorical_columns = ['gender', 'Contract', 'PaymentMethod']
for col in categorical_columns:
    data[col] = data[col].astype('category')

# Encoding categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)


import matplotlib.pyplot as plt
import seaborn as sns

# Exploring relationships and visualize data
sns.countplot(x='Churn', data = data)
plt.show(block=False)

# Dropping non-numeric columns before calculating the correlation matrix

data_sample = data.sample(frac=0.1, random_state=42)

# Dropping non-numeric columns before calculating the correlation matrix
data_numeric = data_sample.drop(['customerID', 'Churn'], axis=1)


# Encoding categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data_numeric)

correlation_matrix = data_encoded.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show(block=False)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Creating a label encoder object
label_encoder = LabelEncoder()

# Applying label encoding to a binary categorical column
data['Churn'] = label_encoder.fit_transform(data['Churn'])
data = data.fillna(-999)

print(data['Churn'].unique())
print(data.dtypes)

# Converting 'TotalCharges' to numeric, errors='coerce' will convert non-numeric values to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Checking for any missing values (NaN) in the 'TotalCharges' column
missing_values = data['TotalCharges'].isnull().sum()
print(f"Number of missing values in 'TotalCharges' : {missing_values}")

# Splitting the data into training and testing sets

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Perform one-hot encoding
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the encoded data into training and testing sets
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_encoded, y_train)


# Evaluating the Model
y_pred = model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy : {accuracy}')
print(f'Confusion Matrix : \n {conf_matrix}')
print(f'Classification Report : \n {classification_rep}')


import joblib

# Saving the model
pkl = "model.sav"

joblib.dump(model, pkl)