# -*- coding: utf-8 -*-
"""Untitled.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rA0X1AbR6P--hXbd6jkX2y_2QLJsAzPp
"""

import pandas as pd
import numpy as np




#from google.colab import drive

#drive.mount('/content/drive')

file_path = '/content/drive/My Drive/World_development_mesurement .xlsx'
data = pd.read_excel(file_path)

data.head()



data.tail()

data.columns

data.info(20)

data.head()

data.tail()

data['GDP']=data['GDP'].astype(str).str.replace('$','',regex=True).str.replace(',','')
data['GDP']=pd.to_numeric(data['GDP'],errors='coerce')

data['Health Exp/Capita']=data['Health Exp/Capita'].astype(str).str.replace('$','',regex=True)
data['Health Exp/Capita']=pd.to_numeric(data['Health Exp/Capita'], errors='coerce')

data['Tourism Inbound']=data['Tourism Inbound'].astype(str).str.replace('$','',regex=True).str.replace(',', '')
data['Tourism Inbound'] = pd.to_numeric(data['Tourism Inbound'], errors='coerce')

data['Tourism Outbound']=data['Tourism Outbound'].astype(str).str.replace('$','',regex=True).str.replace(',','')
data['Tourism Outbound']=pd.to_numeric(data['Tourism Outbound'],errors='coerce')


## Remove %
data['Business Tax Rate']=data['Business Tax Rate'].astype(str).str.replace('%','',regex=True)
data['Business Tax Rate'] = pd.to_numeric(data['Business Tax Rate'], errors='coerce')

data.info()

data.head(20)

print(data.describe())

# Check for extreme values and potential outliers
print(data[['Birth Rate', 'CO2 Emissions', 'GDP', 'Population Total']].quantile([0.25, 0.5, 0.75]))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Country_encoded'] = le.fit_transform(data['Country'])
data['Country_encoded']=data['Country_encoded'].astype(float)
data.drop(['Country'],axis=1,inplace=True)

data.info(10)

data.head(20)

data.isnull().sum()

data['Tourism Inbound'] = data['Tourism Inbound'].fillna(data['Tourism Inbound'].median())
data['Tourism Outbound'] = data['Tourism Outbound'].fillna(data['Tourism Outbound'].median())

data.head(20)

data['Tourism Outbound'] = data['Tourism Outbound'].astype(str).str.replace(r'[^\d.]', '', regex=True)
data['Tourism Outbound'] = pd.to_numeric(data['Tourism Outbound'], errors='coerce')
data['Tourism Outbound'] = data['Tourism Outbound'].fillna(data['Tourism Outbound'].median())

data['Tourism Inbound'] = data['Tourism Inbound'].astype(str).str.replace(r'[^\d.]', '', regex=True)
data['Tourism Inbound'] = pd.to_numeric(data['Tourism Inbound'], errors='coerce')
data['Tourism Inbound'] = data['Tourism Inbound'].fillna(data['Tourism Inbound'].median())

data = pd.read_excel(file_path)

# Define a function to clean numeric columns
def clean_numeric(col):
  col = col.astype(str).str.replace(r'[$,%]', '', regex=True).str.replace(',', '', regex=True)
  return pd.to_numeric(col, errors='coerce')

# Apply the cleaning function to the relevant columns
for col in ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound', 'Business Tax Rate']:
  data[col] = clean_numeric(data[col])

# Fill NaN values with the median for specified columns
for col in ['Tourism Inbound', 'Tourism Outbound']:
    data[col] = data[col].fillna(data[col].median())

data.info()

data.tail(20)

data.describe()

data = data.drop(['Number of Records'],axis=1)

data.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

if 'Ease of Business' in data.columns:
  data = data.drop('Ease of Business', axis=1)

import missingno as msno
msno.bar(data)
plt.show()

data = data.rename(columns={'Birth Rate': 'BirthRate', 'Business Tax Rate': 'BusinessTaxRate','CO2 Emissions':'CO2Emissions','Days to Start Business':'DaystoStartBusiness','Ease of Business':'EaseofBusiness','Energy Usage':'EnergyUsage',
                            'Health Exp % GDP':'HealthExpGDP','Health Exp/Capita':'HealthExpCapita','Hours to do Tax':'HourstodoTax','Infant Mortality Rate':'InfantMortalityRate','Internet Usage':'InternetUsage','Lending Interest':'LendingInterest',
                            'Life Expectancy Female':'LifeExpectancyFemale','Life Expectancy Male':'LifeExpectancyMale','Mobile Phone Usage':'MobilePhoneUsage','Number of Records':'NumberofRecords','Population 0-14':'Population0to14',
                            'Population 15-64':'Population15to64','Population 65+':'Populationmorethan65','Population Total':'PopulationTotal','Population Urban':'PopulationUrban','Tourism Inbound':'TourismInbound','Tourism Outbound':'TourismOutbound'})
data.columns

fig, axes=plt.subplots(6,4,figsize=(14,12),sharex=False,sharey=False)
sns.distplot(data.BirthRate,ax=axes[0,0])
sns.distplot(data.BusinessTaxRate,ax=axes[0,1])
sns.distplot(data.CO2Emissions,ax=axes[0,2])
sns.distplot(data.DaystoStartBusiness,ax=axes[0,3])
sns.distplot(data.EnergyUsage,ax=axes[1,1])
sns.distplot(data.GDP,ax=axes[1,2])
sns.distplot(data.HealthExpGDP,ax=axes[1,3])
sns.distplot(data.HealthExpCapita,ax=axes[2,0])
sns.distplot(data.HourstodoTax,ax=axes[2,1])
sns.distplot(data.InfantMortalityRate,ax=axes[2,2])
sns.distplot(data.InternetUsage,ax=axes[2,3])
sns.distplot(data.LendingInterest,ax=axes[3,0])
sns.distplot(data.LifeExpectancyFemale,ax=axes[3,1])
sns.distplot(data.LifeExpectancyMale,ax=axes[3,2])
sns.distplot(data.MobilePhoneUsage,ax=axes[3,3])
sns.distplot(data.MobilePhoneUsage,ax=axes[4,0])
sns.distplot(data.Population0to14,ax=axes[4,1])
sns.distplot(data.Population15to64,ax=axes[4,2])
sns.distplot(data.Populationmorethan65,ax=axes[4,3])
sns.distplot(data.PopulationTotal,ax=axes[5,0])
sns.distplot(data.PopulationUrban,ax=axes[5,1])
sns.distplot(data.TourismInbound,ax=axes[5,2])
sns.distplot(data.TourismOutbound,ax=axes[5,3])
plt.tight_layout(pad=2.0)

data['BusinessTaxRate'] = data['BusinessTaxRate'].fillna(data['BusinessTaxRate'].mean())

data['HealthExpGDP'] = data['HealthExpGDP'].fillna(data['HealthExpGDP'].mean())
data['HourstodoTax'] = data['HourstodoTax'].fillna(data['HourstodoTax'].mean())
data['Population0to14'] = data['Population0to14'].fillna(data['Population0to14'].mean())

data['BirthRate'] = data['BirthRate'].fillna(data['BirthRate'].median())
data['CO2Emissions'] = data['CO2Emissions'].fillna(data['CO2Emissions'].median())
data['DaystoStartBusiness'] = data['DaystoStartBusiness'].fillna(data['DaystoStartBusiness'].median())
data['EnergyUsage'] = data['EnergyUsage'].fillna(data['EnergyUsage'].median())
data['HealthExpCapita']=data['HealthExpCapita'].fillna(data['HealthExpCapita'].mean())
data['GDP'] = data['GDP'].fillna(data['GDP'].median())
data['InfantMortalityRate'] = data['InfantMortalityRate'].fillna(data['InfantMortalityRate'].median())
data['InternetUsage'] = data['InternetUsage'].fillna(data['InternetUsage'].median())
data['LendingInterest'] = data['LendingInterest'].fillna(data['LendingInterest'].median())
data['LifeExpectancyFemale'] = data['LifeExpectancyFemale'].fillna(data['LifeExpectancyFemale'].median())
data['LifeExpectancyMale'] = data['LifeExpectancyMale'].fillna(data['LifeExpectancyMale'].median())
data['MobilePhoneUsage'] = data['MobilePhoneUsage'].fillna(data['MobilePhoneUsage'].median())
data['TourismInbound'] = data['TourismInbound'].fillna(data['TourismInbound'].median())
data['TourismOutbound'] = data['TourismOutbound'].fillna(data['TourismOutbound'].median())
data['Population15to64'] = data['Population15to64'].fillna(data['Population15to64'].median())
data['Populationmorethan65'] = data['Populationmorethan65'].fillna(data['Populationmorethan65'].median())

print("{} missing values present in whole data.".format(data.isnull().sum().sum()))

data.isnull()

for col in data.select_dtypes(include=np.number):
    data[col] = data[col].fillna(data[col].median())

print("{} missing values present in whole data.".format(data.isnull().sum().sum()))

plt.figure(figsize=(20,15))
sns.heatmap(data.corr(),annot=True)
plt.xticks(rotation=45)
plt.title("Correlation Map of variables", fontsize=15)
plt.show()

def handle_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = np.where((data[column] < lower_bound) | (data[column] > upper_bound),
                           data[column].median(),  # Replace outliers with the median
                           data[column])
    return data

for col in ['GDP', 'HealthExpGDP','InfantMortalityRate', 'TourismInbound', 'TourismOutbound', 'PopulationTotal']:
    data = handle_outliers_iqr(data, col)

# prompt: perform standarisation on the above code

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assuming 'data' is your DataFrame after all the preprocessing steps

# Select numerical columns for standardization
numerical_cols = data.select_dtypes(include=np.number).columns

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the numerical columns
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Now 'data' contains standardized numerical features
print(data.head())

data.head()

!pip install streamlit
!pip install pyngrok

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# ... (your existing data preprocessing code) ...

# Assuming 'Country_encoded' is your target
X = data.drop('Country_encoded', axis=1)
y = data['Country_encoded']

# Convert 'Country_encoded' to integer for classification if it's not already
y = y.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# --- Streamlit app ---
st.title("KNN Model Deployment")

# --- Streamlit app ---
st.title("KNN Model Deployment")

# Input features for prediction
input_features = {}
for col in X.columns:
    # Use mean as default value, but specify data type implicitly
    if data[col].dtype == np.float64:
        input_features[col] = st.number_input(f"Enter {col}:", value=data[col].mean())  # Removed type="float"
    elif data[col].dtype == np.int64:
        input_features[col] = st.number_input(f"Enter {col}:", value=int(data[col].mean()))  # Removed type="int"
    else:  # Handle other data types if needed
        input_features[col] = st.text_input(f"Enter {col}:", value=str(data[col].mode()[0]))

# ... (rest of the code remains the same) ...
# Create a DataFrame from input features
input_df = pd.DataFrame([input_features])

# Make prediction
if st.button("Predict"):
    prediction = knn.predict(input_df)
    st.write(f"Prediction: {le.inverse_transform([int(prediction[0])])[0]}")  # Decode the prediction back to country name


# ... (pyngrok code for deployment remains the same) ...

# --- Add this before the Streamlit app ---
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()  # Create a LabelEncoder object
# Fit the LabelEncoder with your original 'Country' column (from the raw data)
le.fit(pd.read_excel(file_path)['Country'])

