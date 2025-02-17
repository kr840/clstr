#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[ ]:





# In[3]:


#from google.colab import drive


# In[4]:


#drive.mount('/content/drive')


# In[5]:


data=pd.read_excel("WDMDataset.xlsx")


# In[6]:


data.head()


# In[ ]:





# In[7]:


data.tail()


# In[8]:


data.columns


# In[9]:


data.info(20)


# In[10]:


data.head()


# In[11]:


data.tail()


# In[12]:


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


# In[13]:


data.info()


# In[14]:


data.head(20)


# In[15]:


print(data.describe())


# In[16]:


# Check for extreme values and potential outliers
print(data[['Birth Rate', 'CO2 Emissions', 'GDP', 'Population Total']].quantile([0.25, 0.5, 0.75]))


# In[41]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['Country_encoded'] = le.fit_transform(data['Country'])
data['Country_encoded']=data['Country_encoded'].astype(float)
data.drop(['Country'],axis=1,inplace=True)


# In[18]:


data.info(10)


# In[19]:


data.head(20)


# In[20]:


data.isnull().sum()


# In[21]:


data['Tourism Inbound'] = data['Tourism Inbound'].fillna(data['Tourism Inbound'].median())
data['Tourism Outbound'] = data['Tourism Outbound'].fillna(data['Tourism Outbound'].median())


# In[22]:


data.head(20)


# In[23]:


data['Tourism Outbound'] = data['Tourism Outbound'].astype(str).str.replace(r'[^\d.]', '', regex=True)
data['Tourism Outbound'] = pd.to_numeric(data['Tourism Outbound'], errors='coerce')
data['Tourism Outbound'] = data['Tourism Outbound'].fillna(data['Tourism Outbound'].median())

data['Tourism Inbound'] = data['Tourism Inbound'].astype(str).str.replace(r'[^\d.]', '', regex=True)
data['Tourism Inbound'] = pd.to_numeric(data['Tourism Inbound'], errors='coerce')
data['Tourism Inbound'] = data['Tourism Inbound'].fillna(data['Tourism Inbound'].median())


# In[24]:


data = pd.read_excel("/Users/abhisreeravela/Desktop/clusteranlysis/WDMDataset.xlsx")

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




# In[25]:


data.info()


# In[26]:


data.tail(20)


# In[27]:


data.describe()


# In[28]:


data = data.drop(['Number of Records'],axis=1)


# In[29]:


data.isnull().sum()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


# In[31]:


if 'Ease of Business' in data.columns:
  data = data.drop('Ease of Business', axis=1)


# In[32]:


import missingno as msno
msno.bar(data)
plt.show()


# In[33]:


data = data.rename(columns={'Birth Rate': 'BirthRate', 'Business Tax Rate': 'BusinessTaxRate','CO2 Emissions':'CO2Emissions','Days to Start Business':'DaystoStartBusiness','Ease of Business':'EaseofBusiness','Energy Usage':'EnergyUsage',
                            'Health Exp % GDP':'HealthExpGDP','Health Exp/Capita':'HealthExpCapita','Hours to do Tax':'HourstodoTax','Infant Mortality Rate':'InfantMortalityRate','Internet Usage':'InternetUsage','Lending Interest':'LendingInterest',
                            'Life Expectancy Female':'LifeExpectancyFemale','Life Expectancy Male':'LifeExpectancyMale','Mobile Phone Usage':'MobilePhoneUsage','Number of Records':'NumberofRecords','Population 0-14':'Population0to14',
                            'Population 15-64':'Population15to64','Population 65+':'Populationmorethan65','Population Total':'PopulationTotal','Population Urban':'PopulationUrban','Tourism Inbound':'TourismInbound','Tourism Outbound':'TourismOutbound'})
data.columns


# In[34]:


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


# In[35]:


data['BusinessTaxRate'] = data['BusinessTaxRate'].fillna(data['BusinessTaxRate'].mean())

data['HealthExpGDP'] = data['HealthExpGDP'].fillna(data['HealthExpGDP'].mean())
data['HourstodoTax'] = data['HourstodoTax'].fillna(data['HourstodoTax'].mean())
data['Population0to14'] = data['Population0to14'].fillna(data['Population0to14'].mean())


# In[36]:


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


# In[37]:


print("{} missing values present in whole data.".format(data.isnull().sum().sum()))


# In[38]:


data.isnull()


# In[39]:


for col in data.select_dtypes(include=np.number):
    data[col] = data[col].fillna(data[col].median())

print("{} missing values present in whole data.".format(data.isnull().sum().sum()))


# In[42]:


plt.figure(figsize=(20,15))
sns.heatmap(data.corr(),annot=True)
plt.xticks(rotation=45)
plt.title("Correlation Map of variables", fontsize=15)
plt.show()


# In[43]:


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


# In[44]:


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


# In[45]:


data.head()


# In[46]:


data1 = data.copy()    # For method 1
data2 = data.copy()    # For method 2
data3 = data.copy()    # For method 3


# In[47]:


fig, axes=plt.subplots(6,4,figsize=(14,12),sharex=False,sharey=False)
sns.boxplot(data1.BirthRate,ax=axes[0,0])
sns.boxplot(data1.BusinessTaxRate,ax=axes[0,1])
sns.boxplot(data1.CO2Emissions,ax=axes[0,2])
sns.boxplot(data1.DaystoStartBusiness,ax=axes[0,3])
#sns.boxplot(data1.EaseofBusiness,ax=axes[1,0])
sns.boxplot(data1.EnergyUsage,ax=axes[1,1])
sns.boxplot(data1.GDP,ax=axes[1,2])
sns.boxplot(data1.HealthExpGDP,ax=axes[1,3])
sns.boxplot(data1.HealthExpCapita,ax=axes[2,0])
sns.boxplot(data1.HourstodoTax,ax=axes[2,1])
sns.boxplot(data1.InfantMortalityRate,ax=axes[2,2])
sns.boxplot(data1.InternetUsage,ax=axes[2,3])
sns.boxplot(data1.LendingInterest,ax=axes[3,0])
sns.boxplot(data1.LifeExpectancyFemale,ax=axes[3,1])
sns.boxplot(data1.LifeExpectancyMale,ax=axes[3,2])
sns.boxplot(data1.MobilePhoneUsage,ax=axes[3,3])
sns.boxplot(data1.MobilePhoneUsage,ax=axes[4,0])
sns.boxplot(data1.Population0to14,ax=axes[4,1])
sns.boxplot(data1.Population15to64,ax=axes[4,2])
sns.boxplot(data1.Populationmorethan65,ax=axes[4,3])
sns.boxplot(data1.PopulationTotal,ax=axes[5,0])
sns.boxplot(data1.PopulationUrban,ax=axes[5,1])
sns.boxplot(data1.TourismInbound,ax=axes[5,2])
sns.boxplot(data1.TourismOutbound,ax=axes[5,3])
plt.tight_layout(pad=2.0)


# In[48]:


# Outlier removal
Q1 = data1.quantile(0.25)
Q3 = data1.quantile(0.75)
IQR = Q3 - Q1

data1 = data1[~((data1 < (Q1 - 1.5 * IQR)) | (data1 > (Q3 + 1.5 * IQR))).any(axis=1)]
data1.shape


# In[49]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale_data = scaler.fit_transform(data1)
#Apply PCA
from sklearn.decomposition import PCA
pc = PCA()
pc_components = pc.fit_transform(scale_data)
# The amount of variance that each PCA explains is
pc.explained_variance_


# In[50]:


var = pc.explained_variance_ratio_
var


# In[51]:


var1 = np.cumsum(np.round(var,decimals=4)*100)
var1


# In[52]:


# Variance plot for PCA components obtained
plt.plot(var1,color='red')
plt.xlabel('Index')
plt.ylabel('Cumulative Percentage')
plt.show()


# In[53]:


data_pca = pc_components[:,:15]
## Plot between PCA's
x=pc_components[:,0]
y=pc_components[:,1]
z=pc_components[:,2]
plt.scatter(x,y)
plt.scatter(x,z)
plt.scatter(y,z)
plt.show()


# In[54]:


#K-means Clustring
#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[55]:


## creating clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(data_pca)
plt.scatter(data_pca[y_kmeans == 0, 0], data_pca[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data_pca[y_kmeans == 1, 0], data_pca[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(data_pca[y_kmeans == 2, 0], data_pca[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of measurements')
plt.legend()
plt.show()


# In[56]:


## Accuracy check
from sklearn.metrics import silhouette_score
s1_kmeans = silhouette_score(data_pca, y_kmeans)
print('Silhouette Score for K-means clustring :', s1_kmeans)


# In[57]:


#Hierarchy Clustring
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(data_pca, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[58]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(data_pca)
plt.scatter(data_pca[y_hc == 0, 0], data_pca[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data_pca[y_hc == 1, 0], data_pca[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(data_pca[y_hc == 2, 0], data_pca[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(data_pca[y_hc == 3, 0], data_pca[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.title('Clusters of measurments')
plt.legend()
plt.show()


# In[59]:


## Accuracy check
s1_hierarchy = silhouette_score(data_pca,y_hc)
print('Silhouette Score for Hierarchy clustring :',s1_hierarchy)


# In[60]:


#DBSCAN
from sklearn.cluster import DBSCAN

eps = 0.5
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the data to obtain clustering labels
dbscan_labels = dbscan.fit_predict(data_pca)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels)
plt.show()


# In[61]:


data1['cluster']=dbscan.labels_
data1.head()


# In[62]:


data1[data1['cluster']==-1]


# In[63]:


s1_dbscan = silhouette_score(data_pca, dbscan_labels)
print("Silhouette Score for DBSCAN is:", s1_dbscan)


# In[ ]:





# In[ ]:





# In[73]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ... (Your existing code) ...

# Assuming 'data_pca' is your PCA-transformed data and 'y_kmeans' are your k-means cluster labels
# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_pca, y_kmeans, test_size=0.2, random_state=42)


# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the KNN model
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy}")

import streamlit as st
import pandas as pd
import numpy as np
# ... (Your existing code for data preprocessing and model training) ...

# Streamlit app
st.title("World Development Measurement Analysis")

# Display the preprocessed data
if st.checkbox("Show Preprocessed Data"):
    st.write(data.head())

# ... (Add other interactive elements for visualization, etc.) ...

# Display the KNN accuracy
st.write(f"KNN Accuracy: {accuracy}")

# Example: Display a cluster plot (replace with your actual plot)
# import matplotlib.pyplot as plt
# plt.scatter(data_pca[:, 0], data_pca[:, 1], c=y_kmeans)
# st.pyplot(plt)

import pickle


with open("model_KNN", "wb") as model_file:
    pickle.dump(knn, model_file)
    
    
 
    pickle.dump(knn,open('Model_KNN','wb'))


# In[68]:


knn.fit(X_train, y_train)


# In[ ]:




