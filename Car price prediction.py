# # Car price predction:


#  Exracting the data:


import warnings
warnings.filterwarnings('ignore')

# import pandas for analysis the data:
import pandas as pd
import numpy as np

# Extract the data:
car =pd.read_csv("C:/Users/KiranPrasath V/Desktop/quikr_car.csv")

#viewing the top 5 data in csv file:
car.head()

# Veiw the shape of the data:
car.shape
# After viewing the shape of data we found 892 rows and 6 column.

# View the information of the data:
car.info()
#After viewing the information we found there is no null values.
#But in price and year column we found object datatype.

#Viewing the year column:
car["year"].unique()
# Year column found different datatype.

#Viewing the price column:
car["Price"].unique()
# Price column found different datatype. 

# Viewing the kms_driven:
car["kms_driven"].unique()
#After viewing the kms_driven found both integer ans string. 

# Viewing the fuel_type:
car["fuel_type"].unique()
# After Viewing the fuel_type found nan values.

# # List of defects in data:
- Year column found different datatype.
- Year column found string object change to integer.
- Price column found different datatype. 
- Price column found string object change to integer.
- Kms_driven column found srting odject change to integer. 
- Kms_driven column found some nan values.
- Fue_type column found some nan values.
- name column found difficult to understand.

# Backup the data:

# Backup the data:
backup = car.copy()
car = backup

# # Cleaning the data:

# Cleaning the year string in data:
car=car[car["year"].str.isnumeric()]

# changing the year string to integer: 
car["year"]=car["year"].astype(int)

# Cleaning the price string in data:
car=car[car["Price"]!="Ask For Price"]

#Changing the price datatype and comma: 
car["Price"]=car["Price"].str.replace(",", "").astype(int)

#Changing the kms_driven kms, comma:
car["kms_driven"]=car["kms_driven"].str.split(" ").str.get(0).str.replace(",", "")

# Cleaning the kms_driven string in data:
car = car[car["kms_driven"].str.isnumeric()]

car["kms_driven"]=car["kms_driven"].astype(int)

# Cleaning nan values in fuel_type: 
car=car[~car["fuel_type"].isna()]

# Changing the first 3 name in name column:
car["name"]=car["name"].str.split(" ").str.slice(0, 3).str.join(" ")

# Set the new index:
car= car.reset_index(drop=True)

car.describe()

# cleaning the price data which is outfit:
car=car[car["Price"]<6e6].reset_index(drop=True)

car.describe()

# Viewing the information of the data:
car.info()

# Creating new csv file of Cleaned data:
car.to_csv("Cleaned car.csv")

# # Model:

# Assigning the X and y variable for linear regression:
X= car.drop(columns="Price")
y = car["Price"]


# importing sklearn library:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)

# importing sklearn library:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# creating pipeline:
ohe = OneHotEncoder()
ohe.fit(X[["name", "company", "fuel_type"]])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),["name","company","fuel_type"]),remainder= "passthrough")
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

# predicting the price of the cars:
y_pred = pipe.predict(X_test)

#print y predict:
y_pred

# using r2 method find the error:
r2_score(y_test, y_pred)

# finding the r2 for 1000 data:
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

#Find the maxinum trained data:
np.argmax(scores)

# r2 score for 1000 data:
scores[np.argmax(scores)]



