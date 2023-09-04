#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For numpy Arrays 
import numpy as np

# For using pandas DataFrame
import pandas as pd

# For plotting charts
import matplotlib.pyplot as plt
import seaborn as sns

# For plotting in the current window
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


df = pd.read_csv('train.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# ### I. Exploratory Detail Analysis

# In[7]:


df.isnull().sum()


# In[8]:


(df.isnull().sum()/df.shape[0]) * 100


# In[9]:


df.describe()


# In[10]:


print(df['Airline'].unique())
df['Airline'].value_counts()


# In[11]:


df['Departure_City'].unique()


# In[12]:


df['Arrival_City'].unique()


# In[13]:


df['Distance'].unique()


# In[14]:


df['Aircraft_Type'].unique()


# In[15]:


df['Day_of_Week'].unique()


# In[16]:


df['Month_of_Travel'].unique()


# In[17]:


print(df['Demand'].value_counts())
df['Demand'].unique()


# In[18]:


print(df['Weather_Conditions'].value_counts())
df['Weather_Conditions'].unique()


# In[19]:


print(df['Promotion_Type'].value_counts())
df['Promotion_Type'].unique()


# In[20]:


df['Holiday_Season'].unique()


# In[21]:


print(df['Fuel_Price'].unique())


# In[22]:


df['Number_of_Stops'].unique()


# ### II. Treating Missing Values

# In[23]:


df.isnull().sum()


# In[24]:


df['Airline'] = df['Airline'].fillna('Unknown')
df['Aircraft_Type'] = df['Aircraft_Type'].fillna('Unknown')
df['Day_of_Week'] = df['Day_of_Week'].fillna(df['Day_of_Week'].mode()[0])
df['Month_of_Travel'] = df['Month_of_Travel'].fillna(df['Month_of_Travel'].mode()[0])
df['Demand'] = df['Demand'].fillna(df['Demand'].mode()[0])
df['Weather_Conditions'] = df['Weather_Conditions'].fillna(df['Weather_Conditions'].mode()[0])
df['Promotion_Type'] = df['Promotion_Type'].fillna(df['Promotion_Type'].mode()[0])


# In[25]:


df['Departure_City'] = df['Departure_City'].fillna('Unknown')
df['Arrival_City'] = df['Arrival_City'].fillna('Unknown')


# In[26]:


df['Distance'] = df['Distance'].fillna(round(df['Distance'].median(),1))
df['Fuel_Price'] = df['Fuel_Price'].fillna(round(df['Fuel_Price'].median(),2))


# In[27]:


df.isnull().sum()


# ### III. Encoding Features

# In[28]:


Airline_mapper = { 'Airline A': 0, 'Airline B': 1, 'Airline C': 2, 'Unknown': 3}

AircraftType_mapper = { 'Boeing 787': 0, 'Airbus A320': 1, 'Boeing 737': 2, 'Boeing 777': 3, 'Airbus A380': 4, 'Unknown': 5}

DayOfWeek_mapper = {'Wednesday':3, 'Sunday': 0, 'Thursday': 4, 'Tuesday': 2, 'Friday': 5, 'Monday': 1, 'Saturday': 6}

MonthOfTravel_mapper = {'December': 11, 'March': 2, 'September': 8, 'February': 1, 'January': 0, 'May': 4, 'June': 5, 
                        'July': 6, 'August': 7, 'April': 3, 'October': 9, 'November': 10}

Demand_mapper = {'Low': 0, 'High': 2, 'Medium': 1}

Weather_mapper = {'Rain': 0, 'Cloudy': 1, 'Clear': 2, 'Snow': 3}

PromotionType_mapper = {'Special Offer': 0, 'None': 2, 'Discount': 1}

HolidaySeason_mapper = {'Summer': 1, 'Spring' :4, 'Fall': 2, 'None': 0, 'Winter': 3}


# In[29]:


df['Promotion_Type'] = df['Promotion_Type'].replace(PromotionType_mapper)
df['Weather_Conditions'] = df['Weather_Conditions'].replace(Weather_mapper)
df['Demand'] = df['Demand'].replace(Demand_mapper)
df['Month_of_Travel'] = df['Month_of_Travel'].replace(MonthOfTravel_mapper)
df['Day_of_Week'] = df['Day_of_Week'].replace(DayOfWeek_mapper)
df['Aircraft_Type'] = df['Aircraft_Type'].replace(AircraftType_mapper)
df['Airline'] = df['Airline'].replace(Airline_mapper)
df['Holiday_Season'] = df['Holiday_Season'].replace(HolidaySeason_mapper)


# ### III. Feature Engineering

# In[30]:


from datetime import datetime
df['DepartureHour'] = pd.to_datetime(df['Departure_Time']).dt.hour
df['DepartureMinute'] = pd.to_datetime(df['Departure_Time']).dt.minute


# In[31]:


from datetime import datetime
df['ArrivalHour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['ArrivalMinute'] = pd.to_datetime(df['Arrival_Time']).dt.minute


# In[32]:


df['Duration_hour'] = df['Duration'].astype(int)
df['Duration_Minute'] = round((df['Duration'] - df['Duration'].astype(int))*60,0).astype(int)


# In[33]:


df.info()


# In[34]:


df.head()


# ### IV. Feature Scaling

# In[35]:


featured_df = df.drop(['Flight_ID','Departure_City','Arrival_City','Departure_Time','Arrival_Time'], axis=1)
featured_df.head()


# In[36]:


from sklearn.preprocessing import StandardScaler

ss_scaling = StandardScaler()
df_ss_scaled = ss_scaling.fit_transform(featured_df.drop('Flight_Price', axis=1))
df_ss_scaled = pd.DataFrame(df_ss_scaled, columns = featured_df.drop('Flight_Price', axis=1).columns)
df_ss_scaled.head()


# In[37]:


df.info()


# In[38]:


df.head()


# ### V. Train and Vlidation Split

# In[39]:


from sklearn.model_selection import train_test_split

X = df_ss_scaled
Y = df['Flight_Price']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=1234)
print(f'Shape of Training DataSet- X_train {X_train.shape} and X_test {X_test.shape}')
print(f'Shape of Training DataSet- Y_train {Y_train.shape} and Y_test {Y_test.shape}')


# ### VI. Linear Regression Model

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics

linear_regression = LinearRegression()
linear_regression.fit(X_train, Y_train)

y_predict_train = linear_regression.predict(X_train)

print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_train))}')

y_predict_test = linear_regression.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_test))}')


# In[41]:


pd.DataFrame(zip(linear_regression.coef_.T, X_train.columns))


# ### VI. Ridge Regression Model

# In[42]:


from sklearn.linear_model import Ridge

ridge_regression = Ridge()
ridge_regression.fit(X_train,Y_train)

y_predict_ridge_train = ridge_regression.predict(X_train)
print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_ridge_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_ridge_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_ridge_train))}')

y_predict_ridge_test = ridge_regression.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_ridge_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_ridge_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_ridge_test))}')


# In[43]:


pd.DataFrame(zip(ridge_regression.coef_.T, X_train.columns))


# ### VII. Lasso Regression Model

# In[44]:


from sklearn.linear_model import Lasso

lasso_regression = Lasso()
lasso_regression.fit(X_train,Y_train)

y_predict_lasso_train = lasso_regression.predict(X_train)
print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_lasso_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_lasso_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_lasso_train))}')

y_predict_lasso_test = lasso_regression.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_lasso_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_lasso_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_lasso_test))}')


# ### VIII ExtraTree Regressor

# In[45]:


from sklearn.ensemble import ExtraTreesRegressor

extratree_regression = ExtraTreesRegressor()
extratree_regression.fit(X_train,Y_train)

y_predict_extratree_train = extratree_regression.predict(X_train)
print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_extratree_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_extratree_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_extratree_train))}')

y_predict_extratree_test = extratree_regression.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_extratree_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_extratree_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_extratree_test))}')


# ### IX. RandomForest Regressor

# In[46]:


from sklearn.ensemble import RandomForestRegressor

randomforest_regression = RandomForestRegressor()
randomforest_regression.fit(X_train,Y_train)

y_predict_randomforest_train = randomforest_regression.predict(X_train)
print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_randomforest_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_randomforest_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_randomforest_train))}')

y_predict_randomforest_test = randomforest_regression.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_randomforest_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_randomforest_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_randomforest_test))}')


# ### X. Voting Regressor

# In[47]:


from sklearn.ensemble import VotingRegressor

reg_estimator = [('lr',linear_regression),('rid',ridge_regression),('lasso',lasso_regression), ('extra', extratree_regression), 
                 ('rf', randomforest_regression)]

voting_regressor = VotingRegressor(estimators=reg_estimator)
voting_regressor.fit(X_train, Y_train)

y_predict_voting_train = voting_regressor.predict(X_train)

print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_voting_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_voting_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_voting_train))}')

y_predict_voting_test = voting_regressor.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_voting_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_voting_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_voting_test))}')


# ### XI. KNN Regressor

# In[48]:


from sklearn.neighbors import KNeighborsRegressor

knn_regressor = KNeighborsRegressor(n_neighbors= 17)
knn_regressor.fit(X_train, Y_train)

y_predict_knn_train = knn_regressor.predict(X_train)

print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_knn_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_knn_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_knn_train))}')

y_predict_knn_test = knn_regressor.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_knn_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_knn_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_knn_test))}')


# ### XII. GridSearch RandomForest Regressor

# In[53]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
        'max_features': [3,5,7,9],
        'n_estimators': [25,30,35,40,45], # number of trees in the random forest
        'max_depth' : [5,7,10,15], # maximum number of levels allowed in each decision tree
        'min_samples_split' : [3,6,9,12], #,15,20,25,30, # minimum sample number to split a node
        'min_samples_leaf' : [10,12,15,18] # minimum sample number that can be stored in a leaf node
}

rf_tuned = RandomForestRegressor(random_state=123)

gs_rf = GridSearchCV(estimator = rf_tuned, 
                      param_grid = param_grid, 
                      cv = 3,
                      verbose = True,
                      n_jobs= -1
                      )

gs_rf.fit(X_train, Y_train)


# In[55]:


gs_rf.best_estimator_


# In[54]:


y_predict_randomforest_train = gs_rf.predict(X_train)
print(f'R2 Score for Training Dataset is {metrics.r2_score(Y_train,y_predict_randomforest_train)}')
print(f'MSE for Training Dataset is {metrics.mean_absolute_error(Y_train, y_predict_randomforest_train)}')
print(f'RMSE Score for Training Dataset is {np.sqrt(metrics.mean_squared_error(Y_train, y_predict_randomforest_train))}')

y_predict_randomforest_test = gs_rf.predict(X_test)

print(f'\n\nR2 Score for Test Dataset is {metrics.r2_score(Y_test,y_predict_randomforest_test)}')
print(f'MSE for Test Dataset is {metrics.mean_absolute_error(Y_test, y_predict_randomforest_test)}')
print(f'RMSE Score for Test Dataset is {np.sqrt(metrics.mean_squared_error(Y_test, y_predict_randomforest_test))}')


# ## Treating Test data

# In[56]:


df_test = pd.read_csv('test.csv')


# In[57]:


df_test.head()


# In[58]:


df_test.tail()


# In[59]:


df_test.info()


# ### I. Exploratory Detail Analysis

# In[60]:


df_test.isnull().sum()


# In[61]:


(df_test.isnull().sum()/df_test.shape[0]) * 100


# In[62]:


df_test.describe()


# In[63]:


print(df_test['Airline'].unique())
df_test['Airline'].value_counts()


# In[64]:


df_test['Departure_City'].unique()


# In[65]:


df_test['Arrival_City'].unique()


# In[66]:


df_test['Distance'].unique()


# In[67]:


df_test['Aircraft_Type'].unique()


# In[68]:


df_test['Day_of_Week'].unique()


# In[69]:


df_test['Month_of_Travel'].unique()


# In[70]:


print(df_test['Demand'].value_counts())
df_test['Demand'].unique()


# In[71]:


print(df_test['Weather_Conditions'].value_counts())
df_test['Weather_Conditions'].unique()


# In[72]:


print(df_test['Promotion_Type'].value_counts())
df_test['Promotion_Type'].unique()


# In[73]:


df_test['Holiday_Season'].unique()


# In[74]:


print(df_test['Fuel_Price'].unique())


# In[75]:


df_test['Number_of_Stops'].unique()


# ### II. Treating Missing Values

# In[76]:


df_test.isnull().sum()


# In[77]:


df_test['Airline'] = df_test['Airline'].fillna('Unknown')
df_test['Aircraft_Type'] = df_test['Aircraft_Type'].fillna('Unknown')
df_test['Day_of_Week'] = df_test['Day_of_Week'].fillna(df_test['Day_of_Week'].mode()[0])
df_test['Month_of_Travel'] = df_test['Month_of_Travel'].fillna(df_test['Month_of_Travel'].mode()[0])
df_test['Demand'] = df_test['Demand'].fillna(df_test['Demand'].mode()[0])
df_test['Weather_Conditions'] = df_test['Weather_Conditions'].fillna(df_test['Weather_Conditions'].mode()[0])
df_test['Promotion_Type'] = df_test['Promotion_Type'].fillna(df_test['Promotion_Type'].mode()[0])


# In[78]:


df_test['Departure_City'] = df_test['Departure_City'].fillna('Unknown')
df_test['Arrival_City'] = df_test['Arrival_City'].fillna('Unknown')


# In[79]:


df_test['Distance'] = df_test['Distance'].fillna(round(df_test['Distance'].median(),1))
df_test['Fuel_Price'] = df_test['Fuel_Price'].fillna(round(df_test['Fuel_Price'].median(),2))


# In[80]:


df_test.isnull().sum()


# ### III. Encoding Features

# In[81]:


Airline_mapper = { 'Airline A': 0, 'Airline B': 1, 'Airline C': 2, 'Unknown': 3}

AircraftType_mapper = { 'Boeing 787': 0, 'Airbus A320': 1, 'Boeing 737': 2, 'Boeing 777': 3, 'Airbus A380': 4, 'Unknown': 5}

DayOfWeek_mapper = {'Wednesday':3, 'Sunday': 0, 'Thursday': 4, 'Tuesday': 2, 'Friday': 5, 'Monday': 1, 'Saturday': 6}

MonthOfTravel_mapper = {'December': 11, 'March': 2, 'September': 8, 'February': 1, 'January': 0, 'May': 4, 'June': 5, 
                        'July': 6, 'August': 7, 'April': 3, 'October': 9, 'November': 10}

Demand_mapper = {'Low': 0, 'High': 2, 'Medium': 1}

Weather_mapper = {'Rain': 0, 'Cloudy': 1, 'Clear': 2, 'Snow': 3}

PromotionType_mapper = {'Special Offer': 0, 'None': 2, 'Discount': 1}

HolidaySeason_mapper = {'Summer': 1, 'Spring' :4, 'Fall': 2, 'None': 0, 'Winter': 3}


# In[82]:


df_test['Promotion_Type'] = df_test['Promotion_Type'].replace(PromotionType_mapper)
df_test['Weather_Conditions'] = df_test['Weather_Conditions'].replace(Weather_mapper)
df_test['Demand'] = df_test['Demand'].replace(Demand_mapper)
df_test['Month_of_Travel'] = df_test['Month_of_Travel'].replace(MonthOfTravel_mapper)
df_test['Day_of_Week'] = df_test['Day_of_Week'].replace(DayOfWeek_mapper)
df_test['Aircraft_Type'] = df_test['Aircraft_Type'].replace(AircraftType_mapper)
df_test['Airline'] = df_test['Airline'].replace(Airline_mapper)
df_test['Holiday_Season'] = df_test['Holiday_Season'].replace(HolidaySeason_mapper)


# ### III. Feature Engineering

# In[83]:


from datetime import datetime
df_test['DepartureHour'] = pd.to_datetime(df_test['Departure_Time']).dt.hour
df_test['DepartureMinute'] = pd.to_datetime(df_test['Departure_Time']).dt.minute


# In[84]:


from datetime import datetime
df_test['ArrivalHour'] = pd.to_datetime(df_test['Arrival_Time']).dt.hour
df_test['ArrivalMinute'] = pd.to_datetime(df_test['Arrival_Time']).dt.minute


# In[85]:


df_test['Duration_hour'] = df_test['Duration'].astype(int)
df_test['Duration_Minute'] = round((df_test['Duration'] - df_test['Duration'].astype(int))*60,0).astype(int)


# In[86]:


df_test.info()


# In[87]:


df_test.head()


# In[88]:


df_test_final = df_test.drop(['Flight_ID','Departure_City','Arrival_City','Departure_Time','Arrival_Time'], axis=1)


# In[89]:


df_test_final.head()


# In[90]:


df_test_final.info()


# ### IV. Feature Scaling

# In[91]:


# df_test_final = df_test.drop(['Flight_ID','Departure_City','Arrival_City','Departure_Time','Arrival_Time'], axis=1)
df_test_final.head()


# In[93]:


from sklearn.preprocessing import StandardScaler

ss_scaling = StandardScaler()
df_ss_scaled = ss_scaling.fit_transform(df_test_final)
df_ss_scaled = pd.DataFrame(df_ss_scaled, columns = df_test_final.columns)
df_ss_scaled.head()


# In[94]:


df_test_final.info()


# In[95]:


df_ss_scaled.head()


# In[96]:


X_test_prediction_set = df_ss_scaled
y_predict_set = gs_rf.predict(X_test_prediction_set)


# In[98]:


data = {
    'Flight_ID':list(df_test['Flight_ID']),
    'Flight_Price':list(y_predict_set)
}

df_prediction_01 = pd.DataFrame(data)
df_prediction_01.to_csv('submission-scaled.csv', index=False)

