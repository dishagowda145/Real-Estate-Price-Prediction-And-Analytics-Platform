#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("kc_house_data.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
sns.despine()
plt.show()


# In[ ]:


plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")


# In[ ]:


plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")


# In[ ]:


plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")


# In[ ]:


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


# In[ ]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[ ]:


plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")


# In[ ]:


train1 = data.drop(['id', 'price'],axis=1)


# In[ ]:


train1.head()


# In[ ]:


data.floors.value_counts().plot(kind='bar')


# In[ ]:


plt.scatter(data.floors,data.price)


# In[ ]:


plt.scatter(data.condition,data.price)


# In[ ]:


plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg = LinearRegression()


# In[ ]:


labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)


# In[ ]:


reg.fit(x_train,y_train)


# In[ ]:


reg.score(x_test,y_test)


# In[ ]:


from sklearn import ensemble   # <--- Add this line!

clf = ensemble.GradientBoostingRegressor(
    n_estimators=400, 
    max_depth=5, 
    min_samples_split=2, 
    learning_rate=0.1, 
    loss='squared_error',
    verbose=1   # <--- This tells it to print the training status
)


# In[ ]:


clf.fit(x_train, y_train)


# In[ ]:


clf.score(x_test,y_test)


# In[ ]:


t_sc = np.zeros((clf.n_estimators,), dtype=np.float64)


# In[ ]:


y_pred = reg.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

for i, y_pred in enumerate(clf.staged_predict(x_test)):
    t_sc[i] = mean_squared_error(y_test, y_pred)


# In[ ]:


testsc = np.arange((clf.n_estimators)) + 1


# In[ ]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')


# In[ ]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[ ]:


pca = PCA()


# In[ ]:


pca.fit_transform(scale(train1))


# In[ ]:





# In[ ]:




