#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import chart_studio.plotly as py
import plotly.graph_objs as go 
from plotly.offline import plot,iplot

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)


# In[3]:


Britannia=pd.read_csv('STOCK_VAL.csv')


# In[4]:


Britannia


# In[5]:


Britannia.info()


# In[1]:


Britannia['DATE'] = pd.to_datetime(Britannia['DATE']) 


# In[11]:


# exploring the data 

print(f'dataframe contains stock ptices between {Britannia.DATE.min()} {Britannia.DATE.max()}')
print(f'total days = {(Britannia.DATE.max()  - Britannia.DATE.min()).days} days')


# In[12]:


Britannia.describe()


# In[6]:


Britannia[['OPEN', 'HIGH', 'LOW', 'CLOSE']].plot(kind='box')


# In[7]:


# setting the layout for our plot
layout = go.Layout(
    title='stock prices of Britannia',
    xaxis=dict(
        title='YEAR',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#000000'
        )
    ),

 yaxis=dict(
    title='Price',
    titlefont=dict(
       family='Courier New, monospace',
       size=18,
       color='#000000'
    )
 ),
    width=900,
    height=500
)

Britannia_data=[{'x':Britannia['DATE'], 'y':Britannia['CLOSE']}]
plot=go.Figure(data=Britannia_data, layout=layout)


# In[15]:


#PLOT(PLOT) #plotting offline
iplot(plot)


# In[16]:


# Building the regression model 
from sklearn.model_selection import train_test_split

#for preprocessing 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#for model evaluation 
from sklearn.metrics import mean_squared_error 


# In[17]:


# split data into train and test sets
X=np.array(Britannia.index).reshape(-1,1)
y=Britannia['CLOSE']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[18]:


# feature scaling 
scaler=StandardScaler().fit(X_train)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


#creating a liear model 
lm= LinearRegression()
lm.fit(X_train,y_train)


# In[21]:


trace0 = go.Scatter(
     x=X_train.T[0],
     y=y_train,
     mode='markers',
     name='Actual'
)

# Add a small offset to the x values of the predicted data
x_predicted = X_train.T[0] + 0.1

trace1 = go.Scatter(
     x=x_predicted,
     y=y_train,
     mode='markers',
     name='Predicted'
)

Britannia_data = [trace0, trace1]
layout.xaxis.title.text = 'year'
plot2 = go.Figure(data=Britannia_data, layout=layout)

iplot(plot2)


# In[26]:


from sklearn.metrics import mean_squared_error

# Calculate the score for the training and test data
train_score = mean_squared_error(y_train, lm.predict(X_train))
test_score = mean_squared_error(y_test, lm.predict(X_test))

# Create the score report
score_report = f'''
{'Metrics'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'MSE'.ljust(10)}{train_score}\t{test_score}
'''

print(score_report)


# In[28]:


presicion_Score = test_score/train_score
print(presicion_Score)


# In[ ]:




