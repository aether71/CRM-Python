#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Abhishek\Desktop\bank.csv')
df.head()


# In[2]:


df.info()


# In[3]:


df.describe()


# In[9]:



df.groupby('deposit').count()


# In[6]:


# DEPOSIT VALUE COUNTS
df_deposit=pd.DataFrame({'YES':[5289],'NO':[5873]})
sns.barplot(data=df_deposit)


# In[3]:


plt.hist(x='age', data=df, bins = 15)


# In[6]:



df_age = pd.DataFrame()
df_age['age_y'] = (df[df['deposit'] == 'yes'][['deposit','age']].describe())['age']
df_age['age_n'] = (df[df['deposit'] == 'no'][['deposit','age']].describe())['age']
df_age.plot.bar()
#df_age.drop(['count','std', '25%', '50%', '75%']).plot.bar()


# In[112]:


plt.figure(figsize=(20,8))
sns.countplot(x=df['job'],hue=df['deposit'],palette="Set2")


# Customers with management profile and students are more inclined towards a term deposit.

# In[110]:


sns.countplot(x=df['marital'],hue=df['deposit'] ,palette="Set3")


# Married customers are less inclined towards a term deposit.

# In[117]:


sns.countplot(x=df['education'],hue=df['deposit'],palette="Set1")


# Customers with tertiary education background are more inclined towards a term deposit.

# In[3]:


sns.countplot(x=df['default'],hue=df['deposit'])


# Defaulter customers are more reluctant towards a term deposit. 

# In[4]:



df_bal = pd.DataFrame()
df_bal['bal_y'] = (df[df['deposit'] == 'yes'][['deposit','balance']].describe())['balance']
df_bal['bal_n'] = (df[df['deposit'] == 'no'][['deposit','balance']].describe())['balance']
df_bal.drop(['count','std', '25%', '50%', '75%']).plot.bar()


# In[5]:


df['deposit'] == 'yes'


# Customers with greater bank balance are more inclined towards a term deposit.

# In[118]:


sns.countplot(x=df['housing'],hue=df['deposit'],palette="Set2")


# Customers with no Housing loan are more inclined towards a term deposit.

# In[8]:



pd.crosstab(df['loan'],df['deposit'])


# In[119]:


sns.countplot(x=df['loan'],hue=df['deposit'],palette="Set1")


# Customers with no personal loan are more inclined towards a term deposit.

# In[10]:


sns.countplot(x=df['contact'],hue=df['deposit'])


# Customers who were approached via 'cellular' are more inclined towards a term deposit.

# In[11]:



df_day = pd.DataFrame()
df_day['day_y'] = (df[df['deposit'] == 'yes'][['deposit','day']].describe())['day']
df_day['day_n'] = (df[df['deposit'] == 'no'][['deposit','day']].describe())['day']

df_day.drop(['count','std', '25%', '50%', '75%']).plot.bar()


# Customers are more reluctant towards a term deposit in the beginning of the month.

# In[126]:


sns.countplot(x=df['month'],hue=df['deposit'],palette="Set2")


# Customers are more reluctant towards a term deposit during the summer seasons (May to August)

# In[13]:



df_dur = pd.DataFrame()
df_dur['day_y'] = (df[df['deposit'] == 'yes'][['deposit','duration']].describe())['duration']
df_dur['day_n'] = (df[df['deposit'] == 'no'][['deposit','duration']].describe())['duration']


# In[14]:



df_dur.drop(['count','std', '25%', '50%', '75%']).plot.bar()


# Chances of successfully locking a customer for a term deposit substantially increase with higher duration of conversation

# In[15]:



df_camp = pd.DataFrame()
df_camp['day_y'] = (df[df['deposit'] == 'yes'][['deposit','campaign']].describe())['campaign']
df_camp['day_n'] = (df[df['deposit'] == 'no'][['deposit','campaign']].describe())['campaign']

df_camp.drop(['count','std', '25%', '50%', '75%']).plot.bar()


# Chances of successfully locking a customer for a term deposit decreases with more number of contacts.

# In[16]:



sns.barplot(y=df['pdays'],x=df['deposit'])


# More the number of days that passed by after the customer was last contacted from a previous campaign, the customer is inclined towards term deposit

# In[17]:



sns.barplot(y=df['previous'],x=df['deposit'])


# Lesser the number of contacts performed before this campaign and for this customer,  the customer is inclined towards term deposit

# In[19]:


# FINDING NULL VALUES 
df.isnull().sum()

print("Number of null values in the columns:\n\n {}".format(df.isnull().sum()))


# In[23]:


df_columns = df.select_dtypes(include = 'int64')
for i in df_columns:
    print("Column Name: ",i)
#list of all numerical columns


# In[24]:


sns.boxplot(y=df['age'],x=df['deposit'])


# In[25]:


sns.boxplot(y=df['balance'],x=df['deposit'])


# In[26]:


sns.boxplot(y=df['day'],x=df['deposit'])


# In[27]:


sns.boxplot(y=df['duration'],x=df['deposit'])


# In[28]:


sns.boxplot(y=df['campaign'],x=df['deposit'])


# In[29]:


sns.boxplot(y=df['pdays'],x=df['deposit'])


# In[30]:


sns.boxplot(y=df['previous'],x=df['deposit'])


# the day with deposit boxplot clearly has no anomalies. so we will drop pdays.

# In[31]:



df.drop('pdays', axis=1, inplace=True)


# creating dummy varaibles for categorical variables

# In[32]:


df_new = df.select_dtypes(include = 'object')
for i in df_new:
    print("Column Name: ",i)


# In[33]:



work = pd.get_dummies(df['job'],prefix='work')
work


# In[34]:


work = work.drop('work_admin.', axis=1)
work.head()


# In[35]:


#CONVERTING CATEGORICAL FEATURES INTO DUMMY VARIABLES

marital_status = pd.get_dummies(df['marital'],prefix='marital_status')

# DROPPING marital_status_divorced COLUMN

marital_status = marital_status.drop('marital_status_divorced', axis=1)
marital_status.head()


# In[36]:



qualification = pd.get_dummies(df['education'],prefix='quali')

# DROPPING quali_primary COLUMN

qualification = qualification.drop('quali_primary', axis=1)
qualification.head()


# In[37]:



defaulter = pd.get_dummies(df['default'],prefix='defaulter')

# DROPPING quali_primary COLUMN

defaulter = defaulter.drop('defaulter_yes', axis=1)
defaulter.head()


# In[38]:



hloan = pd.get_dummies(df['housing'],prefix='hloan')

# DROPPING quali_primary COLUMN

hloan = hloan.drop('hloan_yes', axis=1)
hloan.head()


# In[39]:



ploan = pd.get_dummies(df['loan'],prefix='ploan')

# DROPPING quali_primary COLUMN

ploan = ploan.drop('ploan_yes', axis=1)
ploan.head()


# In[40]:



contacted = pd.get_dummies(df['contact'],prefix='contacted')

# DROPPING quali_primary COLUMN

contacted = contacted.drop('contacted_cellular', axis=1)
contacted.head()


# In[41]:



mon = pd.get_dummies(df['month'],prefix='mon')

# DROPPING quali_primary COLUMN

mon = mon.drop('mon_nov', axis=1)
mon.head()


# In[42]:



pout = pd.get_dummies(df['poutcome'],prefix='pout')

# DROPPING quali_primary COLUMN

pout = pout.drop('pout_failure', axis=1)
pout.head()


# In[43]:



depo = pd.get_dummies(df['deposit'],prefix='depo')

# DROPPING quali_primary COLUMN

depo = depo.drop('depo_yes', axis=1)
depo.head(50)


# In[44]:



df = pd.concat([df,work,marital_status,qualification,defaulter,hloan,ploan,contacted,mon,pout,depo], axis=1)


# In[45]:


# VERIFYING UPDATES
df.head()


# In[46]:



data_1 = df.drop(['job','marital','education','default','housing','loan','contact','month','poutcome','deposit'],axis=1)


# In[47]:



data_1.head()


# In[129]:



data_1.head(10)


# In[ ]:


logistic Regression 


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(data_1.drop("depo_no", axis=1), data_1["depo_no"], train_size = 0.8,                                                    random_state=101)


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


logit = LogisticRegression()


# In[54]:


logit.fit(X_train, y_train)


# In[55]:



predictions = logit.predict(X_test)


# In[56]:



logit.score(X_test, y_test)


# this is the accuracy score

# In[57]:


from sklearn.metrics import accuracy_score


# In[58]:


accuracy_score(y_test, predictions)


# In[59]:



accuracy_score(y_test, predictions, normalize=False)


# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


rf_model = RandomForestClassifier(n_estimators=500, n_jobs=-1)


# In[63]:


rf_model.fit(X_train, y_train)


# In[64]:


from sklearn.metrics import accuracy_score


# In[65]:


rf_predictions = rf_model.predict(X_test)


# In[66]:


ACS_RF=accuracy_score(y_test, rf_predictions)
print("RANDOM FOREST ACCURACY SCORE : ",ACS_RF)


# In[70]:


from sklearn.model_selection import train_test_split
X=data_1.drop(['depo_no'],axis=1)
y=data_1['depo_no']


# In[71]:


import sklearn.model_selection as model_selection

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=101)


# In[68]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)


# In[73]:


feature_list_RF = []
for Class, score in zip(X.columns, rf_model.feature_importances_):
    feature_list_RF.append((score, Class))


# In[74]:


sorted(feature_list_RF, reverse=True)


# In[78]:



plt.figure(figsize=(21,5))

FI_RF = pd.DataFrame(sorted(zip(X.columns, rf_model.feature_importances_),                             key=lambda x: x[1], reverse= True), columns = ["Features", "Score"])

index = np.arange(len(FI_RF))
plt.bar(index, FI_RF['Score'], color = 'yellow')
plt.xticks(index, FI_RF['Features'], rotation=90)
plt.title('Feature Importances (Random Forest)')

plt.show()


# In[79]:



df_2 = data_1.copy()


# In[80]:



df_2['contact_duration'] = pd.qcut(df_2['duration'], q=10, labels=False, duplicates = 'drop')


# In[81]:



#GROUPING THE 'CONTACT_DURATION' AND FINDING AVERAGE CONTACT DURATION
df_2['contact_duration'] = pd.qcut(df_2['duration'], q=10, labels=False, duplicates = 'drop')
mean_contact_duration = df_2.groupby(['contact_duration'])['depo_no'].mean()


# In[87]:


mean_contact_duration


# In[134]:


df_2[df_2['contact_duration'] == 4]['duration'].max()


# In[135]:


df_2[df_2['contact_duration'] == 6]['duration'].min()


# In[82]:


# PLOTTING THE MEAN % SUBSCRIPTION VS CONTACT DURATION
plt.figure(figsize=(21,9))
plt.plot(mean_contact_duration.index, mean_contact_duration.values)
plt.title('Mean % subscription depending on Contact Duration')
plt.xlabel('Contact Duration Bin')
plt.ylabel('% subscription')
plt.show()


# In[84]:


df_2[df_2['contact_duration'] == 4]['duration'].max()


# In[85]:


df_2[df_2['contact_duration'] == 6]['duration'].min()


# In[86]:


df_2.describe()


# In[89]:


df_2['cust_bal'] = pd.qcut(df_2['balance'], q=50, labels=False, duplicates = 'drop')
mean_cust_bal = df_2.groupby(['cust_bal'])['depo_no'].mean()


# In[90]:


mean_cust_bal


# In[92]:


df_2[df_2['cust_bal'] == 28]['balance'].max()


# In[102]:


df_2[df_2['cust_bal'] == 33]['balance'].min()


# In[93]:



df_2['cust_age'] = pd.qcut(df_2['age'], q=10, labels=False, duplicates = 'drop')


# In[94]:



mean_cust_age = df_2.groupby(['cust_age'])['depo_no'].mean()


# In[95]:


mean_cust_age


# In[96]:


# PLOTTING THE MEAN % SUBSCRIPTION VS AVERAGE CUSTOMER BALANCE
plt.figure(figsize=(21,9))
plt.plot(mean_cust_age.index, mean_cust_age.values)
plt.title('Mean % subscription depending on Average Customer Age')
plt.xlabel('Average Customer Age Bin')
plt.ylabel('% subscription')
plt.show()


# In[131]:


mean_cust_age


# In[97]:


df_2[df_2['cust_age'] == 1]['age'].max()


# In[98]:


df_2[df_2['cust_age'] == 8]['age'].min()


# In[99]:


df_2[df_2['cust_age'] == 4]['age'].min()


# In[103]:



df_2['app_day'] = pd.qcut(df_2['day'], q=4, labels=False, duplicates = 'drop')


# In[104]:



mean_app_day = df_2.groupby(['app_day'])['depo_no'].mean()


# In[105]:


mean_app_day


# In[106]:


# PLOTTING THE MEAN % SUBSCRIPTION VS AVERAGE CUSTOMER BALANCE
plt.figure(figsize=(21,9))
plt.plot(mean_app_day.index, mean_app_day.values)
plt.title('Mean % subscription depending on Approach Day of the Month')
plt.xlabel('Approach Day of the Month Bin')
plt.ylabel('% subscription')
plt.show()


# In[130]:



mean_app_day


# In[107]:


df_2[df_2['app_day'] == 0]['day'].max()


# In[108]:


df_2[df_2['app_day'] == 3]['day'].min()


# In[ ]:




