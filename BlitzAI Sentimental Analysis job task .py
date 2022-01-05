#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE LIBRARIES

# In[1]:


import pandas as pd


# # LOAD THE DATASET

# In[2]:


df=pd.read_parquet('classificationData.parquet',engine='pyarrow')
df


# # EDA

# In[3]:


df.count()


# In[4]:


df.isnull().sum()


# + We drop review_body because it have lot of missing values

# In[5]:


df.dropna(subset=['review_body'],inplace = True)


# # dropping unneccesory columns

# In[6]:


df.columns


# In[7]:


df1 = df.drop(columns=['marketplace', 'customer_id', 'review_id', 'product_id',
       'product_parent', 'product_title', 'product_category','helpful_votes', 'total_votes','review_body', 'vine', 'verified_purchase','review_date'])
df1.head()


# In[8]:


#df['customer_id'].nunique()


# In[9]:


#df['review_id'].nunique()


# In[10]:


#df['product_id'].nunique()


# In[11]:


df1.duplicated()


# In[12]:


#Count of duplicated rows
df1[df1.duplicated()].shape


# In[13]:


df1.drop_duplicates()


# In[14]:


df1.shape


# In[15]:


df1.info()


# In[16]:


df1.describe()


# # Converting the rating to Positive or Negative

# In[17]:


def f(row):
    
    '''This function returns sentiment value based on the overall ratings from the user'''
    
    if row['star_rating'] == 1 or row['star_rating']  == 2 :
        val = 'Negative'
    elif row['star_rating'] == 3 or row['star_rating'] == 4 :
        val = 'Neutral'
    elif row['star_rating'] == 5  :
        val = 'Positive'
    else:
        val = -1
    return val


# # Applying the function in our new column

# In[18]:


df1['sentiment'] = df1.apply(f, axis=1)
df1.head()


# In[ ]:





# In[19]:


df1.head(20)


# # Removing Punctuation and Brackets and etc.

# In[20]:


import nltk
import re


# In[21]:


WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('english')
#stop_word_list


# In[22]:


df1['review_headline'] = df1['review_headline'].apply(lambda x: re.sub('[,\.!?:()"]', '', x))
df1['review_headline'] = df1['review_headline'].apply(lambda x: re.sub('[^a-zA-Z"]', ' ', x))

df1['review_headline'] = df1['review_headline'].apply(lambda x: x.lower())

df1['review_headline'] = df1['review_headline'].apply(lambda x: x.strip())

"""
I closed the stopword process because it took a long time.
If you want, you can try opening the codes in the comment line.
"""
#def token(values):
   # words = nltk.tokenize.word_tokenize(values)
    #filtered_words = [word for word in words if word not in set(stopwords.words("english"))]
    #not_stopword_doc = " ".join(filtered_words)
    #return not_stopword_doc
#data['review content'] = data['review content'].apply(lambda x: token(x))


# # Visualisation for sentiments

# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df1["sentiment"], palette = ["green","red","blue"])
plt.show()
print(df1.sentiment.value_counts())


# # FEATURE EXTRACTION

# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()


# # BALANCE THE DATA

# In[25]:


x = df1['review_headline']
y = df1['sentiment']
y.value_counts()


# In[26]:


from imblearn.under_sampling import RandomUnderSampler


# In[27]:


import numpy as np
rus = RandomUnderSampler()
y = np.array(y)
x = np.array(x)
y = np.reshape(y, (-1,1))
x = np.reshape(x, (-1,1))
x_r, y_r = rus.fit_resample(x, y)
print(len(y_r))


# In[28]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_r,y_r,test_size=0.2,random_state=42)
x_test = x_test.reshape(-1)
x_train = Vectorizer.fit_transform(x_train.reshape(-1))
x_test = Vectorizer.transform(x_test)
rus.fit(x_train, y_train)


# # MODEL BUILDING and PREDICTIION

# # MULTINOMIAL NAIVE BAYES

# In[29]:


from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import pickle
import seaborn as sns


# In[30]:


mnb = MultinomialNB()
mnb.fit(x_train , y_train)


# In[31]:


pred = mnb.predict(x_test)


# In[32]:


print(confusion_matrix(y_test , pred))
print(classification_report(y_test , pred))
print(accuracy_score(y_test , pred))


# # LOGISTIC REGRESSION

# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


Model = LogisticRegression(max_iter = 4000)
Model.fit(x_train, y_train)


# In[35]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[36]:


result = Model.predict(x_test)
print(confusion_matrix(y_test, result))


# In[37]:


accuracy = accuracy_score(y_test, result)
print(accuracy)


# ## Test the Model

# In[38]:


pred1 = Model.predict(Vectorizer.transform(["Do not buy it. Horrible product"]))
pred2 = Model.predict(Vectorizer.transform(["This is probably the best thing i have ever bought"]))
pred3 = Model.predict(Vectorizer.transform(["It works fine! Thank you!"]))
pred4 = Model.predict(Vectorizer.transform(["Many disadvantages, i do not recommend it"]))
pred5 = Model.predict(Vectorizer.transform(["My delivery is 2 months delayed. Pure lack of profissionalism"]))
pred6 = Model.predict(Vectorizer.transform(["Excellent product!"]))

print(pred1, pred2, pred3, pred4, pred5, pred6)


# + accuracy is less,so to improve this we are appling randomsearchcv hyperparameter tuning

# # HYPERPARAMETER TUNING

# ## Gridsearch CV
from sklearn.model_selection import GridSearchCV# Creating the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}# Instantiating the GridSearchCV object
logreg_cv = GridSearchCV(Model, param_grid, cv = 5)
  
logreg_cv.fit(x_train,y_train)# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))
# # RANDOM SEARCH CV

# In[39]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# In[40]:


distributions = dict(C=uniform(loc=0, scale=4))


# In[41]:


clf = RandomizedSearchCV(Model, distributions, random_state=0)


# In[42]:


search = clf.fit(x_train,y_train)
search.best_params_


# In[43]:


pd.DataFrame(np.c_[y_test , pred] , columns=["Actual" , "Predicted"])


# # SAVE THE PICKLE FILE

# In[44]:


import pickle
pickle.dump(Vectorizer, open("count-Vectorizer.pkl" , "wb"))
pickle.dump(Model, open("BlitzAI_xentimental_Analysis.pkl" , "wb"))


# # LOAD THE PICKLE FILE

# In[45]:


save_cv = pickle.load(open('count-Vectorizer.pkl','rb'))
model = pickle.load(open('BlitzAI_xentimental_Analysis.pkl','rb'))


# # TEST THE MODEL

# In[46]:


def test_model(sentence):
    sen = save_cv.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 'Positive':
        return 'Positive review'
    else:
        return 'Negative review'


# In[47]:


sen = 'didnt work recieved bought ps4 stayed put til gave son birthday excitement didnt work blue light stayed'
res = test_model(sen)
print(res)


# In[48]:


sen = 'It works fine! Thank you!'
res = test_model(sen)
print(res)


# In[49]:


sen = 'Do not buy it. Horrible product'
res = test_model(sen)
print(res)


# In[50]:


sen = 'Came with the original xbox controller box all smashed up. No padding nothing. Damaged all the way......'
res = test_model(sen)
print(res)


# In[ ]:





# In[ ]:





# # PIPELINE

# In[51]:


#define X and Y


# In[52]:


x = df1['review_headline']
y = df1['sentiment']


# In[53]:


## Pipelines Creation
## 1. Hyperparameter tuning using randomsearch cv
## 2. Apply  Classifier


# # 1.MultiNomial Naive Bayes

# In[54]:


from sklearn.pipeline import Pipeline
pipeline_nb=Pipeline([('model_nb',MultinomialNB())])


# # 2. Logistic Regression

# In[55]:


pipeline_lr=Pipeline([('model_lr',LogisticRegression())])


# In[56]:


## LEts make the list of pipelines
pipelines = [pipeline_nb, pipeline_lr]


# In[57]:


best_accuracy=0.0
best_classifier=0
best_pipeline=""


# In[61]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Create a pipeline
pipe = Pipeline([("classifier", MultinomialNB())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(0, 4, 10)
                 },
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2'],
                 "classifier__C": np.logspace(0, 4, 10),
                 "classifier__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty
                 },
                {"classifier": [MultinomialNB()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}]
# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(x_train,y_train)


# In[ ]:




