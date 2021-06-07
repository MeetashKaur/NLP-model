#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import nltk
#nltk.download()


# Load libraries and dataset

# In[2]:


import pandas as pd
import pickle
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns 
data=pd.read_csv("E:/offenseval-training-v1.tsv",sep = "\t", header = 0)
data.head()


# In[3]:


dataframe=data.drop(['id','subtask_b','subtask_c'],axis=1)
dataframe.head()


# In[4]:


dataframe.shape


# In[5]:


dataframe.isnull().sum()


# In[6]:


sns.countplot(x=dataframe['subtask_a'])


# # Data Pre-processing

# Function to remove user tag

# In[7]:


mustBeRemovedList = ["@USER", "url"]
def remove_userTag():
    datasetwithoutUserTag = []
    for line in dataframe['tweet']:
        finalListOfWords = []
        tweets = []
        words = line.split()
        for word in words:
            if word not in mustBeRemovedList:
                finalListOfWords.append(word)
        tweets = " ".join(finalListOfWords)
        datasetwithoutUserTag.append(tweets)
    return datasetwithoutUserTag


# Funtion to remove stop words

# In[8]:


noise_list = set(stopwords.words("english"))
# noise detection
def remove_noise(input_text):
    words = word_tokenize(input_text)
    noise_free_words = list()
    i = 0;
    for word in words:
        if word.lower() not in noise_list:
            noise_free_words.append(word)
        i += 1
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text


# Lemmatization

# In[9]:


def lemetize_words(input_text):
    words = word_tokenize(input_text)
    new_words = []
    lem = WordNetLemmatizer()
    for word in words:
        word = lem.lemmatize(word, "v")
        new_words.append(word)
    new_text = " ".join(new_words)
    return new_text


# Function to clean the text

# In[10]:


def cleaning():
    corpus = []
    datasetwithoutUserTag = remove_userTag()
    for line in datasetwithoutUserTag:
        review = re.sub('[^a-zA-Z]', ' ', line)
        review = review.lower()
        # remove non segnificant words
        review = remove_noise(review)
        review = lemetize_words(review)
        corpus.append(review)
    return corpus


# # Baseline: Bag-of-Words model

# In[11]:


def bagOfWordsCreation(corpus):
    cv = CountVectorizer(max_features=12000)
    bagOfWords = cv.fit_transform(corpus).toarray()
    rowsValues = []
    for line in dataframe['subtask_a']:
        if line == "OFF":
            rowsValues.append(1)
        else:
            rowsValues.append(0)
    return (bagOfWords, rowsValues)


# # Cleaning of text

# In[12]:


corpus = cleaning()
bagOfWords,rowsValues=bagOfWordsCreation(corpus)
# splitting data into training and testing data
bagOfWords_train,bagOfWords_test,rowsValues_train,rowsValues_test=train_test_split(bagOfWords,
                                                            rowsValues,test_size=0.2,random_state=0)


# In[13]:


rowsValues_train


# # Oversampling 

# In[14]:


oversample = SMOTE()
x_train_res,y_train_res =oversample.fit_resample(bagOfWords_train,rowsValues_train)


# In[15]:


l1=[]
count=0
for item in y_train_res:
    if item not in l1:
        count += 1
        l1.append(item) 
# printing the output
print("No of unique items are:", count)
def unique(list1): 
    # intilize a null list
    unique_list = []     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print(x),
print(unique(y_train_res))


# Count the number of instanges associated with class 0 in balanced dataset

# In[16]:


occu_0=y_train_res.count(0) 
print(occu_0)


# Count the number of instanges associated with class 1 in balanced dataset

# In[17]:


occu_1=y_train_res.count(1) 
print(occu_1)


# In[18]:


sns.countplot(x=y_train_res)


# # Fitting XG Boost model on unbalanced dataset

# In[19]:


xg_un=XGBClassifier(objective='binary:logistic',n_estimators=50,seed=123,learning_rate=0.5)
xg_un.fit(bagOfWords_train,rowsValues_train)


# In[20]:


pred_xg=xg_un.predict(bagOfWords_test)


# In[21]:


print("Confusion matix\n",confusion_matrix(rowsValues_test,pred_xg))


# In[22]:


print('Acuuracy of model: -',accuracy_score(rowsValues_test,pred_xg))


# In[23]:


print("Classification Report:",classification_report(rowsValues_test,pred_xg))


# # Fitting XG Boost model on balanced dataset

# In[24]:


xg_model=XGBClassifier(objective='binary:logistic',n_estimators=50,seed=123,learning_rate=0.5)
xg_model.fit(x_train_res,y_train_res)


# In[25]:


prediction=xg_model.predict(bagOfWords_test)


# In[26]:


print("Confusion matix\n",confusion_matrix(rowsValues_test,prediction))


# In[27]:


print('Acuuracy of model: -',accuracy_score(rowsValues_test,prediction))


# In[28]:


print("Classification Report:",classification_report(rowsValues_test,prediction))


# In[29]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(dataframe['tweet']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)


# In[30]:


pickle.dump(xg_un,open('model.pkl','wb'))

