#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sys
import os
from pathlib import Path

str =""
str = input("Enter the name of your text file - please use / backslash when typing in directory path: ");

f = open(str,"r+")

dataset = pd.read_csv(f, header = 0)

drop_col = ['First Name', 'Last Name', 'State', 'Zip Code', 'DOB [DD/MM/YYYY]', 'Age', 'Gender', 'Email Address', 'Contact Number', 'Emergency Contact Number', 'University Name', 'Degree', 'Course Type', 'Current Employment Status', 'Certifications/Achievement/ Research papers', 'Link to updated Resume (Google/ One Drive link preferred)', 'link to Linkedin profile']

dataset.drop(drop_col, inplace=True, axis = 1)

dataset['En_Label'] = LabelEncoder().fit_transform(dataset['Label'])
dataset['En_Year'] = LabelEncoder().fit_transform(dataset['Which-year are you studying in?'])
dataset['En_CGPA'] = LabelEncoder().fit_transform(dataset['CGPA/ percentage'])
dataset['En_Major'] = LabelEncoder().fit_transform(dataset['Major/Area of Study'])
dataset['En_Prog_Languages'] = LabelEncoder().fit_transform(dataset['Programming Language Known other than Java (one major)'])
dataset['En_Interest'] = LabelEncoder().fit_transform(dataset['Areas of interest'])
dataset['En_written_score'] = LabelEncoder().fit_transform(dataset['Rate your written communication skills [1-10]'])
dataset['En_verbal_score'] = LabelEncoder().fit_transform(dataset['Rate your verbal communication skills [1-10]'])
dataset['En_SQL'] = LabelEncoder().fit_transform(dataset['Have you worked on MySQL or Oracle database'])
dataset['En_Java'] = LabelEncoder().fit_transform(dataset['Have you worked core Java'])
dataset['En_OOP'] = dataset['Have you studied OOP Concepts'].map({'Yes':1, 'No':0})

features = ['En_Year','En_CGPA','En_Major','En_Prog_Languages','En_Interest','En_written_score' ,'En_verbal_score','En_SQL','En_Java','En_OOP']
X = dataset[features] 
y = dataset['En_Label'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


model=RandomForestClassifier(n_estimators=1)


model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn import metrics
print("F1:",metrics.f1_score(y_test, y_pred))


# In[ ]:




