# -*- coding: utf-8 -*-
"""Data_Visualisation_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fcy78NT67y-D1gUScfqTS5tFFFBYq5qX
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os

# loading dataset
dataset = pd.read_csv('dataset.csv')

#dropping columns with null values
dataset = dataset.drop(['Certifications/Achievement/ Research papers','Link to updated Resume (Google/ One Drive link preferred)','link to Linkedin profile'], axis = 1)

with PdfPages('data_visualization.pdf') as pdf:
    
    # A. THE NUMBER OF STUDENTS APPLIED TO DIFFERENT TECHNOLOGIES
    plt.figure(1)
    sns.set_style('whitegrid')
    plt.figure(figsize=(15,8))
    a=sns.countplot(y = "Areas of interest", data = dataset, orient = 'h')
    for i in a.patches:
        a.text(i.get_width() + 5, i.get_y() + 0.5, str(round((i.get_width()), 2)), fontsize=13, color='dimgrey')
    plt.title("Students Applying For Different Technologies", fontsize=20)
    pdf.savefig()  
    plt.close()
    
    # B. THE NUMBER OF STUDENTS APPLIED FOR DATA SCIENCE WHO KNEW PYTHON AND WHO DIDN'T
    df1 = dataset.filter(['Areas of interest','Programming Language Known other than Java (one major)'], axis=1)
    df1 = df1.loc[df1['Areas of interest']== 'Data Science ']
    df1.rename(columns = {'Programming Language Known other than Java (one major)':'Knew Python'}, inplace = True)
    df1['Knew Python'] = np.where(df1['Knew Python'] == 'Python', 'Yes','No')
    df1.reset_index()
    plt.figure(2)
    b = plt.pie(x=df1['Knew Python'].value_counts(), labels=["Don't know python", "Know Python"],autopct='%1.1f%%')
    plt.title("Students have applied for data science", fontsize=20)
    pdf.savefig()  
    plt.close()
    
    # C. THE DIFFERENT WAYS STUDENTS LEARNED ABOUT THIS PROGRAM
    df2 = dataset.filter(['How Did You Hear About This Internship?'],axis=1)
    plt.figure(3)
    plt.figure(figsize=(10,5))
    sns.set_style('darkgrid')
    c=sns.countplot(y='How Did You Hear About This Internship?',data = dataset)
    for i in c.patches:
        c.text(i.get_width() + 5, i.get_y() + 0.5, str(round((i.get_width()), 2)), fontsize=13, color='dimgrey')
    plt.title(label='The different ways students learned about this program', fontsize=20)
    pdf.savefig()  
    plt.close()
    
    # D. STUDENTS WHO ARE IN THE FOURTH YEAR AND HAVE A CGPA GREATER THAN 8.0
    df3 = dataset.filter(['Which-year are you studying in?','CGPA/ percentage'], axis=1)
    df3 = df3.loc[df3['Which-year are you studying in?']=='Fourth-year']
    students_above_8 = len([x for x in df3['CGPA/ percentage'] if x >= 8.0])
    total_CategoryWise = [students_above_8, len(df3)-students_above_8]
    plt.figure(4)
    d= plt.pie(total_CategoryWise, labels=["Having CGPA >= 8", "Having CGPA < 8"], colors=['g', 'y'], autopct='%1.1f%%')
    plt.title("Students who are in Fourth-year", fontsize=20)
    pdf.savefig()  
    plt.close()
    
    # E. STUDENTS WHO APPLIED FOR DIGITAL MARKETING WITH VERBAL AND WRITTEN COMMUNICATION SCORE GREATER THAN 8
    df4 = dataset.filter(['Areas of interest','Rate your written communication skills [1-10]','Rate your verbal communication skills [1-10]'])
    df4 = df4.loc[df4['Areas of interest']=='Digital Marketing ']
    df4.rename(columns = {'Rate your verbal communication skills [1-10]':'Verbal Score'}, inplace = True)
    df4.rename(columns = {'Rate your written communication skills [1-10]':'Written Score'}, inplace = True)
    score_above_8 = len([x for x in zip(df4['Verbal Score'], df4['Written Score']) if (x[0] >= 8 and x[1] >= 8)])
    tot_cat_wise = [score_above_8, len(df4)-score_above_8]
    plt.figure(5)
    e = plt.pie(tot_cat_wise, labels=["Written and Verbal Scores Above or Equal to 8", "Written and Verbal Scores Below 8"], colors=['lightseagreen', 'aquamarine'], autopct='%1.1f%%')
    plt.title("Students who applied for Digital Marketing", fontsize=20)
    pdf.savefig()  
    plt.close()
    
    # F. YEAR-WISE AND AREA OF STUDY WISE CLASSIFICATION OF STUDENTS
    df5 = dataset.filter(['Which-year are you studying in?','Major/Area of Study'])
    df5.rename(columns = {'Which-year are you studying in?':'Study year'}, inplace = True)
    df5.rename(columns = {'Major/Area of Study':'Area of Study'}, inplace = True)
    plt.figure(6)
    sns.set_style('darkgrid')
    f = sns.countplot(x='Study year',hue = 'Area of Study',data = df5, palette='YlGnBu')
    plt.xticks(rotation=90)
    plt.title(label='Classification on basis of year and area of study')
    pdf.savefig()  
    plt.close()
    
    # G. CITY AND COLLEGE WISE CLASSIFICATION OF STUDENTS
    df6 = dataset.filter(['City','College name'])
    plt.figure(7)
    sns.set_style('darkgrid')
    g = sns.countplot(x='City',data = df6, palette = 'Purples')
    plt.xticks(rotation=90)
    plt.title(label='Classification on basis of city')
    pdf.savefig()  
    plt.close()
    
    plt.figure(8)
    sns.set_style('darkgrid')
    h = sns.countplot(x='College name',data = df6, palette = 'PuBuGn')
    plt.xticks(rotation=90)
    plt.title(label='Classification on basis of college')
    pdf.savefig()  
    plt.close()
    
    # H. Plot the relationship between the CGPA and the target variable
    data1 = dataset.filter(['CGPA/ percentage','Label'])
    eligible = data1[data1['Label']=='eligible']
    s1 = len([x for x in eligible['CGPA/ percentage'] if (x<8 and x>=7)])
    s2 = len([x for x in eligible['CGPA/ percentage'] if (x<9 and x>=8)])
    s3 = len([x for x in eligible['CGPA/ percentage'] if (x>=9)])

    s1t = len([x for x in data1['CGPA/ percentage'] if (x<8 and x>=7)])
    s2t = len([x for x in data1['CGPA/ percentage'] if (x<9 and x>=8)])
    s3t = len([x for x in data1['CGPA/ percentage'] if (x>=9)])

    df7 = pd.DataFrame({'eligible': [s1,s2,s3], 'ineligible': [s1t-s1,s2t-s2,s3t-s3]})
    plt.figure(9)
    plt.figure(figsize = (10,5))
    h = df7.plot(kind='bar')
    plt.title("Relationship between CGPA and eligibility", fontsize=20)
    h.set_ylabel("Number of Students")
    h.set_xlabel("Range of CGPA")
    h.set_xticklabels(["7 - 8", "8 - 9", "9 - 10"],rotation=0)
    pdf.savefig()  
    plt.close()
    
    # I. Plot the relationship between the Area of Interest and the target variable
    data2 = dataset.filter(['Areas of interest','Label'])
    eligible = data2[data2['Label'].str.contains('eligible')]
    eligible_students = eligible['Areas of interest'].value_counts().tolist()
    ineligible = data2[data2['Label'].str.contains('ineligible')]
    ineligible_students = ineligible['Areas of interest'].value_counts().tolist()
    df8 = pd.DataFrame({'Eligible': eligible_students, 'Ineligible': ineligible_students})
    plt.figure(10)
    i = df8.plot(kind='barh', figsize=(15,8),fontsize=10);
    plt.title("Eligibilty for the Applied Technology", fontsize=20)
    i.set_xlabel("Number of Students")
    i.set_ylabel("Areas of interest")
    categories = dataset['Areas of interest'].value_counts().keys().tolist()
    i.set_yticklabels(categories)
    pdf.savefig()  
    plt.close()
    
    # J. Plot the relationship between the year of study, major, and the target variable
    data3 = dataset.filter(['Which-year are you studying in?','Major/Area of Study','Label'])
    plt.figure(11)
    j = sns.FacetGrid(data3,hue="Label",size=5).map(plt.scatter,"Which-year are you studying in?","Major/Area of Study").add_legend();
    plt.xticks(rotation=90)
    plt.title("Relationship between the year of study, major, and the target variable", fontsize=20)
    pdf.savefig()  
    plt.close()

