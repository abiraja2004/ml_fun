import sys as sys
import os as os
import pandas as pd
import numpy as np
import csv as csv
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# load the test and training data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)
ids = test_df['PassengerId'].values

# convert str to binary int
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# strip out title
train_df["Title"] = train_df['Name'].map(lambda x: re.search('.*, (.*?\.).*',x).group(1))
train_df = pd.concat([train_df, pd.get_dummies(train_df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
test_df["Title"] = test_df['Name'].map(lambda x: re.search('.*, (.*?\.).*',x).group(1))
test_df = pd.concat([test_df, pd.get_dummies(test_df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)

# encode the Embarked categorical values
a = pd.concat([train_df['PassengerId'], pd.get_dummies(train_df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
train_df = pd.merge(train_df,a,on='PassengerId',how='outer')
a = pd.concat([test_df['PassengerId'], pd.get_dummies(test_df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
test_df = pd.merge(test_df,a,on='PassengerId',how='outer')

# All missing ages -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# encode the Cabin values
if len(train_df.Cabin[ train_df.Cabin.isnull() ]) > 0:
    train_df.loc[ (train_df.Cabin.isnull()), 'Cabin'] = 'Unknown'
train_df['Cabin'] = train_df['Cabin'].map(lambda x: str(x)[:1])
if len(test_df.Cabin[ test_df.Cabin.isnull() ]) > 0:
    test_df.loc[ (test_df.Cabin.isnull()), 'Cabin'] = 'Unknown'
test_df['Cabin'] = test_df['Cabin'].map(lambda x: str(x)[:1])
a = pd.concat([train_df['PassengerId'], pd.get_dummies(train_df['Cabin']).rename(columns=lambda x: 'Cabin_' + str(x))], axis=1)
train_df = pd.merge(train_df,a,on='PassengerId',how='outer')
a = pd.concat([test_df['PassengerId'], pd.get_dummies(test_df['Cabin']).rename(columns=lambda x: 'Cabin_' + str(x))], axis=1)
test_df = pd.merge(test_df,a,on='PassengerId',how='outer')

# add some unique fields
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"]
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"]

# only take the columns that are in common between the two
common_cols = [col for col in set(train_df.columns).intersection(test_df.columns)]
test_df = test_df[common_cols]
common_cols.insert(0,'Survived')
train_df = train_df[common_cols]

# remove unused columns
a = ['Name', 'Sex', 'Ticket',
     'PassengerId', 'Embarked',
     'Cabin', 'Title']
train_df = train_df.drop(a, axis=1) 
test_df = test_df.drop(a, axis=1) 

# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

# Run the routines
print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
print 'Predicting...'
output = forest.predict(test_data).astype(int)
print 'Scoring...'
score = forest.score( train_data[0::,1::], train_data[0::,0] )
print 'Dataset scored %f' % score

# write results
predictions_file = open("prediciton.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
