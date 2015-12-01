from numpy import *
from sklearn import svm

#nbaData = genfromtxt('NBA12_14.csv', delimiter=',')
nbaData = genfromtxt('NBA97_98_playoffs.csv', delimiter=',')
nba15test = genfromtxt('NBA15_playoffs.csv', delimiter=',')
 #convert NBA12_14.csv to array of values

# SVM may perform better with standardize features
# For easy implementation, I convert data to the base 10 logarithm
nba15test_scaled = nba15test
nba15test_scaled[:,1:] = log10(nba15test_scaled[:,1:])

label = nbaData[:,0]
features = log10(nbaData[:,1:])

clf = svm.SVC(kernel='rbf', probability=True, C=1)
prob15 = clf.fit(features, label).predict_proba(nba15test[:,1:])
#1st column, the W/L, is nba15test[:,0]

print prob15
#print self.classes_.