# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:35:48 2018

@author: LIPPA2
"""

""" Load Packages """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

""" Import Data """
zoo_animal = pd.read_csv('zoo.csv')
animal_class = pd.read_csv('class.csv')

""" Test if Data is correct imported """
zoo_animal.head()
animal_class.head()

""" Test quality of the data """
zoo_animal.isnull().sum()
# --> Data is clean! There are no NULL Values.

""" Join data sets """
# Data can be joined by field class_type (zoo_animal) and "Class_Number" (animal_class)
df = pd.merge(zoo_animal, animal_class, how='left', left_on = 'class_type', right_on = 'Class_Number')
df.head()

""" Create some plots """
sns.factorplot('Class_Type', data = df, kind="count", aspect = 1.5)

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = zoo_animal.corr()
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            #cmap="YlGnBu",
            vmin=0.5, vmax=0.9) 

""" Definition Features & Labels """
# Define wich attributes are features
features = zoo_animal.iloc[:,1:17]
#print list(features) // ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']

# Select the defined labels
labels = df['Class_Type']
#print labels.unique() // ['Mammal' 'Fish' 'Bird' 'Invertebrate' 'Bug' 'Amphibian' 'Reptile']

""" Select best Features """
# Select best features for prediction wit k-best
from sklearn.feature_selection import SelectKBest

k_best = SelectKBest(k=10)
# fit selector to data
k_best.fit(features, labels)
# calculate scores, for ranking which feature is important
scores = k_best.scores_
# create a tuple that returns feature and score
tuples = zip(features[1:], scores)
# sort the tuble, that shows the feature with the highest score first
k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)
print k_best_features[:10]
#[('feathers', inf), 
# ('milk', inf), 
# ('backbone', inf), 
# ('toothed', 197.48931926159673), 
# ('eggs', 127.99263818815812), 
# ('hair', 83.465717821782093), 
# ('breathes', 74.193353818140707), 
# ('fins', 45.696191240745705), 
# ('tail', 28.999226178695697), 
# ('airborne', 27.749632111345598)]


""" Cross Validation split data """
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size = 0.3, random_state = 42)

""" Gaussian Naive Bayes """
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(features_train, labels_train)
pred_nb = clf_nb.predict(features_test)

nb_acc = accuracy_score(pred_nb, labels_test)

from sklearn.model_selection import cross_val_score
nb_vc = cross_val_score(clf_nb, features, labels, cv=5)

""" Decision Tree """
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

accuracy_tree = accuracy_score(pred, labels_test)
tree_cv = cross_val_score(clf, features, labels, cv=5)


""" Support Vector Machine """
from sklearn.svm import SVC
clf_svm = SVC()
clf_svm.fit(features_train, labels_train)
pred = clf_svm.predict(features_test)

accuracy_svm = accuracy_score(pred, labels_test)
svm_cv = cross_val_score(clf_svm, features, labels, cv=5)


""" Adaboost """
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier()
clf_ada.fit(features_train, labels_train)
pred = clf_ada.predict(features_test)

accuracy_adaboost = accuracy_score(pred, labels_test)
ada_cv = cross_val_score(clf_ada, features, labels, cv=5)

""" Random Forest """
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
clf_rf.fit(features_train, labels_train)
pred = clf_rf.predict(features_test)

accuracy_randomforest = accuracy_score(pred, labels_test)
rf_cv = cross_val_score(clf_rf, features, labels, cv=5)


""" Table Overview """
models = pd.DataFrame({
    'Learning Model': ['Gaussian Naive Bayes', 'Decision Tree', 'Support Vector Machines', 'Adaboost', 'Random Forest'],
    'Score Cross Validation': [nb_vc.mean(), tree_cv.mean(), svm_cv.mean(), ada_cv.mean(), rf_cv.mean()],
    'Score Accuracy': [nb_acc, accuracy_tree, accuracy_svm, accuracy_adaboost, accuracy_randomforest]
})
print models

"""
            Learning Model  Score Accuracy  Score Cross Validation
0     Gaussian Naive Bayes        0.935484                0.960902
1            Decision Tree        0.935484                0.952381
2  Support Vector Machines        0.870968                0.920650
3                 Adaboost        0.645161                0.734093
4            Random Forest        0.935484                0.970426
"""


""" Classification Report """
from sklearn.metrics import classification_report
#target_names = ['Mammal', 'Fish', 'Bird', 'Invertebrate', 'Bug', 'Amphibian', 'Reptile']
print "Classification Report:"
print classification_report(y_true=labels_test, y_pred=pred, target_names=labels.unique())

"""
Classification Report:
              precision    recall  f1-score   support

      Mammal       1.00      1.00      1.00         2
        Fish       1.00      1.00      1.00         3
        Bird       0.83      1.00      0.91         5
Invertebrate       0.67      1.00      0.80         2
         Bug       1.00      0.67      0.80         3
   Amphibian       1.00      1.00      1.00        15
     Reptile       0.00      0.00      0.00         1

 avg / total       0.92      0.94      0.92        31
"""




