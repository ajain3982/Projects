#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import my_resources as mrs
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,precision_score,recall_score,make_scorer


def load_data_in_pandas(data_dict):
	data = pd.DataFrame.from_dict(data_dict)
	data = data.T
	enron_data = data.iloc[:,data.columns!='email_address'].apply(lambda x: pd.to_numeric(x, errors='force'))
	enron_data['email_address'] = data['email_address']
	enron_data.fillna(0,inplace=True)
	return enron_data

def compute_fraction(ser1,ser2):
    """
    This function takes in two pandas series and simply return fraction of these two.
    """
    return 1.*ser1 / ser2


def create_new_features(enron_data):

	'''This function takes pandas dataframe and add new features'''


	enron_data['fraction_from_poi'] = compute_fraction(enron_data['from_poi_to_this_person'],enron_data['to_messages'])
	enron_data['fraction_to_poi'] = compute_fraction(enron_data['from_this_person_to_poi'],enron_data['from_messages'])
	enron_data['expenses_salary_ratio'] = compute_fraction(enron_data['expenses'],enron_data['salary'])
	enron_data['bonus_salary_ratio'] = compute_fraction(enron_data['bonus'],enron_data['salary'])
	enron_data.replace([np.inf,-np.inf],np.nan,inplace=True)
	enron_data.fillna(0,inplace=True)
	return enron_data

def scaler(arr):

'''This function is used to scale the features'''
    max_v = arr.max()
    min_v = arr.min()
    if max_v == min_v:
            arr = arr / min_v
    else:
        arr = (arr - min_v)/(max_v - min_v) 
    return arr

scoring = {'R':'recall', 'P':'precision'}

'''This function used grid search to train different classifiers and return the best one with scoring/validation done using
recall and precision measure.'''
def train_algo(algo,params,train_data,train_labels):   
    clf = GridSearchCV(algo,param_grid=params,scoring=scoring,refit='R')
    clf.fit(train_data,train_labels)
    return clf

'''This function takes in a k value and select k best features of the data provided. This function is called recurvisely
by select_features_and_train method. '''
def select_k_best(k,data,labels):
    k_best = SelectKBest(k=k)
    data = k_best.fit_transform(data,labels)
    return data,labels,k_best.get_support()


def select_features_and_train(classifier,params,new_data,target):
    for k in range(19):
        data,labels,mask= select_k_best(k+1,new_data,target)
        clf = train_algo(classifier,params,data,labels)
        print 'k: {} \nBest Parameter: {}\nBest Score: {}'.format(k+1,clf.best_params_,clf.best_score_)
        f_list = [f for f in new_data.columns[mask]]
        print 'Features Selected: {}\n\n'.format(f_list)
        print "-----------------------------------------------------------------------------------------"

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

enron_data = load_data_in_pandas(data_dict)


### Task 2: Remove outliers
enron_data.drop('TOTAL',inplace=True)
print enron_data.shape


### Task 3: Create new feature(s)
enron_data = create_new_features(enron_data)

### Extract features and labels from dataset for local testing
target = enron_data['poi']
enron_data.drop('poi',axis=1,inplace=True)
enron_data.drop('email_address',axis=1,inplace=True)

## Scaling Features
enron_data = enron_data.apply(scaler)

### Store to my_dataset for easy export below.
enron_data['poi'] = target
d2 = enron_data.T
my_dataset = d2.to_dict()
enron_data.drop('poi',axis=1,inplace=True)


### Task 4: Try a varity of classifiers
### Task 5: Tuning also done here
### Tried different classifiers with testing hyperparameter tuning done via GridSearchCV. 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.svm import SVC
print "-------------------------SVM-------------------------"
svc = SVC(random_state=42)
params = {'C': [1, 10, 50,100],
          'gamma': [0.001,0.0001,0.002,0.005], 
          'kernel': ['linear','rbf']}
mrs.main(svc,params,enron_data,target)


print "-----------------Decision Tree-----------------------"
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
params = {'min_samples_split':[2,5,10,15,20,50,70,100],
         'max_depth':[2,5,7,10,14],
         }
select_features_and_train(clf_tree,params,enron_data,target)


print "---------------------Random Forest--------------------"
from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier()
params = {'n_estimators':[10,20,30,40,50,60,100],
         'min_samples_split':[2,5,10,15,20,30,50],
         }
select_features_and_train(rfc_clf,params,enron_data,target)


print "-----------------------Naive Bayes--------------------"
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
params = {}
select_features_and_train(nb_clf,params,enron_data,target)

print "------------------------Adaboost----------------------"
from sklearn.ensemble import AdaBoostClassifier
adb_clf = AdaBoostClassifier()
params = {'n_estimators':[1,10,20,50,100,150,200,500],
         'learning_rate':[1,0.9,0.8,0.5,0.3],
         'algorithm':['SAMME.R']}
select_features_and_train(adb_clf,params,enron_data,target)

### final model and feature_list used for validating result
features_list = ['bonus', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'salary', 'total_stock_value', 'fraction_to_poi', 'bonus_salary_ratio']
clf = tree.DecisionTreeClassifier(min_samples_split = 2, max_depth = 14)



# Validation of model using accuracy and recall with two strategies given below
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

''' testing on final model using train_test_split and classification report '''
features_train, features_test, labels_train, labels_test = \
    train_test_split(enron_data, target, test_size=0.3, random_state=42)
clf.fit(features_train,labels_train)
pred_labels = clf.predict(features_test)
print classification_report(labels_test,pred_labels)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
enron_data = enron_data[features_list]
for train_index, test_index in sss.split(enron_data, target):
	features_train, features_test = enron_data.iloc[train_index,], enron_data.iloc[test_index,]
	labels_train, labels_test = target.iloc[train_index,], target.iloc[test_index,]
	clf.fit(features_train,labels_train)
	predictions = clf.predict(features_test)

	for prediction, truth in zip(predictions, labels_test):
		if prediction == 0 and truth == 0:
			true_negatives += 1
		elif prediction == 0 and truth == 1:
			false_negatives += 1
		elif prediction == 1 and truth == 0:
			false_positives += 1
		elif prediction == 1 and truth == 1:
			true_positives += 1

try:
	''' Here precision and recall are calculated using stratified shuffle split'''
	total_predictions = true_negatives + false_negatives + false_positives + true_positives
	accuracy = 1.0*(true_positives + true_negatives)/total_predictions
	precision = 1.0*true_positives/(true_positives+false_positives)
	recall = 1.0*true_positives/(true_positives+false_negatives)
	f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
	f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
	print clf
	print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
	print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
	print ""

except:
	print "Got a divide by zero when trying out:", clf
	print "Precision or recall may be undefined due to a lack of true positive predicitons."

	print classification_report(labels_test,pred_labels)

# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.

# final feature_list dumped
features_list = ['poi','bonus', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'salary', 'total_stock_value', 'fraction_to_poi', 'bonus_salary_ratio']

dump_classifier_and_data(clf, my_dataset, features_list)