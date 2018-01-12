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
	enron_data['fraction_from_poi'] = compute_fraction(enron_data['from_poi_to_this_person'],enron_data['to_messages'])
	enron_data['fraction_to_poi'] = compute_fraction(enron_data['from_this_person_to_poi'],enron_data['from_messages'])
	enron_data['expenses_salary_ratio'] = compute_fraction(enron_data['expenses'],enron_data['salary'])
	enron_data['bonus_salary_ratio'] = compute_fraction(enron_data['bonus'],enron_data['salary'])
	enron_data.replace([np.inf,-np.inf],np.nan,inplace=True)
	enron_data.fillna(0,inplace=True)
	return enron_data

def scaler(arr):
    max_v = arr.max()
    min_v = arr.min()
    if max_v == min_v:
            arr = arr / min_v
    else:
        arr = (arr - min_v)/(max_v - min_v) 
    return arr

scoring = {'R':'recall', 'P':'precision'}

'''This function used grid search to train different classifiers and return the best one'''
def train_algo(algo,params,train_data,train_labels):   
    clf = GridSearchCV(algo,param_grid=params,scoring=scoring,refit='R')
    clf.fit(train_data,train_labels)
    return clf

'''This function takes in a k value and select k best features of the data provided'''
def select_k_best(k,data,labels):
    k_best = SelectKBest(k=k)
    data = k_best.fit_transform(data,labels)
    return data,labels,k_best.get_support()

def main(classifier,params,new_data,target):
    #k_values = [2,5,7,10,14,18,'all']
    for k in range(19):
        data,labels,mask= select_k_best(k+1,new_data,target)
        #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
        clf = train_algo(classifier,params,data,labels)
        print 'k: {} \nBest Parameter: {}\nBest Score: {}'.format(k+1,clf.best_params_,clf.best_score_)
        f_list = [f for f in new_data.columns[mask]]
        print 'Features Selected: {}\n\n'.format(f_list)
        print "-----------------------------------------------------------------------------------------"