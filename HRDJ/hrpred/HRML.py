import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing #scaling

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv('turnover.csv')

def pre_process (df):
    df = pd.get_dummies(df)
    return df    


def training (df):
    df = pre_process(df)
    y=df['left']
    X = df.drop('left', axis=1)
    dummyRow = pd.DataFrame(np.zeros(len(X.columns)).reshape(1,len(X.columns)), columns=X.columns) #creates this for test prediction.
    dummyRow.to_csv("dummyRow.csv", index=False)

    model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=250,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

    #Split the dataset (using smote sampled data).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train,y_train)

    pickle_file = "pickle_model.pkl"
    with open (pickle_file, 'wb') as file: #wb - write/open file in binary mode.
        pickle.dump(model,file)


    #Test Results
#     yp = model.predict(X_test)
#     print (accuracy_score(y_test, yp))
    
    print("Train Result:\n===========================================")    
    train_pred = model.predict(X_train)
    print(f"accuracy score: {accuracy_score(y_train, train_pred):.4f}\n")
    print(f"Classification Report: \n \tPrecision: {precision_score(y_train, train_pred)}\n\tRecall Score: {recall_score(y_train, train_pred)}\n\tF1 score: {f1_score(y_train, train_pred)}\n")
    print(f"Confusion Matrix: \n {confusion_matrix(y_train, model.predict(X_train))}\n")

    test_pred = model.predict(X_test)
    print("Test Result:\n===========================================")        
    print(f"accuracy score: {accuracy_score(y_test, test_pred)}\n")
    print(f"Classification Report: \n \tPrecision: {precision_score(y_test, test_pred)}\n\tRecall Score: {recall_score(y_test, test_pred)}\n\tF1 score: {f1_score(y_test, test_pred)}\n")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, test_pred)}\n")

    print ("Attrition",sum(test_pred!=0))
    print ("Stayed", sum(test_pred==0))
#   
    
def pred(ob):
    d1 = ob.to_dict() #convert object into dictionary and create a df.
    df = pd.DataFrame(d1, index=[0])
    df.drop("left", axis=1, inplace=True) #Dropping target feature before pre-processing.
    df = pre_process(df)
    # dummyrow_filename = 'dummyRow.csv'
    # dummyrow_filename = os.path.dirname(__file__)+"/" + dummyrow_filename  
    df2 = pd.read_csv('dummyRow.csv')                                       #dummyRow is all the columns during training with  1 row of 0 values.
    for c1 in df.columns: #Add each column from df to df2.
        df2[c1]=df[c1]
   #Load the pickled model.
    pickle_filename = "pickle_model.pkl"
    #pickle_filename=os.path.dirname(__file__)+"/"+pickle_filename 
    with open (pickle_filename, 'rb') as file:
        model = pickle.load(file) 
    pred = model.predict(df2)
    return pred

if __name__ == "__main__":
    df = pd.read_csv("turnover.csv")
    training(df)
    pred(df)


# #Split into predictors and response variables.
# X = df.drop('left', axis=1)
# y = df['left']

# #Getting dummies for numerical features. Same as One-Hot Encoder for nominal features.
# X = pd.get_dummies(X)

# #Using SMOTE to create synthetic sample for equal yes's and no's
# smote = SMOTE()
# X, y = smote.fit_sample(X,y)
# plot_2d_space(pca.fit_transform(X), y, 'Imbalanced dataset (2 PCA components)')

# #Scale the features.
# #Since the data is not standardized, it needs to be scaled to the same level.
# scaler = preprocessing.MinMaxScaler()#(feature_range=(0,1)) can choose range too. can get constant values 
# X = scaler.fit_transform(X)

# #Set aside 25% of the data for test. Keep 75% for training.
# X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=.80, random_state = 200)


# rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=None, max_features='auto',
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=250,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
# rf.fit(X_train,y_train)


# y_pred = rf.predict(X_test)
# print (classification_report(y_test, y_pred))

# #Pickle the model.
# import pickle
# from sklearn.externals import joblib

# filename = 'pickle_model.pkl'#name the pickle file
# joblib.dump (rf, filename) #Stores the weights along with the ml model.

# #Saving the scaler.
# joblib.dump(scaler, 'pickledScaler.pkl')

