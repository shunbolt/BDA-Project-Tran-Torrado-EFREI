import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import argparse
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

#example of command : python Model_Training_RandomForest.py --nb-estimators 22


df_train = pd.read_csv('../../../data/processed/processed_application_train.csv')
df_test = pd.read_csv('../../../data/processed/processed_application_test.csv')

def parse_args():
    parser = argparse.ArgumentParser(description="RandomForestClassifier example")
    parser.add_argument(
        "--nb-estimators",
        type=int,
        default=10,
        help="int, default=10, The number of trees in the forest ",
    )
    return parser.parse_args()

args = parse_args()
estimators = args.nb_estimators

#mlflow.xgboost.autolog()


#estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 10

#print("Hyperparameter: estimators = " + str(estimators))        

#estimators = int(sys.argv[1]) if ((int(sys.argv[1]) > 1) and (int(sys.argv[1]) < 100)) else 10

# Separate majority and minority classes
df_majority = df_train[df_train["TARGET"] == 0]
df_minority = df_train[df_train["TARGET"] == 1]


# Downsample majority class

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=50000,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

X = df_downsampled.drop(columns="TARGET")
y = df_downsampled['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#from __future__ import print_function

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

# Impute impour data, then train
X_train_imp = imp.transform(X_train)

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=estimators)
    clf = clf.fit(X_train_imp, y_train)

    X_test_imp = imp.transform(X_test)
    y_pred_test = clf.predict(X_test_imp)

    print("This is the accuracy score for Random Forest Classifier : ")
    acc = accuracy_score(y_test, y_pred_test)
    print(accuracy_score(y_test, y_pred_test))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print("\nThis is the confusion matrix for Random Forest Classifier : ")
    cm = confusion_matrix(y_test, y_pred_test)
    print(confusion_matrix(y_test, y_pred_test))
    
    mlflow.log_metrics({"accuracy": acc})
    mlflow.log_param("estimators",estimators)
    
    t_n, f_p, f_n, t_p = cm.ravel()
    mlflow.log_metric("true_negative", t_n)
    mlflow.log_metric("false_positive", f_p)
    mlflow.log_metric("false_negative", f_n)
    mlflow.log_metric("true_positive", t_p)
    
    
    
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(clf, "model", registered_model_name="Random Forest Classifier")
    else:
        mlflow.sklearn.log_model(clf, "model")    
    
    