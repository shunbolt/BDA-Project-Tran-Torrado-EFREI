import pandas as pd
import numpy as np
import argparse
import mlflow

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from urllib.parse import urlparse

df_train = pd.read_csv('../../../data/processed/processed_application_train.csv')
df_test = pd.read_csv('../../../data/processed/processed_application_test.csv')

# get argument for the model 
def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=10,
        help="Number of boosting rounds. (default: 10)",
    )
    return parser.parse_args()

args = parse_args() 

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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

# Impute impour data, then train
X_train_imp = imp.transform(X_train)


#Run mlflow 
with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators=args.n_estimators,learning_rate=args.learning_rate,random_state=42)
    clf.fit(X_train_imp, y_train)

    X_test_imp = imp.transform(X_test)
    clf_pred = clf.predict(X_test_imp)

    clf_score = clf.score(X_test_imp, y_test)

    print("This is the accuracy score for Gradient Boosting Classifier : ")
    print(clf_score)
    print("This is the confusion matrix score for Gradient Boosting Classifier : ")
    cm = confusion_matrix(y_test, clf_pred)
    print(cm)

    mlflow.log_metrics({"accuracy": clf_score})
    mlflow.log_param("learning_rate",args.learning_rate)
    mlflow.log_param("estimators",args.n_estimators)
    
    #log metric confusion metrix
    t_n, f_p, f_n, t_p = cm.ravel()
    mlflow.log_metric("true_negative", t_n)
    mlflow.log_metric("false_positive", f_p)
    mlflow.log_metric("false_negative", f_n)
    mlflow.log_metric("true_positive", t_p)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        mlflow.sklearn.log_model(clf, "model", registered_model_name="Gradient Boosting Classifier")
    else:
        mlflow.sklearn.log_model(clf, "model")    