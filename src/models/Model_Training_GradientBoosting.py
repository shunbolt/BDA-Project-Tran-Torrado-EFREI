from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.utils import resample

df_train = pd.read_csv('../data/processed/processed_application_train.csv')
df_test = pd.read_csv('../data/processed/processed_application_test.csv')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
clf = GradientBoostingClassifier(n_estimators=args.n_estimators,learning_rate=args.learning_rate,random_state=42)
clf.fit(X_train_imp, y_train)

X_test_imp = imp.transform(X_test)
clf_pred = clf.predict(X_test_imp)

clf_score = clf.score(X_test_imp, y_test)

print("This is the accuracy score for Gradient Boosting Classifier : ")
print(clf_score)
print("This is the confusion matrix score for Gradient Boosting Classifier : ")
print(confusion_matrix(y_test, clf_pred))

