import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd
import argparse
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)



xgb_model = xgb.XGBClassifier(n_estimators=args.n_estimators,learning_rate=args.learning_rate,random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("This is the accuracy score for XGBClassifier : ")
print(accuracy_score(y_test, xgb_pred))
print("This is the confusion matrix score for XGBClassifier : ")
print(confusion_matrix(y_test, xgb_pred))


#params = {
#            "num_class": 3,
#            "learning_rate": args.learning_rate,
#            "eval_metric": "mlogloss",
#            "colsample_bytree": args.colsample_bytree,
#            "subsample": args.subsample,
#            "seed": 42,
#            "n_estimators": args.n_estimators,
#            "verbosity": args.verbosity,
#        }
#model = xgb.train(params, dtrain, evals=[(dtrain, "train")])

# evaluate model
#y_proba = model.predict(dtest)
#y_pred = y_proba.argmax(axis=1)
#loss = log_loss(y_test, y_proba)
#acc = accuracy_score(y_test, y_pred)


#print("the loss is : " + str(loss) + " , the accuracy is : " + str(acc))


#print(confusion_matrix(y, y_pred))

