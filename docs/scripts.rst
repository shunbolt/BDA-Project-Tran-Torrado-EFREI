Scripts
=======

All scripts are stored under the src directory

You must execute the following scripts in the specified order :

* src/data/Data_preparation.py --> This will be used to process the raw data into processed data

* src/models/<anymodel>/Model-Training-<modelname>-Mlflow.py --> This will train the model from the processed data and output a prediction


Random Forest
#############

To use the model you can use the command :


:code:`python Model_Training_RandomForest-MLflow.py (--nb-estimators) {int}`

``--nb-estimators`` is an optional argument


Gradient Boosting
#################

To use the model you can use the command :

:code:`python Model_Training_GradientBoosting-MLflow.py (--learning-rate) {float} (--n-estimators) {int}`

``--learning-rate`` and ``--n-estimators`` are optional arguments


XGBoost
########

To use the model you can use the command :

:code:`python Model_Training_XGboost-MLflow.py (--learning-rate) {float} (--n-estimators) {int}`

``--learning-rate`` and ``--n-estimators`` are optional arguments

