MLFlow 
======

Make sure that Mlflow is installed in your environment

MlFlow Tracking
###############

After running the script appended with MLflow, run in the same directory :

:code:`mlflow ui`

Then check the model results at **localhost:5000**

MLFlow Project
##############

Once the model has been run, you can inpect the artifact under the **mlruns** directory and retrieve the conda.yaml.

Each model project should contain an **MLProject** file. If so then run :

:code:`mlflow run .`

A dedicated conda environment will be created and run the project.

MLFLow Models
##############

You can deploy the model in a dedicated HTTP server using the **MLmodel** file. Use the following code :

:code:`mlflow models serve -m mlruns/0/<artifact-id>/artifacts/model -p 1234`

You can retrieve the artifact ID using the MLFlow Tracking web interface at port 5000.

Once the server is deployed you can passdata as serialized JSON through a CURL request at port 1234 like this one :

:code:`curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["DAYS_BIRTH","EXT_SOURCE_3","EXT_SOURCE_2","EXT_SOURCE_1"],"data":[[-9461,0.13937578009978951,0.2629485927471776,0.08303696739132256]]}' http://127.0.0.1:1234/invocations`


.. warning:: MLFLow models is not supported for XGBoost model

