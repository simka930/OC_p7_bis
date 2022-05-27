# Imports
from fastapi import FastAPI, Path
from customLib.functions_API import*


app = FastAPI()


# custom function of importing all the needed files to do prediction
data_test,model,data_global_feat, valid_ids, explainer = chargement_data()


# endpoint to get the prediction and explaination (via LIME) of a specific client via his ID
@app.get("/get-proba/{client_id}")
def get_proba_explaination_api(client_id: int = Path(None, description="Expecting client ID")):
    return get_prediction_and_explaination(data_test, client_id, model, explainer)


# endpoint to get the means of all features for client in default, repayment, and all clients from
# the dataset
@app.get("/get_means")
def get_means():
    return get_means_features(data_test), get_means_repay(data_test, model), get_means_default(data_test, model)


# endpoint to get the global feature importance (that is NOT specific to a client)
@app.get("/get_global_feat_imp")
def get_global_feat():
    return data_global_feat

# endpoint to get all possible (then valid) IDs to check if entered ID is valid in dashboard
@app.get("/get_valid_IDs")
def get_valid_ids():
    return valid_ids.to_json()