import pandas as pd
import pickle
import dill

def chargement_data():

    path = r"generated_files"

    path_data_test = path + r'\data_test_preprocessed.csv'
    path_data_global_feat = path +  r'\global_feature_importance_sorted_bis.csv'
    path_model =  path + r'\modele_LR_precise'
    path_valid_id = path + r'\valid_IDs_bis.csv'
    path_explainer = path + r'\explainer_file'

    data_test = pd.read_csv(path_data_test)
    data_global_feat = pd.read_csv(path_data_global_feat)
    model = pickle.load(open(path_model, 'rb'))
    valid_ids = pd.read_csv(path_valid_id)

    with open(path_explainer, 'rb') as f:
        explainer = dill.load(f)

    return data_test,model,data_global_feat, valid_ids, explainer



def proba_to_class(y_pred_proba_local, threshold):
    """convert a probability of class 0 and class 1 vector to a boolean
    that indicates the class in which the sample is placed, depending on the threshold
    given """

    y_class = y_pred_proba_local[:, 0] < threshold
    if len(y_class) == 1:
        return y_class[0]
    else:
        return y_class



def get_means_features(data):
    """return for each feature of a dataset, the vector of means for each feature"""
    return data.drop(["SK_ID_CURR"], axis=1).mean()


def get_means_default(data, model):
    """return for each feature of a dataset, the vector of means for clients of class 1
    for each feature """
    data_no_id = data.drop(["SK_ID_CURR"], axis=1)
    y_pred_proba_local = model.predict_proba(data_no_id)
    y_pred_class = proba_to_class(y_pred_proba_local,0.8)
    data_default = data_no_id[y_pred_class == 1]
    return data_default.mean()


def get_means_repay(data, model):
    """return for each feature of a dataset, the vector of means for clients of class 0
    for each feature """
    data_no_id = data.drop(["SK_ID_CURR"], axis=1)
    y_pred_proba_local = model.predict_proba(data_no_id)
    y_pred_class = proba_to_class(y_pred_proba_local,0.8)
    data_repay = data_no_id[y_pred_class == 0]
    return data_repay.mean()



def get_prediction_and_explaination(data, client_id, model_local, explainer):
    """return the whole prediction and LIME explaination for a client in the dataset"""

    try:
        index_client = data[data["SK_ID_CURR"] == client_id].index[0]
    except:
        return "This client ID does not exist"

    data_no_id = data.drop(["SK_ID_CURR"], axis=1)

    data_client = data[data["SK_ID_CURR"] == client_id]

    data_client = data_client.drop(["SK_ID_CURR"], axis=1)

    y_pred_proba = model_local.predict_proba(data_client)

    # proba_remboursement = y_pred_proba[0,1]
    proba_remboursement = y_pred_proba

    class_target = int(proba_to_class(y_pred_proba, threshold=0.8))

    local_explaination = explainer.explain_instance(data_client.squeeze(), model_local.predict_proba,
                                                    num_features=239)
    local_explaination_list = local_explaination.as_list()
    local_explaination_map = local_explaination.as_map()

    feature_values=[]
    feature_names=[]

    for feat in local_explaination_map[1]:
        index_feature = feat[0]
        feature_values.append(data_client.iloc[:,index_feature].item())
        feature_name = data_client.columns.to_list()[index_feature]
        feature_names.append(feature_name)


    return proba_remboursement.tolist(), class_target, local_explaination_list, feature_values, feature_names

