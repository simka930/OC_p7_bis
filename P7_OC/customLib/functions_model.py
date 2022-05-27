# Load packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lime import lime_tabular


def custom_SMOTE_func(X, y, frac_over=0.2, frac_under=0.5):
    """balance a dataset by using SMOTE for oversapling
    the minority class by 20%, and undersampling (if needed)
    the majority class so that it is 2x bigger than the minority
    class """

    # Combine SMOTE with random under-sampling of the majority class

    # summarize class distribution
    counter = Counter(y)
    print("old count : ", counter)

    # define pipeline
    over = SMOTE(sampling_strategy=frac_over)
    under = RandomUnderSampler(sampling_strategy=frac_under)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    X_bal, y_bal = pipeline.fit_resample(X, y)

    # summarize the new class distribution
    counter = Counter(y_bal)
    print("new count : ", counter)

    return X_bal, y_bal




def custom_score_fct(y_true, y_pred):
    """ custom function that acts as a scorer for or problem.
    it will take into account the fact that a loan payment default is much more important than a payment.
     the severity of default is arbitrary set to 10 times the severity of a payment
     it is then better to not give a loan to sbdy who would have repay it, than giving it to sbdy who will not.

     The function will then be an adjusted f1 score where false negative will have more impact than false positive"""

    try:
        CM = confusion_matrix(y_true, y_pred)

        # True positives, False positives, True negatives,  False negatives
        TP = CM[1][1]
        FP = CM[0][1]
        TN = CM[0][0]
        FN = CM[1][0]

    except:
        return 0

    adjusted_f1 = TP / (TP + 1 / 2 * (FP + 10 * FN))

    return adjusted_f1



def get_explainer(data, model_local):
    """ Use LIME librairy to generate an explainer associated with the tuple (data, model)"""

    data_no_id = data.drop(["SK_ID_CURR"], axis=1)
    explainer = lime_tabular.LimeTabularExplainer(data_no_id.values, mode="classification",
                                                  class_names=None, feature_names=data_no_id.columns.to_list())

    return explainer



def get_feature_importance(pipe_model, list_features):
    """ return : array (1x230) that gives feature importance of each 230 variables using
    pca feature importances and model feature importance"""

    classification_array = pipe_model.named_steps["classification"].coef_
    pca_array = pipe_model.named_steps["pca"].components_
    np_feat_importance = np.matmul(classification_array, pca_array)

    df_feat_importance = pd.DataFrame(np_feat_importance, columns=list_features)

    return df_feat_importance























# %%
