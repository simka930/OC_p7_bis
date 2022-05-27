#APP STREAMLIT : (commande : streamlit run XX/dashboard.py depuis le dossier python)

# Imports
import streamlit as st
import requests
import json
from customLib.functions_dashboard import *
import ast


# URLs
API_url = "http://127.0.0.1:8000/"

API_url_proba = API_url + "get-proba/"
API_url_means = API_url + "get_means/"
API_url_global = API_url + "get_global_feat_imp"
API_url_valid_ids = API_url + "get_valid_IDs"



# Title
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:')

st.markdown("Exemples d'identifiants à prêt accordé : 100042, 289158, 399953")
st.markdown("Exemples d'identifiants à prêt refusé : 418601, 100005, 100117")
# all instructions if ID is entered
if id_input:

    # get valied IDs from API
    valid_ids = ast.literal_eval(json.loads(requests.get(API_url_valid_ids).text))["SK_ID_CURR"].values()

    # check if ID entered can be manipulated. if error : print "not valid id"
    # other condition if id is indeed an integer but is not valid
    try:
        if int(id_input) in valid_ids:
            id_is_valid = True
        else:
            st.write("Identifiant client non valide")
            id_is_valid = False
            if "data" in st.session_state:
                del st.session_state["data"]
    except:
        st.write("Identifiant client non valide")
        id_is_valid = False
        if "data" in st.session_state:
            del st.session_state["data"]


    # if ID is valid
    if id_is_valid == True:


        API_url_client = API_url_proba + id_input

        # loading spinner
        with st.spinner('Chargement du score du client (Temps estimé : 40 secondes)...'):

            # st.session state is used to NOT reload the whole prediction
            # if a widget state is changed, if ID has not changed.
            # this is an issue because it takes lot of time
            if "data" not in st.session_state:

                # Return from API
                predict_explain = json.loads(requests.get(API_url_client).text)
                st.session_state["data"] = predict_explain

            elif "id" in st.session_state :
                if st.session_state["id"] == id_input:
                    predict_explain = st.session_state["data"]
                else:
                    predict_explain = json.loads(requests.get(API_url_client).text)
                    st.session_state["data"] = predict_explain

            else:
                predict_explain = st.session_state["data"]

            st.session_state["id"] = id_input
            proba_remboursement = predict_explain[0][0][0]
            classe_predite = predict_explain[1]


            # Print result of prediction
            if classe_predite==1:
                etat = 'client à risque'
                chaine_pred = 'Prédiction : **' + etat +  '** avec **' + str(round((1-proba_remboursement)*100)) + '%** de risque de défaut '
                chaine_answer_bank = f'<p style="color:Red; font-size: 20px;">Prêt Refusé</p>'

            else:
                etat = 'client peu risqué'
                chaine_pred = 'Prédiction : **' + etat +  '** avec **' + str(round((1-proba_remboursement)*100)) + '%** de risque de défaut '
                chaine_answer_bank = f'<p style="color:Green; font-size: 18px;">Prêt Accepté</p>'


            chaine = 'Prédiction : **' + etat +  '** avec **' + str(round((1-proba_remboursement)*100)) + '%** de risque de défaut '
            #affichage de la prédiction
            st.markdown(chaine_pred, unsafe_allow_html=False)
            st.markdown(chaine_answer_bank, unsafe_allow_html=True)




            st.subheader("Caractéristiques principales client influençant le score")
            st.markdown("Affichage des 6 caractéristiques propres au client qui ont le plus d'importance dans la prédiction")


            with st.spinner('Chargement des détails de la prédiction...'):


                list_explaination = predict_explain[2]
                feature_values = predict_explain[3]
                feature_names = predict_explain[4]
                means_global = json.loads(requests.get(API_url_means).text)[0]
                means_repay = json.loads(requests.get(API_url_means).text)[1]
                means_default = json.loads(requests.get(API_url_means).text)[2]

                fig, axes = get_graphs(list_explaination, feature_values,feature_names, means_global, means_repay, means_default)

                st.pyplot(fig)

                st.markdown("**Note** : Des caractéristiques améliorant le score indiquent qu'elles augmentent la probabilité de remboursement. Il est possible "+
                            "qu'un client désigné comme à risque aie toutes ses caractéristiques améliorant le score, mais pas suffisamment pour être en dessous de "+
                            "**la valeur seuil de 20% de risque de défaut** pour se voir accorder un prêt")

            st.subheader("Caractéristiques globales les plus importantes ")

            st.markdown("liste des 8 caractéristiques ayant le plus de poids dans le modèle.  \n"
                        "**VERT** : valeur haute => probabilité de remboursement plus élevée  \n"
                        "**ROUGE** : valeur haute => probabilité de remboursement plus faible")

            dict_global_feat_importance = json.loads(requests.get(API_url_global).text)
            fig2,axes2 = get_graph_global(dict_global_feat_importance)
            st.pyplot(fig2)



            with st.sidebar:


                st.header("Choisir nouvelle variable à afficher")
                st.markdown('Cette section permet de sélectionner une nouvelle variable client à afficher et comparer aux autres clients')

                liste_features = tuple([''] + [dict_global_feat_importance["feature"][f"{i}"] for i in range(len(dict_global_feat_importance["0"]))])


                feature_to_update = st.sidebar.selectbox('Quelle caractéristique souhaitez vous afficher', liste_features)

                if feature_to_update != "":

                    fig3,ax3 = get_unique_feature_graph(list_explaination, feature_values,feature_to_update,
                                                        feature_names, means_global, means_repay, means_default)

                    st.pyplot(fig3)
