import matplotlib.pyplot as plt
import numpy as np



def get_graph_global(data_global):
    """
    return the graph that explains global feature importance
    """

    fig,ax= plt.subplots()
    color = ["red", "red", "green", "red", "green", "red", "green", "red"]
    list_most_imp_features = [data_global["feature"][f"{i}"] for i in range(8)]
    list_values = [data_global["0"][f"{i}"] for i in range(8)]
    ax.barh(np.arange(len(list_values)), width= list_values, align='center', tick_label = list_most_imp_features, color= color)
    return fig,ax


def get_graphs(list_expl, feature_values,feature_names , means_global, means_repay, means_default):
    """return figures of graphs that will be used in dashboard
    ARGS:
        list_expl : explaination retured by get_explaination_prediction for one client
        feature_values : list of values of each feature of the client
        feature_names : list of names of each feature of the client
        means_global : return of get_means fct
        means_repay : return of get_means_repay fct
        means_default : return of get_means_default fct
    """

    fig, axes = plt.subplots(2,3)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    types = ["Client","Moyenne", "En Règle", "En Défaut"]



    for i in range(len(list_expl)):

        value = feature_values[i]
        title = feature_names[i]

        participation = list_expl[i][1]

        if i>5:
            break
        if i<3:
            ax = axes[0,i]
        else :
            ax = axes[1,i-3]



        if participation <= 0 :
            title_graph = title + "\n (Améliore le score)"
            fontdict = {"color" : "green"}
        else:
            title_graph = title + "\n (Dégrade le score)"
            fontdict = {"color" : "red"}

        title_graph += f"\n Poids dans la prédiction : {abs(round(participation*100, ndigits=3))} %"
        ax.set_title(title_graph, fontdict = fontdict)
        value_mean_global = means_global[title]
        value_mean_repay = means_repay[title]
        value_mean_default = means_default[title]
        plt.subplots_adjust(wspace=0.4,
                            hspace=0.4)
        ax.bar(types,[value, value_mean_global, value_mean_repay, value_mean_default])

    return fig, axes


def get_unique_feature_graph(list_expl, feature_values ,feature_name, feature_names , means_global, means_repay, means_default):
    """
    very similar as get_graphs but this is for a unique feature and not the whole set of
    features
    """

    fig, ax = plt.subplots()
    types = ["Client","Moyenne", "En Règle", "En Défaut"]

    index = feature_names.index(feature_name)

    value = feature_values[index]
    title = feature_name

    participation = list_expl[index][1]


    if participation <= 0 :
        title_graph = title + "\n (Améliore le score)"
        fontdict = {"color" : "green"}
    else:
        title_graph = title + "\n (Dégrade le score)"
        fontdict = {"color" : "red"}

    title_graph += f"\n Poids dans la prédiction : {abs(round(participation*100, ndigits=3))} %"
    ax.set_title(title_graph, fontdict = fontdict)
    value_mean_global = means_global[title]
    value_mean_repay = means_repay[title]
    value_mean_default = means_default[title]
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.4)
    ax.bar(types,[value, value_mean_global, value_mean_repay, value_mean_default])

    return fig, ax


