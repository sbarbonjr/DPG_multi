from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.metrics import mean_squared_error
from sklearn.base import is_classifier, is_regressor
import networkx as nx
import pickle

import networkx as nx
from networkx.drawing.nx_agraph import from_agraph
import pygraphviz as pgv

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpg.core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics
from dpg.visualizer import plot_dpg


def remove_columns_with_nans(df):
    cleaned_df = df.dropna(axis=1)
    return cleaned_df

def list_to_dict(input_list):
    """
    Convert a list into a dictionary where elements at even indices are keys 
    and elements at odd indices are values.

    Args:
        input_list (list): A list with alternating keys and values.

    Returns:
        dict: A dictionary created from the input list.
    """
    return {item[0]: item[1] for item in input_list if len(item) == 2}      



dataset_name = ["Pitaya", "Carambola", "Papaya"]
dataset_url = ["Dragonfruit.csv", "Carambola.csv", "Papaya.csv"]
ensemble_classifier = [RandomForestClassifier(n_estimators=5), AdaBoostClassifier(n_estimators=5), ExtraTreesClassifier(n_estimators=5), BaggingClassifier(n_estimators=5)]
#ensemble_classifier = [RandomForestClassifier(n_estimators=7), AdaBoostClassifier(n_estimators=7)]

PATH = '/home/barbon/PycharmProjects/DPG/DPG/MDPI_fruit/'

for id, dataset in enumerate(dataset_name):
    fruit_dataFrame = remove_columns_with_nans(pd.read_csv(PATH + dataset_url[id], encoding="latin1"))
    
    data = fruit_dataFrame.iloc[:, 1:]
    fruit_dataFrame["Label"] = ["Class " + str(i) for i in pd.factorize(fruit_dataFrame.iloc[:, 0])[0]]
    num_c = len(np.unique(fruit_dataFrame[["Label"]]))
    feature_names = data.columns.values
 
    X_train, X_test, y_train, y_test = train_test_split(data, fruit_dataFrame["Label"], test_size=0.3, random_state=42)
    

    for classifier in ensemble_classifier: 
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print('model', model)
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        file_name = dataset+classifier.__class__.__name__+"_perf_"
        with open(file_name, "w") as f:
            f.write(f'Accuracy: {accuracy:.2f}\n')
            f.write('\nConfusion Matrix:\n')
            for i in confusion:
                f.write(f'{str(i)}\n')
            f.write('\nClassification Report:')
            f.write(classification_rep)

        print(f'Accuracy: {accuracy:.2f}')
        print('Confusion Matrix:')
        print(confusion)
        print('Classification Report:')
        print(classification_rep)


        dot = get_dpg(X_train.values,feature_names, model, 0.001, 2, num_classes=num_c)
        
        # Convert Graphviz Digraph to NetworkX DiGraph
        dpg_model, nodes_list = digraph_to_nx(dot)

        # Get metrics from the DPG
        df_dpg = get_dpg_metrics(dpg_model, nodes_list)
        df_metrics = get_dpg_node_metrics(dpg_model, nodes_list)


        #Exporting to GEPHI
        new_labels = list_to_dict(nodes_list)
        G = nx.relabel_nodes(dpg_model, new_labels)
        nx.write_gexf(G, PATH+dataset+classifier.__class__.__name__+".gexf")


        #dpg_model.save(PATH+dataset+classifier.__class__.__name__+'.graph')
        plot_dpg(dataset+classifier.__class__.__name__, dot, df_metrics, df_dpg, save_dir=PATH,attribute=None,communities=True,class_flag=False)    
        #plot_dpg(dataset+classifier.__class__.__name__, dot, df_metrics, df_dpg, save_dir=PATH,attribute=None,communities=False,class_flag=True)    
        #pd.DataFrame(df_dpg).to_csv(dataset+classifier.__class__.__name__+"_dpg.csv")
        pd.DataFrame(df_metrics).to_csv(PATH+dataset+classifier.__class__.__name__+"_graph_metrics.csv")
        

