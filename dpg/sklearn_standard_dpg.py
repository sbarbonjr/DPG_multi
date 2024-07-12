from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer,
    load_diabetes,
)

from .core import digraph_to_nx, get_dpg, get_dpg_node_metrics, get_dpg_metrics
from .visualizer import plot_dpg

def select_dataset(name):
    """
    Selects a standard sklearn dataset based on the provided name.

    Args:
    name: The name of the dataset to load.

    Returns:
    The selected dataset.
    """
    datasets = {
        "iris": load_iris(),
        "diabetes": load_diabetes(),
        "digits": load_digits(),
        "wine": load_wine(),
        "cancer": load_breast_cancer(),
    }

    return datasets.get(name.lower(), None)


def test_base_sklearn(datasets, n_learners, perc_var, decimal_threshold, file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, class_flag=False):
    """
    Trains a Random Forest classifier on a selected dataset, evaluates its performance, and optionally plots the DPG.

    Args:
    datasets: The name of the dataset to use.
    n_learners: The number of trees in the Random Forest.
    perc_var: Threshold value indicating the desire to retain only those paths that occur with a frequency exceeding a specified proportion across the trees.
    decimal_threshold: Decimal precision of each feature. 
    file_name: The name of the file to save the evaluation results. If None, prints the results to the console.
    plot: Boolean indicating whether to plot the DPG. Default is False.
    save_plot_dir: Directory to save the plot image. Default is "examples/".
    attribute: A specific node attribute to visualize. Default is None.
    communities: Boolean indicating whether to visualize communities. Default is False.
    class_flag: Boolean indicating whether to highlight class nodes. Default is False.

    Returns:
    df: A pandas DataFrame containing node metrics.
    df_dpg: A pandas DataFrame containing DPG metrics.
    """
    
    # Load dataset
    dt = select_dataset(datasets)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        dt.data, dt.target, test_size=0.3, random_state=42
    )
    
    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_learners, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Print or save the evaluation results
    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(f'Accuracy: {accuracy:.2f}\n')
            f.write('\nConfusion Matrix:\n')
            for i in confusion:
                f.write(f'{str(i)}\n')
            f.write('\nClassification Report:')
            f.write(classification_rep)
    else:
        print(f'Accuracy: {accuracy:.2f}')
        print('Confusion Matrix:')
        print(confusion)
        print('Classification Report:')
        print(classification_rep)

    # Extract DPG
    dot = get_dpg(X_train, dt.feature_names, rf_classifier, perc_var, decimal_threshold)
    
    # Convert Graphviz Digraph to NetworkX DiGraph
    dpg_model, nodes_list = digraph_to_nx(dot)

    if len(nodes_list) < 2:
        print("Warning: Less than two nodes resulted.")
        return
    
    # Get metrics from the DPG
    df_dpg = get_dpg_metrics(dpg_model, nodes_list)
    df = get_dpg_node_metrics(dpg_model, nodes_list)
    
    # Plot the DPG if requested
    if plot:
        plot_name = (
            datasets
            + "_bl"
            + str(n_learners)
            + "_perc"
            + str(perc_var)
            + "_dec"
            + str(decimal_threshold)
        )

        plot_dpg(
            plot_name,
            dot,
            df,
            df_dpg,
            save_dir=save_plot_dir,
            attribute=attribute,
            communities=communities,
            class_flag=class_flag
        )
    
    return df, df_dpg