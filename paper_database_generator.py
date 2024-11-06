import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import os

def generate_datasets():
    configurations = []
    datasets = {}
    
    # Define ranges for different parameters
    n_features_options = [2, 5, 10, 20]  # Number of total features
    n_informative_options = [2, 3, 5]  # Number of informative features
    n_redundant_options = [0, 2, 5]  # Number of redundant features
    n_classes_options = [2, 3, 5]  # Number of classes
    n_clusters_per_class_options = [1, 2, 3]  # Number of clusters per class
    base_weights_options = [[0.5, 0.5], [0.7, 0.3], None]  # Base weights for 2 classes

    # Directory to save plots
    plot_dir = "dataset_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Generate configurations
    for n_features in n_features_options:
        for n_informative in n_informative_options:
            if n_informative <= n_features:  # n_informative must be <= n_features
                for n_redundant in n_redundant_options:
                    if n_informative + n_redundant <= n_features:  # Sum must not exceed total features
                        for n_classes in n_classes_options:
                            for n_clusters_per_class in n_clusters_per_class_options:
                                for base_weights in base_weights_options:
                                    if n_classes * n_clusters_per_class <= 2 ** n_informative:
                                        weights = None if base_weights is None else np.random.dirichlet(np.ones(n_classes)*10,1)[0]
                                        config = {
                                            'n_features': n_features,
                                            'n_informative': n_informative,
                                            'n_redundant': n_redundant,
                                            'n_classes': n_classes,
                                            'n_clusters_per_class': n_clusters_per_class,
                                            'weights': weights
                                        }
                                        configurations.append(config)
    
    # Generate datasets based on configurations
    for i, config in enumerate(configurations):
        X, y = make_classification(n_samples=1000, **config)
        datasets[f"dataset_{i}"] = (X, y)
        
        # Apply PCA for visualization if more than 2 features
        if config['n_features'] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X
        
        # Generate a descriptive filename from the configuration
        filename = f"dataset_nf{config['n_features']}_ni{config['n_informative']}_nr{config['n_redundant']}_nc{config['n_classes']}_ncpc{config['n_clusters_per_class']}"
        if config['weights'] is not None:
            weights_str = "_".join(f"{w:.2f}" for w in config['weights'])
            filename += f"_w{weights_str}"
        
        # Save plots
        plt.figure(figsize=(5, 4))
        plt.title(f"Config {i}: nf{config['n_features']}, ni{config['n_informative']}, nr{config['n_redundant']}, nc{config['n_classes']}, ncpc{config['n_clusters_per_class']}")
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
        plt.savefig(f"{plot_dir}/{filename}.png")
        plt.close()
    
    return datasets

# Generate and retrieve datasets
datasets = generate_datasets()

