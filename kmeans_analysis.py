#!/usr/bin/env python3
"""
K-means clustering analysis on the multishapes dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
import os

warnings.filterwarnings('ignore')

def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Created 'plots' directory")

def load_data(filepath='data/multishapes.csv'):
    """Load the multishapes dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully: {len(df)} samples")
        print(f"Features: {df.columns.tolist()}")
        print(f"Shape distribution:\n{df['shape'].value_counts().sort_index()}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return None

def find_optimal_k(data, max_k=15):
    """Find optimal number of clusters using elbow method and silhouette score."""
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved optimal k analysis plot to plots/optimal_k_analysis.png")
    
    # Find optimal k based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal k based on silhouette score: {optimal_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")
    
    return optimal_k

def perform_clustering(data, k):
    """Perform K-means clustering."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
    print(f"K-means clustering completed with k={k}")
    print(f"Silhouette score: {silhouette_score(data, cluster_labels):.3f}")
    
    return kmeans, cluster_labels

def visualize_results(data, cluster_labels, true_labels, kmeans):
    """Create comprehensive visualizations of clustering results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data with true labels
    scatter1 = axes[0, 0].scatter(data['x'], data['y'], 
                                 c=true_labels, cmap='tab10', alpha=0.7)
    axes[0, 0].set_title('Original Data (True Labels)')
    axes[0, 0].set_xlabel('X coordinate')
    axes[0, 0].set_ylabel('Y coordinate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Clustered data
    scatter2 = axes[0, 1].scatter(data['x'], data['y'], 
                                 c=cluster_labels, cmap='viridis', alpha=0.7)
    
    # Add cluster centers
    centers = kmeans.cluster_centers_
    axes[0, 1].scatter(centers[:, 0], centers[:, 1],
                      c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[0, 1].set_title('K-means Clustering Results')
    axes[0, 1].set_xlabel('X coordinate')
    axes[0, 1].set_ylabel('Y coordinate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cluster distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    axes[1, 0].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='skyblue')
    axes[1, 0].set_title('Cluster Size Distribution')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Number of Points')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Comparison matrix
    comparison_df = pd.crosstab(true_labels, cluster_labels, margins=True)
    sns.heatmap(comparison_df.iloc[:-1, :-1], annot=True, fmt='d', 
                cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('True Labels vs Cluster Labels')
    axes[1, 1].set_xlabel('Predicted Cluster')
    axes[1, 1].set_ylabel('True Shape')
    
    plt.tight_layout()
    plt.savefig('plots/clustering_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved clustering results plot to plots/clustering_results.png")

def analyze_performance(true_labels, cluster_labels, data):
    """Analyze clustering performance."""
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    sil_score = silhouette_score(data, cluster_labels)
    
    print("\n=== Clustering Performance ===")
    print(f"Adjusted Rand Index: {ari_score:.3f}")
    print(f"Silhouette Score: {sil_score:.3f}")
    
    # Cluster statistics
    print(f"\nCluster Statistics:")
    for i in range(len(np.unique(cluster_labels))):
        cluster_size = np.sum(cluster_labels == i)
        print(f"Cluster {i}: {cluster_size} points ({cluster_size/len(cluster_labels)*100:.1f}%)")

if __name__ == "__main__":
    print("=== K-means Clustering Analysis ===\n")
    
    # Create plots directory
    create_plots_directory()
    
    # Load data
    df = load_data()
    if df is None:
        exit(1)
    
    # Extract features (x, y coordinates) - data is already scaled
    features = df[['x', 'y']].copy()
    print("Using data as-is (already scaled)")
    
    # Find optimal number of clusters
    print("\nFinding optimal number of clusters...")
    optimal_k = find_optimal_k(features)
    
    # Perform clustering with optimal k
    print(f"\nPerforming K-means clustering with k={optimal_k}...")
    kmeans, cluster_labels = perform_clustering(features, optimal_k)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(features, cluster_labels, df['shape'], kmeans)
    
    # Analyze performance
    analyze_performance(df['shape'], cluster_labels, features)
    
    print("\nAnalysis complete!")