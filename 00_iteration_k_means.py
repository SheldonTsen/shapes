#!/usr/bin/env python3
"""
Iteration K-means clustering analysis on the multishapes dataset.
Runs k-means with k=100 for 20 iterations and stores all results.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath='data/multishapes.csv'):
    """Load the multishapes dataset."""
    try:
        df = pd.read_csv(filepath)
        # Add unique ID for each row
        df['unique_id'] = range(len(df))
        print(f"Dataset loaded successfully: {len(df)} samples")
        print(f"Features: {df.columns.tolist()}")
        print(f"Shape distribution:\n{df['shape'].value_counts().sort_index()}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return None

def run_iteration_kmeans(data, k=100, n_iterations=100):
    """Run k-means clustering multiple times and store all results."""
    print(f"Running k-means with k={k} for {n_iterations} iterations...")
    
    # Prepare features (x, y coordinates)
    features = data[['x', 'y']].values
    
    # List to store all iteration results
    all_results = []
    
    for iteration in range(n_iterations):
        print(f"Running iteration {iteration + 1}/{n_iterations}...")
        
        # Run k-means clustering with high variability
        kmeans = KMeans(n_clusters=k, random_state=None, n_init=1, max_iter=1, init='random')
        cluster_labels = kmeans.fit_predict(features)
        
        # Create dataframe for this iteration
        iteration_df = data[['x', 'y', 'shape', 'unique_id']].copy()
        iteration_df['iteration_id'] = iteration + 1  # 1-indexed
        iteration_df['cluster_id'] = cluster_labels
        
        # Add to results list
        all_results.append(iteration_df)
        
        print(f"  Iteration {iteration + 1}: {len(np.unique(cluster_labels))} unique clusters found")
    
    # Combine all iterations into single dataframe
    final_df = pd.concat(all_results, ignore_index=True)
    
    print(f"\nCombined results:")
    print(f"  Total rows: {len(final_df)}")
    print(f"  Expected rows: {len(data)} x {n_iterations} = {len(data) * n_iterations}")
    print(f"  Columns: {final_df.columns.tolist()}")
    
    return final_df

def save_results(df, filepath='data/iteration_kmeans_results.csv'):
    """Save the results dataframe to CSV."""
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    print(f"File size: {df.shape[0]} rows x {df.shape[1]} columns")

if __name__ == "__main__":
    print("=== Iteration K-means Analysis ===\n")
    
    # Step 1: Load data
    print("Step 1: Loading multishapes.csv...")
    df = load_data()
    if df is None:
        exit(1)
    
    # Step 2 & 3: Run k-means with k=25 for 50 iterations
    print("\nStep 2-3: Running k-means iterations...")
    results_df = run_iteration_kmeans(df, k=100, n_iterations=100)
    
    # Step 4: Save to CSV
    print("\nStep 4: Saving results to CSV...")
    save_results(results_df)
    
    print("\nAnalysis complete!")
    
    # Show sample of final results
    print("\nSample of final dataframe:")
    print(results_df.head(10))
    print("\nDataframe info:")
    print(results_df.info())