#!/usr/bin/env python3
"""
Plot iterations from k-means clustering results.
Visualizes 3 different iterations with cluster coloring.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_iteration_data(filepath='data/iteration_kmeans_results.csv'):
    """Load the iteration k-means results."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Iterations available: {sorted(df['iteration_id'].unique())}")
        print(f"Number of unique clusters per iteration: {df.groupby('iteration_id')['cluster_id'].nunique().iloc[0]}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return None

def plot_iterations(df, iterations_to_plot=[1, 10, 20]):
    """Plot specified iterations with cluster coloring."""
    print(f"Plotting iterations: {iterations_to_plot}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, iteration_id in enumerate(iterations_to_plot):
        # Filter data for this iteration
        iteration_data = df[df['iteration_id'] == iteration_id]
        
        if len(iteration_data) == 0:
            print(f"Warning: No data found for iteration {iteration_id}")
            continue
            
        # Create scatter plot
        scatter = axes[i].scatter(
            iteration_data['x'], 
            iteration_data['y'], 
            c=iteration_data['cluster_id'], 
            cmap='tab20',  # Good colormap for many clusters
            alpha=0.7,
            s=20
        )
        
        axes[i].set_title(f'Iteration {iteration_id}\n({len(iteration_data)} points, {iteration_data["cluster_id"].nunique()} clusters)')
        axes[i].set_xlabel('X coordinate')
        axes[i].set_ylabel('Y coordinate')
        axes[i].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[i])
        cbar.set_label('Cluster ID')
        
        print(f"  Iteration {iteration_id}: {len(iteration_data)} points, {iteration_data['cluster_id'].nunique()} clusters")
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'plots/iterations_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_individual_iterations(df, iterations_to_plot=[1, 10, 20]):
    """Create individual plots for each iteration."""
    print(f"Creating individual plots for iterations: {iterations_to_plot}")
    
    for iteration_id in iterations_to_plot:
        # Filter data for this iteration
        iteration_data = df[df['iteration_id'] == iteration_id]
        
        if len(iteration_data) == 0:
            print(f"Warning: No data found for iteration {iteration_id}")
            continue
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        scatter = plt.scatter(
            iteration_data['x'], 
            iteration_data['y'], 
            c=iteration_data['cluster_id'], 
            cmap='tab20',
            alpha=0.7,
            s=30
        )
        
        plt.title(f'K-means Clustering - Iteration {iteration_id}\n'
                 f'{len(iteration_data)} points, {iteration_data["cluster_id"].nunique()} clusters')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster ID')
        
        # Save individual plot
        output_path = f'plots/iteration_{iteration_id:02d}_clusters.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Individual plot saved to {output_path}")

def analyze_iteration_statistics(df):
    """Analyze and display statistics about the iterations."""
    print("\n=== Iteration Statistics ===")
    
    # Basic stats
    total_iterations = df['iteration_id'].nunique()
    points_per_iteration = len(df) // total_iterations
    clusters_per_iteration = df.groupby('iteration_id')['cluster_id'].nunique().iloc[0]
    
    print(f"Total iterations: {total_iterations}")
    print(f"Points per iteration: {points_per_iteration}")
    print(f"Clusters per iteration: {clusters_per_iteration}")
    
    # Cluster size distribution for first iteration
    first_iteration = df[df['iteration_id'] == 1]
    cluster_sizes = first_iteration['cluster_id'].value_counts().sort_index()
    
    print(f"\nCluster size statistics (Iteration 1):")
    print(f"  Min cluster size: {cluster_sizes.min()}")
    print(f"  Max cluster size: {cluster_sizes.max()}")
    print(f"  Mean cluster size: {cluster_sizes.mean():.1f}")
    print(f"  Std cluster size: {cluster_sizes.std():.1f}")

if __name__ == "__main__":
    print("=== Plot K-means Iterations ===\n")
    
    # Step 1: Load data
    print("Step 1: Loading iteration results...")
    df = load_iteration_data()
    if df is None:
        exit(1)
    
    # Step 2: Analyze data
    analyze_iteration_statistics(df)
    
    # Step 3: Create plots
    print("\nStep 2: Creating comparison plot...")
    plot_iterations(df, iterations_to_plot=[1, 10, 20])
    
    print("\nStep 3: Creating individual plots...")
    plot_individual_iterations(df, iterations_to_plot=[1, 10, 20])
    
    print("\nPlotting complete!")
    print("Generated plots:")
    print("  - plots/iterations_comparison.png (side-by-side comparison)")
    print("  - plots/iteration_01_clusters.png (individual plot)")
    print("  - plots/iteration_10_clusters.png (individual plot)")
    print("  - plots/iteration_20_clusters.png (individual plot)")