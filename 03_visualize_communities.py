#!/usr/bin/env python3
"""
Visualize communities detected from k-means clustering analysis.
Creates plots showing all detected communities with different colors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_communities_data(filepath='data/data_with_communities.csv'):
    """Load the data with community assignments."""
    try:
        df = pd.read_csv(filepath)
        # Ensure isolated points have community_id = -1
        df['community_id'] = df['community_id'].fillna(-1).astype(int)
        
        print(f"Data loaded successfully: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Community statistics:")
        community_stats = df['community_id'].value_counts().sort_index()
        for comm_id, count in community_stats.items():
            if comm_id == -1:
                print(f"  Isolated points: {count}")
            else:
                print(f"  Community {comm_id}: {count} points")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return None

def plot_all_communities(df):
    """Create a plot showing all communities with different colors."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create a custom colormap that handles -1 (isolated points) specially
    unique_communities = sorted(df['community_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))
    
    # Plot 1: All communities colored by community_id
    scatter1 = axes[0].scatter(df['x'], df['y'], c=df['community_id'], 
                              cmap='tab20', alpha=0.7, s=30)
    
    num_communities = len(df[df['community_id'] >= 0]['community_id'].unique())
    num_isolated = len(df[df['community_id'] == -1])
    
    axes[0].set_title(f'All Detected Communities\n({num_communities} communities, {num_isolated} isolated points)')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Community ID')
    
    # Plot 2: Original shapes for comparison
    scatter2 = axes[1].scatter(df['x'], df['y'], c=df['shape'], 
                              cmap='viridis', alpha=0.7, s=30)
    axes[1].set_title('Original Shape Labels')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Shape ID')
    
    plt.tight_layout()
    plt.savefig('plots/all_communities.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved all communities plot to plots/all_communities.png")

def plot_community_sizes(df):
    """Plot community size distribution."""
    community_counts = df['community_id'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    
    # Plot all community sizes including isolated points (-1)
    colors = ['red' if idx == -1 else 'skyblue' for idx in community_counts.index]
    bars = plt.bar(community_counts.index, community_counts.values, alpha=0.7, color=colors)
    
    # Add legend only if we have multiple bar types
    if len(bars) > 1:
        plt.legend([bars[0], bars[1]], ['Isolated Points', 'Communities'], loc='upper right')
    
    plt.title(f'Community Size Distribution\n(All communities and isolated points)')
    plt.xlabel('Community ID')
    plt.ylabel('Number of Points')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/community_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved community sizes plot to plots/community_sizes.png")

def generate_summary_statistics(df):
    """Generate and print summary statistics."""
    print("\n=== Community Analysis Summary ===")
    
    total_points = len(df)
    isolated_points = len(df[df['community_id'] == -1])
    community_points = total_points - isolated_points
    num_communities = len(df[df['community_id'] >= 0]['community_id'].unique())
    
    print(f"Total data points: {total_points}")
    print(f"Points in communities: {community_points} ({community_points/total_points*100:.1f}%)")
    print(f"Isolated points: {isolated_points} ({isolated_points/total_points*100:.1f}%)")
    print(f"Number of communities: {num_communities}")
    
    # Community size statistics
    if num_communities > 0:
        community_sizes = df[df['community_id'] >= 0]['community_id'].value_counts()
        print(f"\nCommunity size statistics:")
        print(f"  Largest community: {community_sizes.max()} points")
        print(f"  Smallest community: {community_sizes.min()} points")
        print(f"  Mean community size: {community_sizes.mean():.1f} points")
        print(f"  Median community size: {community_sizes.median():.1f} points")

if __name__ == "__main__":
    print("=== Community Visualization ===\n")
    
    # Load data
    print("Loading community data...")
    df = load_communities_data()
    if df is None:
        exit(1)
    
    # Generate visualizations
    print("\nCreating community visualization...")
    plot_all_communities(df)
    
    print("\nCreating community size distribution...")
    plot_community_sizes(df)
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    print("\nVisualization complete!")
    print("Generated plots:")
    print("  - plots/all_communities.png")
    print("  - plots/community_sizes.png")