#!/usr/bin/env python3
"""
Community detection analysis on k-means clustering results.
Performs self join on iteration_id and cluster_id to find co-clustered points.
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
import warnings

warnings.filterwarnings('ignore')

def load_iteration_results(filepath='data/iteration_kmeans_results.csv'):
    """Load the iteration k-means results."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Iterations: {sorted(df['iteration_id'].unique())}")
        print(f"Unique data points: {df['unique_id'].nunique()}")
        print(f"Sample data:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return None

def perform_self_join(df):
    """Perform self join on iteration_id and cluster_id to find co-clustered pairs."""
    print("\nPerforming self join on iteration_id and cluster_id...")
    
    # Self join: find pairs of points that are in the same cluster in the same iteration
    joined = df.merge(
        df, 
        on=['iteration_id', 'cluster_id'],
        suffixes=('_1', '_2')
    )
    
    print(f"Self join result: {len(joined)} rows")
    
    # Filter to only keep pairs where unique_id_1 < unique_id_2 to avoid duplicates and self-pairs
    pairs = joined[joined['unique_id_1'] < joined['unique_id_2']].copy()
    
    print(f"After removing duplicates and self-pairs: {len(pairs)} rows")
    
    # Select relevant columns
    result = pairs[['unique_id_1', 'unique_id_2', 'iteration_id', 'cluster_id', 
                    'x_1', 'y_1', 'shape_1', 'x_2', 'y_2', 'shape_2']].copy()
    
    print(f"Final result columns: {result.columns.tolist()}")
    print(f"Sample of co-clustered pairs:")
    print(result.head(10))
    
    return result

def analyze_co_clustering(pairs_df):
    """Analyze co-clustering patterns."""
    print("\n=== Co-clustering Analysis ===")
    
    # Count how many times each pair appears together across all iterations
    pair_counts = pairs_df.groupby(['unique_id_1', 'unique_id_2']).size().reset_index(name='co_cluster_count')
    
    print(f"Total unique pairs that co-clustered at least once: {len(pair_counts)}")
    print(f"Co-clustering frequency distribution:")
    print(pair_counts['co_cluster_count'].value_counts().sort_index())
    
    # Find pairs that are frequently co-clustered (50+ times = >50% of iterations)
    frequent_pairs = pair_counts[pair_counts['co_cluster_count'] >= 50]  # Appear together in 50+ iterations
    print(f"\nPairs that co-cluster in 50+ iterations (>50% of the time): {len(frequent_pairs)}")
    
    # Analyze by shape similarity
    pairs_with_shapes = pairs_df[['unique_id_1', 'unique_id_2', 'shape_1', 'shape_2']].drop_duplicates()
    same_shape_pairs = pairs_with_shapes[pairs_with_shapes['shape_1'] == pairs_with_shapes['shape_2']]
    diff_shape_pairs = pairs_with_shapes[pairs_with_shapes['shape_1'] != pairs_with_shapes['shape_2']]
    
    print(f"\nShape analysis:")
    print(f"  Pairs with same shape: {len(same_shape_pairs)}")
    print(f"  Pairs with different shapes: {len(diff_shape_pairs)}")
    print(f"  Percentage same shape: {len(same_shape_pairs) / len(pairs_with_shapes) * 100:.1f}%")
    
    return pair_counts

def detect_communities(pair_counts, threshold=25):
    """Apply greedy community detection based on co-clustering pairs."""
    print(f"\n=== Community Detection ===")
    
    # Filter pairs that meet the threshold (50+ co-occurrences = >50% of iterations)
    strong_pairs = pair_counts[pair_counts['co_cluster_count'] >= threshold]
    print(f"Strong pairs (>={threshold} co-occurrences): {len(strong_pairs)}")
    
    if len(strong_pairs) == 0:
        print("No strong pairs found. Cannot perform community detection.")
        return None
    
    # Create a graph from the strong pairs
    G = nx.Graph()
    
    # Add edges with weights based on co-clustering frequency
    for _, row in strong_pairs.iterrows():
        G.add_edge(row['unique_id_1'], row['unique_id_2'], weight=row['co_cluster_count'])
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Apply connected components - any co-occurrence creates same community
    communities = list(nx.connected_components(G))
    
    print(f"Communities detected: {len(communities)}")
    
    # Create community assignments
    community_assignments = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_assignments[node] = i
        print(f"  Community {i}: {len(comm)} nodes")
    
    return community_assignments

def assign_communities_to_data(df, community_assignments):
    """Assign community IDs to the original data points."""
    print(f"\n=== Assigning Communities to Data ===")
    
    # Create a mapping of unique_id to community_id
    df['community_id'] = df['unique_id'].map(community_assignments)
    
    # Points not in any strong pair get community_id = -1 (isolated)
    df['community_id'] = df['community_id'].fillna(-1).astype(int)
    
    community_stats = df['community_id'].value_counts().sort_index()
    print(f"Community assignment statistics:")
    for comm_id, count in community_stats.items():
        if comm_id == -1:
            print(f"  Isolated points (no community): {count}")
        else:
            print(f"  Community {comm_id}: {count} points")
    
    return df

def save_results(pairs_df, pair_counts, df_with_communities=None, 
                 pairs_filepath='data/co_clustered_pairs.csv', 
                 counts_filepath='data/pair_co_cluster_counts.csv',
                 communities_filepath='data/data_with_communities.csv'):
    """Save the results to CSV files."""
    pairs_df.to_csv(pairs_filepath, index=False)
    pair_counts.to_csv(counts_filepath, index=False)
    
    print(f"\nResults saved:")
    print(f"  Co-clustered pairs: {pairs_filepath} ({pairs_df.shape[0]} rows)")
    print(f"  Pair counts: {counts_filepath} ({pair_counts.shape[0]} rows)")
    
    if df_with_communities is not None:
        df_with_communities.to_csv(communities_filepath, index=False)
        print(f"  Data with communities: {communities_filepath} ({df_with_communities.shape[0]} rows)")

if __name__ == "__main__":
    print("=== Community Detection Analysis ===\n")
    
    # Step 1: Load data
    print("Step 1: Loading iteration results...")
    df = load_iteration_results()
    if df is None:
        exit(1)
    
    # Step 2: Perform self join
    print("\nStep 2: Performing self join...")
    co_clustered_pairs = perform_self_join(df)
    
    # Step 3: Analyze co-clustering patterns
    print("\nStep 3: Analyzing co-clustering patterns...")
    pair_counts = analyze_co_clustering(co_clustered_pairs)
    
    # Step 4: Detect communities using greedy algorithm
    print("\nStep 4: Detecting communities...")
    community_assignments = detect_communities(pair_counts, threshold=50)
    
    # Step 5: Assign communities to original data
    if community_assignments is not None:
        print("\nStep 5: Assigning communities to data...")
        # Get unique data points (one per unique_id)
        unique_data = df.drop_duplicates(subset=['unique_id'])[['x', 'y', 'shape', 'unique_id']].copy()
        df_with_communities = assign_communities_to_data(unique_data, community_assignments)
        
        # Step 6: Save results
        print("\nStep 6: Saving results...")
        save_results(co_clustered_pairs, pair_counts, df_with_communities)
    else:
        print("\nStep 5: Saving results without communities...")
        save_results(co_clustered_pairs, pair_counts)
    
    print("\nCommunity detection analysis complete!")