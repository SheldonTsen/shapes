
# Prompt 1
Great. This is the protocol I want you to follow:

0. Do this in a new file called iteration_k_means.py
1. Given the multishapes.csv file
2. Set k to be very large, like 100. And run k-means with that k for 20 times. 
3. For each iteration, store the results of the n-th iteration as an addtional column, as well as the cluster-id for that iteration. 
    Meaning, I expect a dataframe with: ` x, y, shape, iteration_id, cluster_id`
4. Store this dataframe, which should contain number of rows multiplied by 20 iterations to disk. Store it as CSV. 

Take this step by step. 

Manually edits:
* Update data path
* update script name

# Prompt 2
Next:

1. Create a new script called 01_plot_iterations.py
2. Read the data/iteration_k_means_results.csv
3. Plot several of these iterations, let's say 3 iterations, where x = x, y=y, but colour it by the cluster-id. 
4. Save these plots to the plots/ folder. 

# Prompt 3
Next:

1. Create a second script called 02_apply_community_detection.py
2. In this script, read the output that we just created (iteration_kmeans_results.csv) 
3. Perform a self join on iteration_id and cluster_id. 


# Prompt 4

Now - write a script to visualize the communities in the data_with_communities.csv file. 
Call this 03_visualize_communities.py
I want the plots saved to plots/ folder as usual. 
