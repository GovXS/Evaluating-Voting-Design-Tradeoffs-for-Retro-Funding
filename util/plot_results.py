def plot_aggregation_results(mean_funds, median_funds, quadratic_funds, voting_matrix, num_top_projects=10):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Calculate summary statistics for each aggregation method
    total_mean_funds = np.sum(mean_funds)
    total_median_funds = np.sum(median_funds)
    total_quadratic_funds = np.sum(quadratic_funds)

    max_mean_funds = np.max(mean_funds)
    max_median_funds = np.max(median_funds)
    max_quadratic_funds = np.max(quadratic_funds)

    min_mean_funds = np.min(mean_funds)
    min_median_funds = np.min(median_funds)
    min_quadratic_funds = np.min(quadratic_funds)

    # Display the results
    print("Summary Statistics:")
    print(f"Mean Aggregation: Total Funds: {total_mean_funds}, Max Funds: {max_mean_funds}, Min Funds: {min_mean_funds}")
    print(f"Median Aggregation: Total Funds: {total_median_funds}, Max Funds: {max_median_funds}, Min Funds: {min_median_funds}")
    print(f"Quadratic Aggregation: Total Funds: {total_quadratic_funds}, Max Funds: {max_quadratic_funds}, Min Funds: {min_quadratic_funds}")

    # Histogram Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(mean_funds, bins=50)
    plt.xlabel('Funds Allocated')
    plt.ylabel('Number of Projects')
    plt.title('Mean Aggregation')

    plt.subplot(1, 3, 2)
    plt.hist(median_funds, bins=50)
    plt.xlabel('Funds Allocated')
    plt.ylabel('Number of Projects')
    plt.title('Median Aggregation')

    plt.subplot(1, 3, 3)
    plt.hist(quadratic_funds, bins=50)
    plt.xlabel('Funds Allocated')
    plt.ylabel('Number of Projects')
    plt.title('Quadratic Aggregation')
    plt.tight_layout()
    plt.show()

    # Box Plot Visualization
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    sns.boxplot(mean_funds)
    plt.xlabel('Funds Allocated')
    plt.title('Mean Aggregation')

    plt.subplot(1, 3, 2)
    sns.boxplot(median_funds)
    plt.xlabel('Funds Allocated')
    plt.title('Median Aggregation')

    plt.subplot(1, 3, 3)
    sns.boxplot(quadratic_funds)
    plt.xlabel('Funds Allocated')
    plt.title('Quadratic Aggregation')
    plt.tight_layout()
    plt.show()

    # Density Plot Visualization
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    sns.kdeplot(mean_funds, fill=True)
    plt.xlabel('Funds Allocated')
    plt.title('Mean Aggregation')

    plt.subplot(1, 3, 2)
    sns.kdeplot(median_funds, fill=True)
    plt.xlabel('Funds Allocated')
    plt.title('Median Aggregation')

    plt.subplot(1, 3, 3)
    sns.kdeplot(quadratic_funds, fill=True)
    plt.xlabel('Funds Allocated')
    plt.title('Quadratic Aggregation')
    plt.tight_layout()
    plt.show()

    # Comparison Bar Chart Visualization
    top_mean_projects = np.argsort(mean_funds)[-num_top_projects:]
    top_median_projects = np.argsort(median_funds)[-num_top_projects:]
    top_quadratic_projects = np.argsort(quadratic_funds)[-num_top_projects:]

    top_projects_data = {
        "Project": range(num_top_projects),
        "Mean": mean_funds[top_mean_projects],
        "Median": median_funds[top_median_projects],
        "Quadratic": quadratic_funds[top_quadratic_projects]
    }

    df_top_projects = pd.DataFrame(top_projects_data)
    df_top_projects.plot(kind="bar", figsize=(18, 6))
    plt.xticks(range(num_top_projects), [f"Project {i+1}" for i in range(num_top_projects)], rotation=0)
    plt.ylabel('Funds Allocated')
    plt.title('Top 10 Projects Fund Allocation Comparison')
    plt.show()

    # Convert the voting matrix to a DataFrame for visualization
    voting_df = pd.DataFrame(voting_matrix, columns=[f'Project {i}' for i in range(1, voting_matrix.shape[1] + 1)])
    voting_df['Voter'] = [f'Voter {i}' for i in range(1, voting_matrix.shape[0] + 1)]

    # Heatmap Visualization
    plt.figure(figsize=(15, 10))
    sns.heatmap(voting_df.drop(columns=['Voter']), cmap="YlGnBu", cbar=True)
    plt.xlabel('Projects')
    plt.ylabel('Voters')
    plt.title('Heatmap of Voter Allocations')
    plt.show()

    # Histogram Visualization (For a single voter)
    voter_id = 0  # Change this to visualize different voters
    plt.figure(figsize=(10, 6))
    sns.histplot(voting_df.iloc[voter_id, :-1], bins=30, kde=True)
    plt.xlabel('Votes Allocated')
    plt.ylabel('Frequency')
    plt.title(f'Vote Distribution for Voter {voter_id + 1}')
    plt.show()

    # Pairplot Visualization (For a subset of projects)
    num_projects_subset = 5
    subset_projects = voting_df.drop(columns=['Voter']).iloc[:, :num_projects_subset]
    sns.pairplot(subset_projects)
    plt.suptitle('Pairplot of Votes for Subset of Projects', y=1.02)
    plt.show()
