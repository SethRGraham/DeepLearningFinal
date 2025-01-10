# DeepLearningFinal
Final project for a Deep Learning course taken in 2024. Involves combining Applied Topological methods with Graph Neural Networks to attempt hierarchical clustering.


**Description:**

This project applies Topological Data Analysis (TDA) with the Mapper algorithm and Deep Graph Infomax (DGI) Graph Neural Networks (GNNs) to detect anomalies in a news posts dataset. The Mapper algorithm transforms high-dimensional data into a graph structure where nodes are clusters of news posts, and edges connect clusters with shared elements. DGI generates node embeddings that capture both local and global graph structures, enabling unsupervised anomaly detection. From here, everything is traditional statistical analysis.

The aim of this project was to investigate if a GNN can perform meaningful global analysis on a graph constructed through the Mapper algorithm, while statistical analysis is used to analyze the clusters that exist within each node.


**Overview:**

1) Data Preparation: Encode news posts using TF-IDF to convert text into numerical features.

2) Mapper Algorithm: Construct a graph from data clusters using dimensionality reduction, cover, and clustering.

3) DGI GNN: Generate node embeddings using a GCN encoder; cluster embeddings to identify anomalies.

4) Statistical Analysis: Evaluate clusters using metrics such as density, purity, inertia, and feature-level statistics.



**Results:**

Cluster Statistics:
Analysis shows that anomalous clusters have higher average dispersion, inertia, and cohesion amongst data points, while the average purity is lower. This implies that the topology of anomalous clusters has a higher spread amongst data points, there is more diversity amongst the data, and there is a higher total distance from the cluster’s centroid to other data points.

Feature Analysis:
Anomalous clusters have average feature means that are relatively close in value, but there is a notable difference in feature standard deviation. Words in the anomalous and non-anomalous clusters have the same frequency in the news posts they appear in, but the higher standard deviation in anomalous clusters indicates a more unique vocabulary.

Vocabulary Insights:
Upon verifying the words in anomalous clusters, many with the highest standard deviation were names of states and countries, people’s names, nouns relating to religion, objects, and organizations (e.g., NASA, IBM, government). Additionally, domain extensions (.com, .edu, .org) had high standard deviations, suggesting that anomalous clusters contained significant email-related data with varied corporate, non-profit, and academic domains. Depending on the context, this could be seen as noise or signal for identifying persons of interest.


**Usage Instructions:**
1) Ensure all dependencies are installed.

2) Run Final_Project.ipynb to reproduce the results. It is preferred to run this in a Google Colab Notebook.

3) That’s it!


**Core Dependencies:**
- Python 3.x

- PyTorch Geometric

- scikit-learn

- sciPy

- matplotlib

- seaborn

- kmapper



