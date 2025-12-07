# Clustering Algorithms

This repository contains a collection of Google Colab notebooks demonstrating a wide range of clustering algorithms, evaluation metrics, visualization techniques, and modern embedding-based clustering methods.  
The purpose of these experiments is to understand classical clustering approaches, explore modern embedding techniques, and apply clustering to structured, text, image, audio, and time-series datasets.

---

### a) K-Means Clustering (from scratch)
- Implement K-Means manually in Python.
- Demonstrate centroid initialization, iterative updates, convergence, and visualizations.
- Include clustering evaluation metrics such as SSE, silhouette score, etc.

---

### b) Hierarchical Clustering (library-based)
- Use SciPy or Scikit-learn.
- Plot dendrograms, explore different linkage types, and visualize clusters.
- Include evaluation metrics.

---

### c) Gaussian Mixture Models (GMM)
- Use `sklearn.mixture.GaussianMixture`.
- Visualize components, probability contours, and cluster assignments.
- Evaluate using AIC, BIC, and log-likelihood.

---

### d) DBSCAN Clustering (PyCaret)
- Use PyCaret’s clustering module.
- Demonstrate cluster formation, noise identification, and hyperparameter effects.

---

### e) Anomaly Detection using PyOD
- Use PyOD models such as IsolationForest, LOF, AutoEncoder, etc.
- Apply on univariate or multivariate datasets.
- Visualize anomalies and compare with clustering-based anomaly detection.

---

### f) Time-Series Clustering using Pretrained Models
- Use pretrained time-series encoders such as TS2Vec, TimeGPT, TST, etc.
- Extract embeddings and cluster them using methods such as K-Means, DBSCAN, or GMM.
- Visualize cluster structures using PCA, TSNE, or UMAP.

---

### g) Document Clustering using LLM Embeddings
- Use state-of-the-art text embeddings such as Sentence Transformers or OpenAI embeddings.
- Cluster embedded vectors and visualize cluster groups.
- Include cluster quality metrics and example document groups.

---

### h) Image Clustering using ImageBind Embeddings
- Use Meta’s ImageBind to extract multimodal image embeddings.
- Cluster embeddings using algorithms such as K-Means or DBSCAN.
- Visualize sample images within each cluster.

---

### i) Audio Clustering using ImageBind or Other Embedding Models
- Extract audio embeddings using ImageBind, VGGish, Wav2Vec, or other pretrained models.
- Cluster and visualize audio groups using waveform or spectrogram plots.


End of README.
