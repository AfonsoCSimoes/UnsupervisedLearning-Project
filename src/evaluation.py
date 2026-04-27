import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.modeling import ikmeans_initialize
from sklearn.mixture import GaussianMixture

def evaluate_models(X_processed, rep_id, k_range=range(3, 9), seeds=[42, 123, 456, 789, 999]):
    """
    Execute the tests of stability and generate the experiments.csv
    """
    logs = []

    print("\nStarting Standard K-Means evaluation")
    for k in k_range:
        for seed in seeds:
            print(f"Training K-Means: K={k:02d} | Seed={seed}")

            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(X_processed)

            logs.append({
                'representation_id': rep_id,
                'method': 'k-means',
                'k': k,
                'seed': seed,
                'silhouette': silhouette_score(X_processed, labels, sample_size=30000),
                'calinski_harabasz': calinski_harabasz_score(X_processed, labels),
                'davies_bouldin': davies_bouldin_score(X_processed, labels)
            })

    print("\nInitializing iK-MEANS")
    try:
        _, init_centroids = ikmeans_initialize(
            X=X_processed,
            min_cluster_size=100,
            use_unit_ranges=True
        )

        k_ik = len(init_centroids)
        print(f"Success! iK-Means determined K={k_ik} clusters.")

        if k_ik >= 2:
            kmeans_ik = KMeans(n_clusters=k_ik, init=init_centroids, n_init=1)
            labels_ik = kmeans_ik.fit_predict(X_processed)

            logs.append({
                'representation_id': rep_id,
                'method': 'ik-means',
                'k': k_ik,
                'seed': 'deterministic',
                'silhouette': silhouette_score(X_processed, labels_ik, sample_size=30000),
                'calinski_harabasz': calinski_harabasz_score(X_processed, labels_ik),
                'davies_bouldin': davies_bouldin_score(X_processed, labels_ik)
            })
    
    except Exception as e:
        print(f"Aviso: iK-Means encontrou um erro: {e}")
        
    print("\nStarting Gaussian Mixture Model (GMM) evaluation")
    aic_bic_logs = []
    
    for k in k_range:
        for seed in seeds:
            print(f"Training GMM: K={k:02d} | Seed={seed}")

            gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type='full', n_init=1)
            labels_gmm = gmm.fit_predict(X_processed)

            logs.append({
                'representation_id': rep_id,
                'method': 'gmm',
                'k': k,
                'seed': seed,
                'silhouette': silhouette_score(X_processed, labels_gmm, sample_size=30000, random_state=seed),
                'calinski_harabasz': calinski_harabasz_score(X_processed, labels_gmm),
                'davies_bouldin': davies_bouldin_score(X_processed, labels_gmm)
            })

            if seed == 42:
                aic_bic_logs.append({
                    'k': k,
                    'aic': gmm.aic(X_processed),
                    'bic': gmm.bic(X_processed)
                })

    df_diagnostics = pd.DataFrame(aic_bic_logs)
    safe_rep_id = rep_id.replace('-', '_')
    df_diagnostics.to_csv(f'tables/gmm_diagnostics_{safe_rep_id}.csv', index=False)

    df_results = pd.DataFrame(logs)
    print("\nEvaluation successed!")

    return df_results
