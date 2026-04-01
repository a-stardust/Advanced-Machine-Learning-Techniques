import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load marketing campaign data and preprocess for clustering."""
    df = pd.read_csv(path, sep=';')

    # drop identifiers and date
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    if 'Dt_Customer' in df.columns:
        df = df.drop(columns=['Dt_Customer'])

    # numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # drop columns with too many missing values or low variance
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # categorical columns: one-hot encode
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_mat = enc.fit_transform(df[cat_cols])
        cat_df = pd.DataFrame(cat_mat, columns=enc.get_feature_names_out(cat_cols), index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), cat_df], axis=1)

    # scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled


def fit_gmm(df_scaled: pd.DataFrame, n_components: int = 3) -> GaussianMixture:
    """Fit a Gaussian mixture model to the scaled dataframe."""
    gm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gm.fit(df_scaled)
    return gm


def add_cluster_labels(df: pd.DataFrame, gm: GaussianMixture) -> pd.DataFrame:
    """Attach cluster labels to dataframe."""
    labels = gm.predict(df)
    out = df.copy()
    out['cluster'] = labels
    return out


def plot_clusters(df_scaled: pd.DataFrame, labels: np.ndarray, n_components: int = 3):
    """Visualize clusters using the first two PCA components."""
    pca = PCA(n_components=2)
    xy = pca.fit_transform(df_scaled)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(xy[:,0], xy[:,1], c=labels, cmap='tab10', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'GMM clustering (k={n_components}) projected to 2D PCA')
    plt.legend(*scatter.legend_elements(), title="clusters")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # load dataset
    csv_path = r"c:\Users\LENOVO\Desktop\Sem2\Advanced Machine Learning Techniques\5. Gaussian Mixture Models\Data\marketing_campaign.csv"
    print('loading and preprocessing data...')
    data_scaled = load_and_preprocess(csv_path)
    print(f'data shape after preprocessing: {data_scaled.shape}')

    # fit model
    n_clusters = 4
    print(f'fitting GaussianMixture with {n_clusters} components...')
    model = fit_gmm(data_scaled, n_components=n_clusters)

    # assign labels
    labels = model.predict(data_scaled)
    print('cluster counts:')
    print(pd.Series(labels).value_counts())

    # plot
    plot_clusters(data_scaled, labels, n_components=n_clusters)

    # save labeled dataset
    labeled = add_cluster_labels(data_scaled, model)
    labeled.to_csv('marketing_campaign_with_clusters.csv', index=False)
    print('saved labeled dataset to marketing_campaign_with_clusters.csv')
