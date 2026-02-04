from sklearn.datasets import load_breast_cancer, make_classification


def load_breast_cancer_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y, data


def generate_high_dimensional_data(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=random_state
    )
    return X, y
