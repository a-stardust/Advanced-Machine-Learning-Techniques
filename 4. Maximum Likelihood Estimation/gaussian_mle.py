import numpy as np
from data import generate_gaussian_data
from likelihood import gaussian_log_likelihood
from visualization import plot_likelihood


def main():
    data = generate_gaussian_data()

    mu_values = np.linspace(-1, 5, 200)
    log_likelihoods = [
        gaussian_log_likelihood(data, mu, sigma=np.std(data))
        for mu in mu_values
    ]

    mle_mu = mu_values[np.argmax(log_likelihoods)]

    print("Estimated Mean (MLE):", mle_mu)
    print("True Mean approx:", np.mean(data))

    plot_likelihood(
        mu_values,
        log_likelihoods,
        xlabel="Mean (μ)",
        ylabel="Log-Likelihood",
        title="Gaussian Log-Likelihood vs Mean"
    )


if __name__ == "__main__":
    main()
