import numpy as np
from data import generate_bernoulli_data
from likelihood import bernoulli_log_likelihood
from visualization import plot_likelihood


def main():
    data = generate_bernoulli_data()

    p_values = np.linspace(0.01, 0.99, 200)
    log_likelihoods = [
        bernoulli_log_likelihood(data, p)
        for p in p_values
    ]

    mle_p = p_values[np.argmax(log_likelihoods)]

    print("Estimated Probability (MLE):", mle_p)
    print("Sample Mean:", np.mean(data))

    plot_likelihood(
        p_values,
        log_likelihoods,
        xlabel="Probability (p)",
        ylabel="Log-Likelihood",
        title="Bernoulli Log-Likelihood vs p"
    )


if __name__ == "__main__":
    main()
