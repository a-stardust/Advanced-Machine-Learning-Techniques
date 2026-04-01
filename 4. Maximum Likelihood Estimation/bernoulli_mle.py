import numpy as np
from data import generate_bernoulli_data
from likelihood import bernoulli_log_likelihood
from visualization import plot_likelihood, plot_convergence

def bernoulli_convergence():
    true_p = 0.7
    sample_sizes = [10, 30, 50, 100, 500, 1000]
    p_mle_values = []
    
    for n in sample_sizes:
        data = generate_bernoulli_data(p=true_p, n=n)
        p_mle = np.mean(data)
        p_mle_values.append(p_mle)
        
    plot_convergence(
        sample_sizes, 
        p_mle_values, 
        true_value=true_p, 
        xlabel="Sample Size", 
        ylabel="MLE of p", 
        title="Bernoulli MLE Convergence with Increasing Data"
    )


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

    bernoulli_convergence()


if __name__ == "__main__":
    main()
