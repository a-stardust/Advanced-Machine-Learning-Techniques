import numpy as np
from data import generate_gaussian_data
from likelihood import gaussian_log_likelihood
from visualization import plot_likelihood, plot_convergence

def gaussian_convergence():
    true_mu = 2.0
    true_sigma = 1.5
    sample_sizes = [10, 30, 50, 100, 500, 1000]
    mu_mle_values = []
    sigma_mle_values = []
    
    for n in sample_sizes:
        data = generate_gaussian_data(mu=true_mu, sigma=true_sigma, n=n)
        mu_hat = np.mean(data)
        sigma_hat = np.sqrt(np.mean((data - mu_hat) ** 2))
        
        mu_mle_values.append(mu_hat)
        sigma_mle_values.append(sigma_hat)
        
    plot_convergence(
        sample_sizes, 
        mu_mle_values, 
        true_value=true_mu, 
        xlabel="Sample Size", 
        ylabel="MLE of Mean", 
        title="Mean MLE Convergence"
    )
    
    plot_convergence(
        sample_sizes, 
        sigma_mle_values, 
        true_value=true_sigma, 
        xlabel="Sample Size", 
        ylabel="MLE of Std", 
        title="Std MLE Convergence"
    )

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

    sigma_values = np.linspace(0.5, 3.5, 100)
    ll_sigma = [gaussian_log_likelihood(data, mle_mu, s) for s in sigma_values]
    
    plot_likelihood(
        sigma_values,
        ll_sigma,
        xlabel="Standard Deviation (σ)",
        ylabel="Log-Likelihood",
        title="Gaussian Log-Likelihood vs Std"
    )

    gaussian_convergence()


if __name__ == "__main__":
    main()
