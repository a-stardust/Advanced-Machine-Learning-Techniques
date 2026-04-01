from data import generate_gmm_data
from gmm_scratch import fit_gmm_em
from visualization import plot_gmm_scratch

def main():
    print("Generating toy data for GMM...")
    X = generate_gmm_data()
    
    print("Fitting GMM from scratch using EM algorithm...")
    responsibilities, means, covariances, weights = fit_gmm_em(X, k=2, iterations=2)
    
    print("Learned means:\n", means)
    
    print("Plotting results...")
    plot_gmm_scratch(X, responsibilities, means)

if __name__ == "__main__":
    main()
