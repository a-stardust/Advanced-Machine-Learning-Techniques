from data import generate_linear_data, generate_xor_data
from visual import plot_2d_datat

def main():
    X, y = generate_xor_data()
    plot_2d_datat(X, y, title="Non Linearly separable Data")

if __name__ == "__main__":
    main()