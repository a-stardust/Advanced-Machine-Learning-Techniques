from data import generate_linear_data
from visual import plot_2d_datat

def main():
    X, y = generate_linear_data()
    plot_2d_datat(X, y, title="Linearly separable Data")

if __name__ == "__main__":
    main()