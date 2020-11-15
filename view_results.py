import main
import pandas as pd
data = pd.read_csv("results/results_11_15_2020_12_42_20.csv")
main.plot_orientation(data)