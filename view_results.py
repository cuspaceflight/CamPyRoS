import main
import pandas as pd
data = pd.read_csv("results/stat_model_20201115/3.csv")

main.plot_position(data)