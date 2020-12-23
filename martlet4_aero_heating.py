import trajectory, trajectory.post, trajectory.aero, csv
import numpy as np
from martlet4 import martlet4

trajectory_data = trajectory.from_json("output.json")

tangent_ogive = trajectory.post.TangentOgive(xprime = 73.7e-2, yprime = (19.7e-2)/2)
analysis = trajectory.post.HeatTransfer(tangent_ogive, trajectory_data, martlet4)
analysis.step(print_style="metric")

#analysis.run(iterations = 150, starting_index = 120, print_style="minimal")
#analysis.to_json("martlet4_heating.json")

analysis.from_json("martlet4_heating.json")

analysis.plot_station(station_number = 9, imax = 300)
#analysis.plot_heat_transfer(automatic_rescaling=True)
#analysis.plot_fluid_properties(automatic_rescaling=True)