from main import plot_altitude_time
import pandas as pd
record = pd.DataFrame({"Time":[0,1,2,3,4,5],"x":[0,1,2,3,4,5],"y":[0,1,2,3,4,5],"z":[0,1,2,3,4,5],"v_x":[0,1,2,3,4,5],"v_y":[0,1,2,3,4,5],"v_z":[0,1,2,3,4,5]})
plot_altitude_time(record)