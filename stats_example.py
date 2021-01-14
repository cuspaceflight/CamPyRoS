import trajectory.statistical as stats 
import time
t_start=time.time()
model = stats.StatisticalModel("stats_settings.json")

model.run_model()
print(time.time()-t_start)
