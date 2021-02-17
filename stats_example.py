import campyros.statistical as stats

model = stats.StatisticalModel("stats_settings.json")

model.run_model(test_mode=False, debug=True, num_cpus=3)
