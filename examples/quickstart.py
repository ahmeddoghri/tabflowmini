from tabflowmini import evaluate_synth, fit_synth, make_table, sample_synth

real = make_table(n=120)
synth = sample_synth(fit_synth(real), 120)
print(f"mean_ks={evaluate_synth(real, synth)['mean_ks']:.3f}")
