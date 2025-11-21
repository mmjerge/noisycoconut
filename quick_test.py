import sys
sys.path.append('.')
from test import run_noise_experiment, analyze_results, save_results

print("\n" + "="*70)
print("COMPLETE NOISE EXPERIMENT")
print("Per question per noise level: 2 Baseline + 1 Coconut")
print("="*70)

# Run experiment with all three methods
results = run_noise_experiment(
    num_questions=25,
    noise_scales=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0],
    max_new_tokens=512,
    model_name='/scratch/mj6ux/.cache/hf_models/gpt-oss-20b',
    noise_type='gaussian_scaled',
    noise_direction=None,
    apply_noise_to_all_passes=False
)

# Analyze and save
analyze_results(results)
save_results(results, 'results_llama_gaussian_scaled.json')

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("Results saved to: results_llama_gaussian_scaled.json")
print("="*70)