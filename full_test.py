import sys
sys.path.append('.')
from test import run_noise_experiment, analyze_results, save_results

# Run on ALL questions in GSM8K test set (change num_questions to desired amount)
results = run_noise_experiment(
    num_questions=100,  # Run on 100 questions (or change to dataset size)
    noise_scales=[0.0, 1.0, 2.0, 5.0, 10.0],
    max_new_tokens=512,
    model_name='Qwen/Qwen2.5-7B-Instruct',
    noise_type='gaussian_scaled',
    noise_direction=None,
    apply_noise_to_all_passes=True
)

analyze_results(results)
save_results(results, 'results_full_dataset.json')
