import sys
sys.path.append('.')
from test import run_noise_experiment, analyze_results, save_results

# Run on ALL questions in GSM8K test set (change num_questions to desired amount)
results = run_noise_experiment(
    num_questions=100,  # Run on 100 questions (or change to dataset size)
    noise_scales=[0.0, 1.0, 2.0, 5.0, 10.0],
    max_new_tokens=512,
    model_name='meta-llama/Llama-3.1-8B-Instruct',
    noise_type='targeted',
    noise_direction='opposite',
    apply_noise_to_all_passes=True
)

analyze_results(results)
save_results(results, 'results_full_dataset.json')
