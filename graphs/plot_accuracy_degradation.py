"""
Accuracy Degradation vs Noise Scale - Llama-3.1-8B
Reads from combined experimental results JSON file and fits sigmoid curves.

Usage:
    python plot_accuracy_degradation.py [path_to_json]
    
If no path is provided, defaults to 'results_llama_combined.json' in the same directory.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def standard_sigmoid(x, L, k, x0, b):
    """Standard sigmoid: L / (1 + exp(-k*(x-x0))) + b"""
    return L / (1 + np.exp(-k * (x - x0))) + b


def log_sigmoid(x, L, k, x0, b):
    """Log sigmoid: operates on log(x) internally"""
    x_safe = np.maximum(x, 1e-10)
    return L / (1 + np.exp(-k * (np.log(x_safe) - x0))) + b


def fit_sigmoid(x_data, y_data, sigmoid_func, p0=None):
    """Fit sigmoid function and return parameters and R²"""
    try:
        if p0 is None:
            p0 = [-0.3, 0.5, 5, 0.5]
        popt, _ = curve_fit(sigmoid_func, x_data, y_data, p0=p0, maxfev=10000)
        y_pred = sigmoid_func(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, r_squared
    except Exception as e:
        print(f"Fitting error: {e}")
        return None, 0

def load_and_compute_accuracy(json_path):
    """Load JSON file and compute accuracy at each noise level."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    noise_scales = data['experiment_info']['noise_scales']
    model_name = data['experiment_info'].get('model', 'Unknown Model')
    
    correct = {ns: 0 for ns in noise_scales}
    total = {ns: 0 for ns in noise_scales}
    
    for question in data['questions']:
        for test in question['noise_tests']:
            ns = test['noise_scale']
            total[ns] += 1
            if test['is_correct']:
                correct[ns] += 1
    
    accuracy = {ns: correct[ns] / total[ns] for ns in noise_scales}
    
    noise_arr = np.array(noise_scales)
    acc_arr = np.array([accuracy[ns] for ns in noise_scales])
    
    return noise_arr, acc_arr, model_name, total[noise_scales[0]]

def create_plot(noise_scale, accuracy, model_name, n_questions, output_prefix='accuracy_degradation'):
    """Create the accuracy degradation plot with sigmoid fits."""
    
    noise_scale_offset = noise_scale.copy()
    if noise_scale[0] == 0.0:
        noise_scale_offset[0] = 0.1
    
    popt_std, r2_std = fit_sigmoid(noise_scale, accuracy, standard_sigmoid, p0=[-0.3, 0.5, 3, 0.5])
    popt_log, r2_log = fit_sigmoid(noise_scale_offset, accuracy, log_sigmoid, p0=[-0.4, 1.5, 0.5, 0.4])
    
    print(f"Model: {model_name}")
    print(f"Questions per noise level: {n_questions}")
    print(f"Standard Sigmoid R² = {r2_std:.3f}")
    print(f"Log Sigmoid R² = {r2_log:.3f}")
    
    x_smooth = np.linspace(0.1, 55, 500)
    y_std = standard_sigmoid(x_smooth, *popt_std) if popt_std is not None else None
    y_log = log_sigmoid(x_smooth, *popt_log) if popt_log is not None else None
    
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(10, 4))
    
    data_color = '#9467bd'  # Purple for Llama
    grey_dashed = '#404040'  # Dark grey for standard sigmoid
    grey_solid = '#707070'   # Medium grey for log sigmoid
    
    display_name = model_name.split('/')[-1] if '/' in model_name else model_name
    
    ax_linear.scatter(noise_scale, accuracy, c=data_color, s=70, zorder=5,
                      label=display_name, edgecolors='white', linewidth=0.5)
    
    if y_std is not None:
        ax_linear.plot(x_smooth, y_std, color=grey_dashed, linestyle='--', linewidth=1.5,
                       label=f'Standard Sigmoid (R²={r2_std:.3f})')
    if y_log is not None:
        ax_linear.plot(x_smooth, y_log, color=grey_solid, linestyle='-', linewidth=1.5,
                       label=f'Log Sigmoid (R²={r2_log:.3f})')
    
    ax_linear.axvline(x=1, color='grey', linestyle=':', alpha=0.5, linewidth=1)
    
    ax_linear.set_xlim(-2, 55)
    ax_linear.set_ylim(0.3, 0.80)
    ax_linear.set_xlabel(r'Noise Scale ($\sigma$)')
    ax_linear.set_ylabel('Accuracy')
    ax_linear.grid(True, alpha=0.3)
    ax_linear.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    ax_log.scatter(noise_scale_offset, accuracy, c=data_color, s=70, zorder=5,
                   label=display_name, edgecolors='white', linewidth=0.5)
    
    if y_std is not None:
        ax_log.plot(x_smooth, y_std, color=grey_dashed, linestyle='--', linewidth=1.5,
                    label=f'Standard Sigmoid (R²={r2_std:.3f})')
    if y_log is not None:
        ax_log.plot(x_smooth, y_log, color=grey_solid, linestyle='-', linewidth=1.5,
                    label=f'Log Sigmoid (R²={r2_log:.3f})')
    
    ax_log.axvline(x=1, color='grey', linestyle=':', alpha=0.5, linewidth=1)
    
    ax_log.set_xscale('log')
    ax_log.set_xlim(0.08, 100)
    ax_log.set_ylim(0.3, 0.80)
    ax_log.set_xlabel(r'Noise Scale ($\sigma$, log)')
    ax_log.set_ylabel('Accuracy')
    ax_log.grid(True, alpha=0.3, which='both')
    ax_log.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\nFigures saved: {output_prefix}.png, {output_prefix}.pdf")
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot accuracy degradation from experimental results')
    parser.add_argument('json_path', nargs='?', default='results_llama_combined.json',
                        help='Path to the combined JSON results file')
    parser.add_argument('-o', '--output', default='accuracy_degradation',
                        help='Output filename prefix (default: accuracy_degradation)')
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        exit(1)
    
    print(f"Loading data from: {json_path}")
    noise_scale, accuracy, model_name, n_questions = load_and_compute_accuracy(json_path)
    
    print("\nAccuracy by noise level:")
    print("-" * 30)
    for ns, acc in zip(noise_scale, accuracy):
        print(f"  σ = {ns:<5} : {acc:.4f}")
    print("-" * 30)
    
    fig = create_plot(noise_scale, accuracy, model_name, n_questions, args.output)
    plt.show()
