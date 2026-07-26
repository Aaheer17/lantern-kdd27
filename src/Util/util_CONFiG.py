import matplotlib.pyplot as plt
import numpy as np
import torch  


# ============================================================================
# Gradient Analysis Helper Functions
# ============================================================================

def compute_gradient_stats(grad_vector):
    """Compute statistics for a gradient vector"""
    return {
        'norm': torch.norm(grad_vector).item(),
        'mean': grad_vector.mean().item(),
        'std': grad_vector.std().item(),
        'max': grad_vector.max().item(),
        'min': grad_vector.min().item()
    }


def cosine_similarity(grad1, grad2):
    """Compute cosine similarity between two gradient vectors"""
    dot_product = torch.dot(grad1, grad2)
    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim.item()


def gradient_conflict_score(grad1, grad2):
    """
    Compute conflict score: 
    - Positive cosine = aligned gradients (no conflict)
    - Negative cosine = opposing gradients (conflict)
    - Close to 0 = orthogonal (independent)
    """
    cos_sim = cosine_similarity(grad1, grad2)
    return {
        'cosine_similarity': cos_sim,
        'conflict_magnitude': -cos_sim if cos_sim < 0 else 0.0,
        'alignment': 'aligned' if cos_sim > 0.1 else ('conflict' if cos_sim < -0.1 else 'orthogonal')
    }


# ============================================================================
# Analysis and Visualization Functions
# ============================================================================

def analyze_gradient_conflicts(gradient_history):
    """Provide statistical summary of gradient conflicts"""
    cosine_sims = np.array(gradient_history['cosine_similarity'])
    
    analysis = {
        'mean_cosine': np.mean(cosine_sims),
        'std_cosine': np.std(cosine_sims),
        'conflict_percentage': (cosine_sims < -0.1).sum() / len(cosine_sims) * 100,
        'aligned_percentage': (cosine_sims > 0.1).sum() / len(cosine_sims) * 100,
        'orthogonal_percentage': (np.abs(cosine_sims) <= 0.1).sum() / len(cosine_sims) * 100,
        'min_cosine': np.min(cosine_sims),
        'max_cosine': np.max(cosine_sims)
    }
    
    print("\n" + "="*50)
    print("GRADIENT CONFLICT ANALYSIS")
    print("="*50)
    print(f"Mean Cosine Similarity: {analysis['mean_cosine']:.4f} ± {analysis['std_cosine']:.4f}")
    print(f"Range: [{analysis['min_cosine']:.4f}, {analysis['max_cosine']:.4f}]")
    print(f"\nGradient Relationship Breakdown:")
    print(f"  Conflict (cos < -0.1):     {analysis['conflict_percentage']:.1f}%")
    print(f"  Orthogonal (|cos| ≤ 0.1):  {analysis['orthogonal_percentage']:.1f}%")
    print(f"  Aligned (cos > 0.1):       {analysis['aligned_percentage']:.1f}%")
    print("="*50 + "\n")
    
    return analysis


def plot_gradient_analysis(gradient_history, save_path='gradient_analysis.png'):
    """Create comprehensive gradient analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    steps = range(len(gradient_history['grad_fm_norm']))
    
    # Plot 1: Gradient Norms
    ax1 = axes[0, 0]
    ax1.plot(steps, gradient_history['grad_fm_norm'], label='FM Loss', alpha=0.7)
    ax1.plot(steps, gradient_history['grad_energy_norm'], label='Energy Loss', alpha=0.7)
    if 'grad_config_norm' in gradient_history:
        ax1.plot(steps, gradient_history['grad_config_norm'], label='ConFIG Combined', 
                 alpha=0.7, linewidth=2, linestyle='--')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Magnitudes Over Training')
    ax1.legend()
    ax1.set_yscale('log')  # Log scale often helpful for gradient norms
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cosine Similarity
    ax2 = axes[0, 1]
    ax2.plot(steps, gradient_history['cosine_similarity'], color='purple', alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Orthogonal')
    ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Moderate Alignment')
    ax2.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Conflict')
    ax2.fill_between(steps, -1, 0, alpha=0.1, color='red', label='Conflict Region')
    ax2.fill_between(steps, 0, 1, alpha=0.1, color='green', label='Aligned Region')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Gradient Alignment (FM vs Energy)')
    ax2.set_ylim([-1.1, 1.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Values
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(steps, gradient_history['loss_fm'], label='FM Loss', color='blue', alpha=0.7)
    l2 = ax3_twin.plot(steps, gradient_history['loss_energy'], label='Energy Loss', 
                       color='red', alpha=0.7)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('FM Loss', color='blue')
    ax3_twin.set_ylabel('Energy Loss', color='red')
    ax3.set_title('Loss Components')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Conflict Magnitude and ConFIG Alignment
    ax4 = axes[1, 1]
    if 'conflict_magnitude' in gradient_history:
        ax4.plot(steps, gradient_history['conflict_magnitude'], 
                label='Conflict Magnitude', color='red', alpha=0.7)
    if 'config_vs_fm_cos' in gradient_history and 'config_vs_energy_cos' in gradient_history:
        ax4.plot(steps, gradient_history['config_vs_fm_cos'], 
                label='ConFIG vs FM', alpha=0.7, linestyle='--')
        ax4.plot(steps, gradient_history['config_vs_energy_cos'], 
                label='ConFIG vs Energy', alpha=0.7, linestyle='--')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Similarity / Conflict')
    ax4.set_title('Gradient Conflict and ConFIG Alignment')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gradient analysis saved to {save_path}")