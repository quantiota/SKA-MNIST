"""
Pure SKA Implementation Based ONLY on the Research Papers
========================================================

Implements the SKA framework EXACTLY as described in the research papers.
- H^(l) = -1/ln(2) * Σ_k z^(l)_k · ΔD^(l)_k
- Forward-only learning without backpropagation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import pickle
import json
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

class PureSKAGenerator(nn.Module):
    """Pure SKA implementation following research papers exactly"""
    def __init__(self, input_size=784, layer_size=10, K=50, eta=0.01):
        super(PureSKAGenerator, self).__init__()
        self.input_size = input_size
        self.layer_size = layer_size
        self.K = K
        self.eta = eta

        self.weight = nn.Parameter(torch.randn(input_size, layer_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(layer_size))
        self.reset_for_new_digit()

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        z = torch.mm(x_flat, self.weight) + self.bias
        D = torch.sigmoid(z)
        self.z_history.append(z.clone())
        self.D_history.append(D.clone())
        z_norm = torch.norm(z, p='fro').item()
        self.frobenius_evolution.append(z_norm)
        return D

    def compute_ska_entropy(self, k):
        if k == 0: return 0.0
        z_k = self.z_history[k]
        D_k = self.D_history[k]
        D_k_minus_1 = self.D_history[k-1]
        delta_D_k = D_k - D_k_minus_1
        self.delta_D_history.append(delta_D_k.clone())
        H_lk = (-1 / np.log(2)) * (z_k * delta_D_k)
        H_layer = torch.sum(H_lk).item()
        self.H_evolution.append(H_layer)
        dot_product = torch.sum(z_k * delta_D_k)
        z_norm = torch.norm(z_k)
        delta_D_norm = torch.norm(delta_D_k)
        if z_norm > 0 and delta_D_norm > 0:
            cos_theta = (dot_product / (z_norm * delta_D_norm)).item()
        else:
            cos_theta = 0.0
        self.cos_theta_evolution.append(cos_theta)
        return H_layer

    def ska_weight_update(self, x, k):
        if k == 0 or len(self.delta_D_history) == 0: return
        x_flat = x.view(x.shape[0], -1)
        z_k = self.z_history[k]
        D_k = self.D_history[k]
        delta_D_k = self.delta_D_history[-1]
        D_prime = D_k * (1 - D_k)
        grad_z_H = (-1 / np.log(2)) * (z_k * D_prime + delta_D_k)
        dW = torch.matmul(x_flat.t(), grad_z_H) / x_flat.shape[0]
        db = grad_z_H.mean(dim=0)
        self.weight.data = self.weight.data - self.eta * dW
        self.bias.data = self.bias.data - self.eta * db

    def reset_for_new_digit(self):
        self.z_history = []
        self.D_history = []
        self.delta_D_history = []
        self.H_evolution = []
        self.cos_theta_evolution = []
        self.frobenius_evolution = []

    def generate_ska_signature(self, data, digit_label, verbose=False):
        self.reset_for_new_digit()
        for k in range(self.K):
            self.forward(data)
            self.compute_ska_entropy(k)
            self.ska_weight_update(data, k)
        return self.create_paper_signature(digit_label)

    def create_paper_signature(self, digit_label):
        H_seq = self.H_evolution
        if H_seq:
            min_H_idx = np.argmin(H_seq)
            min_H_value = min(H_seq)
            initial_H = np.mean(H_seq[:5]) if len(H_seq) >= 5 else H_seq[0]
            final_H = np.mean(H_seq[-5:]) if len(H_seq) >= 5 else H_seq[-1]
            H_variance = np.var(H_seq)
        else:
            min_H_idx = 0
            min_H_value = 0
            initial_H = 0
            final_H = 0
            H_variance = 0
        has_u_shape = (min_H_idx > 5 and min_H_idx < len(H_seq) - 5) if len(H_seq) > 10 else False
        final_output = self.D_history[-1].mean(dim=0).detach().cpu().numpy() if self.D_history else np.zeros(self.layer_size)
        signature = {
            'digit_label': digit_label,
            'ska_version': 'papers_exact',
            'parameters': {
                'K': self.K,
                'eta': self.eta,
                'layer_size': self.layer_size
            },
            'entropy_evolution': H_seq,
            'cosine_evolution': self.cos_theta_evolution,
            'frobenius_evolution': self.frobenius_evolution,
            'min_entropy_value': min_H_value,
            'min_entropy_step': min_H_idx + 1,
            'initial_entropy': initial_H,
            'final_entropy': final_H,
            'entropy_variance': H_variance,
            'has_u_shape': has_u_shape,
            'entropy_reduction': initial_H - min_H_value,
            'entropy_recovery': final_H - min_H_value,
            'final_output_pattern': final_output.tolist(),
            'signature_summary': {
                'depth': abs(min_H_value),
                'timing': min_H_idx + 1,
                'recovery': final_H - min_H_value,
                'stability': 1 / (H_variance + 1e-6)
            }
        }
        return signature

class SKADatasetGenerator:
    """Generate MNIST entropy signatures using pure SKA"""
    def __init__(self, samples_per_digit=100):
        self.samples_per_digit = samples_per_digit
        self.signatures = {}

    def generate_mnist_signatures(self, K=50, eta=0.01, save_path="ska_signatures.pkl"):
        print("🔬 GENERATING SKA SIGNATURES FROM RESEARCH PAPERS")
        print("=" * 60)
        print(f"Parameters: K={K}, η={eta}")
        print(f"Formula: H^(l) = -1/ln(2) * Σ_k z^(l)_k · ΔD^(l)_k")
        print("=" * 60)
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        signatures = {}
        for digit in range(10):
            print(f"Processing digit {digit}")
            digit_indices = [i for i, (_, label) in enumerate(mnist_data) if label == digit][:self.samples_per_digit]
            signatures[digit] = []
            for img_idx in digit_indices:
                img = mnist_data[img_idx][0].unsqueeze(0)
                ska_gen = PureSKAGenerator(K=K, eta=eta)
                sig = ska_gen.generate_ska_signature(img, digit, verbose=False)
                signatures[digit].append(sig)
            print(f"  ✓ Stored {len(signatures[digit])} single-image signatures for digit {digit}")

            # Print simple stats for this digit
            if len(signatures[digit]) > 0:
                summaries = [s['signature_summary'] for s in signatures[digit]]
                mean_depth = np.mean([s['depth'] for s in summaries])
                mean_timing = np.mean([s['timing'] for s in summaries])
                mean_recovery = np.mean([s['recovery'] for s in summaries])
                print(f"  ✓ Mean Entropy Depth: {mean_depth:.2f}")
                print(f"  ✓ Mean Minimum at Step: {mean_timing:.2f}")
                print(f"  ✓ Mean Recovery: {mean_recovery:.2f}")

        # Save signatures
        with open(save_path, 'wb') as f:
            pickle.dump(signatures, f)
        def to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_json_safe(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj
        json_signatures = {str(k): to_json_safe(v) for k, v in signatures.items()}
        with open(save_path.replace('.pkl', '.json'), 'w') as f:
            json.dump(json_signatures, f, indent=2)
        print(f"\n💾 Signatures saved:")
        print(f"  • {save_path}")
        print(f"  • {save_path.replace('.pkl', '.json')}")
        self.signatures = signatures
        return signatures

def visualize_ska_signatures(signatures):
    """Visualize SKA signatures (averaged per digit)"""
    print("\n🎨 Creating SKA signature visualizations...")
    plt.figure(figsize=(15, 10))

    # 1. Entropy Evolution
    plt.subplot(2, 3, 1)
    for digit, sig_list in signatures.items():
        H_seqs = [sig['entropy_evolution'] for sig in sig_list if len(sig['entropy_evolution']) > 0]
        if H_seqs:
            H_seqs = np.array([np.array(h) for h in H_seqs])
            avg_H_seq = np.mean(H_seqs, axis=0)
            plt.plot(range(1, len(avg_H_seq)+1), avg_H_seq, label=f'Digit {digit}', linewidth=2)
    plt.title('SKA Entropy Evolution (Mean)')
    plt.xlabel('Step k')
    plt.ylabel('H^(l)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Depth vs Timing
    plt.subplot(2, 3, 2)
    for digit, sig_list in signatures.items():
        depths = [sig['signature_summary']['depth'] for sig in sig_list]
        timings = [sig['signature_summary']['timing'] for sig in sig_list]
        if depths and timings:
            plt.scatter(np.mean(timings), np.mean(depths), s=100, label=f'Digit {digit}')
            plt.annotate(str(digit), (np.mean(timings), np.mean(depths)), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Minimum Step (mean)')
    plt.ylabel('Entropy Depth (mean)')
    plt.title('SKA Signature Characteristics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Recovery Analysis
    plt.subplot(2, 3, 3)
    for digit, sig_list in signatures.items():
        recoveries = [sig['signature_summary']['recovery'] for sig in sig_list]
        stabilities = [sig['signature_summary']['stability'] for sig in sig_list]
        if recoveries and stabilities:
            plt.scatter(np.mean(recoveries), np.mean(stabilities), s=100, label=f'Digit {digit}')
            plt.annotate(str(digit), (np.mean(recoveries), np.mean(stabilities)), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Entropy Recovery (mean)')
    plt.ylabel('Stability (mean)')
    plt.title('Recovery vs Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Cosine Evolution
    plt.subplot(2, 3, 4)
    for digit, sig_list in signatures.items():
        cos_seqs = [sig['cosine_evolution'] for sig in sig_list if len(sig['cosine_evolution']) > 0]
        if cos_seqs:
            cos_seqs = np.array([np.array(c) for c in cos_seqs])
            avg_cos_seq = np.mean(cos_seqs, axis=0)
            plt.plot(range(1, len(avg_cos_seq)+1), avg_cos_seq, label=f'Digit {digit}', alpha=0.7)
    plt.title('Cosine Alignment Evolution (Mean)')
    plt.xlabel('Step k')
    plt.ylabel('cos(θ)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Frobenius Norm Evolution
    plt.subplot(2, 3, 5)
    for digit, sig_list in signatures.items():
        frob_seqs = [sig['frobenius_evolution'] for sig in sig_list if len(sig['frobenius_evolution']) > 0]
        if frob_seqs:
            frob_seqs = np.array([np.array(f) for f in frob_seqs])
            avg_frob_seq = np.mean(frob_seqs, axis=0)
            plt.plot(range(1, len(avg_frob_seq)+1), avg_frob_seq, label=f'Digit {digit}', alpha=0.7)
    plt.title('Knowledge Magnitude ||z||_F (Mean)')
    plt.xlabel('Step k')
    plt.ylabel('||z||_F')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Summary Heatmap
    plt.subplot(2, 3, 6)
    summary_data = []
    for digit in range(10):
        sig_list = signatures[digit]
        if len(sig_list) == 0:
            summary_data.append([0, 0, 0, 0])
        else:
            summaries = [sig['signature_summary'] for sig in sig_list]
            mean_depth = np.mean([s['depth'] for s in summaries])
            mean_timing = np.mean([s['timing'] for s in summaries])
            mean_recovery = np.mean([s['recovery'] for s in summaries])
            mean_stability = np.mean([s['stability'] for s in summaries])
            summary_data.append([mean_depth, mean_timing, mean_recovery, mean_stability])
    summary_data = np.array(summary_data)
    for col in range(summary_data.shape[1]):
        col_data = summary_data[:, col]
        if col_data.max() != col_data.min():
            summary_data[:, col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
    sns.heatmap(summary_data,
                xticklabels=['Depth', 'Timing', 'Recovery', 'Stability'],
                yticklabels=[f'Digit {i}' for i in range(10)],
                cmap='viridis',
                annot=True,
                fmt='.2f')
    plt.title('SKA Signature Summary (Mean)')
    plt.tight_layout()
    plt.savefig('ska_signatures_from_papers.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Visualization saved as 'ska_signatures_from_papers.png'")

def run_pure_ska_analysis():
    print("🔬 PURE SKA ANALYSIS FROM RESEARCH PAPERS")
    print("=" * 50)
    generator = SKADatasetGenerator(samples_per_digit=100)
    signatures = generator.generate_mnist_signatures(K=50, eta=0.01)
    visualize_ska_signatures(signatures)
    print("\n📊 SKA SIGNATURE SUMMARY (MEAN PER DIGIT):")
    for digit, sig_list in signatures.items():
        if len(sig_list) == 0: continue
        summaries = [sig['signature_summary'] for sig in sig_list]
        mean_depth = np.mean([s['depth'] for s in summaries])
        mean_timing = np.mean([s['timing'] for s in summaries])
        mean_recovery = np.mean([s['recovery'] for s in summaries])
        mean_stability = np.mean([s['stability'] for s in summaries])
        print(f"Digit {digit}: Depth={mean_depth:.2f}, Timing={mean_timing:.2f}, Recovery={mean_recovery:.2f}, Stability={mean_stability:.2f}")
    return signatures

if __name__ == "__main__":
    signatures = run_pure_ska_analysis()
