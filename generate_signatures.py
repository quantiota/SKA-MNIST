"""
Pure SKA Implementation Based ONLY on the Research Papers
========================================================

This implements the SKA framework EXACTLY as described in the research papers:
- Paper 1: "Structured Knowledge Accumulation: An Autonomous Framework..."
- Paper 2: "The Principle of Entropic Least Action in Forward-Only Neural Learning"

Key formulas from papers:
- H^(l) = -1/ln(2) * Σ_k z^(l)_k · ΔD^(l)_k
- ΔD^(l)_k = D^(l)_k - D^(l)_{k-1}
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
        self.K = K  # Number of forward steps
        self.eta = eta  # Learning rate (time step in continuous formulation)
        
        # Initialize weights (as in papers)
        self.weight = nn.Parameter(torch.randn(input_size, layer_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(layer_size))
        
        # Knowledge and decision tracking (from papers)
        self.z_history = []  # Knowledge vectors z^(l)_k
        self.D_history = []  # Decision probabilities D^(l)_k
        self.delta_D_history = []  # Decision shifts ΔD^(l)_k
        
        # Entropy signature tracking
        self.H_evolution = []  # Layer entropy H^(l) over steps
        self.cos_theta_evolution = []  # Cosine alignment
        self.frobenius_evolution = []  # ||z||_F magnitude
        
    def forward(self, x):
        """Forward pass computing knowledge z and decisions D"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Compute knowledge tensor: z = Wx + b (from papers)
        z = torch.mm(x_flat, self.weight) + self.bias
        
        # Compute decision probabilities: D = σ(z) (from papers)
        D = torch.sigmoid(z)
        
        # Store for entropy computation
        self.z_history.append(z.clone())
        self.D_history.append(D.clone())
        
        # Track Frobenius norm ||z||_F
        z_norm = torch.norm(z, p='fro').item()
        self.frobenius_evolution.append(z_norm)
        
        return D
    
    def compute_ska_entropy(self, k):
        """
        Compute SKA entropy exactly as in papers:
        H^(l) = -1/ln(2) * Σ_k z^(l)_k · ΔD^(l)_k
        """
        if k == 0:
            return 0.0  # No previous step for ΔD computation
        
        # Get current and previous states
        z_k = self.z_history[k]  # Current knowledge
        D_k = self.D_history[k]  # Current decisions  
        D_k_minus_1 = self.D_history[k-1]  # Previous decisions
        
        # Compute decision shift: ΔD^(l)_k = D^(l)_k - D^(l)_{k-1}
        delta_D_k = D_k - D_k_minus_1
        self.delta_D_history.append(delta_D_k.clone())
        
        # SKA entropy formula from papers: H^(l) = -1/ln(2) * Σ_k z^(l)_k · ΔD^(l)_k
        H_lk = (-1 / np.log(2)) * (z_k * delta_D_k)  # Element-wise product
        H_layer = torch.sum(H_lk).item()  # Sum over all elements
        
        # Store entropy
        self.H_evolution.append(H_layer)
        
        # Compute cosine alignment cos(θ) between z and ΔD
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
        """
        SKA weight update without backpropagation (from papers)
        """
        if k == 0 or len(self.delta_D_history) == 0:
            return  # No update on first step
        
        x_flat = x.view(x.shape[0], -1)
        z_k = self.z_history[k]
        D_k = self.D_history[k]
        delta_D_k = self.delta_D_history[-1]
        
        # Sigmoid derivative: D'(z) = D(1-D)
        D_prime = D_k * (1 - D_k)
        
        # Entropy gradient (from papers)
        grad_z_H = (-1 / np.log(2)) * (z_k * D_prime + delta_D_k)
        
        # Weight update via outer product
        dW = torch.matmul(x_flat.t(), grad_z_H) / x_flat.shape[0]
        db = grad_z_H.mean(dim=0)
        
        # Update parameters
        self.weight.data = self.weight.data - self.eta * dW
        self.bias.data = self.bias.data - self.eta * db
    
    def reset_for_new_digit(self):
        """Reset all tracking for new digit"""
        self.z_history = []
        self.D_history = []
        self.delta_D_history = []
        self.H_evolution = []
        self.cos_theta_evolution = []
        self.frobenius_evolution = []
    
    def generate_ska_signature(self, data, digit_label, verbose=True):
        """
        Generate entropy signature following SKA papers exactly
        """
        if verbose:
            print(f"Generating SKA signature for digit {digit_label}...")
        
        self.reset_for_new_digit()
        
        # SKA forward-only learning loop (K steps as in papers)
        for k in range(self.K):
            # Forward pass
            outputs = self.forward(data)
            
            # Compute entropy (only after step 0)
            entropy = self.compute_ska_entropy(k)
            
            # SKA weight update (forward-only)
            self.ska_weight_update(data, k)
            
            if verbose and k % 10 == 0 and k > 0:
                print(f"  Step {k}: H = {entropy:.4f}")
        
        # Create signature following papers
        signature = self.create_paper_signature(digit_label)
        
        if verbose and self.H_evolution:
            min_H = min(self.H_evolution)
            min_step = np.argmin(self.H_evolution) + 1  # +1 because entropy starts at step 1
            print(f"  Minimum entropy: {min_H:.4f} at step {min_step}")
        
        return signature
    
    def create_paper_signature(self, digit_label):
        """Create signature following research papers"""
        
        # Entropy characteristics
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
        
        # U-shape pattern detection (from papers)
        has_u_shape = (min_H_idx > 5 and min_H_idx < len(H_seq) - 5) if len(H_seq) > 10 else False
        
        # Final output pattern
        final_output = self.D_history[-1].mean(dim=0).detach().cpu().numpy() if self.D_history else np.zeros(self.layer_size)
        
        signature = {
            'digit_label': digit_label,
            'ska_version': 'papers_exact',
            'parameters': {
                'K': self.K,
                'eta': self.eta,
                'layer_size': self.layer_size
            },
            
            # Core SKA entropy evolution
            'entropy_evolution': H_seq,
            'cosine_evolution': self.cos_theta_evolution,
            'frobenius_evolution': self.frobenius_evolution,
            
            # Entropy characteristics
            'min_entropy_value': min_H_value,
            'min_entropy_step': min_H_idx + 1,  # Step numbering starts at 1
            'initial_entropy': initial_H,
            'final_entropy': final_H,
            'entropy_variance': H_variance,
            'has_u_shape': has_u_shape,
            
            # Recovery metrics
            'entropy_reduction': initial_H - min_H_value,
            'entropy_recovery': final_H - min_H_value,
            
            # Output fingerprint
            'final_output_pattern': final_output.tolist(),
            
            # Summary metrics
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
        """Generate SKA signatures for all MNIST digits"""
        
        print("🔬 GENERATING SKA SIGNATURES FROM RESEARCH PAPERS")
        print("=" * 60)
        print(f"Parameters: K={K}, η={eta}")
        print(f"Formula: H^(l) = -1/ln(2) * Σ_k z^(l)_k · ΔD^(l)_k")
        print("=" * 60)
        
        # Load MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        signatures = {}
        
        for digit in range(10):
            print(f"\n📊 Processing Digit {digit}")
            print("-" * 30)
            
            # Get digit samples
            digit_indices = [i for i, (_, label) in enumerate(mnist_data) if label == digit][:self.samples_per_digit]
            digit_data = torch.stack([mnist_data[i][0] for i in digit_indices])
            
            # Create SKA generator
            ska_gen = PureSKAGenerator(K=K, eta=eta)
            
            # Generate signature
            signature = ska_gen.generate_ska_signature(digit_data, digit)
            signatures[digit] = signature
            
            # Print results
            summary = signature['signature_summary']
            print(f"  ✓ Entropy Depth: {summary['depth']:.2f}")
            print(f"  ✓ Minimum at Step: {summary['timing']}")
            print(f"  ✓ Recovery: {summary['recovery']:.2f}")
        
        # Save signatures
        with open(save_path, 'wb') as f:
            pickle.dump(signatures, f)
        
        # JSON export
        def to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_json_safe(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):  # FIX: Added np.bool_ handling
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
    """Visualize SKA signatures"""
    print("\n🎨 Creating SKA signature visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Entropy Evolution
    plt.subplot(2, 3, 1)
    for digit, sig in signatures.items():
        H_seq = sig['entropy_evolution']
        if H_seq:
            plt.plot(range(1, len(H_seq)+1), H_seq, label=f'Digit {digit}', linewidth=2)
            
            # Mark minimum
            min_step = sig['min_entropy_step']
            min_val = sig['min_entropy_value']
            plt.scatter(min_step, min_val, s=50, alpha=0.7)
    
    plt.title('SKA Entropy Evolution')
    plt.xlabel('Step k')
    plt.ylabel('H^(l)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Depth vs Timing
    plt.subplot(2, 3, 2)
    depths = [sig['signature_summary']['depth'] for sig in signatures.values()]
    timings = [sig['signature_summary']['timing'] for sig in signatures.values()]
    
    for i, (depth, timing) in enumerate(zip(depths, timings)):
        plt.scatter(timing, depth, s=100)
        plt.annotate(str(i), (timing, depth), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Minimum Step')
    plt.ylabel('Entropy Depth')
    plt.title('SKA Signature Characteristics')
    plt.grid(True, alpha=0.3)
    
    # 3. Recovery Analysis
    plt.subplot(2, 3, 3)
    recoveries = [sig['signature_summary']['recovery'] for sig in signatures.values()]
    stabilities = [sig['signature_summary']['stability'] for sig in signatures.values()]
    
    for i, (recovery, stability) in enumerate(zip(recoveries, stabilities)):
        plt.scatter(recovery, stability, s=100)
        plt.annotate(str(i), (recovery, stability), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Entropy Recovery')
    plt.ylabel('Stability')
    plt.title('Recovery vs Stability')
    plt.grid(True, alpha=0.3)
    
    # 4. Cosine Evolution
    plt.subplot(2, 3, 4)
    for digit, sig in signatures.items():
        cos_seq = sig['cosine_evolution']
        if cos_seq:
            plt.plot(range(1, len(cos_seq)+1), cos_seq, label=f'Digit {digit}', alpha=0.7)
    
    plt.title('Cosine Alignment Evolution')
    plt.xlabel('Step k')
    plt.ylabel('cos(θ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Frobenius Norm Evolution  
    plt.subplot(2, 3, 5)
    for digit, sig in signatures.items():
        frob_seq = sig['frobenius_evolution']
        if frob_seq:
            plt.plot(frob_seq, label=f'Digit {digit}', alpha=0.7)
    
    plt.title('Knowledge Magnitude ||z||_F')
    plt.xlabel('Step k')
    plt.ylabel('||z||_F')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Summary Heatmap
    plt.subplot(2, 3, 6)
    summary_data = []
    for digit in range(10):
        sig = signatures[digit]
        summary = sig['signature_summary']
        summary_data.append([summary['depth'], summary['timing'], summary['recovery'], summary['stability']])
    
    summary_data = np.array(summary_data)
    
    # Normalize for heatmap
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
    plt.title('SKA Signature Summary')
    
    plt.tight_layout()
    plt.savefig('ska_signatures_from_papers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'ska_signatures_from_papers.png'")

def run_pure_ska_analysis():
    """Run complete SKA analysis based on research papers"""
    
    print("🔬 PURE SKA ANALYSIS FROM RESEARCH PAPERS")
    print("=" * 50)
    
    # Generate signatures
    generator = SKADatasetGenerator(samples_per_digit=100)
    signatures = generator.generate_mnist_signatures(K=50, eta=0.01)
    
    # Visualize results
    visualize_ska_signatures(signatures)
    
    # Print summary
    print("\n📊 SKA SIGNATURE SUMMARY:")
    for digit, sig in signatures.items():
        summary = sig['signature_summary']
        print(f"Digit {digit}: Depth={summary['depth']:.2f}, "
              f"Timing={summary['timing']}, Recovery={summary['recovery']:.2f}")
    
    return signatures

if __name__ == "__main__":
    signatures = run_pure_ska_analysis()