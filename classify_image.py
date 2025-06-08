"""
SKA Image Classifier
===================

This script classifies test images using pre-generated SKA entropy signatures.

Usage:
    python classify_image.py --image path/to/image.png
    python classify_image.py --image test_digit.png --verbose
    
Features:
- Loads canonical entropy signatures
- Scans test image to extract entropy signature
- Compares with canonical signatures
- Returns classification with confidence metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os

class SKAClassifier(nn.Module):
    """SKA classifier for test images"""
    
    def __init__(self, input_size=784, layer_size=10, K=50, eta=0.01):
        super(SKAClassifier, self).__init__()
        self.input_size = input_size
        self.layer_size = layer_size
        self.K = K
        self.eta = eta
        
        # Initialize fresh weights for scanning (no training)
        self.weight = nn.Parameter(torch.randn(input_size, layer_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(layer_size))
        
        # Tracking for signature extraction
        self.z_history = []
        self.D_history = []
        self.delta_D_history = []
        self.H_evolution = []
        self.cos_theta_evolution = []
        self.frobenius_evolution = []
        
    def forward(self, x):
        """Forward pass for scanning"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Compute knowledge and decisions
        z = torch.mm(x_flat, self.weight) + self.bias
        D = torch.sigmoid(z)
        
        # Store for entropy computation
        self.z_history.append(z.clone())
        self.D_history.append(D.clone())
        
        # Track knowledge magnitude
        z_norm = torch.norm(z, p='fro').item()
        self.frobenius_evolution.append(z_norm)
        
        return D
    
    def compute_entropy(self, k):
        """Compute entropy at step k"""
        if k == 0:
            return 0.0
        
        z_k = self.z_history[k]
        D_k = self.D_history[k]
        D_k_minus_1 = self.D_history[k-1]
        
        # Decision shift
        delta_D_k = D_k - D_k_minus_1
        self.delta_D_history.append(delta_D_k.clone())
        
        # SKA entropy
        H_lk = (-1 / np.log(2)) * (z_k * delta_D_k)
        H = torch.sum(H_lk).item()
        self.H_evolution.append(H)
        
        # Cosine alignment
        dot_product = torch.sum(z_k * delta_D_k)
        z_norm = torch.norm(z_k)
        delta_D_norm = torch.norm(delta_D_k)
        
        if z_norm > 0 and delta_D_norm > 0:
            cos_theta = (dot_product / (z_norm * delta_D_norm)).item()
        else:
            cos_theta = 0.0
        self.cos_theta_evolution.append(cos_theta)
        
        return H
    
    def reset_state(self):
        """Reset for new image"""
        self.z_history = []
        self.D_history = []
        self.delta_D_history = []
        self.H_evolution = []
        self.cos_theta_evolution = []
        self.frobenius_evolution = []
    
    def extract_signature(self, image):
        """Extract entropy signature from test image"""
        self.reset_state()
        
        # SKA scanning (no weight updates)
        for k in range(self.K):
            _ = self.forward(image)
            entropy = self.compute_entropy(k)
        
        # Create signature
        if self.H_evolution:
            min_H_idx = np.argmin(self.H_evolution)
            min_H_value = min(self.H_evolution)
            initial_H = np.mean(self.H_evolution[:5]) if len(self.H_evolution) >= 5 else self.H_evolution[0]
            final_H = np.mean(self.H_evolution[-5:]) if len(self.H_evolution) >= 5 else self.H_evolution[-1]
            H_variance = np.var(self.H_evolution)
        else:
            min_H_idx, min_H_value, initial_H, final_H, H_variance = 0, 0, 0, 0, 0
        
        signature = {
            'entropy_evolution': self.H_evolution,
            'min_entropy_value': min_H_value,
            'min_entropy_step': min_H_idx + 1,
            'initial_entropy': initial_H,
            'final_entropy': final_H,
            'entropy_recovery': final_H - min_H_value,
            'entropy_variance': H_variance,
            'signature_summary': {
                'depth': abs(min_H_value),
                'timing': min_H_idx + 1,
                'recovery': final_H - min_H_value,
                'stability': 1 / (H_variance + 1e-6)
            }
        }
        
        return signature

def load_canonical_signatures(signature_path="signatures/ska_signatures.pkl"):
    """Load pre-generated canonical signatures"""
    if not os.path.exists(signature_path):
        raise FileNotFoundError(f"Canonical signatures not found at {signature_path}")
        
    with open(signature_path, 'rb') as f:
        signatures = pickle.load(f)
    
    return signatures

def load_and_preprocess_image(image_path):
    """Load and preprocess image for SKA"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Invert if needed (MNIST is white on black)
        if img_array.mean() > 0.5:
            img_array = 1.0 - img_array
        
        # Convert to torch tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
        
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {e}")

def calculate_signature_distance(test_sig, canonical_sig):
    """Calculate distance between test and canonical signatures"""
    test_summary = test_sig['signature_summary']
    canonical_summary = canonical_sig['signature_summary']
    
    # Weighted distance calculation
    depth_diff = abs(test_summary['depth'] - canonical_summary['depth'])
    timing_diff = abs(test_summary['timing'] - canonical_summary['timing'])
    recovery_diff = abs(test_summary['recovery'] - canonical_summary['recovery'])
    
    # Normalize and weight the differences
    distance = (
        depth_diff * 0.5 +      # Depth is most important
        timing_diff * 0.3 +     # Timing is moderately important
        recovery_diff * 0.2     # Recovery is least important
    )
    
    return distance

def classify_image(image_path, signature_path="signatures/ska_signatures.pkl", verbose=False):
    """Classify image using SKA entropy signatures"""
    
    # Load canonical signatures
    if verbose:
        print("Loading canonical signatures...")
    canonical_signatures = load_canonical_signatures(signature_path)
    
    # Load and preprocess image
    if verbose:
        print(f"Loading image: {image_path}")
    image = load_and_preprocess_image(image_path)
    
    # Create SKA classifier
    classifier = SKAClassifier()
    
    # Extract test image signature
    if verbose:
        print("Scanning image with SKA (50 steps)...")
    test_signature = classifier.extract_signature(image)
    
    if verbose:
        test_summary = test_signature['signature_summary']
        print(f"Test signature: Depth={test_summary['depth']:.2f}, "
              f"Timing={test_summary['timing']}, Recovery={test_summary['recovery']:.2f}")
    
    # Compare with canonical signatures
    distances = {}
    for digit, canonical_sig in canonical_signatures.items():
        distance = calculate_signature_distance(test_signature, canonical_sig)
        distances[digit] = distance
    
    # Find best match
    predicted_digit = min(distances, key=distances.get)
    confidence = 1.0 / (distances[predicted_digit] + 1e-6)
    
    # Calculate relative confidence
    sorted_distances = sorted(distances.values())
    if len(sorted_distances) > 1:
        relative_confidence = (sorted_distances[1] - sorted_distances[0]) / sorted_distances[1]
    else:
        relative_confidence = 1.0
    
    results = {
        'predicted_digit': predicted_digit,
        'confidence': confidence,
        'relative_confidence': relative_confidence,
        'test_signature': test_signature,
        'distances': distances,
        'canonical_signatures': canonical_signatures
    }
    
    return results

def visualize_comparison(results, save_path=None):
    """Visualize signature comparison"""
    test_sig = results['test_signature']
    canonical_sigs = results['canonical_signatures']
    predicted_digit = results['predicted_digit']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Entropy evolution comparison
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(test_sig['entropy_evolution'])+1), 
             test_sig['entropy_evolution'], 
             'r-', linewidth=3, label='Test Image')
    
    for digit in [predicted_digit]:
        canonical_entropy = canonical_sigs[digit]['entropy_evolution']
        ax1.plot(range(1, len(canonical_entropy)+1), 
                canonical_entropy, 
                '--', linewidth=2, label=f'Canonical Digit {digit}')
    
    ax1.set_title('Entropy Evolution Comparison')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Entropy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signature characteristics
    ax2 = axes[0, 1]
    test_summary = test_sig['signature_summary']
    canonical_summary = canonical_sigs[predicted_digit]['signature_summary']
    
    metrics = ['depth', 'timing', 'recovery']
    test_values = [test_summary[m] for m in metrics]
    canonical_values = [canonical_summary[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, test_values, width, label='Test Image', alpha=0.8)
    ax2.bar(x + width/2, canonical_values, width, label=f'Canonical Digit {predicted_digit}', alpha=0.8)
    
    ax2.set_title('Signature Characteristics')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Values')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # Plot 3: Distance comparison
    ax3 = axes[1, 0]
    digits = list(results['distances'].keys())
    distances = list(results['distances'].values())
    
    colors = ['red' if d == predicted_digit else 'lightblue' for d in digits]
    bars = ax3.bar(digits, distances, color=colors, alpha=0.7)
    
    ax3.set_title('Signature Distance Comparison')
    ax3.set_xlabel('Digit')
    ax3.set_ylabel('Distance')
    ax3.grid(True, alpha=0.3)
    
    # Highlight predicted digit
    min_idx = distances.index(min(distances))
    bars[min_idx].set_color('red')
    bars[min_idx].set_alpha(1.0)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
Classification Results:

Predicted Digit: {predicted_digit}
Confidence: {results['confidence']:.2f}
Relative Confidence: {results['relative_confidence']:.2f}

Test Signature:
• Depth: {test_summary['depth']:.2f}
• Timing: {test_summary['timing']}
• Recovery: {test_summary['recovery']:.2f}

Canonical Signature:
• Depth: {canonical_summary['depth']:.2f}
• Timing: {canonical_summary['timing']}
• Recovery: {canonical_summary['recovery']:.2f}
    """
    
    ax4.text(0.1, 0.9, summary_text, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Classify images using SKA entropy signatures')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--signatures', default='signatures/ska_signatures.pkl', 
                       help='Path to canonical signatures file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--save-viz', help='Save visualization to file')
    
    args = parser.parse_args()
    
    try:
        # Classify image
        results = classify_image(args.image, args.signatures, args.verbose)
        
        # Print results
        print(f"\n🎯 SKA Classification Results:")
        print(f"📸 Image: {args.image}")
        print(f"🔢 Predicted Digit: {results['predicted_digit']}")
        print(f"📊 Confidence: {results['confidence']:.4f}")
        print(f"📈 Relative Confidence: {results['relative_confidence']:.4f}")
        
        test_summary = results['test_signature']['signature_summary']
        print(f"\n🧬 Test Image Signature:")
        print(f"   Depth: {test_summary['depth']:.2f}")
        print(f"   Timing: {test_summary['timing']}")
        print(f"   Recovery: {test_summary['recovery']:.2f}")
        
        if args.verbose:
            print(f"\n📏 Distances to all digits:")
            for digit, distance in sorted(results['distances'].items()):
                print(f"   Digit {digit}: {distance:.4f}")
        
        # Visualize if requested
        if args.visualize or args.save_viz:
            visualize_comparison(results, args.save_viz)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())