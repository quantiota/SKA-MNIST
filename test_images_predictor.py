"""
SKA Test Image Predictor
========================

This script predicts test images using SKA entropy signatures.
It computes the entropy signature for a test image and compares it 
with stored signatures to classify the digit.

Based on the research papers:
- H^(l) = -1/ln(2) * Σ_k z^(l)_k · ΔD^(l)_k
- Forward-only learning without backpropagation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import json
import os
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr
import torchvision.transforms as transforms
from torchvision import datasets

# Set random seeds for reproducibility (same as signature generation)
torch.manual_seed(42)
np.random.seed(42)

class PureSKAGenerator(nn.Module):
    """Pure SKA implementation following research papers exactly - copied from signature generation"""
    
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

class SKATestPredictor:
    """SKA-based test image predictor using entropy signatures"""
    
    def __init__(self, signatures_path="ska_signatures.pkl"):
        """Initialize predictor with stored signatures"""
        self.signatures_path = signatures_path
        self.reference_signatures = self.load_signatures()
        self.K = 50  # Default steps from training
        self.eta = 0.01  # Default learning rate from training
        
    def load_signatures(self):
        """Load reference signatures from file"""
        try:
            with open(self.signatures_path, 'rb') as f:
                signatures = pickle.load(f)
            print(f"✅ Loaded reference signatures from {self.signatures_path}")
            return signatures
        except FileNotFoundError:
            print(f"❌ Signatures file {self.signatures_path} not found!")
            print("Please run the signature generation script first.")
            return None
    
    def compute_test_signature(self, test_image, verbose=False):
        """
        Compute SKA entropy signature for a test image
        Using the same random seed as signature generation for consistency
        """
        if verbose:
            print("Computing test image entropy signature...")
        
        # Use the same seed as signature generation for consistency
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate SKA signature using the exact same class as training
        ska_gen = PureSKAGenerator(K=self.K, eta=self.eta)
        test_signature = ska_gen.generate_ska_signature(test_image, digit_label=-1, verbose=verbose)
        
        if verbose:
            print(f"  Using seed: 42 (same as training)")
            print(f"  Entropy depth: {test_signature['signature_summary']['depth']:.3f}")
            print(f"  Minimum at step: {test_signature['signature_summary']['timing']}")
            print(f"  Recovery: {test_signature['signature_summary']['recovery']:.3f}")
        
        return test_signature
    
    def compare_signatures(self, test_signature, method='combined'):
        """
        Compare test signature with reference signatures
        Using multiple comparison methods
        """
        if self.reference_signatures is None:
            return None
        
        similarities = {}
        
        print(f"\n🔍 DEBUGGING: Test Signature Analysis")
        test_summary = test_signature['signature_summary']
        print(f"Test - Depth: {test_summary['depth']:.3f}")
        print(f"Test - Timing: {test_summary['timing']}")
        print(f"Test - Recovery: {test_summary['recovery']:.3f}")
        print(f"Test - Stability: {test_summary['stability']:.3f}")
        print(f"Test - Min Entropy: {test_signature['min_entropy_value']:.3f}")
        
        print(f"\n📊 Comparing with Reference Signatures:")
        print("-" * 80)
        
        for digit in range(10):
            if digit not in self.reference_signatures or len(self.reference_signatures[digit]) == 0:
                similarities[digit] = 0.0
                continue
                
            # Get mean reference signature for this digit
            ref_sigs = self.reference_signatures[digit]
            ref_summaries = [sig['signature_summary'] for sig in ref_sigs]
            
            # Compute mean reference characteristics
            mean_depth = np.mean([s['depth'] for s in ref_summaries])
            mean_timing = np.mean([s['timing'] for s in ref_summaries])
            mean_recovery = np.mean([s['recovery'] for s in ref_summaries])
            mean_stability = np.mean([s['stability'] for s in ref_summaries])
            
            print(f"\nDigit {digit} Reference (mean of {len(ref_sigs)} samples):")
            print(f"  Depth: {mean_depth:.3f} | Diff: {abs(test_summary['depth'] - mean_depth):.3f}")
            print(f"  Timing: {mean_timing:.1f} | Diff: {abs(test_summary['timing'] - mean_timing):.1f}")
            print(f"  Recovery: {mean_recovery:.3f} | Diff: {abs(test_summary['recovery'] - mean_recovery):.3f}")
            print(f"  Stability: {mean_stability:.3f} | Diff: {abs(test_summary['stability'] - mean_stability):.3f}")
            
            if method == 'entropy_curve':
                # Compare entropy evolution curves with mean reference curve
                test_H = test_signature['entropy_evolution']
                ref_H_curves = [sig['entropy_evolution'] for sig in ref_sigs if len(sig['entropy_evolution']) > 0]
                
                if len(test_H) == 0 or len(ref_H_curves) == 0:
                    similarities[digit] = 0.0
                    print(f"  Entropy Curve Similarity: 0.000 (empty curves)")
                    continue
                
                # Compute mean reference curve
                min_len = min(len(test_H), min(len(curve) for curve in ref_H_curves))
                if min_len > 1:
                    ref_H_mean = np.mean([curve[:min_len] for curve in ref_H_curves], axis=0)
                    test_H_norm = test_H[:min_len]
                    
                    try:
                        corr, _ = pearsonr(test_H_norm, ref_H_mean)
                        similarities[digit] = max(0, corr) if not np.isnan(corr) else 0
                        print(f"  Entropy Curve Similarity: {similarities[digit]:.3f} (correlation)")
                    except:
                        similarities[digit] = 0
                        print(f"  Entropy Curve Similarity: 0.000 (correlation failed)")
                else:
                    similarities[digit] = 0
                    print(f"  Entropy Curve Similarity: 0.000 (insufficient data)")
                    
            elif method == 'summary_features':
                # Compare summary features using mean reference
                test_features = [
                    test_summary['depth'],
                    test_summary['timing'],
                    test_summary['recovery'],
                    test_summary['stability']
                ]
                ref_features = [mean_depth, mean_timing, mean_recovery, mean_stability]
                
                print(f"  Test Features: {[f'{f:.3f}' for f in test_features]}")
                print(f"  Ref Features:  {[f'{f:.3f}' for f in ref_features]}")
                
                # Euclidean distance (convert to similarity)
                try:
                    distance = euclidean(test_features, ref_features)
                    similarities[digit] = 1 / (1 + distance)
                    print(f"  Feature Distance: {distance:.3f} | Similarity: {similarities[digit]:.3f}")
                except:
                    similarities[digit] = 0
                    print(f"  Feature Similarity: 0.000 (distance calculation failed)")
                
            elif method == 'combined':
                # Combine both methods
                # Calculate entropy similarity
                test_H = test_signature['entropy_evolution']
                ref_H_curves = [sig['entropy_evolution'] for sig in ref_sigs if len(sig['entropy_evolution']) > 0]
                
                if len(test_H) == 0 or len(ref_H_curves) == 0:
                    entropy_sim = 0.0
                else:
                    min_len = min(len(test_H), min(len(curve) for curve in ref_H_curves))
                    if min_len > 1:
                        ref_H_mean = np.mean([curve[:min_len] for curve in ref_H_curves], axis=0)
                        test_H_norm = test_H[:min_len]
                        try:
                            corr, _ = pearsonr(test_H_norm, ref_H_mean)
                            entropy_sim = max(0, corr) if not np.isnan(corr) else 0
                        except:
                            entropy_sim = 0
                    else:
                        entropy_sim = 0
                
                # Calculate feature similarity
                test_features = [
                    test_summary['depth'],
                    test_summary['timing'],
                    test_summary['recovery'],
                    test_summary['stability']
                ]
                ref_features = [mean_depth, mean_timing, mean_recovery, mean_stability]
                try:
                    distance = euclidean(test_features, ref_features)
                    feature_sim = 1 / (1 + distance)
                except:
                    feature_sim = 0
                
                similarities[digit] = 0.6 * entropy_sim + 0.4 * feature_sim
                print(f"  Combined: 0.6×{entropy_sim:.3f} + 0.4×{feature_sim:.3f} = {similarities[digit]:.3f}")
        
        print("-" * 80)
        print(f"🏆 FINAL SIMILARITIES:")
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for digit, sim in sorted_sims:
            print(f"  Digit {digit}: {sim:.3f}")
        
        return similarities
    
    def predict_digit(self, test_image, method='combined', verbose=True):
        """
        Predict digit for test image using SKA entropy signatures
        """
        if self.reference_signatures is None:
            return None, None
        
        # Compute test signature
        test_signature = self.compute_test_signature(test_image, verbose=verbose)
        
        # Compare with reference signatures
        similarities = self.compare_signatures(test_signature, method=method)
        
        # Find best match
        predicted_digit = max(similarities, key=similarities.get)
        confidence = similarities[predicted_digit]
        
        if verbose:
            print(f"\n🔍 SKA Prediction Results:")
            print(f"Predicted digit: {predicted_digit}")
            print(f"Confidence: {confidence:.3f}")
            print(f"\nTop 3 similarities:")
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            for i, (digit, sim) in enumerate(sorted_sims[:3]):
                print(f"  {i+1}. Digit {digit}: {sim:.3f}")
        
        return predicted_digit, similarities, test_signature
    
    def predict_image_file(self, image_path, method='combined', verbose=True):
        """Predict digit from image file"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28
            
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            img_tensor = transform(img)
            
            if verbose:
                print(f"📸 Loaded image: {image_path}")
                print(f"Image shape: {img_tensor.shape}")
            
            return self.predict_digit(img_tensor, method=method, verbose=verbose)
            
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None, None, None
    
    def test_mnist_samples(self, num_samples=10, method='combined'):
        """Test on random MNIST samples"""
        print(f"🧪 Testing SKA predictor on {num_samples} MNIST samples")
        print("=" * 50)
        
        # Load MNIST test data
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Random samples (different from training)
        torch.manual_seed(999)  # Different seed from training
        indices = torch.randperm(len(test_dataset))[:num_samples]
        
        correct = 0
        results = []
        
        for i, idx in enumerate(indices):
            test_image, true_label = test_dataset[idx]
            
            print(f"\nTest {i+1}/{num_samples} - True label: {true_label}")
            
            predicted_digit, similarities, test_sig = self.predict_digit(
                test_image, method=method, verbose=False
            )
            
            is_correct = predicted_digit == true_label
            if is_correct:
                correct += 1
            
            confidence = similarities[predicted_digit]
            print(f"Predicted: {predicted_digit} (confidence: {confidence:.3f}) {'✅' if is_correct else '❌'}")
            
            # Show top 3 predictions for debugging
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 3: {sorted_sims[:3]}")
            
            results.append({
                'true_label': true_label,
                'predicted': predicted_digit,
                'confidence': confidence,
                'correct': is_correct,
                'similarities': similarities
            })
        
        accuracy = correct / num_samples
        print(f"\n📊 Final Results:")
        print(f"Accuracy: {accuracy:.1%} ({correct}/{num_samples})")
        
        return results

def visualize_prediction(test_image, predicted_digit, similarities, test_signature, true_label=None):
    """Visualize prediction results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Test image
    ax1 = axes[0, 0]
    if len(test_image.shape) == 3:
        img_display = test_image.squeeze(0)
    else:
        img_display = test_image
    
    ax1.imshow(img_display, cmap='gray')
    title = f'Test Image'
    if true_label is not None:
        title += f' (True: {true_label})'
    ax1.set_title(title)
    ax1.axis('off')
    
    # 2. Similarity scores
    ax2 = axes[0, 1]
    digits = list(similarities.keys())
    scores = list(similarities.values())
    bars = ax2.bar(digits, scores)
    
    # Highlight predicted digit
    max_idx = scores.index(max(scores))
    bars[max_idx].set_color('red')
    
    ax2.set_title(f'SKA Similarity Scores\nPredicted: {predicted_digit}')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Similarity')
    ax2.grid(True, alpha=0.3)
    
    # 3. Test entropy evolution
    ax3 = axes[1, 0]
    H_evolution = test_signature['entropy_evolution']
    if H_evolution:
        steps = range(1, len(H_evolution) + 1)
        ax3.plot(steps, H_evolution, 'b-', linewidth=2, label='Test Image')
        
        # Mark minimum
        min_step = test_signature['min_entropy_step']
        min_val = test_signature['min_entropy_value']
        ax3.scatter(min_step, min_val, color='red', s=100, zorder=5)
        ax3.annotate(f'Min: {min_val:.3f}', 
                    (min_step, min_val), 
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    ax3.set_title('Test Image Entropy Evolution')
    ax3.set_xlabel('Step k')
    ax3.set_ylabel('H^(l)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Signature characteristics
    ax4 = axes[1, 1]
    summary = test_signature['signature_summary']
    
    characteristics = ['Depth', 'Timing', 'Recovery', 'Stability']
    values = [summary['depth'], summary['timing'], summary['recovery'], summary['stability']]
    
    ax4.bar(characteristics, values, color=['blue', 'green', 'orange', 'purple'])
    ax4.set_title('SKA Signature Summary')
    ax4.set_ylabel('Value')
    
    # Rotate labels
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('ska_test_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Prediction visualization saved as 'ska_test_prediction.png'")

def main():
    """Main testing function"""
    print("🔬 SKA TEST IMAGE PREDICTOR")
    print("=" * 40)
    
    # Initialize predictor
    predictor = SKATestPredictor("ska_signatures.pkl")
    
    if predictor.reference_signatures is None:
        print("Please run the signature generation script first!")
        return
    
    # Test on MNIST samples
    print("\n1. Testing on MNIST samples:")
    results = predictor.test_mnist_samples(num_samples=10)
    
    # Demonstrate single prediction with visualization
    print("\n2. Demonstrating single prediction with visualization:")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Pick a random test image
    torch.manual_seed(123)
    test_idx = torch.randint(0, len(test_dataset), (1,)).item()
    test_image, true_label = test_dataset[test_idx]
    
    print(f"Analyzing test image with true label: {true_label}")
    predicted, similarities, test_sig = predictor.predict_digit(test_image, verbose=True)
    
    # Visualize the prediction
    visualize_prediction(test_image, predicted, similarities, test_sig, true_label)
    
    # Test on saved images (if available)
    print("\n3. Testing on saved PNG images:")
    images_dir = "./mnist_test_images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')][:5]
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            
            # Extract true label from filename
            true_label = None
            if '_label_' in img_file:
                try:
                    true_label = int(img_file.split('_label_')[1].split('.')[0])
                except:
                    pass
            
            print(f"\n📸 Testing: {img_file}")
            predicted, similarities, test_sig = predictor.predict_image_file(
                img_path, verbose=False
            )
            
            if predicted is not None:
                confidence = similarities[predicted]
                is_correct = predicted == true_label if true_label is not None else "Unknown"
                print(f"Predicted: {predicted} (confidence: {confidence:.3f}) {is_correct}")
    else:
        print(f"No images found in {images_dir}")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()