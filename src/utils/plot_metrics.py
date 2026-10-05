import sys
import re
import matplotlib.pyplot as plt

def plot_learning_curves(log_file):
    epochs, train_loss, val_loss, mae_k, mae_g = [], [], [], [], []
    
    # Regex to extract metrics from your SLURM output format
    pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)\s+\|\s+Val MAE_K:\s+([\d.]+)\s+GPa\s+\|\s+Val MAE_G:\s+([\d.]+)\s+GPa")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_loss.append(float(match.group(2)))
                val_loss.append(float(match.group(3)))
                mae_k.append(float(match.group(4)))
                mae_g.append(float(match.group(5)))
                
    if not epochs:
        print("No training metrics found in log.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Losses
    ax1.plot(epochs, train_loss, label='Train Loss (Smooth L1)', color='blue')
    ax1.plot(epochs, val_loss, label='Validation Loss', color='orange')
    ax1.set_title("Learning Curves (Overfitting Check)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Normalized Loss")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Physical MAE
    ax2.plot(epochs, mae_k, label='MAE K (Bulk)', color='red')
    ax2.plot(epochs, mae_g, label='MAE G (Shear)', color='green')
    ax2.set_title("Mean Absolute Error (GPa)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error in GPa")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    print("Saved learning curves to training_curves.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <slurm_log_file>")
    else:
        plot_learning_curves(sys.argv[1])
