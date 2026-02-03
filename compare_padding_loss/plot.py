import re
import matplotlib.pyplot as plt


def parse_log_file(filepath):
    """Parse log file and extract iteration and loss values."""
    iterations = []
    losses = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Extract iteration number
            iter_match = re.search(r'iteration\s+(\d+)/', line)
            # Extract lm loss value
            loss_match = re.search(r'lm loss:\s+([\d.E+-]+)', line)
            
            if iter_match and loss_match:
                iteration = int(iter_match.group(1))
                loss = float(loss_match.group(1))
                iterations.append(iteration)
                losses.append(loss)
    
    return iterations, losses


def main():
    # Parse both log files
    # Define log files and their labels
    log_files = [
        ('./with_padding.log', 'With Padding'),
        ('./wo_padding.log', 'Without Padding'),
    ]
    
    
    # Parse all log files
    all_data = []
    for filepath, label in log_files:
        iters, losses = parse_log_file(filepath)
        all_data.append((iters, losses, label))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    for i, (iters, losses, label) in enumerate(all_data):
        plt.plot(iters, losses, label=label)
    
    plt.xlabel('Iteration')
    plt.ylabel('LM Loss')
    plt.title('Loss Curve Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig('./loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics for each log file
    for iters, losses, label in all_data:
        print(f"\n{label}:")
        print(f"  Total iterations: {len(iters)}")
        print(f"  Final loss: {losses[-1] if losses else 'N/A'}")
    
    # Calculate difference between two loss curves at corresponding iterations
    if len(all_data) >= 2:
        iters1, losses1, label1 = all_data[0]
        iters2, losses2, label2 = all_data[1]
        
        # Create dictionaries for quick lookup
        loss_dict1 = dict(zip(iters1, losses1))
        loss_dict2 = dict(zip(iters2, losses2))
        
        # Find common iterations
        common_iters = sorted(set(iters1) & set(iters2))
        
        if common_iters:
            differences = []
            for it in common_iters:
                diff = (loss_dict1[it] - loss_dict2[it])/loss_dict1[it]
                differences.append(diff)
            
            print(f"\n{'='*50}")
            print(f"Difference ({label1} - {label2}):")
            print(f"  Common iterations: {len(common_iters)}")
            print(f"  Max difference: {max(differences):.6f}")
            print(f"  Min difference: {min(differences):.6f}")
            print(f"  Mean difference: {sum(differences)/len(differences):.6f}")
            print(f"{'='*50}")
            
            # Plot the difference
            plt.figure(figsize=(10, 6))
            plt.plot(common_iters, differences, label=f'({label1} - {label2})/{label1}')
            plt.xlabel('Iteration')
            plt.ylabel('Loss Difference ratio')
            plt.title('Loss Difference ratio Between Two Runs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('./loss_difference_ratio.png', dpi=150, bbox_inches='tight')
            plt.show()



if __name__ == '__main__':
    main()
