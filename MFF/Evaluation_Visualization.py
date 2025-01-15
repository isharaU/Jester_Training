import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def get_processing_times(content):
    time_pattern = r"average ([\d.]+) sec/video"
    return [float(x) for x in re.findall(time_pattern, content)]

def parse_log_file(file_path):
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract class accuracies
    class_pattern = r"Class (\d+) \((.*?)\) Accuracy: ([\d.]+)%"
    class_matches = re.findall(class_pattern, content)
    
    # Create DataFrame for class accuracies
    classes_df = pd.DataFrame(class_matches, columns=['Class_ID', 'Class_Name', 'Accuracy'])
    classes_df['Accuracy'] = classes_df['Accuracy'].astype(float)
    
    # Extract overall metrics
    overall_accuracy = float(re.search(r"Class Accuracy ([\d.]+)%", content).group(1))
    
    # Extract Prec@1 and Prec@5 from the final video processing line
    final_metrics = re.search(r"moving Prec@1 ([\d.]+) Prec@5 ([\d.]+)", content)
    prec_at_1 = float(final_metrics.group(1))
    prec_at_5 = float(final_metrics.group(2))
    
    # Extract unique prediction and label values
    pred_pattern = r"Unique prediction values: \[([\d\s]+)\]"
    label_pattern = r"Unique label values: \[([\d\s]+)\]"
    
    pred_values = np.array([int(x) for x in re.search(pred_pattern, content).group(1).split()])
    label_values = np.array([int(x) for x in re.search(label_pattern, content).group(1).split()])
    
    # Get processing times
    processing_times = get_processing_times(content)
    
    return classes_df, overall_accuracy, prec_at_1, prec_at_5, pred_values, label_values, processing_times

def create_class_accuracies_plot(classes_df):
    plt.figure(figsize=(15, 8))
    x = range(len(classes_df))
    plt.bar(x, classes_df['Accuracy'])
    plt.title('Accuracy by Class', fontsize=14, pad=20)
    plt.xlabel('Class ID', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(x, classes_df['Class_ID'], rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_accuracies.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_metrics_plot(overall_accuracy, prec_at_1, prec_at_5):
    plt.figure(figsize=(10, 6))
    metrics = ['Class Accuracy', 'Precision@1', 'Precision@5']
    values = [overall_accuracy, prec_at_1, prec_at_5]
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'coral'])
    plt.title('Overall Performance Metrics', fontsize=14, pad=20)
    plt.ylabel('Percentage (%)', fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_plot(pred_values, label_values):
    plt.figure(figsize=(12, 10))
    matrix_size = max(max(pred_values), max(label_values)) + 1
    confusion_data = np.zeros((matrix_size, matrix_size))
    
    # Mark the predictions
    for pred, label in zip(pred_values, label_values):
        confusion_data[pred, label] = 1
    
    # Only show relevant rows and columns
    relevant_indices = sorted(list(set(pred_values) | set(label_values)))
    confusion_data = confusion_data[relevant_indices][:, relevant_indices]
    
    im = plt.imshow(confusion_data, cmap='Blues')
    plt.colorbar(im)
    
    plt.xticks(range(len(relevant_indices)), relevant_indices)
    plt.yticks(range(len(relevant_indices)), relevant_indices)
    
    # Add text annotations
    for i in range(len(relevant_indices)):
        for j in range(len(relevant_indices)):
            plt.text(j, i, f'{confusion_data[i, j]:.0f}',
                    ha="center", va="center", color="black")
    
    plt.title('Prediction vs Ground Truth Matrix', fontsize=14, pad=20)
    plt.xlabel('True Labels', fontsize=12)
    plt.ylabel('Predicted Labels', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_processing_time_plot(times):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(times) + 1), times, marker='o', linewidth=2, markersize=8)
    plt.title('Processing Time per Video', fontsize=14, pad=20)
    plt.xlabel('Video Number', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True)
    
    # Add value labels
    for i, time in enumerate(times):
        plt.text(i + 1, time, f'{time:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('processing_times.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(file_path):
    # Parse the log file
    classes_df, overall_accuracy, prec_at_1, prec_at_5, pred_values, label_values, times = parse_log_file(file_path)
    
    # Create individual plots
    create_class_accuracies_plot(classes_df)
    create_overall_metrics_plot(overall_accuracy, prec_at_1, prec_at_5)
    create_confusion_matrix_plot(pred_values, label_values)
    create_processing_time_plot(times)

# Example usage
file_path = '/content/evaluation_results.txt'  # Save log output to this file
create_visualizations(file_path)