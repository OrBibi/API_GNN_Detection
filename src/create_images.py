import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set the output directory for report figures
output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

# 1. Performance Data based on latest evaluation
models = ["GNN Only", "Random Forest", "Isolation Forest", "Stacking Ensemble"]
metrics = ["Accuracy", "Attack Recall", "Benign Recall", "Global F1"]

data = {
    "Model": models,
    "Accuracy": [0.9729, 0.9567, 0.8299, 0.9728],
    "Attack Recall": [0.9485, 0.8002, 0.5636, 0.9491],
    "Benign Recall": [0.9764, 0.9786, 0.8673, 0.9762],
    "Global F1": [0.9736, 0.9562, 0.8440, 0.9735]
}
df_metrics = pd.DataFrame(data)

# 2. Stacking Model Weights data provided by user
weights_data = {
    "Model Component": ["GNN", "RF", "IF"],
    "Weight": [6.4642, 2.3647, -0.7592]
}
df_weights = pd.DataFrame(weights_data)

# 3. Confusion Matrices mapping [ [TN, FP], [FN, TP] ]
cms = {
    "GNN Only": np.array([[92046, 2228], [681, 12553]]),
    "Random Forest": np.array([[92260, 2014], [2644, 10590]]),
    "Isolation Forest": np.array([[81766, 12508], [5775, 7459]]),
    "Stacking Ensemble": np.array([[92027, 2247], [673, 12561]])
}

# --- Visualizing Functions ---

def save_confusion_matrix(cm, model_name):
    """Generates and saves a confusion matrix heatmap with integer formatting."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Benign', 'Pred Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()

def save_comparison_chart(df):
    """Generates a grouped bar chart comparing all models across all metrics."""
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Model Performance Metrics Comparison", fontsize=16)
    plt.ylim(0.5, 1.05)
    plt.ylabel("Score (0.0 - 1.0)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_metrics.png"))
    plt.close()

def save_weights_chart(df):
    """Generates a bar chart showing the importance (weights) of each model in the ensemble."""
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    df_sorted = df.sort_values(by="Weight", ascending=False)
    # Color positive weights green and negative weights red
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_sorted['Weight']]
    ax = sns.barplot(data=df_sorted, x="Model Component", y="Weight", palette=colors)
    plt.title("Stacking Ensemble: Component Weights", fontsize=14)
    plt.ylabel("Learned Weight Value")
    plt.axhline(0, color='black', linewidth=0.8) # Reference line at zero
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacking_weights.png"))
    plt.close()

def save_radar_chart(df):
    """Generates a radar (spider) chart to compare model 'profiles' across metrics."""
    categories = list(df.columns[1:])
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the circle
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='grey', size=11)
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["0.6","0.7","0.8","0.9","1.0"], color="grey", size=8)
    plt.ylim(0.5, 1.0)
    
    # Plot each model's profile
    colors = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71']
    for i, model in enumerate(df['Model']):
        values = df.loc[i].drop('Model').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    plt.title("Performance Profiles: Radar Chart Analysis", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart_comparison.png"))
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting figure generation...")
    # 1. Confusion Matrices
    for name, matrix in cms.items():
        save_confusion_matrix(matrix, name)
    # 2. Comparison Bar Chart
    save_comparison_chart(df_metrics)
    # 3. Component Weights
    save_weights_chart(df_weights)
    # 4. Radar Profile
    save_radar_chart(df_metrics)
    print(f"Success! All charts are saved in: {output_dir}")