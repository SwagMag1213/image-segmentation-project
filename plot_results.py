import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for clean plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_and_prepare_data(csv_file):
    """Load and prepare the results data"""
    df = pd.read_csv(csv_file)
    
    # IMPORTANT: Rank by CV performance for proper model selection
    df_cv_ranked = df.sort_values('CV_IoU_Mean', ascending=False).reset_index(drop=True)
    df_cv_ranked['CV_Rank'] = range(1, len(df_cv_ranked) + 1)
    
    # Clean up experiment names for better display
    df_cv_ranked['Experiment_Short'] = (df_cv_ranked['Experiment']
                                       .str.replace('Focal (α=', 'Focal α=')
                                       .str.replace('Combo (α=', 'Combo α=')
                                       .str.replace('Dice (smooth=', 'Dice s=')
                                       .str.replace(')', ''))
    
    # Create architecture labels
    df_cv_ranked['Architecture'] = (df_cv_ranked['Backbone'] + ' + ' + 
                                   df_cv_ranked['Attention'].map({'Yes': 'Attention', 'No': 'No Attention'}) + 
                                   ' + BS=' + df_cv_ranked['Batch_Size'].astype(str))
    
    return df_cv_ranked

def plot_1_cv_ranking(df, save_name="01_cv_performance_ranking"):
    """Plot 1: Top 10 by CV Performance (Model Selection)"""
    
    top_10_cv = df.head(10)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create color map by experiment type
    unique_experiments = top_10_cv['Experiment'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_experiments)))
    color_map = dict(zip(unique_experiments, colors))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(top_10_cv))
    bars = ax.barh(y_pos, top_10_cv['CV_IoU_Mean'], 
                   xerr=top_10_cv['CV_IoU_Std'],
                   color=[color_map[exp] for exp in top_10_cv['Experiment']], 
                   alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    
    # Customize labels
    labels = []
    for _, row in top_10_cv.iterrows():
        label = f"#{row['CV_Rank']} {row['Experiment_Short']}\n{row['Architecture']}"
        labels.append(label)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    
    ax.set_xlabel('Cross-Validation IoU (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_title('Top 10 Configurations by CV Performance\n(Proper Model Selection Ranking)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, top_10_cv['CV_IoU_Mean'], top_10_cv['CV_IoU_Std'])):
        ax.text(bar.get_width() + std_val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean_val:.3f}±{std_val:.3f}', ha='left', va='center', 
                fontsize=11, fontweight='bold')
    
    # Add legend for experiments
    handles = [plt.Rectangle((0,0),1,1, color=color_map[exp], alpha=0.8) 
               for exp in unique_experiments]
    ax.legend(handles, unique_experiments, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("📊 Plot 1: CV Performance Ranking - SAVED")
    return top_10_cv

def plot_2_generalization_analysis(df, save_name="02_generalization_analysis"):
    """Plot 2: CV vs Test Performance (Generalization Analysis)"""
    
    top_10_cv = df.head(10)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all data points in background
    ax.scatter(df['CV_IoU_Mean'], df['Test_IoU'], alpha=0.3, s=50, color='lightgray', 
              label=f'All configs (n={len(df)})')
    
    # Highlight top 10 by CV
    unique_experiments = top_10_cv['Experiment'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_experiments)))
    color_map = dict(zip(unique_experiments, colors))
    
    scatter = ax.scatter(top_10_cv['CV_IoU_Mean'], top_10_cv['Test_IoU'], 
                        c=[color_map[exp] for exp in top_10_cv['Experiment']], 
                        s=150, alpha=0.9, edgecolors='black', linewidth=2, 
                        label='Top 10 by CV')
    
    # Add error bars for CV performance
    ax.errorbar(top_10_cv['CV_IoU_Mean'], top_10_cv['Test_IoU'], 
               xerr=top_10_cv['CV_IoU_Std'], fmt='none', 
               ecolor='black', alpha=0.6, capsize=4, linewidth=1.5)
    
    # Add diagonal line (perfect correlation)
    min_val = min(df['CV_IoU_Mean'].min(), df['Test_IoU'].min()) - 0.01
    max_val = max(df['CV_IoU_Mean'].max(), df['Test_IoU'].max()) + 0.01
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2,
           label='Perfect CV-Test correlation')
    
    # Annotate top 3 by CV
    for i, (_, row) in enumerate(top_10_cv.head(3).iterrows()):
        ax.annotate(f'CV #{i+1}', 
                   (row['CV_IoU_Mean'], row['Test_IoU']),
                   xytext=(10, 10), textcoords='offset points', 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Cross-Validation IoU (Mean)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test IoU (Generalization)', fontsize=14, fontweight='bold')
    ax.set_title('CV vs Test Performance\n(Generalization Analysis)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Calculate and display correlation
    correlation = np.corrcoef(df['CV_IoU_Mean'], df['Test_IoU'])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: r = {correlation:.3f}', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("📊 Plot 2: Generalization Analysis - SAVED")

def plot_3_loss_function_comparison(df, save_name="03_loss_function_comparison"):
    """Plot 3: Loss Function Performance Comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Group by experiment and calculate statistics
    exp_stats = df.groupby('Experiment').agg({
        'CV_IoU_Mean': ['mean', 'std', 'count'],
        'Test_IoU': ['mean', 'std']
    }).round(4)
    
    exp_stats.columns = ['CV_Mean', 'CV_Std', 'Count', 'Test_Mean', 'Test_Std']
    exp_stats = exp_stats.sort_values('CV_Mean', ascending=False)
    
    # Plot 1: CV Performance by Loss Function
    colors = plt.cm.viridis(np.linspace(0, 1, len(exp_stats)))
    
    bars1 = ax1.bar(range(len(exp_stats)), exp_stats['CV_Mean'], 
                   yerr=exp_stats['CV_Std'], capsize=5,
                   color=colors, alpha=0.8, error_kw={'linewidth': 2})
    
    # Customize labels
    short_names = [exp.replace('Focal (α=', 'F α=').replace('Combo (α=', 'C α=').replace('Dice (smooth=', 'D s=').replace(')', '') 
                   for exp in exp_stats.index]
    ax1.set_xticks(range(len(exp_stats)))
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('CV IoU (Mean ± Std)', fontsize=12, fontweight='bold')
    ax1.set_title('CV Performance by Loss Function\n(Model Selection Criterion)', fontsize=14, fontweight='bold')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars1, exp_stats['Count'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + exp_stats['CV_Std'].iloc[i] + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Test Performance by Loss Function
    bars2 = ax2.bar(range(len(exp_stats)), exp_stats['Test_Mean'], 
                   yerr=exp_stats['Test_Std'], capsize=5,
                   color=colors, alpha=0.8, error_kw={'linewidth': 2})
    
    ax2.set_xticks(range(len(exp_stats)))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('Test IoU (Mean ± Std)', fontsize=12, fontweight='bold')
    ax2.set_title('Test Performance by Loss Function\n(Generalization Performance)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("📊 Plot 3: Loss Function Comparison - SAVED")
    return exp_stats

def plot_4_architecture_effects(df, save_name="04_architecture_effects"):
    """Plot 4: Architecture Component Effects"""
    
    top_10_cv = df.head(10)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Backbone Effect
    backbone_stats = df.groupby('Backbone').agg({
        'CV_IoU_Mean': ['mean', 'std', 'count'],
        'Test_IoU': ['mean', 'std']
    })
    backbone_stats.columns = ['CV_Mean', 'CV_Std', 'Count', 'Test_Mean', 'Test_Std']
    
    x = np.arange(len(backbone_stats))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, backbone_stats['CV_Mean'], width, 
                   yerr=backbone_stats['CV_Std'], capsize=5, 
                   label='CV Performance', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, backbone_stats['Test_Mean'], width,
                   yerr=backbone_stats['Test_Std'], capsize=5,
                   label='Test Performance', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Backbone Architecture', fontweight='bold')
    ax1.set_ylabel('IoU Performance', fontweight='bold')
    ax1.set_title('Backbone Architecture Effect', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(backbone_stats.index)
    ax1.legend()
    
    # 2. Attention Effect
    attention_stats = df.groupby('Attention').agg({
        'CV_IoU_Mean': ['mean', 'std', 'count'],
        'Test_IoU': ['mean', 'std']
    })
    attention_stats.columns = ['CV_Mean', 'CV_Std', 'Count', 'Test_Mean', 'Test_Std']
    
    x = np.arange(len(attention_stats))
    bars1 = ax2.bar(x - width/2, attention_stats['CV_Mean'], width,
                   yerr=attention_stats['CV_Std'], capsize=5,
                   label='CV Performance', alpha=0.8, color='skyblue')
    bars2 = ax2.bar(x + width/2, attention_stats['Test_Mean'], width,
                   yerr=attention_stats['Test_Std'], capsize=5,
                   label='Test Performance', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Attention Mechanism', fontweight='bold')
    ax2.set_ylabel('IoU Performance', fontweight='bold')
    ax2.set_title('Attention Mechanism Effect', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['No Attention', 'With Attention'])
    ax2.legend()
    
    # 3. Batch Size Effect
    batch_stats = df.groupby('Batch_Size').agg({
        'CV_IoU_Mean': ['mean', 'std', 'count'],
        'Test_IoU': ['mean', 'std']
    })
    batch_stats.columns = ['CV_Mean', 'CV_Std', 'Count', 'Test_Mean', 'Test_Std']
    
    x = np.arange(len(batch_stats))
    bars1 = ax3.bar(x - width/2, batch_stats['CV_Mean'], width,
                   yerr=batch_stats['CV_Std'], capsize=5,
                   label='CV Performance', alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x + width/2, batch_stats['Test_Mean'], width,
                   yerr=batch_stats['Test_Std'], capsize=5,
                   label='Test Performance', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Batch Size', fontweight='bold')
    ax3.set_ylabel('IoU Performance', fontweight='bold')
    ax3.set_title('Batch Size Effect', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'BS={bs}' for bs in batch_stats.index])
    ax3.legend()
    
    # 4. Top 10 Architecture Composition
    arch_composition = {
        'ResNet50': (top_10_cv['Backbone'] == 'resnet50').sum(),
        'ResNet34': (top_10_cv['Backbone'] == 'resnet34').sum(),
        'With Attention': (top_10_cv['Attention'] == 'Yes').sum(),
        'No Attention': (top_10_cv['Attention'] == 'No').sum(),
        'Batch Size 4': (top_10_cv['Batch_Size'] == 4).sum(),
        'Batch Size 2': (top_10_cv['Batch_Size'] == 2).sum()
    }
    
    categories = list(arch_composition.keys())
    values = list(arch_composition.values())
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    wedges, texts, autotexts = ax4.pie(values, labels=categories, autopct='%1.0f/10',
                                      colors=colors_pie, startangle=90)
    ax4.set_title('Architecture Composition\n(Top 10 by CV)', fontweight='bold')
    
    # Make text larger
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("📊 Plot 4: Architecture Effects - SAVED")

def plot_5_generalization_gap(df, save_name="05_generalization_gap"):
    """Plot 5: Generalization Gap Analysis"""
    
    top_10_cv = df.head(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Generalization Gap for Top 10
    gaps = top_10_cv['CV_Test_Gap']
    colors_gap = ['darkgreen' if gap <= 0 else 'darkred' for gap in gaps]
    
    bars = ax1.barh(range(len(top_10_cv)), gaps, color=colors_gap, alpha=0.7)
    
    # Labels
    labels = [f"CV #{i+1}: {row['Experiment_Short'][:20]}" 
              for i, (_, row) in enumerate(top_10_cv.iterrows())]
    ax1.set_yticks(range(len(top_10_cv)))
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.invert_yaxis()
    
    ax1.set_xlabel('CV IoU - Test IoU Gap', fontsize=12, fontweight='bold')
    ax1.set_title('Generalization Gap Analysis\n(Top 10 by CV Performance)', 
                 fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, gaps)):
        x_pos = bar.get_width() + 0.005 if val >= 0 else bar.get_width() - 0.005
        ha = 'left' if val >= 0 else 'right'
        color = 'darkred' if val > 0 else 'darkgreen'
        ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', ha=ha, va='center', fontsize=10, 
                fontweight='bold', color=color)
    
    # Add legend
    ax1.text(0.02, 0.98, 'Green: Good generalization (Test > CV)\nRed: Possible overfitting (CV > Test)', 
            transform=ax1.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Plot 2: Gap Distribution
    ax2.hist(df['CV_Test_Gap'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No gap (CV = Test)')
    ax2.axvline(x=df['CV_Test_Gap'].mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean gap: {df["CV_Test_Gap"].mean():+.3f}')
    
    ax2.set_xlabel('CV IoU - Test IoU Gap', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Configurations', fontsize=12, fontweight='bold')
    ax2.set_title('Generalization Gap Distribution\n(All Configurations)', 
                 fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("📊 Plot 5: Generalization Gap - SAVED")

def plot_6_precision_recall_analysis(df, save_name="06_precision_recall_analysis"):
    """Plot 6: Precision vs Recall Analysis"""
    
    top_10_cv = df.head(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Precision vs Recall Scatter
    # All data in background
    ax1.scatter(df['Test_Recall'], df['Test_Precision'], alpha=0.3, s=50, 
               color='lightgray', label=f'All configs (n={len(df)})')
    
    # Top 10 highlighted
    unique_experiments = top_10_cv['Experiment'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_experiments)))
    color_map = dict(zip(unique_experiments, colors))
    
    scatter = ax1.scatter(top_10_cv['Test_Recall'], top_10_cv['Test_Precision'], 
                         c=[color_map[exp] for exp in top_10_cv['Experiment']], 
                         s=150, alpha=0.9, edgecolors='black', linewidth=2,
                         label='Top 10 by CV')
    
    # Annotate top 3 by CV
    for i, (_, row) in enumerate(top_10_cv.head(3).iterrows()):
        ax1.annotate(f'CV #{i+1}', 
                    (row['Test_Recall'], row['Test_Precision']),
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.set_xlabel('Test Recall', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision vs Recall Trade-off\n(Test Set Performance)', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score Distribution
    ax2.hist(df['Test_F1'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black',
            label='All configurations')
    
    # Highlight top 10 F1 scores
    ax2.hist(top_10_cv['Test_F1'], bins=10, alpha=0.8, color='darkgreen', 
            edgecolor='black', label='Top 10 by CV')
    
    ax2.axvline(x=df['Test_F1'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Overall mean: {df["Test_F1"].mean():.3f}')
    ax2.axvline(x=top_10_cv['Test_F1'].mean(), color='darkgreen', linestyle='-', linewidth=2,
               label=f'Top 10 mean: {top_10_cv["Test_F1"].mean():.3f}')
    
    ax2.set_xlabel('Test F1 Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Configurations', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score Distribution\n(Test Set Performance)', 
                 fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("📊 Plot 6: Precision-Recall Analysis - SAVED")

def print_key_insights(df):
    """Print key insights from the analysis"""
    
    top_10_cv = df.head(10)
    best_cv = df.iloc[0]
    
    print(f"\n{'='*70}")
    print("🎯 KEY INSIGHTS (Ranked by CV Performance)")
    print(f"{'='*70}")
    
    print(f"\n🏆 BEST MODEL (by CV performance):")
    print(f"   Experiment: {best_cv['Experiment']}")
    print(f"   Architecture: {best_cv['Architecture']}")
    print(f"   CV IoU: {best_cv['CV_IoU_Mean']:.3f} ± {best_cv['CV_IoU_Std']:.3f}")
    print(f"   Test IoU: {best_cv['Test_IoU']:.3f} (generalization)")
    print(f"   Gap: {best_cv['CV_Test_Gap']:+.3f} ({'Good' if best_cv['CV_Test_Gap'] <= 0 else 'Overfitting'})")
    
    print(f"\n📊 TOP 10 COMPOSITION (by CV):")
    print(f"   ResNet50: {(top_10_cv['Backbone']=='resnet50').sum()}/10")
    print(f"   With Attention: {(top_10_cv['Attention']=='Yes').sum()}/10")
    print(f"   Batch Size 4: {(top_10_cv['Batch_Size']==4).sum()}/10")
    
    print(f"\n⚖️ GENERALIZATION ANALYSIS:")
    good_gen = (top_10_cv['CV_Test_Gap'] <= 0).sum()
    print(f"   Good generalization (gap ≤ 0): {good_gen}/10")
    print(f"   Average gap: {top_10_cv['CV_Test_Gap'].mean():+.3f}")
    
    print(f"\n🔬 LOSS FUNCTION RANKING (by CV):")
    loss_stats = df.groupby('Experiment')['CV_IoU_Mean'].mean().sort_values(ascending=False)
    for i, (loss, avg_cv) in enumerate(loss_stats.head(5).items(), 1):
        print(f"   {i}. {loss}: {avg_cv:.3f}")
    
    print(f"\n📈 CORRELATION:")
    correlation = np.corrcoef(df['CV_IoU_Mean'], df['Test_IoU'])[0, 1]
    print(f"   CV-Test correlation: r = {correlation:.3f}")
    
    if correlation > 0.8:
        print("   → Strong correlation: CV is highly reliable for model selection")
    elif correlation > 0.6:
        print("   → Good correlation: CV is reasonably reliable for model selection")
    else:
        print("   → Weak correlation: CV may not be fully reliable for model selection")

def main():
    """Main function to create all individual plots"""
    
    # Load data
    csv_file = "performance_comparison_20250610_121101_detailed.csv"
    df = load_and_prepare_data(csv_file)
    
    print("="*70)
    print("🎨 CREATING INDIVIDUAL CLEAN PLOTS (RANKED BY CV PERFORMANCE)")
    print("="*70)
    print(f"Total configurations: {len(df)}")
    print(f"Unique experiments: {df['Experiment'].nunique()}")
    print(f"Best CV IoU: {df['CV_IoU_Mean'].max():.3f}")
    print(f"Best Test IoU: {df['Test_IoU'].max():.3f}")
    
    # Create individual plots
    print(f"\n📊 Creating individual plots...")
    
    top_10_cv = plot_1_cv_ranking(df)
    plot_2_generalization_analysis(df)
    exp_stats = plot_3_loss_function_comparison(df)
    plot_4_architecture_effects(df)
    plot_5_generalization_gap(df)
    plot_6_precision_recall_analysis(df)
    
    # Print insights
    print_key_insights(df)
    
    print(f"\n✅ ALL PLOTS SAVED:")
    print("   01_cv_performance_ranking.png")
    print("   02_generalization_analysis.png") 
    print("   03_loss_function_comparison.png")
    print("   04_architecture_effects.png")
    print("   05_generalization_gap.png")
    print("   06_precision_recall_analysis.png")

if __name__ == "__main__":
    main()