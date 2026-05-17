import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── 1. Load ──────────────────────────────────────────────────────────────────
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('iris.data', header=None, names=cols)

# ── 2. Clean ─────────────────────────────────────────────────────────────────
print("=== DATA OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")

# ── 3. Encode & Scale ─────────────────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['species'])

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

# ── 5. Train SVM (Linear Kernel) ──────────────────────────────────────────────
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# ── 6. Evaluation Metrics ─────────────────────────────────────────────────────
print("\n=== EVALUATION METRICS (Test Set) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ── 7. Visualisations ─────────────────────────────────────────────────────────
colors       = ['#FF6B6B', '#4ECDC4', '#FFE66D']
species_list = le.classes_
bg_fig       = '#0f1117'
bg_ax        = '#1a1d27'
spine_col    = '#333333'

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(bg_fig)

def style_ax(ax):
    ax.set_facecolor(bg_ax)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_col)

# — Sepal scatter —
ax1 = fig.add_subplot(3, 3, 1)
for i, sp in enumerate(species_list):
    m = df['species'] == sp
    ax1.scatter(df[m]['sepal_length'], df[m]['sepal_width'],
                c=colors[i], label=sp.replace('Iris-', ''), alpha=0.7, s=40, edgecolors='none')
ax1.set_xlabel('Sepal Length', color='white', fontsize=9)
ax1.set_ylabel('Sepal Width',  color='white', fontsize=9)
ax1.set_title('Sepal Length vs Width', color='white', fontsize=10, fontweight='bold')
ax1.legend(fontsize=7, framealpha=0.3)
style_ax(ax1)

# — Petal scatter —
ax2 = fig.add_subplot(3, 3, 2)
for i, sp in enumerate(species_list):
    m = df['species'] == sp
    ax2.scatter(df[m]['petal_length'], df[m]['petal_width'],
                c=colors[i], label=sp.replace('Iris-', ''), alpha=0.7, s=40, edgecolors='none')
ax2.set_xlabel('Petal Length', color='white', fontsize=9)
ax2.set_ylabel('Petal Width',  color='white', fontsize=9)
ax2.set_title('Petal Length vs Width', color='white', fontsize=10, fontweight='bold')
ax2.legend(fontsize=7, framealpha=0.3)
style_ax(ax2)

# — Confusion Matrix heatmap —
ax3 = fig.add_subplot(3, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=[s.replace('Iris-', '') for s in le.classes_],
            yticklabels=[s.replace('Iris-', '') for s in le.classes_],
            ax=ax3, cbar_kws={'shrink': 0.8}, linewidths=0.5)
ax3.set_title('Confusion Matrix', color='white', fontsize=10, fontweight='bold')
ax3.set_xlabel('Predicted', color='white', fontsize=9)
ax3.set_ylabel('Actual',    color='white', fontsize=9)
ax3.tick_params(colors='white', labelsize=8)
ax3.set_facecolor(bg_ax)

# — Box plots for each feature —
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
titles   = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
for idx, (feat, title) in enumerate(zip(features, titles)):
    ax = fig.add_subplot(3, 3, 4 + idx)
    data_by_species = [df[df['species'] == sp][feat].values for sp in species_list]
    bp = ax.boxplot(data_by_species, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'fliers']:
        for item in bp[element]:
            item.set_color('#aaaaaa')
    ax.set_xticklabels([s.replace('Iris-', '') for s in species_list],
                       fontsize=8, color='white')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold')
    style_ax(ax)

# — Per-class metrics bar chart —
ax8 = fig.add_subplot(3, 3, 8)
report     = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
class_names = [s.replace('Iris-', '') for s in le.classes_]
precision   = [report[s]['precision']  for s in le.classes_]
recall      = [report[s]['recall']     for s in le.classes_]
f1          = [report[s]['f1-score']   for s in le.classes_]
x, w        = np.arange(len(class_names)), 0.25
ax8.bar(x - w, precision, w, label='Precision', color='#4ECDC4', alpha=0.85)
ax8.bar(x,     recall,    w, label='Recall',    color='#FF6B6B', alpha=0.85)
ax8.bar(x + w, f1,        w, label='F1-Score',  color='#FFE66D', alpha=0.85)
ax8.set_xticks(x)
ax8.set_xticklabels(class_names, color='white', fontsize=8)
ax8.set_ylim(0, 1.15)
ax8.set_title('Per-Class Metrics', color='white', fontsize=10, fontweight='bold')
ax8.legend(fontsize=7, framealpha=0.3)
style_ax(ax8)

# — Summary card —
ax9 = fig.add_subplot(3, 3, 9)
ax9.set_facecolor(bg_ax)
ax9.axis('off')
acc   = accuracy_score(y_test, y_pred)
macro = report['macro avg']
ax9.text(0.5, 0.85, f'{acc*100:.1f}%',      ha='center', fontsize=36, color='#4ECDC4',
         fontweight='bold', transform=ax9.transAxes)
ax9.text(0.5, 0.65, 'Accuracy',              ha='center', fontsize=12, color='white',
         transform=ax9.transAxes)
ax9.text(0.5, 0.48, f"Macro Precision: {macro['precision']:.3f}", ha='center', fontsize=9,
         color='#aaaaaa', transform=ax9.transAxes)
ax9.text(0.5, 0.36, f"Macro Recall:    {macro['recall']:.3f}",    ha='center', fontsize=9,
         color='#aaaaaa', transform=ax9.transAxes)
ax9.text(0.5, 0.24, f"Macro F1-Score:  {macro['f1-score']:.3f}",  ha='center', fontsize=9,
         color='#aaaaaa', transform=ax9.transAxes)
ax9.text(0.5, 0.10, f"SVM · Linear Kernel · n_test={X_test.shape[0]}", ha='center', fontsize=8,
         color='#555', transform=ax9.transAxes)
for spine in ax9.spines.values():
    spine.set_edgecolor(spine_col)

plt.suptitle('Iris Dataset — SVM Classification (Linear Kernel)',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('iris_svm_results.png', dpi=150, bbox_inches='tight', facecolor=bg_fig)
print("\nPlot saved to iris_svm_results.png")
