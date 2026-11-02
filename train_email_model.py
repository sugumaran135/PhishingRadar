import os
import sys
import io
import glob
import time
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

print("==================================================", flush=True)
print(" 🛡️ CHECKPHISH SOC — EMAIL CLASSIFIER OVERHAUL", flush=True)
print("==================================================", flush=True)

# 1. HARDWARE DETECTION
training_device = "CPU"
gpu_name = "N/A"
try:
    import torch
    if torch.cuda.is_available():
        training_device = "GPU (CUDA)"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA GPU Detected: {gpu_name} (CUDA {torch.version.cuda})", flush=True)
    else:
        print("Running on optimized multi-core CPU", flush=True)
except ImportError:
    print("Running on CPU via scikit-learn", flush=True)

# 2. LOAD INDEPENDENT RAW CORPORA + AUGMENTED SUB-CORPORA
dataset_dir = os.path.join(os.path.dirname(__file__), 'data', 'Email_dataset')
raw_filenames = [
    'CEAS_08.csv', 'Enron.csv', 'Ling.csv', 'Nazario.csv',
    'Nigerian_Fraud.csv', 'SpamAssasin.csv',
    'legit_security_notices_synthetic.csv',
    'augmented_robustness_corpus.csv'
]

raw_file_stats = []
loaded_dfs = []
total_raw = 0

print("\n📁 Loading Independent Corpora & De-Biased Datasets:", flush=True)

for fname in raw_filenames:
    fpath = os.path.join(dataset_dir, fname)
    if not os.path.exists(fpath):
        print(f"⚠️ Warning: File not found: {fname}")
        continue

    df = pd.read_csv(fpath)
    raw_count = len(df)
    total_raw += raw_count

    if 'combined_text' in df.columns:
        df['clean_text'] = df['combined_text'].fillna('')
    elif 'text_combined' in df.columns:
        df['clean_text'] = df['text_combined'].fillna('')
    elif 'subject' in df.columns and 'body' in df.columns:
        df['clean_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    elif 'body' in df.columns:
        df['clean_text'] = df['body'].fillna('')
    else:
        df['clean_text'] = ''

    if 'clean_label' in df.columns:
        df['target_label'] = df['clean_label'].astype(int)
    elif 'label' in df.columns:
        df['target_label'] = df['label'].astype(int)
    else:
        df['target_label'] = 0

    df['source_file'] = fname
    if 'template_group_id' not in df.columns:
        df['template_group_id'] = [f"{fname}_{i}" for i in range(len(df))]

    counts = df['target_label'].value_counts().to_dict()
    raw_file_stats.append({
        'file': fname,
        'rows': raw_count,
        'phishing_or_spam': counts.get(1, 0),
        'legitimate_ham': counts.get(0, 0)
    })
    print(f"  - {fname:<36}: {raw_count:>6,} rows (Label 1: {counts.get(1,0):>6,}, Label 0: {counts.get(0,0):>6,})", flush=True)
    loaded_dfs.append(df[['clean_text', 'target_label', 'source_file', 'template_group_id']])

full_df = pd.concat(loaded_dfs, ignore_index=True)

# 3. LEAKAGE AUDIT & DEDUPLICATION
print("\n==================================================", flush=True)
print(" 🧹 LEAKAGE AUDIT & DEDUPLICATION REPORT", flush=True)
print("==================================================", flush=True)

full_df = full_df[full_df['clean_text'].str.strip() != ''].copy()
full_df['norm_key'] = full_df['clean_text'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()

# Deduplicate normalized text
clean_df = full_df.drop_duplicates(subset=['norm_key']).copy()
usable_total = len(clean_df)
duplicates_removed = total_raw - usable_total

class_dist = clean_df['target_label'].value_counts().to_dict()
p_cnt = class_dist.get(1, 0)
l_cnt = class_dist.get(0, 0)

print(f"Total Raw Rows Loaded:       {total_raw:,}")
print(f"Duplicates Removed:          {duplicates_removed:,}")
print(f"Clean Unique Emails:         {usable_total:,}")
print(f"  - Label 1 (Phishing/Spam): {p_cnt:,} ({p_cnt/usable_total*100:.1f}%)")
print(f"  - Label 0 (Legitimate):    {l_cnt:,} ({l_cnt/usable_total*100:.1f}%)\n")

# 4. STRICT 3-WAY GROUP SPLITTING (70% TRAIN / 10% CALIBRATION / 20% UNSEEN TEST)
groups = clean_df['template_group_id'].values
X_text = clean_df['clean_text'].values
y_labels = clean_df['target_label'].values
source_files = clean_df['source_file'].values

# Split 80% train+calib vs 20% strictly unseen test
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_calib_idx, test_idx = next(gss_test.split(X_text, y_labels, groups))

X_tr_cal, y_tr_cal, groups_tr_cal = X_text[train_calib_idx], y_labels[train_calib_idx], groups[train_calib_idx]
X_test_text, y_test, src_test = X_text[test_idx], y_labels[test_idx], source_files[test_idx]

# Split 70% train vs 10% calibration (1/8 of train_calib)
gss_calib = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
train_idx, calib_idx = next(gss_calib.split(X_tr_cal, y_tr_cal, groups_tr_cal))

X_train_text, y_train = X_tr_cal[train_idx], y_tr_cal[train_idx]
X_calib_text, y_calib = X_tr_cal[calib_idx], y_tr_cal[calib_idx]

print("📊 Dataset Split Summary:")
print(f"  - Training Set (70%):      {len(X_train_text):,} (Phish: {np.sum(y_train==1):,}, Legit: {np.sum(y_train==0):,})")
print(f"  - Calibration Set (10%):   {len(X_calib_text):,} (Phish: {np.sum(y_calib==1):,}, Legit: {np.sum(y_calib==0):,})")
print(f"  - Unseen Test Set (20%):   {len(X_test_text):,} (Phish: {np.sum(y_test==1):,}, Legit: {np.sum(y_test==0):,})")

# 5. TF-IDF VECTORIZATION
print("\n==================================================", flush=True)
print(" 🔤 TF-IDF VECTORIZATION", flush=True)
print("==================================================", flush=True)

t0 = time.time()
vectorizer = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

print("Fitting TfidfVectorizer on 70% Training Set strictly...", flush=True)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_calib_tfidf = vectorizer.transform(X_calib_text)
X_test_tfidf = vectorizer.transform(X_test_text)

v_time = time.time() - t0
print(f"TF-IDF Vectorization complete in {v_time:.2f}s | Vocabulary size: {len(vectorizer.vocabulary_):,}")

# 6. MODEL TRAINING WITH REGULARIZATION
print("\n==================================================", flush=True)
print(" 🤖 TRAINING BASE CLASSIFIER WITH L2 REGULARIZATION", flush=True)
print("==================================================", flush=True)

train_start = time.time()
base_model = LogisticRegression(
    C=1.5,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)
base_model.fit(X_train_tfidf, y_train)
train_time = time.time() - train_start
print(f"Base model fit in {train_time:.2f}s")

# 7. PROBABILITY CALIBRATION COMPARISON (PLATT SIGMOID vs ISOTONIC ON CALIBRATION SPLIT)
print("\n==================================================", flush=True)
print(" 🎯 PROBABILITY CALIBRATION (EVALUATING ON CALIBRATION SPLIT)", flush=True)
print("==================================================", flush=True)

raw_calib_probs = base_model.predict_proba(X_calib_tfidf)[:, 1]
brier_raw_cal = brier_score_loss(y_calib, raw_calib_probs)
logloss_raw_cal = log_loss(y_calib, raw_calib_probs)
ece_raw_cal = compute_ece(y_calib, raw_calib_probs)

print(f"Uncalibrated (Base) on Calibration Set:  Brier={brier_raw_cal:.4f}, LogLoss={logloss_raw_cal:.4f}, ECE={ece_raw_cal:.4f}")

try:
    from sklearn.frozen import FrozenEstimator
    calibrator_sig = CalibratedClassifierCV(FrozenEstimator(base_model), method='sigmoid')
    calibrator_iso = CalibratedClassifierCV(FrozenEstimator(base_model), method='isotonic')
except ImportError:
    calibrator_sig = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv='prefit')
    calibrator_iso = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')

calibrator_sig.fit(X_calib_tfidf, y_calib)
calibrator_iso.fit(X_calib_tfidf, y_calib)

sig_calib_probs = calibrator_sig.predict_proba(X_calib_tfidf)[:, 1]
iso_calib_probs = calibrator_iso.predict_proba(X_calib_tfidf)[:, 1]

brier_sig_cal = brier_score_loss(y_calib, sig_calib_probs)
logloss_sig_cal = log_loss(y_calib, sig_calib_probs)
ece_sig_cal = compute_ece(y_calib, sig_calib_probs)

brier_iso_cal = brier_score_loss(y_calib, iso_calib_probs)
logloss_iso_cal = log_loss(y_calib, iso_calib_probs)
ece_iso_cal = compute_ece(y_calib, iso_calib_probs)

print(f"Sigmoid (Platt) on Calibration Set:      Brier={brier_sig_cal:.4f}, LogLoss={logloss_sig_cal:.4f}, ECE={ece_sig_cal:.4f}")
print(f"Isotonic on Calibration Set:             Brier={brier_iso_cal:.4f}, LogLoss={logloss_iso_cal:.4f}, ECE={ece_iso_cal:.4f}")

# Select best calibration method based strictly on calibration set Brier Score Loss
if brier_sig_cal <= brier_iso_cal:
    chosen_method = "Sigmoid (Platt Scaling)"
    calibrator = calibrator_sig
    best_calib_probs = sig_calib_probs
else:
    chosen_method = "Isotonic Regression"
    calibrator = calibrator_iso
    best_calib_probs = iso_calib_probs

print(f"✅ Selected Calibration Method: {chosen_method} (Lowest Brier Score Loss on Calibration Split)")

# 8. THRESHOLD OPTIMIZATION ON CALIBRATION SPLIT
print("\n==================================================", flush=True)
print(" 🎚️ THRESHOLD OPTIMIZATION ON CALIBRATION SPLIT", flush=True)
print("==================================================", flush=True)

prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_calib, best_calib_probs)
f1_arr = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-10)
best_f1_idx = np.argmax(f1_arr[:-1])
optimal_threshold = float(thresh_arr[best_f1_idx]) if len(thresh_arr) > best_f1_idx else 0.50

# 3-tier operational thresholds:
# Low Risk / Safe: < threshold_safe
# Suspicious / Review: threshold_safe to threshold_phish
# High Risk / Phishing: >= threshold_phish
threshold_safe = 0.30
threshold_phish = max(0.60, round(optimal_threshold, 2))

print(f"Optimal Binary Threshold (Max F1 on Calib): {optimal_threshold:.4f}")
print(f"Locked Operational Thresholds: SAFE < {threshold_safe:.2f} | SUSPICIOUS [{threshold_safe:.2f}, {threshold_phish:.2f}) | PHISHING >= {threshold_phish:.2f}")

# 9. FINAL ONE-PASS EVALUATION ON STRICTLY UNSEEN 20% TEST SET
print("\n==================================================", flush=True)
print(" 🧪 FINAL EVALUATION ON STRICTLY UNSEEN 20% TEST SET", flush=True)
print("==================================================", flush=True)

test_raw_probs = base_model.predict_proba(X_test_tfidf)[:, 1]
test_cal_probs = calibrator.predict_proba(X_test_tfidf)[:, 1]

y_pred_binary = (test_cal_probs >= threshold_phish).astype(int)

# Comprehensive Metrics
test_roc_auc = float(roc_auc_score(y_test, test_cal_probs))
test_pr_auc = float(average_precision_score(y_test, test_cal_probs))
test_acc = float(accuracy_score(y_test, y_pred_binary))
test_bal_acc = float(balanced_accuracy_score(y_test, y_pred_binary))
test_prec = float(precision_score(y_test, y_pred_binary, zero_division=0))
test_rec = float(recall_score(y_test, y_pred_binary, zero_division=0))
test_f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))
test_brier = float(brier_score_loss(y_test, test_cal_probs))
test_logloss = float(log_loss(y_test, test_cal_probs))
test_ece = compute_ece(y_test, test_cal_probs)

cm = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()
fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

print(f"ROC-AUC:                 {test_roc_auc:.4f}")
print(f"PR-AUC:                  {test_pr_auc:.4f}")
print(f"Accuracy:                {test_acc * 100:.2f}%")
print(f"Balanced Accuracy:       {test_bal_acc * 100:.2f}%")
print(f"Precision:               {test_prec * 100:.2f}%")
print(f"Recall:                  {test_rec * 100:.2f}%")
print(f"F1-Score:                {test_f1 * 100:.2f}%")
print(f"Brier Score Loss:        {test_brier:.4f}")
print(f"Log Loss:                {test_logloss:.4f}")
print(f"ECE (Calibration Error): {test_ece:.4f}")
print(f"False Positive Rate:     {fpr * 100:.2f}%")
print(f"False Negative Rate:     {fnr * 100:.2f}%\n")
print("Confusion Matrix:")
print(f"  True Negatives  (Legitimate): {tn:,}")
print(f"  False Positives (FP Phish):   {fp:,}")
print(f"  False Negatives (FN Missed):  {fn:,}")
print(f"  True Positives  (TP Phish):   {tp:,}\n")

# 10. GENERATE PLOTS AND METRICS ARTIFACTS
graphs_dir = os.path.join(os.path.dirname(__file__), 'training_graphs')
os.makedirs(graphs_dir, exist_ok=True)

# Graph 1: ROC Curve
plt.figure(figsize=(7, 6))
fpr_roc, tpr_roc, _ = roc_curve(y_test, test_cal_probs)
plt.plot(fpr_roc, tpr_roc, color='#2563eb', lw=2, label=f'Calibrated ROC (AUC = {test_roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--')
plt.title('Email Model ROC Curve (Unseen 20% Test Set)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
roc_plot_path = os.path.join(graphs_dir, 'email_roc_curve.png')
plt.savefig(roc_plot_path, dpi=200)
plt.close()

# Graph 2: Precision-Recall Curve
plt.figure(figsize=(7, 6))
p_curve, r_curve, _ = precision_recall_curve(y_test, test_cal_probs)
plt.plot(r_curve, p_curve, color='#16a34a', lw=2, label=f'Calibrated PR (PR-AUC = {test_pr_auc:.4f})')
plt.title('Email Model Precision-Recall Curve (Unseen 20% Test Set)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left')
plt.tight_layout()
pr_plot_path = os.path.join(graphs_dir, 'email_pr_curve.png')
plt.savefig(pr_plot_path, dpi=200)
plt.close()

# Graph 3: Calibration Curve (Reliability Diagram)
plt.figure(figsize=(7, 6))
prob_true, prob_pred = calibration_curve(y_test, test_cal_probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', color='#8b5cf6', lw=2, label=f'{chosen_method} (ECE = {test_ece:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', label='Perfect Calibration')
plt.title('Email Model Calibration Curve (Reliability Diagram)')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()
calib_plot_path = os.path.join(graphs_dir, 'email_calibration_curve.png')
plt.savefig(calib_plot_path, dpi=200)
plt.close()

# Graph 4: Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.title('Email Model Confusion Matrix (Unseen Test)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
cm_plot_path = os.path.join(graphs_dir, 'email_confusion_matrix.png')
plt.savefig(cm_plot_path, dpi=200)
plt.close()

# Graph 5: Probability Distribution Histogram
plt.figure(figsize=(8, 5))
plt.hist(test_cal_probs[y_test == 0], bins=30, alpha=0.6, color='#3b82f6', label='Legitimate (Label 0)', density=True)
plt.hist(test_cal_probs[y_test == 1], bins=30, alpha=0.6, color='#ef4444', label='Phishing (Label 1)', density=True)
plt.axvline(threshold_phish, color='#dc2626', linestyle='--', label=f'Phishing Threshold ({threshold_phish:.2f})')
plt.axvline(threshold_safe, color='#16a34a', linestyle=':', label=f'Safe Threshold ({threshold_safe:.2f})')
plt.title('Calibrated Probability Distribution on Unseen Test Set')
plt.xlabel('Predicted Phishing Probability')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
dist_plot_path = os.path.join(graphs_dir, 'email_prob_distribution.png')
plt.savefig(dist_plot_path, dpi=200)
plt.close()

# Graph 6: Top Feature Weights
feat_names = np.array(vectorizer.get_feature_names_out())
coefs = base_model.coef_[0]
top_phish_idx = np.argsort(coefs)[-15:]
top_legit_idx = np.argsort(coefs)[:15]

plt.figure(figsize=(10, 6))
top_indices = np.concatenate([top_legit_idx, top_phish_idx])
top_features = feat_names[top_indices]
top_weights = coefs[top_indices]
colors = ['#22c55e' if w < 0 else '#ef4444' for w in top_weights]
plt.barh(top_features, top_weights, color=colors)
plt.title('Top 15 Legitimate (-) vs Top 15 Phishing (+) Learned Feature Weights')
plt.xlabel('Logistic Regression L2 Weight')
plt.grid(True, alpha=0.3)
plt.tight_layout()
feat_plot_path = os.path.join(graphs_dir, 'email_feature_weights.png')
plt.savefig(feat_plot_path, dpi=200)
plt.close()

print("📈 All 6 evaluation graphs saved to training_graphs/", flush=True)

# 11. SAVE MODEL ARTIFACTS
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'email_classifier.pkl')
vectorizer_path = os.path.join(models_dir, 'email_tfidf_vectorizer.pkl')
meta_path = os.path.join(models_dir, 'email_model_metadata.json')

joblib.dump(calibrator, model_path)
joblib.dump(vectorizer, vectorizer_path)

metadata = {
    'training_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'hardware': {
        'training_device': training_device,
        'gpu_name': gpu_name
    },
    'dataset_breakdown': {
        'files_used': raw_filenames,
        'file_stats': raw_file_stats,
        'total_raw_rows': total_raw,
        'duplicates_removed': duplicates_removed,
        'clean_unique_emails': usable_total,
        'train_samples_70pct': len(X_train_text),
        'calib_samples_10pct': len(X_calib_text),
        'test_samples_20pct': len(X_test_text),
        'class_distribution': {'phishing_1': p_cnt, 'legitimate_0': l_cnt}
    },
    'model_architecture': {
        'vectorizer': 'TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)',
        'vocab_size': len(vectorizer.vocabulary_),
        'classifier': 'LogisticRegression(C=1.5, class_weight=balanced, L2 regularization)',
        'calibration_method': chosen_method,
        'operational_thresholds': {
            'safe_below': threshold_safe,
            'suspicious_range': [threshold_safe, threshold_phish],
            'phishing_above': threshold_phish
        }
    },
    'calibration_selection_metrics': {
        'uncalibrated_brier': round(brier_raw_cal, 4),
        'sigmoid_platt_brier': round(brier_sig_cal, 4),
        'isotonic_brier': round(brier_iso_cal, 4),
        'chosen_method': chosen_method
    },
    'unseen_test_metrics': {
        'roc_auc': round(test_roc_auc, 4),
        'pr_auc': round(test_pr_auc, 4),
        'accuracy': round(test_acc, 4),
        'balanced_accuracy': round(test_bal_acc, 4),
        'precision': round(test_prec, 4),
        'recall': round(test_rec, 4),
        'f1_score': round(test_f1, 4),
        'brier_score_loss': round(test_brier, 4),
        'log_loss': round(test_logloss, 4),
        'expected_calibration_error_ece': round(test_ece, 4),
        'false_positive_rate': round(fpr, 4),
        'false_negative_rate': round(fnr, 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    },
    'training_time_seconds': round(train_time + v_time, 2)
}

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("\n==================================================", flush=True)
print(" 💾 SAVED RETRAINED EMAIL MODEL ARTIFACTS", flush=True)
print("==================================================", flush=True)
print(f"  - Classifier: {model_path}")
print(f"  - Vectorizer: {vectorizer_path}")
print(f"  - Metadata:   {meta_path}")
print("==================================================", flush=True)
