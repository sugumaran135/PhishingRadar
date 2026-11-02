import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

def run_ml_benchmark(dataset_csv_path='dataset/url_features.csv'):
    print("=" * 75)
    print("CHECKPHISH SOC — MULTI-MODEL COMPARATIVE BENCHMARK")
    print("=" * 75)

    # 1. Dataset handling (uses real dataset if found, otherwise synthetic split)
    if not os.path.exists(dataset_csv_path):
        print(f"Dataset file '{dataset_csv_path}' not found.")
        print("Running benchmark on simulated held-out validation distribution...")
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(2000, 15), columns=[f'feat_{i}' for i in range(15)])
        y = np.random.choice([0, 1], size=2000, p=[0.5, 0.5])
    else:
        df = pd.read_csv(dataset_csv_path)
        y = df['label'].values
        X = df.drop(columns=['label', 'url'], errors='ignore')

    # 2. Strict 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training Samples : {len(X_train):,}")
    print(f"Held-Out Test Set: {len(X_test):,}")
    print(f"Feature Dimension: {X_train.shape[1]} features\n")

    # 3. Model Suite Definition
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1),
        "SVM (Linear/RBF)": SVC(probability=True, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, random_state=42, verbose=-1),
        "XGBoost (Active)": xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }

    benchmark_records = []

    print(f"{'Model':<22} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 75)

    for name, clf in models.items():
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
        else:
            probs = clf.decision_function(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        try:
            roc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
        except Exception:
            roc, pr_auc = 0.0, 0.0

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        benchmark_records.append({
            "model": name,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(roc * 100, 2),
            "pr_auc": round(pr_auc * 100, 2),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        })

        print(f"{name:<22} | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {f1*100:>7.2f}% | {roc*100:>7.2f}%")

    # 4. Save results to evaluation/results/benchmark_comparison.json
    results_dir = os.path.join(parent_dir, 'evaluation', 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, 'benchmark_comparison.json')
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_sample_size": len(X_test),
            "models": benchmark_records
        }, f, indent=4)

    print("-" * 75)
    print(f"Benchmark completed successfully. Saved to: {out_file}\n")

if __name__ == "__main__":
    run_ml_benchmark()