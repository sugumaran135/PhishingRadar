import os
import sys
import io
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def split_phiusiil(test_size=0.20, random_state=42, use_domain_group=False, data_dir='data'):
    csv_path = os.path.join(data_dir, 'PhiUSIIL_Phishing_URL_Dataset.csv')
    if not os.path.exists(csv_path):
        print(f"[-] File not found: {csv_path}")
        return

    print(f"[+] Loading PhiUSIIL Phishing URL Dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Total samples: {len(df):,}, Features: {df.shape[1]}")
    print(f"    Class distribution:\n{df['label'].value_counts(normalize=True).map(lambda n: f'{n:.2%}')}")

    if use_domain_group and 'Domain' in df.columns:
        print(f"\n[+] Splitting by Domain Groups (Train ∩ Test Domains = ∅)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, df['label'], df['Domain']))
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
    else:
        print(f"\n[+] Splitting using Stratified Sampling (test_size={test_size:.0%}, random_state={random_state})...")
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )

    train_path = os.path.join(data_dir, 'PhiUSIIL_train.csv')
    test_path = os.path.join(data_dir, 'PhiUSIIL_test.csv')

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[OK] Saved Training Set ({len(df_train):,} rows, {len(df_train)/len(df):.1%}) -> {train_path}")
    print(f"[OK] Saved Testing Set  ({len(df_test):,} rows, {len(df_test)/len(df):.1%}) -> {test_path}")
    print(f"    Train label distribution:\n{df_train['label'].value_counts().to_string()}")
    print(f"    Test label distribution:\n{df_test['label'].value_counts().to_string()}")

def split_dataset_phishing(test_size=0.20, random_state=42, data_dir='data'):
    csv_path = os.path.join(data_dir, 'dataset_phishing.csv')
    if not os.path.exists(csv_path):
        print(f"[-] File not found: {csv_path}")
        return

    print(f"\n[+] Loading dataset_phishing.csv from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Total samples: {len(df):,}, Features: {df.shape[1]}")
    print(f"    Class distribution:\n{df['status'].value_counts(normalize=True).map(lambda n: f'{n:.2%}')}")

    print(f"\n[+] Splitting using Stratified Sampling (test_size={test_size:.0%}, random_state={random_state})...")
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['status']
    )

    train_path = os.path.join(data_dir, 'dataset_phishing_train.csv')
    test_path = os.path.join(data_dir, 'dataset_phishing_test.csv')

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[OK] Saved Training Set ({len(df_train):,} rows, {len(df_train)/len(df):.1%}) -> {train_path}")
    print(f"[OK] Saved Testing Set  ({len(df_test):,} rows, {len(df_test)/len(df):.1%}) -> {test_path}")
    print(f"    Train status distribution:\n{df_train['status'].value_counts().to_string()}")
    print(f"    Test status distribution:\n{df_test['status'].value_counts().to_string()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split Phishing URL datasets into training and testing sets.")
    parser.add_argument('--test-size', type=float, default=0.20, help='Proportion of test set (default: 0.20 for 80/20 split)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state seed (default: 42)')
    parser.add_argument('--domain-group', action='store_true', help='Use domain-grouped splitting for out-of-domain evaluation')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing dataset files')

    args = parser.parse_args()

    print("=" * 65)
    print(" PHISHING URL DATASET - TRAIN/TEST SPLITTER")
    print("=" * 65)
    split_phiusiil(test_size=args.test_size, random_state=args.random_state, use_domain_group=args.domain_group, data_dir=args.data_dir)
    split_dataset_phishing(test_size=args.test_size, random_state=args.random_state, data_dir=args.data_dir)
    print("=" * 65)
    print(" Split completed successfully.")
    print("=" * 65)
