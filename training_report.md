# CheckPhish SOC Phishing Detection System — Technical Overhaul & Validation Report

**Model Release Version:** `4.0.0-generalized-calibrated`  
**System Architecture:** Decoupled Multi-Modal XGBoost & TF-IDF Linear Model with Calibrated Evidence Fusion  
**Training Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.1), Python 3.9, XGBoost 2.1.4  
**Date:** August 30, 2026  

---

## Executive Summary

The CheckPhish SOC Phishing Detection System has undergone a complete architectural and technical overhaul to achieve **generalized, leakage-free, and calibrated phishing detection** for both emails and URLs without reliance on:
- Hardcoded brand lists or company names (e.g. `CORE_BRANDS`, `gmail.com`, `paypal.com`, `linkedin.com`, `microsoft.com`)
- Whitelists, blacklists, or domain override tables
- Artificial score floors (e.g., `85.0`, `90.0`, `92.0`, `base_score = 10.0`)
- Hardcoded test cases or special-case branches

---

## 1. System Audit & Root Cause Analysis

### Identified Failure Modes in Legacy System
1. **False Positives on Short Benign Conversational Emails (`sug@gmail.com` -> 88.95% Phishing):**
   - *Root Cause:* Legacy training corpora (CEAS_08, Nigerian_Fraud, SpamAssassin) contained heavy representations of personal pronouns (`you`, `are you`, `how are`) in spam/phishing, while the only legitimate corpora (Enron) consisted exclusively of corporate memos. Short chats lacked representation in class 0.
   - *Heuristic Distortion:* Arbitrary `base_score = 10.0` and domain penalty (+35 points for domains not in `CORE_BRANDS`).
2. **Path & Numeric Feature Bias in URLs:**
   - *Root Cause:* Training datasets had legitimate samples dominated by root/homepage URLs, while phishing samples contained long paths and numeric IDs.
3. **Scraper Shielding Penalties (LinkedIn HTTP 999, Cloudflare 403):**
   - *Root Cause:* When web content scraping was blocked, legacy code treated missing HTML features as zeros or failed states, skewing HTML probabilities.

---

## 2. Multi-Modal Dataset & Leakage-Free Splitting

### A. Email Subcorpora & Balanced Deduplication
- **Corpora Integrated:**
  - CEAS_08 (39,154 raw rows)
  - Enron Corpus (29,767 raw rows)
  - Ling Spam (2,859 raw rows)
  - Nazario Phishing Corpus (1,565 raw rows)
  - Nigerian Fraud Corpus (3,332 raw rows)
  - SpamAssassin (5,809 raw rows)
  - Synthetic Security Notices with shared vocabulary (1,413 rows)
  - Augmented Robustness Corpus (9,647 rows: conversational, newsletters, receipts, shipping, adversarial overlap)
- **Leakage Prevention & Group Deduplication:**
  - Total Raw Samples: **93,546**
  - Duplicates Removed Across Corpora: **5,593**
  - Clean Unique Emails: **87,953** (48.5% Phishing / 51.5% Legitimate)
- **3-Way Data Partitioning:**
  - **70% Training Set:** 61,478 samples (Phish: 29,792, Legit: 31,686) strictly for vocabulary & estimator fitting.
  - **10% Calibration Set:** 8,766 samples (Phish: 4,245, Legit: 4,521) for probability calibration & threshold tuning.
  - **20% Unseen Test Set:** 17,709 samples (Phish: 8,611, Legit: 9,098) evaluated only once.

### B. URL & HTML Dataset (PhiUSIIL 235k)
- **Domain-Grouped Partitioning (Train ∩ Test Domains = ∅):**
  - URL Split: Train = 516,714 | Calibration = 73,239 | Unseen Test = 147,717
  - HTML Split: Train = 98,536 | Calibration = 14,040 | Unseen Test = 28,139

---

## 3. Empirical Model Performance & Probability Calibration

### Email Content Classifier (TF-IDF + Regularized Linear + Isotonic Calibration)
- **Calibration Comparison on Calibration Split:**
  - Uncalibrated Base: Brier Score = `0.0122`, Log Loss = `0.0604`, ECE = `0.0338`
  - Sigmoid (Platt): Brier Score = `0.0093`, Log Loss = `0.0353`, ECE = `0.0032`
  - **Isotonic Regression:** Brier Score = `0.0088`, Log Loss = `0.0327`, ECE = `0.0000` (Selected)
- **Final Metrics on 20% Unseen Test Set (17,709 Emails):**
  - **Accuracy:** `98.44%` (Balanced: `98.43%`)
  - **Precision:** `98.63%`
  - **Recall:** `98.15%`
  - **F1-Score:** `98.39%`
  - **ROC-AUC:** `99.87%`
  - **PR-AUC:** `99.85%`
  - **Brier Score Loss:** `0.0115`
  - **Expected Calibration Error (ECE):** `0.0028`
  - **False Positive Rate:** `1.29%` (117 / 9,098)
  - **False Negative Rate:** `1.85%` (159 / 8,611)

### Decoupled URL-Only Model (XGBoost GPU + Sigmoid Calibration)
- **Metrics on Unseen Out-of-Domain Test Split (147,717 URLs):**
  - **Accuracy:** `97.75%`
  - **Phishing Precision:** `93.71%`
  - **Phishing Recall:** `88.25%`
  - **Phishing F1-Score:** `90.90%`
  - **ROC-AUC:** `99.19%`
  - **PR-AUC:** `96.54%`
  - **Brier Score:** `0.0188`

### Decoupled HTML-Content Model (XGBoost GPU + Sigmoid Calibration)
- **Metrics on Unseen Out-of-Domain Test Split (28,139 Webpages):**
  - **Accuracy:** `98.81%`
  - **Precision:** `81.77%`
  - **Recall:** `93.44%`
  - **F1-Score:** `87.21%`
  - **ROC-AUC:** `99.55%`
  - **PR-AUC:** `95.64%`
  - **Brier Score:** `0.0090`

### Paired Multi-Modal Evidence Fusion (5,000 Unseen Test Domains)
- **Accuracy:** `99.28%`
- **Precision:** `99.44%`
- **Recall:** `83.64%`
- **F1-Score:** `90.86%`
- **ROC-AUC:** `99.99%`
- **PR-AUC:** `99.72%`
- **Brier Score:** `0.0046`

---

## 4. Probabilistic Evidence Fusion Architecture

Rather than arbitrary arithmetic penalties, the engine combines signals via log-odds likelihood shifts:

$$\text{logit}(P_{\text{final}}) = w_{\text{ML}} \cdot \text{logit}(P_{\text{ML}}) + \sum_{i} w_i \cdot \text{Evidence}_i$$

$$P_{\text{threat}} = \frac{1}{1 + e^{-\text{logit}(P_{\text{final}})}}$$

### Missing Evidence Handling Rules
| Evidence Stream | Condition | Treatment in Fusion Layer |
| :--- | :--- | :--- |
| **Authentication Headers** | Not provided (`provided=False`) | `UNKNOWN` (Shift = `0.0`, Neutral) |
| **Webpage HTML** | Scraper blocked (HTTP 999 / 403 / SPA) | `UNKNOWN` (Fallback cleanly to URL Structural Model without penalty) |
| **Email URLs** | No embedded URLs in body | `UNKNOWN` (Evaluated purely on Email ML + Content Evidence) |
| **Domain Registration** | WHOIS blocked / private | `UNKNOWN` (Neutral evidence) |

---

## 5. Verification Test Suite Results

### A. Email Scanner Functional Suite (`test_email_scanner.py`)
- **12/12 Passed (100%)**
- `sug@gmail.com` with `hi how are you` -> **0.9% SAFE** (PASS)
- `Visit the application portal and submit your password and OTP immediately.` -> **99.1% PHISHING** (PASS)
- `Your order has shipped. No action is required.` -> **0.0% SAFE** (PASS)
- `Your account will be suspended today unless you verify your password and OTP.` -> **100.0% PHISHING** (PASS)

### B. Decoupled URL Benchmark Suite (`test_checkphish.py`)
- **13/13 Passed (100%)**
- Modern Single Page Apps & Established Platforms (`rockstargames.com`, `harvard.edu`, `wikipedia.org`, `apple.com`, `github.com`) -> **SAFE**
- Obfuscated Attacks (`@`-symbol, typosquatting, raw IP) -> **CRITICAL PHISHING**

### C. Real-World False Positive Stress Test (`test_false_positive_real_world.py`)
- **19/20 Passed (95.0%)** across live HTTP/HTTPS connections and bot-shielded platforms.

---

## 6. Generated Visual Artifacts

All training and evaluation graphs are persisted in `training_graphs/`:
- `email_roc_curve.png`
- `email_pr_curve.png`
- `email_calibration_curve.png`
- `email_confusion_matrix.png`
- `email_prob_distribution.png`
- `email_feature_weights.png`
- `url_roc_curve.png`
- `url_pr_curve.png`
- `url_calibration_curve.png`
- `url_confusion_matrix.png`
- `url_feature_importance.png`
- `html_roc_curve.png`
- `html_calibration_curve.png`
- `html_confusion_matrix.png`
- `html_feature_importance.png`
