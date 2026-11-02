import os
import sys
import json
import time
from datetime import datetime

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import analyze_single_url

# Comprehensive test vectors across evasion strategies
ADVERSARIAL_DATASET = [
    # 1. Benign Legitimate Controls (Expected: SAFE)
    {"url": "https://google.com", "expected": "safe", "category": "Legitimate Top-Domain"},
    {"url": "https://github.com", "expected": "safe", "category": "Legitimate Top-Domain"},
    {"url": "https://wikipedia.org", "expected": "safe", "category": "Legitimate Top-Domain"},
    {"url": "https://microsoft.com", "expected": "safe", "category": "Legitimate Top-Domain"},
    {"url": "https://amazon.com", "expected": "safe", "category": "Legitimate Top-Domain"},
    {"url": "https://stackoverflow.com", "expected": "safe", "category": "Legitimate Top-Domain"},
    {"url": "https://cloudflare.com", "expected": "safe", "category": "Legitimate Top-Domain"},
    {"url": "https://python.org", "expected": "safe", "category": "Legitimate Top-Domain"},

    # 2. Typosquatting & Brand Impersonation (Expected: PHISHING / SUSPICIOUS)
    {"url": "http://paypa1-security-update.com/login", "expected": "phishing", "category": "Typosquatting Substitution"},
    {"url": "http://g00gle-account-verification.xyz/auth", "expected": "phishing", "category": "Typosquatting Substitution"},
    {"url": "http://micros0ft-support-alert.com", "expected": "phishing", "category": "Typosquatting Substitution"},
    {"url": "http://app1e-id-recover.top", "expected": "phishing", "category": "Typosquatting Substitution"},
    {"url": "http://login-apple-id-verify.xyz/auth", "expected": "phishing", "category": "Brand Combosquatting"},
    {"url": "http://secure-chase-update.com/signin", "expected": "phishing", "category": "Brand Combosquatting"},

    # 3. Raw IP Address Hosting (Expected: PHISHING / SUSPICIOUS)
    {"url": "http://192.168.1.1/admin/login.php", "expected": "phishing", "category": "Raw IP Address"},
    {"url": "http://45.33.32.156/secure/paypal/", "expected": "phishing", "category": "Raw IP Address"},
    {"url": "http://185.220.101.5/auth/banking", "expected": "phishing", "category": "Raw IP Address"},

    # 4. Userinfo '@' Obfuscation Exploits (Expected: PHISHING)
    {"url": "http://google.com@evil-phishing-host.xyz/login", "expected": "phishing", "category": "Userinfo @ Obfuscation"},
    {"url": "http://paypal.com@verify-credentials-access.top/", "expected": "phishing", "category": "Userinfo @ Obfuscation"},

    # 5. Excessive Subdomain Chains (Expected: PHISHING / SUSPICIOUS)
    {"url": "http://login.verification.account.security.portal.verify-notice.com", "expected": "phishing", "category": "Excessive Subdomains"},
    {"url": "http://secure.auth.bank.login.portal.redirect-alert.xyz", "expected": "phishing", "category": "Excessive Subdomains"},

    # 6. Risky TLD + Credential Harvest Keyword Path (Expected: PHISHING / SUSPICIOUS)
    {"url": "http://account-suspension-billing-update.xyz", "expected": "phishing", "category": "Risky TLD + Keywords"},
    {"url": "http://customer-support-portal-alert.top", "expected": "phishing", "category": "Risky TLD + Keywords"},
    {"url": "http://ebanking-verification-service.site/login", "expected": "phishing", "category": "Risky TLD + Keywords"},
]

def run_adversarial_suite():
    print("=" * 70)
    print("CHECKPHISH SOC — AUTOMATED ADVERSARIAL EVALUATION SUITE")
    print("=" * 70)
    print(f"Total test vectors: {len(ADVERSARIAL_DATASET)}")
    print("Starting automated execution against decoupled pipeline...\n")

    results = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    start_time = time.time()

    for idx, test in enumerate(ADVERSARIAL_DATASET, 1):
        url = test["url"]
        exp = test["expected"]
        cat = test["category"]
        print(f"[{idx:02d}/{len(ADVERSARIAL_DATASET):02d}] Testing: {url}")
        
        try:
            res = analyze_single_url(url)
            actual_result = res.get('result', 'unknown')
            threat_score = res.get('threat_score', 0.0)
            confidence = res.get('confidence', 0.0)
            evidence_qual = res.get('evidence_quality', 'UNKNOWN')

            detected_as_malicious = actual_result in ['phishing', 'suspicious']
            
            is_pass = False
            if exp == 'phishing':
                if detected_as_malicious:
                    tp += 1
                    is_pass = True
                    status_str = "DETECTED (TP)"
                else:
                    fn += 1
                    status_str = "MISSED (FN)"
            else:
                if not detected_as_malicious:
                    tn += 1
                    is_pass = True
                    status_str = "BENIGN (TN)"
                else:
                    fp += 1
                    status_str = "FALSE POSITIVE (FP)"

            print(f"       -> Verdict: {actual_result.upper()} ({threat_score}%) | Confidence: {confidence}% | {status_str}")

            results.append({
                "test_id": idx,
                "url": url,
                "category": cat,
                "expected": exp,
                "actual": actual_result,
                "threat_score": threat_score,
                "confidence": confidence,
                "evidence_quality": evidence_qual,
                "passed": is_pass
            })
        except Exception as e:
            print(f"       -> ERROR: {e}")
            fn += 1

    total_time = round(time.time() - start_time, 2)
    total_tests = len(ADVERSARIAL_DATASET)
    total_phish = tp + fn
    total_safe = tn + fp

    detection_rate = round((tp / total_phish) * 100, 2) if total_phish > 0 else 0
    accuracy = round(((tp + tn) / total_tests) * 100, 2)
    fpr = round((fp / total_safe) * 100, 2) if total_safe > 0 else 0
    fnr = round((fn / total_phish) * 100, 2) if total_phish > 0 else 0

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": total_time,
        "summary": {
            "total_tests": total_tests,
            "accuracy": accuracy,
            "detection_rate": detection_rate,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr
        },
        "test_results": results
    }

    os.makedirs(os.path.join(parent_dir, 'evaluation', 'results'), exist_ok=True)
    report_path = os.path.join(parent_dir, 'evaluation', 'results', 'adversarial_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 70)
    print("ADVERSARIAL TEST REPORT SUMMARY")
    print("=" * 70)
    print(f"Total Tests Executed   : {total_tests}")
    print(f"Overall Accuracy       : {accuracy}%")
    print(f"Phishing Detection Rate: {detection_rate}%")
    print(f"False Positives (FP)   : {fp} (FPR: {fpr}%)")
    print(f"False Negatives (FN)   : {fn} (FNR: {fnr}%)")
    print(f"Report JSON Saved To   : {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_adversarial_suite()