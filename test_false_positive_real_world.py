import sys
import os
import io
import json
import urllib.parse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from app import analyze_single_url

test_cases = [
    # Modern legitimate websites & Deep paths / Profiles
    ("https://www.linkedin.com/in/sugumaran135/", "safe"),
    ("https://azure.microsoft.com/en-in/pricing/offers/ms-azr-0170p", "safe"),
    ("https://sonasoft.sonatech.ac.in/Studentdashboard/#/Menu/Acadmics", "safe"),
    ("https://swayam.gov.in/nc_details/NPTEL", "safe"),
    ("https://gemini.google.com/app", "safe"),
    ("https://github.com/torvalds/linux/tree/master/kernel", "safe"),
    ("https://www.twitch.tv/", "safe"),
    ("https://store.steampowered.com/app/872998/Stream_Games/", "safe"),
    ("https://www.rockstargames.com/", "safe"),
    ("https://leetcode.com/problemset/all/", "safe"),
    ("https://neetcode.io/practice", "safe"),
    ("https://python.org/downloads/release/python-3120/", "safe"),
    ("https://www.wikipedia.org/wiki/Phishing", "safe"),
    ("https://www.apple.com/iphone-15-pro/", "safe"),
    ("https://www.cloudflare.com/learning/access-management/what-is-trust/", "safe"),
    ("https://www.sbi.co.in/", "safe"),
    
    # Real Phishing Attacks
    ("http://paypa1-security-update.xyz/login", "phishing"),
    ("http://account-update-security-check.xyz/login.php", "phishing"),
    ("https://secure-login.paypal.com@evil-attacker.xyz/verify", "phishing"),
    ("http://192.168.1.100/verify/account.html", "phishing")
]

print("=" * 90)
print(f"{'TEST URL':<50} | {'URL ML':<8} | {'HTML ML':<8} | {'FUSED':<8} | {'FINAL':<8} | {'VERDICT':<10}")
print("=" * 90)

passed = 0
for url, expected in test_cases:
    res = analyze_single_url(url)
    u_prob = res.get('url_model_probability', 0.0)
    h_prob = res.get('html_model_probability')
    h_str = f"{h_prob:.1f}%" if h_prob is not None else "N/A"
    f_prob = res.get('fused_ml_probability', 0.0)
    final_score = res.get('final_risk_score', 0.0)
    verdict = res.get('result')
    
    is_ok = (verdict == expected) or (expected == "safe" and verdict in ["safe", "low_risk"])
    status_mark = "PASS" if is_ok else "FAIL"
    if is_ok:
        passed += 1
        
    print(f"{url:<50} | {u_prob:>6.1f}% | {h_str:>8} | {f_prob:>6.1f}% | {final_score:>6.1f}% | {verdict.upper():<10} [{status_mark}]")

print("=" * 90)
print(f"Summary: {passed}/{len(test_cases)} Passed ({passed/len(test_cases)*100:.1f}%)")
print("=" * 90)
