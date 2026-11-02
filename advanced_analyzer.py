import re
from urllib.parse import urlparse
import difflib

# Top high-value targeted brands to monitor
PROTECTED_BRANDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix", "facebook",
    "instagram", "chase", "wellsfargo", "bankofamerica", "binance", "coinbase",
    "dropbox", "linkedin", "adobe", "twitter", "whatsapp", "telegram", "yahoo", "github"
]

def analyze_brand_impersonation(domain: str):
    """
    Evaluates brand lookalike, typosquatting, and character substitution risks.
    Recognizes legitimate brand parent domains and returns 0.0% risk.
    """
    clean = domain.lower().replace('www.', '').split(':')[0]
    parts = clean.split('.')

    if len(parts) >= 3 and parts[-2] in ['co', 'com', 'org', 'edu', 'gov', 'net']:
        domain_body = parts[-3]
    elif len(parts) >= 2:
        domain_body = parts[-2]
    else:
        domain_body = parts[0] if parts else clean

    normalized_body = (
        domain_body.replace('0', 'o')
                   .replace('1', 'l')
                   .replace('3', 'e')
                   .replace('4', 'a')
                   .replace('5', 's')
                   .replace('-', '')
    )

    best_brand = "unknown"
    highest_sim = 0.0
    detected_swap = False

    for brand in PROTECTED_BRANDS:
        sim_raw = difflib.SequenceMatcher(None, brand, domain_body).ratio()
        sim_norm = difflib.SequenceMatcher(None, brand, normalized_body).ratio()

        current_max = max(sim_raw, sim_norm)
        if current_max > highest_sim:
            highest_sim = current_max
            best_brand = brand
            if sim_norm > sim_raw and sim_norm >= 0.75:
                detected_swap = True

    sim_pct = round(highest_sim * 100, 1)
    is_genuine_brand = (domain_body == best_brand)

    if is_genuine_brand:
        brand_risk = 0.0
        status = "LEGITIMATE_BRAND_DOMAIN"
    elif sim_pct >= 80.0 or (detected_swap and sim_pct >= 70.0):
        brand_risk = 90.0
        status = "HIGH_BRAND_IMPERSONATION"
    elif sim_pct >= 60.0:
        brand_risk = 50.0
        status = "SUSPICIOUS_BRAND_LOOKALIKE"
    else:
        brand_risk = 5.0
        status = "NO_BRAND_MATCH"

    return {
        'target_brand': best_brand,
        'similarity_score': 0.0 if is_genuine_brand else sim_pct,
        'raw_match_similarity': sim_pct,
        'has_character_substitution': detected_swap,
        'brand_risk': brand_risk,
        'status': status
    }


def analyze_html_security(html_content: str, base_url: str):
    """
    Forensic DOM and heuristic inspector.
    Distinguishes legitimate security behavior (CSRF, same-domain JS navigations, CSP/HSTS)
    from true malicious credential-harvesting and exfiltration attempts.
    """
    if not html_content:
        return {
            'has_password_field': False,
            'external_form_action': False,
            'hidden_input_count': 0,
            'iframe_count': 0,
            'content_security_risk': 0.0,
            'signals': [],
            'positive_signals': [],
            'available': False
        }

    soup_text = html_content.lower()
    parsed_base = urlparse(base_url)
    base_netloc = parsed_base.netloc.lower().split(':')[0]
    base_parts = base_netloc.replace('www.', '').split('.')
    reg_base = ".".join(base_parts[-2:]) if len(base_parts) >= 2 else base_netloc

    signals = []
    positive_signals = []
    risk_score = 0.0

    # 1. Password Field & Form Action Analysis
    has_password = bool(re.search(r'<input[^>]+type=["\']password["\']', soup_text))
    
    form_actions = re.findall(r'<form[^>]+action=["\']([^"\']*)["\']', soup_text)
    external_exfil = False
    for action in form_actions:
        action_clean = action.strip().lower()
        if action_clean.startswith(('http://', 'https://')):
            act_netloc = urlparse(action_clean).netloc.lower().split(':')[0].replace('www.', '')
            act_parts = act_netloc.split('.')
            act_reg = ".".join(act_parts[-2:]) if len(act_parts) >= 2 else act_netloc
            if act_reg and act_reg != reg_base:
                external_exfil = True
                break

    if has_password and external_exfil:
        risk_score += 85.0
        signals.append("CRITICAL: Password input submits directly to an unauthorized external domain.")
    elif external_exfil:
        risk_score += 45.0
        signals.append("WARNING: Form submissions routed to an external domain.")
    elif has_password:
        risk_score += 5.0
        signals.append("Standard login/credential input detected.")
    else:
        positive_signals.append("No credential-harvesting forms detected.")
        positive_signals.append("No suspicious external form submissions.")

    # 2. Context-Aware Hidden Input Inspection
    hidden_inputs = re.findall(r'<input[^>]+type=["\']hidden["\'][^>]*>', soup_text)
    hidden_count = len(hidden_inputs)
    
    safe_token_patterns = ['csrf', 'token', 'xsrf', 'nonce', 'state', 'session', 'auth_tag', 'viewstate', 'guid', 'authenticity']
    cred_patterns = ['user', 'login', 'pass', 'account', 'email', 'pin', 'ssn', 'credential', 'secret']
    
    suspicious_hidden = 0
    for h in hidden_inputs:
        is_safe = any(token in h for token in safe_token_patterns)
        has_cred = any(cred in h for cred in cred_patterns)
        if not is_safe and has_cred:
            suspicious_hidden += 1

    if suspicious_hidden > 0:
        risk_score += min(suspicious_hidden * 10.0, 25.0)
        signals.append(f"Suspicious hidden credential fields detected ({suspicious_hidden} found).")
    else:
        positive_signals.append(f"Hidden inputs verified benign ({hidden_count} standard session/CSRF tokens).")

    # 3. Destination-Aware JavaScript Navigation / Redirects
    js_nav_targets = re.findall(r'(?:window\.location(?:\.href)?|location\.replace|location\.assign)\s*=\s*["\']([^"\']+)["\']', soup_text)
    external_js_redirect = False
    
    for target in js_nav_targets:
        target_clean = target.strip()
        if target_clean.startswith(('http://', 'https://')):
            tgt_netloc = urlparse(target_clean).netloc.lower().split(':')[0].replace('www.', '')
            tgt_parts = tgt_netloc.split('.')
            tgt_reg = ".".join(tgt_parts[-2:]) if len(tgt_parts) >= 2 else tgt_netloc
            if tgt_reg and tgt_reg != reg_base:
                external_js_redirect = True
                break

    if external_js_redirect:
        risk_score += 40.0
        signals.append("Client-side JavaScript navigates to an external unauthorized domain.")
    else:
        positive_signals.append("No suspicious external JavaScript navigation detected.")

    # 4. Security Controls Detection (CSP, HSTS, X-Frame-Options)
    has_csp_meta = 'http-equiv=["\']content-security-policy["\']' in soup_text
    has_xfo_meta = 'http-equiv=["\']x-frame-options["\']' in soup_text
    if has_csp_meta or has_xfo_meta:
        positive_signals.append("Client-side defense headers/meta tags detected (CSP/XFO).")

    # 5. IFrames Inspection
    iframes = re.findall(r'<iframe[^>]*>', soup_text)
    iframe_count = len(iframes)
    if iframe_count >= 3:
        risk_score += 15.0
        signals.append(f"Multiple iframe tags detected ({iframe_count} present).")

    final_security_risk = round(min(max(risk_score, 0.0), 100.0), 1)

    return {
        'has_password_field': has_password,
        'external_form_action': external_exfil,
        'hidden_input_count': hidden_count,
        'iframe_count': iframe_count,
        'content_security_risk': final_security_risk,
        'signals': signals,
        'positive_signals': positive_signals,
        'available': True
    }


def analyze_redirect_chain(chain_history: list, final_url: str = ""):
    """
    Evaluates HTTP redirect hops.
    Accepts list of strings or list of dicts.
    Canonical redirects (e.g., domain.com -> www.domain.com) yield 0.0% risk.
    """
    if not chain_history or len(chain_history) <= 1:
        return {
            'redirect_count': 0,
            'cross_domain_hops': 0,
            'redirect_risk': 0.0,
            'chain': chain_history or []
        }

    redirect_count = len(chain_history) - 1
    cross_domain_hops = 0

    def get_registered_domain(u):
        if isinstance(u, dict):
            u = u.get('url', '')
        if not u or not isinstance(u, str):
            return ""
        netloc = urlparse(u).netloc.lower().split(':')[0]
        parts = netloc.split('.')
        if len(parts) >= 3 and parts[-2] in ['co', 'com', 'org', 'edu', 'gov', 'net']:
            return ".".join(parts[-3:])
        elif len(parts) >= 2:
            return ".".join(parts[-2:])
        return netloc

    first_domain = get_registered_domain(chain_history[0])

    for step in chain_history[1:]:
        step_domain = get_registered_domain(step)
        if step_domain and step_domain != first_domain:
            cross_domain_hops += 1

    if cross_domain_hops == 0:
        redirect_risk = 0.0
    else:
        redirect_risk = round(min(cross_domain_hops * 30.0 + (redirect_count * 5.0), 90.0), 1)

    return {
        'redirect_count': redirect_count,
        'cross_domain_hops': cross_domain_hops,
        'redirect_risk': redirect_risk,
        'chain': chain_history
    }


# Aliases to guarantee all app.py imports resolve cleanly
inspect_html_security = analyze_html_security

import os
import requests

GSB_API_KEY = os.environ.get("GSB_API_KEY", "")

def check_google_safe_browsing(url: str) -> dict:
    """
    Checks URL against Google Safe Browsing API v4.
    Returns whether it is listed, along with threat types.
    """
    if not GSB_API_KEY or GSB_API_KEY == "YOUR_GOOGLE_SAFE_BROWSING_API_KEY":
        return {"status": "SKIPPED", "is_flagged": False, "threats": []}

    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GSB_API_KEY}"
    payload = {
        "client": {
            "clientId": "checkphish-soc-engine",
            "clientVersion": "1.0.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            if matches:
                threats = list({m.get("threatType") for m in matches})
                return {
                    "status": "FLAGGED",
                    "is_flagged": True,
                    "threats": threats,
                    "evidence_line": f"Listed on Google Safe Browsing as {', '.join(threats)}."
                }
            return {"status": "CLEAN", "is_flagged": False, "threats": []}
    except Exception as e:
        print(f"⚠️ Safe Browsing API Error: {e}")

    return {"status": "UNAVAILABLE", "is_flagged": False, "threats": []}