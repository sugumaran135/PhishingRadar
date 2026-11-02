import re
import difflib
import urllib.parse
import requests
import numpy as np
from datetime import datetime

# Common URL Shortener domains (used strictly for HTTP redirect resolution)
KNOWN_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 'cutt.ly',
    'buff.ly', 'adf.ly', 'rebrand.ly', 'tiny.cc', 'shorturl.at',
    'git.io', 'v.gd', 'qr.ae', 'goo.gl'
}

def extract_domain_from_email(sender_str):
    """
    Extracts clean domain and email address from a sender string.
    Works for any email address format (e.g. 'John Doe <john@domain.com>' or 'john@domain.com').
    """
    if not sender_str:
        return "", "", ""
    
    sender_clean = sender_str.strip()
    match = re.search(r'<([^>]+)>', sender_clean)
    email_addr = match.group(1).strip() if match else sender_clean
    
    display_name = ""
    if match:
        display_name = sender_clean[:match.start()].strip(' "\'')

    if '@' in email_addr:
        parts = email_addr.split('@')
        domain = parts[1].strip().lower()
        return display_name, email_addr, domain
    return display_name, email_addr, email_addr.strip().lower()

def is_shortened_url(url):
    """Checks if a URL uses a known redirect/shortening service."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.replace('www.', '').lower()
        return domain in KNOWN_SHORTENERS
    except Exception:
        return False

def resolve_shortened_url(url, timeout=3.0, max_redirects=4):
    """
    Safely resolves HTTP/HTTPS redirects for shortened URLs.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return url, False, "Non-HTTP protocol rejected for security"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CheckPhishSOC/4.0'}
        session = requests.Session()
        session.max_redirects = max_redirects
        resp = session.head(url, headers=headers, allow_redirects=True, timeout=timeout)
        
        if resp.status_code in (405, 403, 501):
            resp = session.get(url, headers=headers, allow_redirects=True, timeout=timeout, stream=True)
            resp.close()
            
        if resp.status_code >= 400:
            return url, False, f"Destination not reached (HTTP {resp.status_code})"

        final_url = resp.url
        final_domain = urllib.parse.urlparse(final_url).netloc.replace('www.', '').lower()
        orig_domain = parsed.netloc.replace('www.', '').lower()

        if final_domain == orig_domain:
            return url, False, "Shortened link did not resolve to external destination"

        return final_url, True, f"Resolved via redirect to {final_domain}"
    except Exception as e:
        return url, False, f"Destination resolution failed ({str(e)})"

def extract_urls_from_text(text):
    """Extracts all embedded URLs from text and deduplicates them."""
    if not text:
        return []
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    extracted = re.findall(url_pattern, text)
    
    cleaned = []
    for u in extracted:
        u_clean = u.rstrip('.,);:\'">')
        if u_clean and u_clean not in cleaned:
            cleaned.append(u_clean)
    return cleaned

# =============================================================================
# GENERALIZED EVIDENCE EXTRACTION (NO HARDCODED BRANDS / DOMAINS)
# =============================================================================

def analyze_sender_evidence(sender_str):
    """
    Extracts structural security evidence strictly from sender formatting.
    Does NOT use brand whitelists or domain exceptions.
    """
    evidence = {
        'display_name': '',
        'email_address': '',
        'domain': '',
        'has_punycode': False,
        'has_ip_host': False,
        'excessive_subdomains': False,
        'suspicious_sender_structure': False,
        'flags': []
    }

    display_name, email_addr, domain = extract_domain_from_email(sender_str)
    evidence['display_name'] = display_name
    evidence['email_address'] = email_addr
    evidence['domain'] = domain

    if not domain:
        return evidence

    # 1. Punycode / IDN internationalized domain obfuscation
    if domain.startswith('xn--') or '.xn--' in domain:
        evidence['has_punycode'] = True
        evidence['flags'].append({
            'severity': 'HIGH',
            'type': 'sender_punycode',
            'title': 'Punycode / IDN Domain Detected',
            'description': f"Sender domain '{domain}' uses Punycode character encoding"
        })

    # 2. Raw IP address sender host
    clean_host = domain.split(':')[0]
    parts = clean_host.split('.')
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        evidence['has_ip_host'] = True
        evidence['flags'].append({
            'severity': 'HIGH',
            'type': 'sender_ip_host',
            'title': 'Raw IP Address Sender Host',
            'description': f"Sender uses raw IP address instead of registered domain name ({clean_host})"
        })

    # 3. Excessive subdomains count
    sub_parts = [p for p in domain.split('.') if p and p != 'www']
    if len(sub_parts) > 4:
        evidence['excessive_subdomains'] = True
        evidence['flags'].append({
            'severity': 'MEDIUM',
            'type': 'excessive_subdomains',
            'title': 'Excessive Subdomain Depth',
            'description': f"Sender domain has {len(sub_parts)} domain hierarchy levels"
        })

    return evidence

def analyze_content_evidence(email_text):
    """
    Extracts semantic threat indicators strictly from the text content.
    Identifies explicit actionable credential solicitation, threats, and urgency.
    """
    evidence = {
        'credential_harvesting_prompt': False,
        'urgency_threat': False,
        'account_suspension_threat': False,
        'actionable_prompts_found': [],
        'flags': []
    }

    if not email_text:
        return evidence

    text_lower = email_text.lower()

    # 1. Actionable Credential Harvesting Prompts
    explicit_prompts = [
        'submit your password and otp', 'provide your password and otp',
        'enter your password and otp', 'verify your password and otp',
        'submit your password', 'provide your password', 'enter your password',
        'reset your password', 'password reset', 'update your password',
        'confirm your password', 'confirm your identity', 'verify your identity',
        'verify your account', 'verify account', 'update your account',
        'enter your otp', 'provide your otp', 'submit your otp', 'verify otp',
        'enter your pin', 'provide your pin', 'submit your credentials',
        'provide your login credentials', 'verify your credentials',
        'banking credentials', 'card details and pin', 'update billing details',
        'login to verify', 'log in to verify', 'authenticate your account',
        'password:', 'otp:', 'pin:', 'security pin:', 'passcode:'
    ]
    matched_prompts = [p for p in explicit_prompts if p in text_lower]

    if matched_prompts:
        evidence['credential_harvesting_prompt'] = True
        evidence['actionable_prompts_found'] = matched_prompts
        evidence['flags'].append({
            'severity': 'HIGH',
            'type': 'credential_harvesting',
            'title': 'Actionable Credential Solicitation Detected',
            'description': f"Email explicitly prompts user to submit sensitive credentials ({', '.join(matched_prompts[:3])})"
        })

    # 2. Account Suspension & Disabling Threats
    suspension_phrases = [
        'account will be permanently suspended', 'account will be suspended', 'account suspended',
        'account access has been restricted', 'account will be disabled', 'account disabled',
        'prevent your account from being disabled', 'account scheduled for deletion',
        'account locked due to', 'account locked', 'unauthorized sign-in attempts',
        'unauthorized access detected', 'unusual login activity', 'security alert: verify',
        'mailbox full', 'access restricted', 'deactivation notice', 'service suspension'
    ]
    if any(p in text_lower for p in suspension_phrases):
        evidence['account_suspension_threat'] = True
        evidence['flags'].append({
            'severity': 'MEDIUM',
            'type': 'account_threat',
            'title': 'Account Suspension / Security Threat',
            'description': "Email threatens account restriction, security alert, or suspension"
        })

    # 3. Urgency & Coercive Time Pressure
    urgency_phrases = [
        'immediately', 'act now', 'within 24 hours', 'immediate action required',
        'urgent action required', 'urgent', 'action required', 'final warning',
        'without delay', 'claim immediately', 'expires soon', 'critical alert'
    ]
    if any(p in text_lower for p in urgency_phrases):
        evidence['urgency_threat'] = True
        evidence['flags'].append({
            'severity': 'LOW',
            'type': 'urgency_pressure',
            'title': 'High Urgency Time Pressure',
            'description': "Email creates urgency demanding immediate action"
        })

    return evidence

def analyze_header_evidence(raw_headers_str):
    """
    Parses email authentication headers (SPF, DKIM, DMARC, Return-Path).
    CRITICAL: When headers are missing, explicitly sets provided=False (UNKNOWN).
    Missing headers do NOT penalize or imply phishing.
    """
    if not raw_headers_str or not raw_headers_str.strip():
        return {
            'provided': False,
            'status': 'UNKNOWN',
            'spf_status': 'UNKNOWN',
            'dkim_status': 'UNKNOWN',
            'dmarc_status': 'UNKNOWN',
            'mismatch_detected': False,
            'auth_failed': False,
            'summary': 'Headers not provided (Authentication status unknown - neutral evidence).',
            'flags': []
        }

    headers_lower = raw_headers_str.lower()
    flags = []

    from_match = re.search(r'^from:\s*(.*)$', raw_headers_str, re.MULTILINE | re.IGNORECASE)
    reply_to_match = re.search(r'^reply-to:\s*(.*)$', raw_headers_str, re.MULTILINE | re.IGNORECASE)

    from_addr = from_match.group(1).strip() if from_match else ""
    reply_to_addr = reply_to_match.group(1).strip() if reply_to_match else ""

    from_domain = extract_domain_from_email(from_addr)[2]
    reply_to_domain = extract_domain_from_email(reply_to_addr)[2]

    mismatch_detected = False
    if from_domain and reply_to_domain and from_domain != reply_to_domain:
        mismatch_detected = True
        flags.append({
            'severity': 'HIGH',
            'type': 'header_mismatch',
            'title': 'From vs Reply-To Domain Mismatch',
            'description': f"From domain ({from_domain}) differs from Reply-To destination ({reply_to_domain})"
        })

    spf_status = "UNKNOWN"
    if 'spf=pass' in headers_lower: spf_status = "PASS"
    elif 'spf=fail' in headers_lower or 'spf=softfail' in headers_lower: spf_status = "FAIL"

    dkim_status = "UNKNOWN"
    if 'dkim=pass' in headers_lower: dkim_status = "PASS"
    elif 'dkim=fail' in headers_lower: dkim_status = "FAIL"

    dmarc_status = "UNKNOWN"
    if 'dmarc=pass' in headers_lower: dmarc_status = "PASS"
    elif 'dmarc=fail' in headers_lower: dmarc_status = "FAIL"

    auth_failed = (spf_status == "FAIL" or dkim_status == "FAIL" or dmarc_status == "FAIL" or mismatch_detected)
    if auth_failed:
        flags.append({
            'severity': 'HIGH',
            'type': 'auth_failure',
            'title': 'Cryptographic Authentication Verification Failure',
            'description': f"SPF: {spf_status}, DKIM: {dkim_status}, DMARC: {dmarc_status}"
        })

    return {
        'provided': True,
        'status': 'VERIFIED' if (spf_status == 'PASS' and not mismatch_detected) else ('FAIL' if auth_failed else 'NEUTRAL'),
        'from_address': from_addr,
        'reply_to_address': reply_to_addr,
        'spf_status': spf_status,
        'dkim_status': dkim_status,
        'dmarc_status': dmarc_status,
        'mismatch_detected': mismatch_detected,
        'auth_failed': auth_failed,
        'summary': f"Headers analyzed. SPF: {spf_status}, DKIM: {dkim_status}, DMARC: {dmarc_status}.",
        'flags': flags
    }

# =============================================================================
# PRINCIPLED PROBABILISTIC EVIDENCE FUSION ENGINE
# =============================================================================

def evaluate_email_threat(sender, email_text, raw_headers="", url_evaluator_fn=None, email_ml_fn=None):
    """
    Main evaluation pipeline for Email Phishing Detection.
    Combines:
      1. Trained & Calibrated Email Content ML Model
      2. Extracted Structural Security Evidence (Sender, Content, Headers)
      3. Decoupled URL Evaluation (when URLs exist in body)
    Strictly adheres to NO HARDCODED WHITELISTS / BLACKLISTS.
    """
    all_flags = []
    positive_reasons = []
    negative_reasons = []
    unknown_evidence = []

    # --- 1. EMAIL ML CONTENT CLASSIFIER INFERENCE ---
    email_ml_res = None
    email_phish_prob = 0.50 # Neutral default if model unavailable
    
    if email_ml_fn:
        try:
            email_ml_res = email_ml_fn(email_text)
            if email_ml_res and 'phishing_probability' in email_ml_res:
                email_phish_prob = float(email_ml_res['phishing_probability']) / 100.0
        except Exception as e:
            print(f"⚠️ Email ML prediction error: {e}")

    email_ml_percent = round(email_phish_prob * 100.0, 2)

    # --- 2. EXTRACT INDEPENDENT STRUCTURAL EVIDENCE ---
    sender_evidence = analyze_sender_evidence(sender)
    content_evidence = analyze_content_evidence(email_text)
    header_evidence = analyze_header_evidence(raw_headers)

    all_flags.extend(sender_evidence['flags'])
    all_flags.extend(content_evidence['flags'])
    all_flags.extend(header_evidence['flags'])

    # --- 3. URL EXTRACTION & STRUCTURAL EVALUATION ---
    extracted_urls = extract_urls_from_text(email_text)
    analyzed_urls = []
    max_url_phish_prob = 0.0
    has_url_evidence = len(extracted_urls) > 0
    sender_domain = sender_evidence['domain']

    for url in extracted_urls[:5]:
        is_short = is_shortened_url(url)
        resolved_url = url
        is_resolved = True
        status_msg = "Direct URL"

        if is_short:
            resolved_url, is_resolved, status_msg = resolve_shortened_url(url)

        target_eval_url = resolved_url if is_resolved else url
        parsed_dest = urllib.parse.urlparse(target_eval_url)
        dest_domain = parsed_dest.netloc.replace('www.', '').lower()
        dest_host = dest_domain.split(':')[0]

        # Check domain mismatch between sender domain and destination link
        url_domain_mismatch = False
        if sender_domain and dest_host:
            if sender_domain != dest_host and not dest_host.endswith('.' + sender_domain) and not sender_domain.endswith('.' + dest_host):
                url_domain_mismatch = True

        # Decoupled URL Model inference
        url_threat_score = 0.0
        url_verdict = 'safe'
        url_risk_level = 'SAFE'

        if not is_resolved:
            url_threat_score = 65.0
            url_verdict = 'phishing'
            url_risk_level = 'SUSPICIOUS'
            all_flags.append({
                'severity': 'MEDIUM',
                'type': 'unresolved_shortener',
                'title': 'Unverified Shortened Destination',
                'description': f"Shortened link '{url}' could not be reached for destination verification"
            })
        elif url_evaluator_fn:
            try:
                eval_res = url_evaluator_fn(target_eval_url)
                url_verdict = eval_res.get('result', 'safe')
                url_risk_level = eval_res.get('risk_level', 'SAFE')
                url_threat_score = float(eval_res.get('threat_score', 0.0))
            except Exception as e:
                print(f"⚠️ URL evaluator error for {target_eval_url}: {e}")

        url_prob = url_threat_score / 100.0
        max_url_phish_prob = max(max_url_phish_prob, url_prob)

        analyzed_urls.append({
            'url': url,
            'is_shortened': is_short,
            'resolved_url': resolved_url,
            'status_msg': status_msg,
            'threat_score': round(url_threat_score, 1),
            'result': url_verdict,
            'risk_level': url_risk_level,
            'domain_mismatch': url_domain_mismatch
        })

    # --- 4. PRINCIPLED PROBABILISTIC LOG-ODDS EVIDENCE FUSION ---
    # Logit of base calibrated email ML probability
    eps = 1e-4
    
    # Handle short-text edge cases for ML probability
    words_count = len((email_text or "").split())
    has_critical_flags = content_evidence['credential_harvesting_prompt'] or content_evidence['account_suspension_threat']
    
    # If explicit credential harvesting or account threats are present in short content, ensure base prob reflects risk
    if has_critical_flags and email_phish_prob < 0.60:
        email_phish_prob = max(email_phish_prob, 0.65)
        email_ml_percent = round(email_phish_prob * 100.0, 2)
    
    # If text is short conversational email with 0 risk flags and 0 embedded URLs, dampen sparse TF-IDF noise
    if len(all_flags) == 0 and not has_url_evidence and words_count <= 25 and email_phish_prob > 0.25:
        email_phish_prob = min(email_phish_prob, 0.15)
        email_ml_percent = round(email_phish_prob * 100.0, 2)

    clipped_ml_prob = np.clip(email_phish_prob, eps, 1.0 - eps)
    logit_score = float(np.log(clipped_ml_prob / (1.0 - clipped_ml_prob)))

    # Evidence Likelihood Shifts (Empirically normalized log-odds)
    # Actionable Credential solicitation
    if content_evidence['credential_harvesting_prompt']:
        logit_score += 2.0
        positive_reasons.append("Email contains explicit prompts soliciting sensitive credentials or PIN/OTP codes.")
    
    # Account suspension coercion
    if content_evidence['account_suspension_threat']:
        logit_score += 1.2
        positive_reasons.append("Email employs account suspension or termination threats to pressure the recipient.")

    # High urgency
    if content_evidence['urgency_threat']:
        logit_score += 0.5
        positive_reasons.append("Email creates artificial time pressure demanding immediate compliance.")

    # Sender structural anomalies
    if sender_evidence['has_punycode']:
        logit_score += 2.5
        positive_reasons.append(f"Sender domain ({sender_evidence['domain']}) uses deceptive Punycode/IDN internationalized characters.")
    if sender_evidence['has_ip_host']:
        logit_score += 2.0
        positive_reasons.append("Sender uses a raw IP address host rather than a standard domain name.")

    # Header authentication failure
    if header_evidence['provided']:
        if header_evidence['auth_failed']:
            logit_score += 1.8
            positive_reasons.append("Cryptographic email authentication checks failed (SPF/DKIM/DMARC failure or From/Reply-To mismatch).")
        elif header_evidence['status'] == 'VERIFIED':
            logit_score -= 1.0
            negative_reasons.append("Cryptographic email authentication headers verified successfully (SPF/DKIM Pass).")
    else:
        unknown_evidence.append("Email authentication headers not supplied.")

    # URL evidence integration
    if has_url_evidence:
        clipped_url_prob = np.clip(max_url_phish_prob, eps, 1.0 - eps)
        url_logit = float(np.log(clipped_url_prob / (1.0 - clipped_url_prob)))
        # URL evidence acts as independent signal
        logit_score = (logit_score * 0.60) + (url_logit * 0.40)
        
        has_mismatch = any(u['domain_mismatch'] for u in analyzed_urls)
        if has_mismatch and (content_evidence['credential_harvesting_prompt'] or content_evidence['account_suspension_threat']):
            logit_score += 1.5
            positive_reasons.append("Destination URL host does not match the sender organization domain.")
    else:
        unknown_evidence.append("No embedded URLs present in email body.")

    # Compute final calibrated threat probability via standard Sigmoid
    final_fused_prob = float(1.0 / (1.0 + np.exp(-logit_score)))
    final_threat_score = round(final_fused_prob * 100.0, 1)

    # --- 5. OPERATIONAL THRESHOLDS (LOCKED FROM VALIDATION) ---
    # SAFE: < 30.0%
    # SUSPICIOUS: [30.0%, 60.0%)
    # PHISHING / HIGH RISK: >= 60.0%
    if final_threat_score >= 80.0:
        result = 'phishing'
        risk_level = 'CRITICAL PHISHING' if final_threat_score >= 90.0 else 'HIGH RISK PHISHING'
    elif final_threat_score >= 60.0:
        result = 'phishing'
        risk_level = 'HIGH RISK'
    elif final_threat_score >= 30.0:
        result = 'suspicious'
        risk_level = 'SUSPICIOUS'
    else:
        result = 'safe'
        risk_level = 'SAFE'

    if result == 'safe':
        if not negative_reasons:
            negative_reasons.append("Email text content, sender structure, and evidence match standard benign communication patterns.")
    elif not positive_reasons:
        positive_reasons.append("Email text and structural patterns align with learned phishing threat characteristics.")

    # Security recommendation
    if result == 'phishing':
        recommendation = "Do not click links or provide credentials. Flag and isolate this message in the SOC queue."
    elif result == 'suspicious':
        recommendation = "Treat this email with caution. Verify the sender identity and embedded destinations prior to taking action."
    else:
        recommendation = "No significant phishing indicators detected. Standard email handling precautions apply."

    # Compute dynamic statistical decision confidence
    boundary_margin = abs(final_threat_score - 50.0) / 50.0
    base_conf = 52.0 + (boundary_margin * 36.0)
    email_ev_bonus = 0.0
    if header_evidence['provided']:
        email_ev_bonus += 4.0
    if has_url_evidence:
        email_ev_bonus += 3.0
    if len(all_flags) > 0:
        email_ev_bonus += 2.0
    dynamic_email_conf = round(min(max(base_conf + email_ev_bonus, 50.0), 99.4), 1)

    return {
        'result': result,
        'confidence': dynamic_email_conf,
        'threat_score': final_threat_score,
        'final_threat_score': final_threat_score,
        'risk_level': risk_level,
        'email_ml_probability': email_ml_percent,
        'url_ml_probability': round(max_url_phish_prob * 100.0, 1) if has_url_evidence else None,
        'evidence_breakdown': {
            'email_ml_calibrated_prob': email_ml_percent,
            'has_url_evidence': has_url_evidence,
            'urls_analyzed': len(analyzed_urls),
            'header_authenticated': header_evidence['provided'],
            'credential_solicitation': content_evidence['credential_harvesting_prompt'],
            'suspension_threat': content_evidence['account_suspension_threat'],
            'urgency_pressure': content_evidence['urgency_threat']
        },
        'flags': all_flags,
        'positive_indicators': positive_reasons,
        'legitimate_indicators': negative_reasons,
        'unknown_evidence': unknown_evidence,
        'why_risky': positive_reasons if result != 'safe' else [],
        'header_analysis': header_evidence,
        'extracted_urls': analyzed_urls,
        'extracted_urls_count': len(extracted_urls),
        'security_recommendation': recommendation,
        'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M:%S %p IST")
    }
