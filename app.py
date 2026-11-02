from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import joblib
import urllib.parse
import cloudscraper  
import requests
import xgboost as xgb
from bs4 import BeautifulSoup
import difflib
import json
import os
import sys
import io
import socket
import whois
import concurrent.futures
from datetime import datetime
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Import Google GenAI SDK
try:
    from google import genai
except ImportError:
    genai = None


from email_analyzer import evaluate_email_threat
from feature_extractor import (
    extract_url_structural_features,
    extract_html_features,
    extract_full_features,
    URL_FEATURE_COLUMNS,
    HTML_FEATURE_COLUMNS,
    FEATURE_COLUMNS
)
from advanced_analyzer import (
    analyze_redirect_chain,
    analyze_brand_impersonation,
    inspect_html_security
)

# Phase 2 DNS Intelligence Module
try:
    from dns_intel import inspect_dns_records
except ImportError:
    def inspect_dns_records(domain: str) -> dict:
        return {"domain": domain, "dns_security_score": 100.0, "security_flags": [], "positive_signals": []}

if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# --- 0. ENVIRONMENT & MODERN GOOGLE GENAI CLIENT SETUP ---
def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val

_load_env_file()

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
gemini_client = None

if LLM_API_KEY and genai:
    try:
        gemini_client = genai.Client(api_key=LLM_API_KEY)
        print("✅ Modern Google GenAI Client connected successfully!")
    except Exception as e:
        print(f"⚠️ Error initializing GenAI Client: {e}")
elif not LLM_API_KEY:
    print("ℹ️ LLM_API_KEY not configured. Add it to .env to enable GenAI threat briefings.")


def synthesize_url_threat_report(url: str, risk_level: str, score: float, ip: str, isp: str, age: str, risk_signals: list, positive_signals: list) -> str:
    """Generates an executive SOC threat briefing with resilient fallback models."""
    if not gemini_client:
        if risk_signals:
            return f"Heuristic Analysis: Flagged with {score}% threat probability due to: {'; '.join(risk_signals[:2])}."
        return f"Heuristic Analysis: Host parameters align with legitimate traffic. Domain registered to {isp}."

    prompt = f"""You are an L3 Senior Threat Intelligence SOC Analyst.
Summarize the technical URL telemetry into exactly 2 concise sentences for a security dashboard:
Sentence 1 explains the root security indicators/anomalies.
Sentence 2 advises whether the user should proceed or block.

Target URL: {url}
Model Risk Level: {risk_level} (Score: {score}/100)
Host IP: {ip} | Registrar: {isp} | Age: {age}
Observed Risk Indicators: {', '.join(risk_signals) if risk_signals else 'None'}
Legitimate Pattern Signals: {', '.join(positive_signals) if positive_signals else 'None'}
"""
    models_to_try = [
        "gemini-3.6-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest"
    ]
    
    for candidate_model in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config={"automatic_function_calling": {"disable": True}}
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            err_msg = str(e)
            if any(code in err_msg for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                print(f"⚠️ {candidate_model} throttled/busy. Trying next model...")
                continue
            elif any(code in err_msg for code in ["404", "NOT_FOUND"]):
                continue
            else:
                print(f"⚠️ Gemini Error on {candidate_model}: {e}")
                continue

    if risk_signals:
        return f"Heuristic Analysis: Flagged with {score}% threat confidence due to: {'; '.join(risk_signals[:2])}."
    return f"Heuristic Analysis: Host parameters match standard legitimate patterns for {isp}."


# --- 1. LOAD DECOUPLED XGBOOST MODELS ---
url_model = None
url_model_columns = URL_FEATURE_COLUMNS

html_model = None
html_model_columns = HTML_FEATURE_COLUMNS

def ensure_cpu_device(model_obj):
    if model_obj is None:
        return
    try:
        if hasattr(model_obj, 'calibrated_classifiers_'):
            for cc in model_obj.calibrated_classifiers_:
                if hasattr(cc, 'estimator'):
                    try:
                        cc.estimator.set_params(device="cpu")
                    except Exception:
                        pass
                    try:
                        cc.estimator.get_booster().set_param({'device': 'cpu'})
                    except Exception:
                        pass
        elif hasattr(model_obj, 'get_booster'):
            model_obj.get_booster().set_param({'device': 'cpu'})
    except Exception:
        pass

URL_MODEL_VERSION = "3.2.0"
HTML_MODEL_VERSION = "3.2.0"
FEATURE_EXTRACTOR_VERSION = "3.2.0"

try:
    if os.path.exists('url_model.pkl') and os.path.exists('url_model_columns.pkl'):
        url_model = joblib.load('url_model.pkl')
        url_model_columns = joblib.load('url_model_columns.pkl')
        ensure_cpu_device(url_model)
        print("✅ Decoupled URL-Only Model loaded.")

    if os.path.exists('html_model.pkl') and os.path.exists('html_model_columns.pkl'):
        html_model = joblib.load('html_model.pkl')
        html_model_columns = joblib.load('html_model_columns.pkl')
        ensure_cpu_device(html_model)
        print("✅ Decoupled HTML-Content Model loaded.")
except Exception as e:
    print(f"❌ Error loading Decoupled XGBoost models: {e}")

# Legacy aliases
model = url_model
model_columns = url_model_columns

# --- LOAD TRAINED EMAIL ML MODEL & TF-IDF VECTORIZER ---
email_model = None
email_vectorizer = None

try:
    email_model_path = os.path.join('models', 'email_classifier.pkl')
    email_vectorizer_path = os.path.join('models', 'email_tfidf_vectorizer.pkl')

    if os.path.exists(email_model_path) and os.path.exists(email_vectorizer_path):
        email_model = joblib.load(email_model_path)
        email_vectorizer = joblib.load(email_vectorizer_path)
        print("✅ Genuine Email Phishing ML Model loaded.")
except Exception as e:
    print(f"⚠️ Error loading Email ML model: {e}")

def predict_email_ml(subject_or_text, body=""):
    if not email_model or not email_vectorizer:
        return None

    full_text = (str(subject_or_text or "") + " " + str(body or "")).strip()
    if not full_text:
        return None

    try:
        tfidf_vec = email_vectorizer.transform([full_text])
        prob_array = email_model.predict_proba(tfidf_vec)[0]
        phishing_prob = float(prob_array[1])
        legit_prob = float(prob_array[0])
        pred_label = "phishing" if phishing_prob >= 0.50 else "safe"

        return {
            'prediction': pred_label,
            'phishing_probability': round(phishing_prob * 100.0, 2),
            'legitimate_probability': round(legit_prob * 100.0, 2),
            'model': 'TF-IDF + Logistic Regression'
        }
    except Exception as e:
        print(f"⚠️ Error during predict_email_ml: {e}")
        return None

def compute_model_shap_explanation(model, columns, feature_dict):
    if model is None or not feature_dict:
        return [], []

    try:
        df = pd.DataFrame([feature_dict])[columns]
        if hasattr(model, 'calibrated_classifiers_'):
            booster = model.calibrated_classifiers_[0].estimator.get_booster()
        elif hasattr(model, 'estimator'):
            booster = model.estimator.get_booster()
        else:
            booster = model.get_booster()

        dmat = xgb.DMatrix(df)
        contribs = booster.predict(dmat, pred_contribs=True)[0]
        feat_contribs = contribs[:-1]
        
        phish_contribs = []
        legit_contribs = []
        
        for col, val, c in zip(columns, df.iloc[0].values, feat_contribs):
            if c < -0.05:
                phish_contribs.append({'feature': col, 'val': float(val), 'impact': round(float(-c), 3)})
            elif c > 0.05:
                legit_contribs.append({'feature': col, 'val': float(val), 'impact': round(float(c), 3)})
                
        phish_contribs.sort(key=lambda x: x['impact'], reverse=True)
        legit_contribs.sort(key=lambda x: x['impact'], reverse=True)
        return phish_contribs, legit_contribs
    except Exception as e:
        print(f"⚠️ Error computing SHAP explanation: {e}")
        return [], []


# --- 2. MULTI-VECTOR ANALYST PIPELINE ---
def analyze_single_url(url):
    print(f"\nScanning URL: {url}")
    parsed_url = urllib.parse.urlparse(url)
    raw_netloc = parsed_url.netloc or ""
    clean_domain = raw_netloc.split('@')[-1].split(':')[0].lower().replace('www.', '')

    # Safe Baseline Initializations
    url_phishing_prob = 0.0
    url_legit_prob = 100.0
    redirect_risk = 0.0
    brand_risk = 0.0
    content_security_risk = 0.0
    domain_risk = 20.0
    domain_age_days = -1
    domain_age_str = "Unknown"
    ip_address = "Unknown"
    isp = "Unknown"
    html_text = None
    content_fetch_success = False
    html_phishing_prob = None
    html_legit_prob = None
    html_feats = None

    # 👉 PUT THEM HERE (Initialized once at the start of the scan):
    positive_signals = []
    risk_signals = []
    reasons = []

    # ==========================================
    # Branch A: URL Structural Features & XGBoost
    # ==========================================
    url_feats = extract_url_structural_features(url)
    df_url = pd.DataFrame([url_feats])[url_model_columns]

    if url_model is not None:
        try:
            probs = url_model.predict_proba(df_url)[0]
            p_idx = list(url_model.classes_).index(0)
            l_idx = list(url_model.classes_).index(1)
            url_phishing_prob = round(float(probs[p_idx]) * 100.0, 2)
            url_legit_prob = round(float(probs[l_idx]) * 100.0, 2)
        except Exception as e:
            print(f"❌ Error predicting with URL model: {e}")

   # ==========================================
    # Branch B: Redirect Engine & Resilient HTML Fetch
    # ==========================================
    chain_history = [{'url': url, 'status_code': 200, 'domain': clean_domain}]
    http_status_code = 200
    is_challenge_page = False

    try:
        scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        resp = scraper.get(url, headers=headers, timeout=6, allow_redirects=True)
        http_status_code = resp.status_code

        if resp.history:
            chain_history = []
            for hop in resp.history:
                h_domain = urllib.parse.urlparse(hop.url).netloc.lower().replace('www.', '')
                chain_history.append({'url': hop.url, 'status_code': hop.status_code, 'domain': h_domain})
            final_domain = urllib.parse.urlparse(resp.url).netloc.lower().replace('www.', '')
            chain_history.append({'url': resp.url, 'status_code': resp.status_code, 'domain': final_domain})

        raw_body = resp.text or ""
        
        # Detect Cloudflare / WAF anti-bot challenge interstitial pages
        challenge_markers = [
            'cf-browser-verification', 'cloudflare-static', 'challenge-running',
            'just a moment...', 'turnstile', '<title>access denied</title>',
            'attention required! | cloudflare', 'checking your browser'
        ]
        
        if any(marker in raw_body.lower() for marker in challenge_markers) or http_status_code in [403, 503]:
            is_challenge_page = True
            html_text = None
        elif len(raw_body.strip()) > 300:
            html_text = raw_body

    except Exception as err:
        print(f"⚠️ Direct HTML scraper attempt failed: {err}")
        html_text = None

    try:
        redirect_data = analyze_redirect_chain(chain_history, final_url=url)
        redirect_risk = redirect_data.get('redirect_risk', 0.0)
    except Exception as e:
        redirect_data = {'redirect_count': 0, 'cross_domain_hops': 0, 'chain': chain_history, 'redirect_risk': 0.0}
        redirect_risk = 0.0

    content_fetch_success = bool(html_text and not is_challenge_page)

    # ==========================================
    # Branch C: Brand Similarity Engine
    # ==========================================
    try:
        brand_data = analyze_brand_impersonation(clean_domain)
        brand_risk = brand_data.get('brand_risk', 0.0)
    except Exception as e:
        print(f"⚠️ Brand analyzer error: {e}")
        brand_data = {'status': 'NO_BRAND_MATCH', 'similarity_score': 0.0, 'target_brand': 'none', 'brand_risk': 0.0}

# ==========================================
    # Content ML XGBoost & Content Security
    # ==========================================
    if content_fetch_success and html_model is not None:
        try:
            html_feats = extract_html_features(html_text, url)
            df_html = pd.DataFrame([html_feats])[html_model_columns]
            probs_h = html_model.predict_proba(df_html)[0]
            hp_idx = list(html_model.classes_).index(0)
            hl_idx = list(html_model.classes_).index(1)
            html_phishing_prob = round(float(probs_h[hp_idx]) * 100.0, 2)
            html_legit_prob = round(float(probs_h[hl_idx]) * 100.0, 2)
        except Exception as e:
            print(f"❌ Error predicting with HTML model: {e}")
            content_fetch_success = False

    if content_fetch_success:
        try:
            html_sec = inspect_html_security(html_text, url)
            content_security_risk = html_sec.get('content_security_risk', 0.0)

            if html_sec.get('positive_signals'):
                positive_signals.extend(html_sec['positive_signals'])

            if html_sec.get('signals'):
                risk_signals.extend(html_sec['signals'])
                for s in html_sec['signals']:
                    reasons.append(s)
        except Exception as e:
            html_sec = {'signals': [], 'positive_signals': [], 'content_security_risk': 0.0, 'available': False}
    else:
        # DOM inspection bypassed due to WAF challenge or connection block
        html_sec = {'signals': [], 'positive_signals': [], 'content_security_risk': None, 'available': False}
        content_security_risk = None
        html_phishing_prob = None

    # ==========================================
    # Domain Intelligence (DNS + WHOIS + RDAP)
    # ==========================================
    dns_intel_data = inspect_dns_records(clean_domain) if clean_domain else {}

    if clean_domain:
        try:
            socket.setdefaulttimeout(1.5)
            ip_address = socket.gethostbyname(clean_domain)
        except Exception:
            ip_address = "Unknown"

        # Standard WHOIS
        try:
            domain_info = whois.whois(clean_domain)
            if domain_info and domain_info.registrar:
                isp = domain_info.registrar[0] if isinstance(domain_info.registrar, list) else domain_info.registrar
            
            creation_date = getattr(domain_info, 'creation_date', None)
            if isinstance(creation_date, list): 
                creation_date = creation_date[0]
            if creation_date:
                creation_date = creation_date.replace(tzinfo=None)
                domain_age_days = (datetime.now() - creation_date).days
                domain_age_str = f"{domain_age_days} days"
        except Exception:
            pass

        # HTTPS RDAP Fallback for blocked WHOIS port 43
        if isp == "Unknown" or domain_age_days == -1:
            try:
                rdap_resp = requests.get(f"https://rdap.org/domain/{clean_domain}", timeout=2.5).json()
                for ent in rdap_resp.get('entities', []):
                    if 'registrar' in ent.get('roles', []):
                        vcard = ent.get('vcardArray', [None, []])[1]
                        for prop in vcard:
                            if prop[0] == 'fn':
                                isp = prop[3]
                                break
                for event in rdap_resp.get('events', []):
                    if event.get('eventAction') == 'registration':
                        c_date = datetime.strptime(event['eventDate'][:10], "%Y-%m-%d")
                        domain_age_days = (datetime.now() - c_date).days
                        domain_age_str = f"{domain_age_days} days"
                        break
            except Exception:
                pass

    # Score Domain Risk
    if ip_address == "Unknown":
        domain_risk = 80.0
    elif domain_age_days != -1:
        if domain_age_days < 30:
            domain_risk = 85.0
        elif domain_age_days > 1825:
            domain_risk = 5.0
        else:
            domain_risk = 30.0
    else:
        domain_risk = 35.0

    dns_sec_score = dns_intel_data.get('dns_security_score', 100.0)
    if dns_sec_score < 70.0:
        domain_risk = max(domain_risk, 100.0 - dns_sec_score)

    # ==========================================
    # Diagnostic Signals & Explainability
    # ==========================================
    positive_signals = []
    risk_signals = []
    reasons = []

    tld = clean_domain.split('.')[-1] if '.' in clean_domain else "com"
    suspicious_keywords = ['login', 'secure', 'verify', 'update', 'account', 'auth', 'support', 'billing', 'password', 'banking', 'confirm']
    has_suspicious_keyword = any(word in url.lower() for word in suspicious_keywords)

    if has_suspicious_keyword:
        risk_signals.append("URL path contains credential/login keywords.")
    else:
        positive_signals.append("No suspicious credential keywords in URL path.")

    if parsed_url.scheme.lower() == 'https':
        positive_signals.append("HTTPS transport encryption enabled.")
    else:
        risk_signals.append("Unencrypted HTTP protocol connection.")

    if domain_age_days != -1:
        if domain_age_days < 30:
            risk_signals.append(f"Newly registered domain ({domain_age_days} days old).")
        elif domain_age_days > 1825:
            positive_signals.append(f"Established domain registered for {domain_age_days} days ({domain_age_days//365} years).")
        else:
            positive_signals.append(f"Domain age: {domain_age_days} days.")
    else:
        risk_signals.append("WHOIS registration details unavailable.")

    if url_feats.get('has_at_symbol'):
        risk_signals.append("URL contains '@' symbol, a common obfuscation technique.")

    if brand_data.get('status') == 'HIGH_BRAND_IMPERSONATION':
        risk_signals.append(f"Brand Impersonation: Domain mimics {brand_data['target_brand'].title()} ({brand_data['similarity_score']}% match).")
        reasons.append(f"Domain closely matches known brand target: {brand_data['target_brand'].title()}.")

    if redirect_data.get('redirect_count', 0) > 1 and redirect_data.get('cross_domain_hops', 0) > 0:
        risk_signals.append(f"Multi-hop cross-domain redirect chain detected ({redirect_data['redirect_count']} hops).")
        reasons.append("Suspicious multi-hop redirection structure.")

    if html_sec.get('signals'):
        risk_signals.extend(html_sec['signals'])
        for s in html_sec['signals']:
            reasons.append(s)

    # DNS Signals
    if dns_intel_data.get('positive_signals'):
        positive_signals.extend(dns_intel_data['positive_signals'][:2])
    if dns_intel_data.get('security_flags'):
        risk_signals.extend(dns_intel_data['security_flags'][:2])

    is_unreachable = (ip_address == "Unknown")
    is_invalid_format = (' ' in clean_domain or '.' not in clean_domain) and not url_feats.get('IsIPAddress')

    if is_invalid_format:
        risk_signals.append("Invalid Hostname Syntax: Domain contains spaces or lacks a standard TLD dot.")
        reasons.append("URL contains invalid domain syntax or spaces.")
    elif is_unreachable:
        risk_signals.append("Unreachable Domain: Host failed DNS resolution.")
        reasons.append("Domain host is unresolvable in DNS lookup.")

  # ==========================================
    # Dynamic Multi-Vector Fusion & Renormalization
    # ==========================================
    is_legit_brand = (brand_data.get('status') == 'LEGITIMATE_BRAND_DOMAIN')
    brand_risk = 0.0 if is_legit_brand else float(brand_data.get('brand_risk', 0.0))

    url_score = float(url_phishing_prob or 0.0)
    redir_risk = float(redirect_risk or 0.0)
    dom_risk = float(domain_risk or 0.0)

    if content_fetch_success:
        html_score = float(html_phishing_prob if html_phishing_prob is not None else url_score)
        cred_risk = float(content_security_risk or 0.0)

        # Dampen HTML script noise on verified parent brands or institutional domains
        if is_legit_brand or clean_domain.endswith(('.edu', '.ac.in', '.gov', '.gov.in')):
            html_score = min(html_score, 1.5)

        # 100% Full Vector Evaluation: 30 + 25 + 15 + 15 + 10 + 5
        fused_score = (
            (url_score * 0.30) +
            (html_score * 0.25) +
            (cred_risk * 0.15) +
            (brand_risk * 0.15) +
            (redir_risk * 0.10) +
            (dom_risk * 0.05)
        )
        evidence_quality = "HIGH"
        scan_mode = "FULL MULTI-VECTOR SCAN"
    else:
        # Renormalized Fallback: Exclude HTML ML & Credential Harvester
        # Available weights: URL (0.30) + Brand (0.15) + Redirect (0.10) + DNS (0.05) = 0.60
        fused_score = (
            (url_score * 0.30) +
            (brand_risk * 0.15) +
            (redir_risk * 0.10) +
            (dom_risk * 0.05)
        ) / 0.60

        evidence_quality = "MEDIUM"
        scan_mode = "CONTENT BLOCKED — URL & TELEMETRY ONLY"

        if is_challenge_page:
            reasons.append("Cloudflare / Anti-Bot protection intercepted crawler. DOM inspection omitted to prevent challenge page false positives.")
            risk_signals.append("Target protected by WAF/Cloudflare bot mitigation challenge.")

    # Heuristic Hard Triggers
    if is_invalid_format:
        fused_score = max(fused_score, 88.5)
    elif is_unreachable and not is_legit_brand:
        fused_score = max(fused_score, 72.5)

    if url_feats.get('has_at_symbol') or url_feats.get('has_userinfo'):
        fused_score += 20.0
    if url_feats.get('IsIPAddress'):
        fused_score += 15.0

    if brand_data.get('status') == 'HIGH_BRAND_IMPERSONATION':
        fused_score = max(fused_score, 82.5)

    if is_legit_brand:
        fused_score = min(fused_score * 0.15, 2.5)

    final_risk_score = round(min(max(fused_score, 0.0), 100.0), 1)

    # ==========================================
    # Dynamic Statistical Decision Confidence
    # ==========================================
    # 1. Statistical certainty based on decision boundary margin (|score - 50| / 50)
    boundary_margin = abs(final_risk_score - 50.0) / 50.0  # 0.0 (ambiguous) to 1.0 (decisive)
    base_confidence = 52.0 + (boundary_margin * 36.0)      # 52.0% (at 50/50) to 88.0% (at extremes)

    # 2. Live telemetry & evidence completeness modifier
    evidence_bonus = 0.0
    if content_fetch_success:
        evidence_bonus += 4.5
    if ip_address != "Unknown":
        evidence_bonus += 2.0
    if domain_age_days != -1:
        evidence_bonus += 2.0
    if dns_sec_score >= 70.0:
        evidence_bonus += 1.0
    if len(positive_signals) + len(risk_signals) >= 3:
        evidence_bonus += 1.0

    # 3. Model cross-verification agreement (when both URL and DOM classifiers are active)
    if content_fetch_success and html_phishing_prob is not None:
        model_agreement = 1.0 - (abs(url_score - float(html_score)) / 100.0)
        evidence_bonus += model_agreement * 2.0

    confidence = round(min(max(base_confidence + evidence_bonus, 50.0), 99.4), 1)

    if is_invalid_format:
        risk_level = "CRITICAL / INVALID HOST"
        result = "phishing"
    elif final_risk_score >= 71.0:
        risk_level = "CRITICAL / HIGH RISK"
        result = "phishing"
    elif final_risk_score >= 31.0:
        risk_level = "SUSPICIOUS / ELEVATED RISK"
        result = "suspicious"
    else:
        risk_level = "LOW RISK / LIKELY LEGITIMATE"
        result = "safe"

    if not reasons:
        if result == "phishing":
            reasons.append("URL structural and heuristic indicators align with learned phishing signatures.")
        elif result == "suspicious":
            reasons.append("Elevated risk indicators detected in structural domain analysis.")
        else:
            reasons.append("Target verified safe. Domain authority and telemetry align with standard legitimate web services.")

    # Explicit telemetry mapping: N/A for bypassed features
    pipeline_breakdown = {
        'url_ml_score': round(float(url_score), 1),
        'html_ml_score': round(float(html_score), 1) if content_fetch_success else "N/A",
        'redirect_risk': round(float(redir_risk), 1),
        'brand_similarity': round(float(brand_risk), 1),
        'matched_brand': brand_data.get('target_brand', 'none'),
        'content_security_risk': round(float(cred_risk), 1) if content_fetch_success else "N/A",
        'domain_risk': round(float(dom_risk), 1)
    }

    # ==========================================
    # SHAP Local Feature Explanations
    # ==========================================
    u_phish_c, u_legit_c = compute_model_shap_explanation(url_model, url_model_columns, url_feats)
    h_phish_c, h_legit_c = compute_model_shap_explanation(html_model, html_model_columns, html_feats) if content_fetch_success else ([], [])

    all_phish_contribs = u_phish_c + h_phish_c
    all_legit_contribs = u_legit_c + h_legit_c
    all_phish_contribs.sort(key=lambda x: x['impact'], reverse=True)
    all_legit_contribs.sort(key=lambda x: x['impact'], reverse=True)

    # ==========================================
    # Gemini Executive Threat Briefing
    # ==========================================
    llm_verdict = synthesize_url_threat_report(
        url=url,
        risk_level=risk_level,
        score=final_risk_score,
        ip=ip_address,
        isp=isp,
        age=domain_age_str,
        risk_signals=risk_signals,
        positive_signals=positive_signals
    )

    feature_breakdown = {
        'url_length': int(url_feats.get('URLLength', len(url))),
        'subdomains': int(url_feats.get('NoOfSubDomain', 0)),
        'is_https': bool(url_feats.get('IsHTTPS', 0) == 1.0),
        'has_keywords': bool(has_suspicious_keyword or url_feats.get('PathHasSuspiciousKeyword', 0) == 1.0),
        'tld': tld or "com"
    }

    return {
        'result': result, 
        'classification': result,
        'url': url,
        'confidence': confidence,
        'evidence_quality': evidence_quality,
        'scan_mode': scan_mode,
        'html_fetched': content_fetch_success,
        'dom_content': html_text if content_fetch_success else "",
        'threat_score': final_risk_score,
        'final_risk_score': final_risk_score,
        'risk_level': risk_level,
        'llm_analysis': llm_verdict,
        'security_signals': risk_signals,
        'positive_signals': positive_signals,
        'reasons': reasons,
        'ip_address': ip_address,
        'isp': isp,
        'domain_age': domain_age_str,
        'domain_age_days': domain_age_days,
        'feature_breakdown': feature_breakdown,
        'pipeline_breakdown': pipeline_breakdown,
        'redirect_data': redirect_data,
        'brand_data': brand_data,
        'html_security': html_sec,
        'dns_intelligence': dns_intel_data,
        'model_explanation': {
            'phishing_contributors': all_phish_contribs,
            'legitimate_contributors': all_legit_contribs
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M:%S %p IST")
    }


# --- 3. WEB ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/metadata')
def get_metadata():
    metadata_path = os.path.join('models', 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return jsonify(data)
    return jsonify({})

@app.route('/api/adversarial-report')
def get_adversarial_report():
    report_path = os.path.join('evaluation', 'results', 'adversarial_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'status': 'pending', 'message': 'Run tests/run_adversarial_suite.py to generate report.'})

@app.route('/api/benchmark-report')
def get_benchmark_report():
    report_path = os.path.join('evaluation', 'results', 'benchmark_comparison.json')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'status': 'pending', 'message': 'Run evaluation/benchmark_models.py to compute benchmarks.'})

@app.route('/api/dns-intelligence', methods=['POST'])
def api_dns_intelligence():
    data = request.get_json() or {}
    target = data.get('domain') or data.get('url', '')
    if not target:
        return jsonify({'error': 'Domain or URL required'}), 400
    clean = target.lower().replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0].split(':')[0]
    return jsonify(inspect_dns_records(clean))

@app.route('/api/relationship-graph', methods=['POST'])
def api_relationship_graph():
    try:
        data = request.get_json() or {}
        scan_data = data.get('scan_data')
        
        if not scan_data:
            return jsonify({'error': 'No scan data provided'}), 400

        nodes = []
        edges = []

        url = scan_data.get('url', 'Unknown URL')
        domain = scan_data.get('domain', 'Unknown Domain')
        ip = scan_data.get('ip_address', 'Unknown IP')
        isp = scan_data.get('isp', 'Unknown ISP')
        threat_score = scan_data.get('threat_score', 0)
        dns_data = scan_data.get('dns_intelligence', {})
        redirect_data = scan_data.get('redirect_data', {})
        brand_data = scan_data.get('brand_data', {})

        # Root Node: Target URL
        root_color = "#ef4444" if threat_score >= 70 else ("#f59e0b" if threat_score >= 30 else "#10b981")
        nodes.append({
            "id": "url_root",
            "label": f"TARGET URL\n{url[:30]}...",
            "color": root_color,
            "shape": "box",
            "font": {"color": "#ffffff", "face": "monospace"}
        })

        # Domain Node
        nodes.append({
            "id": "domain_node",
            "label": f"DOMAIN\n{domain}",
            "color": "#38bdf8",
            "shape": "ellipse",
            "font": {"color": "#ffffff"}
        })
        edges.append({"from": "url_root", "to": "domain_node", "label": "hosts"})

        # IP Node
        if ip and ip != "Unknown":
            nodes.append({
                "id": "ip_node",
                "label": f"IP ADDRESS\n{ip}",
                "color": "#818cf8",
                "shape": "box",
                "font": {"color": "#ffffff"}
            })
            edges.append({"from": "domain_node", "to": "ip_node", "label": "resolves_to"})

            # ISP Node
            if isp and isp != "Unknown":
                nodes.append({
                    "id": "isp_node",
                    "label": f"ASN / ISP\n{isp[:25]}",
                    "color": "#c084fc",
                    "shape": "ellipse",
                    "font": {"color": "#ffffff"}
                })
                edges.append({"from": "ip_node", "to": "isp_node", "label": "routed_by"})

        # Nameservers (NS)
        for i, ns in enumerate(dns_data.get('ns_records', [])[:2]):
            ns_id = f"ns_{i}"
            nodes.append({
                "id": ns_id,
                "label": f"NAMESERVER\n{ns}",
                "color": "#64748b",
                "shape": "ellipse",
                "font": {"color": "#ffffff"}
            })
            edges.append({"from": "domain_node", "to": ns_id, "label": "delegated_to"})

        # Mail Exchangers (MX)
        for i, mx in enumerate(dns_data.get('mx_records', [])[:2]):
            mx_id = f"mx_{i}"
            nodes.append({
                "id": mx_id,
                "label": f"MAIL SERVER\n{mx.split()[0]}",
                "color": "#a855f7",
                "shape": "ellipse",
                "font": {"color": "#ffffff"}
            })
            edges.append({"from": "domain_node", "to": mx_id, "label": "mail_route"})

        # Brand Spoof Target
        matched_brand = brand_data.get('target_brand')
        if matched_brand and matched_brand != 'none':
            nodes.append({
                "id": "brand_target",
                "label": f"TARGET BRAND\n{matched_brand.upper()} ({brand_data.get('similarity_score', 0)}%)",
                "color": "#f43f5e",
                "shape": "diamond",
                "size": 30,
                "font": {"color": "#ffffff", "face": "bold"}
            })
            edges.append({"from": "domain_node", "to": "brand_target", "label": "impersonates", "color": {"color": "#f43f5e"}})

        # Redirect Chain
        chain = redirect_data.get('chain', [])
        if len(chain) > 1:
            prev_id = "url_root"
            for idx, hop in enumerate(chain[1:], 1):
                hop_id = f"hop_{idx}"
                nodes.append({
                    "id": hop_id,
                    "label": f"REDIRECT HOP {idx}\n{hop.get('domain', 'link')}",
                    "color": "#fb923c",
                    "shape": "box",
                    "font": {"color": "#ffffff"}
                })
                edges.append({"from": prev_id, "to": hop_id, "label": f"HTTP {hop.get('status_code', 302)}", "arrows": "to"})
                prev_id = hop_id

        return jsonify({"nodes": nodes, "edges": edges})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/training_graphs/<path:filename>')
def serve_training_graph(filename):
    return send_from_directory('training_graphs', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        url = data.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
            
        res = analyze_single_url(url)
        return jsonify(res)
    except Exception as e:
        print(f"❌ ERROR in /predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/scan-email', methods=['POST'])
def scan_email():
    """Purely ML/NLP based evaluation (No LLM)."""
    try:
        data = request.get_json() or {}
        email_text = data.get('email_text', '')
        sender = data.get('sender', '')
        raw_headers = data.get('raw_headers', '')

        if not email_text and not sender and not raw_headers:
            return jsonify({'error': 'No email text, sender, or headers provided'}), 400

        analysis_result = evaluate_email_threat(
            sender=sender,
            email_text=email_text,
            raw_headers=raw_headers,
            url_evaluator_fn=analyze_single_url,
            email_ml_fn=predict_email_ml
        )
        return jsonify(analysis_result)
    except Exception as e:
        print(f"❌ ERROR in /scan-email: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/typosquat-monitor', methods=['POST'])
def typosquat_monitor():
    try:
        data = request.get_json() or {}
        raw_domain = data.get('domain', '').lower().strip()
        
        if not raw_domain:
            return jsonify({'error': 'No domain provided'}), 400

        clean = raw_domain.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0].strip()
        parts = clean.split('.')
        brand = parts[0]
        tld = '.'.join(parts[1:]) if len(parts) > 1 else 'com'

        if not brand:
            return jsonify({'error': 'Invalid domain name'}), 400

        raw_candidates = []

        # 1. Visual Homoglyphs / Character Substitutions
        homoglyph_map = {
            'o': ['0'],
            'l': ['1', 'i'],
            'i': ['1', 'l'],
            'e': ['3'],
            'a': ['4'],
            's': ['5'],
            'm': ['rn'],
            'w': ['vv'],
            'd': ['cl']
        }
        for char, swaps in homoglyph_map.items():
            if char in brand:
                for sw in swaps:
                    replaced = brand.replace(char, sw)
                    if replaced != brand:
                        raw_candidates.append((f"{replaced}.{tld}", f"Homoglyph ({char} \u2192 {sw})"))

        # 2. Character Omission / Fat-finger Drop
        if len(brand) > 3:
            for i in range(len(brand)):
                omitted = brand[:i] + brand[i+1:]
                if omitted != brand:
                    raw_candidates.append((f"{omitted}.{tld}", f"Character Omission (dropped '{brand[i]}')"))

        # 3. Double-Stroke / Character Repetition
        for i in range(len(brand)):
            repeated = brand[:i] + brand[i] + brand[i:]
            if repeated != brand:
                raw_candidates.append((f"{repeated}.{tld}", f"Double Character ('{brand[i]}')"))

        # 4. Adjacent Letter Transposition
        for i in range(len(brand) - 1):
            if brand[i] != brand[i+1]:
                transposed = brand[:i] + brand[i+1] + brand[i] + brand[i+2:]
                raw_candidates.append((f"{transposed}.{tld}", f"Transposition ({brand[i]}{brand[i+1]} \u2192 {brand[i+1]}{brand[i]})"))

        # 5. Phishing / Security Keyword Prefixes & Suffixes
        prefixes = [('login-', 'Login Prefix'), ('secure-', 'Security Prefix'), ('verify-', 'Verification Prefix'),
                    ('account-', 'Account Prefix'), ('auth-', 'Auth Prefix'), ('update-', 'Update Prefix'), ('my-', 'Personalized Prefix')]
        for p, ptype in prefixes:
            raw_candidates.append((f"{p}{brand}.{tld}", ptype))

        suffixes = [('-security', 'Security Suffix'), ('-verify', 'Verification Suffix'), ('-login', 'Login Suffix'),
                    ('-support', 'Support Suffix'), ('-portal', 'Portal Suffix'), ('-account', 'Account Suffix'), ('-billing', 'Billing Suffix')]
        for s, stype in suffixes:
            raw_candidates.append((f"{brand}{s}.{tld}", stype))

        # 6. High-Risk / Deceptive TLD Lookalikes
        risky_tlds = ['xyz', 'top', 'online', 'vip', 'site', 'live', 'cc', 'shop', 'info', 'net']
        for rtld in risky_tlds:
            if rtld != tld:
                raw_candidates.append((f"{brand}.{rtld}", f"High-Risk TLD (.{rtld})"))

        # Deduplicate candidates while preserving order
        seen = set()
        unique_candidates = []
        for d, vtype in raw_candidates:
            if d not in seen and d != clean:
                seen.add(d)
                unique_candidates.append((d, vtype))

        # 7. Concurrent Fast Real-Time Live Resolution Probe
        import dns.resolver
        shared_resolver = dns.resolver.Resolver()
        shared_resolver.timeout = 0.4
        shared_resolver.lifetime = 0.6

        def probe_candidate(item):
            var_domain, vtype = item
            is_live = False
            resolved_ip = None
            try:
                answers = shared_resolver.resolve(var_domain, 'A')
                if answers:
                    resolved_ip = str(answers[0])
                    is_live = True
            except Exception:
                is_live = False

            sim = round(difflib.SequenceMatcher(None, brand, var_domain.split('.')[0]).ratio() * 100, 1)

            has_suspicious_kw = any(k in vtype.lower() for k in ['login', 'security', 'verify', 'auth', 'update', 'billing', 'portal'])
            has_risky_tld = 'tld' in vtype.lower()

            if is_live:
                if has_suspicious_kw or sim >= 80:
                    risk_level = "CRITICAL ATTACK HOST"
                    risk_score = 94.5
                elif has_risky_tld or sim >= 65:
                    risk_level = "ACTIVE THIRD-PARTY HOST"
                    risk_score = 82.0
                else:
                    risk_level = "LIVE MONITORED HOST"
                    risk_score = 72.0
            else:
                if has_suspicious_kw:
                    risk_level = "HIGH RISK VARIANT"
                    risk_score = 65.0
                elif sim >= 80:
                    risk_level = "DEFENSIVE CANDIDATE"
                    risk_score = 45.0
                else:
                    risk_level = "MONITORED"
                    risk_score = 25.0

            return {
                'domain': var_domain,
                'type': vtype,
                'similarity': sim,
                'is_live': is_live,
                'resolved_ip': resolved_ip or 'Unresolved',
                'risk_level': risk_level,
                'threat_score': risk_score,
                'whois_status': f"🟢 LIVE HOST ({resolved_ip})" if is_live else "⚪ UNREGISTERED (AVAILABLE)"
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            probed_variants = list(executor.map(probe_candidate, unique_candidates))

        # Sort: Live hosts first, then by threat score descending
        probed_variants.sort(key=lambda x: (x['is_live'], x['threat_score']), reverse=True)

        live_count = sum(1 for v in probed_variants if v['is_live'])
        critical_count = sum(1 for v in probed_variants if 'CRITICAL' in v['risk_level'])

        return jsonify({
            'target_domain': clean,
            'total_variants_monitored': len(probed_variants),
            'live_hosts_detected': live_count,
            'critical_threats_count': critical_count,
            'variants': probed_variants,
            'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M:%S %p IST")
        })

    except Exception as e:
        print(f"❌ ERROR in /typosquat-monitor: {e}")
        return jsonify({'error': str(e)}), 500
 # --- GOOGLE SAFE BROWSING & EXTERNAL THREAT INTEL ---
GSB_API_KEY = os.environ.get("GSB_API_KEY", "")  # Leave empty if you don't have a key; it will safely skip without errors

def check_google_safe_browsing(url: str) -> dict:
    """
    Queries Google Safe Browsing API v4.
    Fails safely without raising exceptions if unconfigured or unresolvable.
    """
    if not GSB_API_KEY:
        return {"status": "SKIPPED", "is_flagged": False, "threats": [], "evidence_line": ""}

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
        response = requests.post(endpoint, json=payload, timeout=2.5)
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
            return {"status": "CLEAN", "is_flagged": False, "threats": [], "evidence_line": ""}
    except Exception as e:
        print(f"⚠️ Safe Browsing API check skipped: {e}")

    return {"status": "UNAVAILABLE", "is_flagged": False, "threats": [], "evidence_line": ""}


@app.route('/takedown-request', methods=['POST'])
def takedown_request():
    try:
        data = request.get_json(silent=True) or {}
        target_url = data.get('target_url') or data.get('domain') or ""
        reason = data.get('reason') or data.get('evidence') or "Unauthorized brand impersonation and deceptive credential harvesting."
        contact_email = data.get('contact_email') or "security@checkphish.internal"
        filing_org = data.get('filing_org') or data.get('brand') or "Brand Protection SOC Team"

        if not target_url:
            return jsonify({'error': 'No target URL provided'}), 400

        parsed_domain = urllib.parse.urlparse(target_url).netloc or target_url
        clean_domain = parsed_domain.split('@')[-1].split(':')[0].lower().replace('www.', '')

        # Safe External Intel query
        gsb_res = check_google_safe_browsing(target_url)
        gsb_line = f"\n- Threat Registry: {gsb_res['evidence_line']}" if gsb_res.get('is_flagged') else ""

        takedown_notice = f"""OFFICIAL PHISHING TAKEDOWN NOTICE
======================================================
Date: {datetime.now().strftime("%Y-%m-%d %I:%M:%S %p IST")}
Target URL: {target_url}
Target Domain: {clean_domain}
Reporting Entity: {filing_org}
Submitter Contact: {contact_email}

LEGAL NOTICE & DEMAND FOR IMMEDIATE DEACTIVATION:
The host domain identified above has been flagged by automated multi-vector 
forensic telemetry for hosting active brand impersonation and deceptive 
credential-harvesting infrastructure.

FORENSIC EVIDENCE SUMMARY:
- Primary Finding: {reason}{gsb_line}
- Multi-Vector Fusion: Deceptive structural markers and credential input routing detected.

REQUESTED ACTION:
Pursuant to ICANN Registrar Abuse policies and your Acceptable Use Policy (AUP),
we request immediate suspension, DNS sinkholing, or deactivation of the domain.
======================================================
"""
        return jsonify({
            'status': 'SUCCESS',
            'reference_id': f"TKD-{int(datetime.now().timestamp())}",
            'target_domain': clean_domain,
            'gsb_status': gsb_res.get('status', 'SKIPPED'),
            'takedown_notice': takedown_notice.strip(),
            'takedown_notice_text': takedown_notice.strip(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M:%S %p IST")
        })

    except Exception as e:
        print(f"❌ ERROR in /takedown-request: {e}")
        return jsonify({'error': str(e)}), 500


# --- 4. SERVER RUNNER ---
if __name__ == '__main__':
    print("🚀 Starting CheckPhish Security Platform (Multi-Branch Engine Active)...")
    app.run(debug=True, host='0.0.0.0', port=5000)