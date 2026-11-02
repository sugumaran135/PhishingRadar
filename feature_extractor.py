import re
import urllib.parse
import bs4
from bs4 import BeautifulSoup

# Decoupled Model Feature Columns (v3.2.0 Robust Generalized Architecture)
URL_FEATURE_COLUMNS = [
    'URLLength',
    'DomainLength',
    'NoOfSubDomain',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'IsHTTPS',
    'PathLength',
    'PathDepth',
    'HasQuery',
    'DigitCount',
    'DigitRatio',
    'HyphenCount',
    'DotCount',
    'HasAtSymbol',
    'DoubleSlashInPath',
    'HostnameHasHyphen',
    'HostnameHasDigit',
    'IsIPAddress',
    'PathHasSuspiciousKeyword'
]

HTML_FEATURE_COLUMNS = [
    'LineOfCode',
    'NoOfCSS',
    'NoOfJS',
    'NoOfImage',
    'NoOfSelfRef',
    'NoOfExternalRef',
    'NoOfiFrame',
    'HasPasswordField',
    'HasTitle',
    'ExternalRefRatio'
]

# Legacy combined feature columns order
FEATURE_COLUMNS = URL_FEATURE_COLUMNS + HTML_FEATURE_COLUMNS


def extract_url_structural_features(url):
    """
    Extracts structural features strictly from the URL string.
    Does NOT attempt network requests or webpage fetching.
    """
    url = str(url or "").strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    parsed = urllib.parse.urlparse(url)
    raw_netloc = parsed.netloc or ""
    
    if '@' in raw_netloc:
        hostname = raw_netloc.split('@')[-1].split(':')[0].lower()
    else:
        hostname = raw_netloc.split(':')[0].lower()
        
    clean_domain = hostname.replace('www.', '')

    url_length = float(len(url))
    domain_length = float(len(clean_domain))

    # Subdomains calculation
    sub_parts = [p for p in hostname.split('.') if p and p != 'www']
    no_of_subdomain = float(max(0, len(sub_parts) - 2))

    no_of_qmark = float(url.count('?'))
    no_of_ampersand = float(url.count('&'))

    is_https = 1.0 if parsed.scheme.lower() == 'https' else 0.0

    path = parsed.path or ""
    clean_path = path.strip('/')
    is_root_domain = 1.0 if not clean_path else 0.0
    has_path = 1.0 if clean_path else 0.0
    path_length = float(len(clean_path))
    path_depth = float(len([s for s in clean_path.split('/') if s]))
    has_query = 1.0 if len(parsed.query) > 0 else 0.0

    digit_count = float(sum(c.isdigit() for c in url))
    digit_ratio = float(digit_count / url_length) if url_length > 0 else 0.0
    hyphen_count = float(url.count('-'))
    dot_count = float(url.count('.'))
    has_at_symbol = 1.0 if '@' in url else 0.0
    double_slash_in_path = 1.0 if '//' in path else 0.0
    hostname_has_hyphen = 1.0 if '-' in hostname else 0.0
    hostname_has_digit = 1.0 if any(c.isdigit() for c in hostname) else 0.0

    # IP Address detection
    is_ip = 0.0
    clean_host = hostname.split(':')[0]
    parts = clean_host.split('.')
    if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        is_ip = 1.0

    # Suspicious path keyword detection
    suspicious_keywords = ['login', 'secure', 'verify', 'update', 'account', 'auth', 'support', 'billing', 'password', 'banking', 'confirm']
    path_has_kw = 1.0 if any(kw in url.lower() for kw in suspicious_keywords) else 0.0

    return {
        'URLLength': url_length,
        'DomainLength': domain_length,
        'NoOfSubDomain': no_of_subdomain,
        'NoOfQMarkInURL': no_of_qmark,
        'NoOfAmpersandInURL': no_of_ampersand,
        'IsHTTPS': is_https,
        'IsRootDomain': is_root_domain,
        'HasPath': has_path,
        'PathLength': path_length,
        'PathDepth': path_depth,
        'HasQuery': has_query,
        'DigitCount': digit_count,
        'DigitRatio': digit_ratio,
        'HyphenCount': hyphen_count,
        'DotCount': dot_count,
        'HasAtSymbol': has_at_symbol,
        'DoubleSlashInPath': double_slash_in_path,
        'HostnameHasHyphen': hostname_has_hyphen,
        'HostnameHasDigit': hostname_has_digit,
        'IsIPAddress': is_ip,
        'PathHasSuspiciousKeyword': path_has_kw,
        'hostname': hostname,
        'scheme': parsed.scheme.lower(),
        'has_at_symbol': '@' in url,
        'has_userinfo': '@' in raw_netloc
    }

def extract_html_features(html_text, target_url):
    """
    Extracts HTML webpage content features that match the PhiUSIIL dataset definition.
    """
    if not html_text:
        return {
            'LineOfCode': 0.0,
            'NoOfCSS': 0.0,
            'NoOfJS': 0.0,
            'NoOfImage': 0.0,
            'NoOfSelfRef': 0.0,
            'NoOfExternalRef': 0.0,
            'NoOfiFrame': 0.0,
            'HasPasswordField': 0.0,
            'HasTitle': 0.0,
            'ExternalRefRatio': 0.0,
            'content_fetch_success': 0.0
        }

    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        target_parsed = urllib.parse.urlparse(target_url)
        target_netloc = target_parsed.netloc or ""
        target_domain = target_netloc.split('@')[-1].split(':')[0].lower().replace('www.', '')
        raw_lines = len(html_text.splitlines())
        tag_count = len(re.findall(r'<[^>]+>', html_text))
        # Robust LineOfCode calculation handling both formatted and single-line minified HTML
        line_of_code = float(max(raw_lines, tag_count // 2, len(html_text) // 80))

        # CSS links + inline <style> blocks
        css_links = len(soup.find_all('link', rel=lambda r: r and 'stylesheet' in r.lower()))
        inline_styles = len(soup.find_all('style'))
        no_of_css = float(css_links + inline_styles)

        # JS script tags
        no_of_js = float(len(soup.find_all('script')))

        # Image tags
        no_of_image = float(len(soup.find_all('img')))

        # iFrame tags
        no_of_iframe = float(len(soup.find_all('iframe')))

        # Password input field indicator
        has_password = 1.0 if soup.find('input', type=lambda t: t and 'password' in str(t).lower()) else 0.0

        # Title presence indicator
        has_title = 1.0 if (soup.find('title') and soup.find('title').text.strip()) else 0.0

        # Reference analysis (a, img, script, link, iframe, form)
        resource_tags = []
        for tag in soup.find_all(['a', 'img', 'script', 'link', 'iframe', 'form']):
            ref = tag.get('href') or tag.get('src') or tag.get('action')
            if ref and isinstance(ref, str) and not ref.startswith(('data:', 'javascript:', 'mailto:', '#')):
                resource_tags.append(ref)

        self_ref_count = 0
        ext_ref_count = 0

        for ref in resource_tags:
            try:
                full_ref = urllib.parse.urljoin(target_url, ref)
                ref_domain = urllib.parse.urlparse(full_ref).netloc.split('@')[-1].split(':')[0].lower().replace('www.', '')
                if not ref_domain or ref_domain == target_domain or target_domain.endswith('.' + ref_domain) or ref_domain.endswith('.' + target_domain):
                    self_ref_count += 1
                else:
                    ext_ref_count += 1
            except Exception:
                ext_ref_count += 1

        ext_ref_ratio = float(ext_ref_count / (self_ref_count + ext_ref_count + 1.0))

        return {
            'LineOfCode': float(line_of_code),
            'NoOfCSS': float(no_of_css),
            'NoOfJS': float(no_of_js),
            'NoOfImage': float(no_of_image),
            'NoOfSelfRef': float(self_ref_count),
            'NoOfExternalRef': float(ext_ref_count),
            'NoOfiFrame': float(no_of_iframe),
            'HasPasswordField': float(has_password),
            'HasTitle': float(has_title),
            'ExternalRefRatio': float(ext_ref_ratio),
            'content_fetch_success': 1.0
        }
    except Exception as e:
        return {
            'LineOfCode': 0.0,
            'NoOfCSS': 0.0,
            'NoOfJS': 0.0,
            'NoOfImage': 0.0,
            'NoOfSelfRef': 0.0,
            'NoOfExternalRef': 0.0,
            'NoOfiFrame': 0.0,
            'HasPasswordField': 0.0,
            'HasTitle': 0.0,
            'ExternalRefRatio': 0.0,
            'content_fetch_success': 0.0
        }

def extract_full_features(url, html_text=None):
    """
    Combines URL and HTML features into a complete dictionary.
    """
    url_feat = extract_url_structural_features(url)
    if html_text:
        html_feat = extract_html_features(html_text, url)
    else:
        html_feat = {
            'LineOfCode': 0.0,
            'NoOfCSS': 0.0,
            'NoOfJS': 0.0,
            'NoOfImage': 0.0,
            'NoOfSelfRef': 0.0,
            'NoOfExternalRef': 0.0,
            'content_fetch_success': 0.0
        }

    combined = {**url_feat, **html_feat}
    return combined
