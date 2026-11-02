import dns.resolver
from datetime import datetime

def inspect_dns_records(domain: str) -> dict:
    """
    Authoritative DNS Intelligence Probe:
    Harvests A, AAAA, MX, NS, TXT records and assesses SPF & DMARC compliance.
    """
    clean_domain = domain.lower().strip().replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]

    records = {
        "domain": clean_domain,
        "a_records": [],
        "aaaa_records": [],
        "mx_records": [],
        "ns_records": [],
        "txt_records": [],
        "has_spf": False,
        "spf_record": None,
        "has_dmarc": False,
        "dmarc_record": None,
        "dns_security_score": 100.0,
        "security_flags": [],
        "positive_signals": []
    }

    if not clean_domain or "." not in clean_domain:
        records["dns_security_score"] = 0.0
        records["security_flags"].append("Invalid hostname for DNS resolution")
        return records

    resolver = dns.resolver.Resolver()
    resolver.timeout = 2.0
    resolver.lifetime = 2.5

    # 1. A Records (IPv4)
    try:
        answers = resolver.resolve(clean_domain, "A")
        records["a_records"] = [str(rdata) for rdata in answers]
        records["positive_signals"].append(f"A records resolvable ({len(records['a_records'])} host IP{'s' if len(records['a_records']) > 1 else ''})")
    except Exception:
        records["security_flags"].append("No A records resolved (Unreachable or sinkholed domain)")
        records["dns_security_score"] -= 40.0

    # 2. AAAA Records (IPv6)
    try:
        answers = resolver.resolve(clean_domain, "AAAA")
        records["aaaa_records"] = [str(rdata) for rdata in answers]
    except Exception:
        pass

    # 3. MX Records (Mail Exchanger)
    try:
        answers = resolver.resolve(clean_domain, "MX")
        records["mx_records"] = [f"{rdata.exchange.to_text().rstrip('.')} (pref {rdata.preference})" for rdata in answers]
        records["positive_signals"].append(f"Configured MX mail exchange ({len(records['mx_records'])} servers)")
    except Exception:
        records["security_flags"].append("No MX records configured (Host cannot accept legitimate email)")
        records["dns_security_score"] -= 15.0

    # 4. NS Records (Authoritative Nameservers)
    try:
        answers = resolver.resolve(clean_domain, "NS")
        records["ns_records"] = [rdata.target.to_text().rstrip(".") for rdata in answers]
    except Exception:
        records["security_flags"].append("No authoritative NS records found")
        records["dns_security_score"] -= 20.0

    # 5. TXT Records & SPF Detection
    try:
        answers = resolver.resolve(clean_domain, "TXT")
        txt_list = []
        for rdata in answers:
            txt_str = b"".join(rdata.strings).decode("utf-8", errors="replace")
            txt_list.append(txt_str)
            if "v=spf1" in txt_str.lower():
                records["has_spf"] = True
                records["spf_record"] = txt_str
        records["txt_records"] = txt_list
        if records["has_spf"]:
            records["positive_signals"].append("Valid SPF email authentication policy configured")
        else:
            records["security_flags"].append("Missing SPF policy (Vulnerable to email spoofing)")
            records["dns_security_score"] -= 10.0
    except Exception:
        records["security_flags"].append("No TXT records published")
        records["dns_security_score"] -= 10.0

    # 6. DMARC Records (_dmarc.domain)
    try:
        dmarc_answers = resolver.resolve(f"_dmarc.{clean_domain}", "TXT")
        for rdata in dmarc_answers:
            txt_str = b"".join(rdata.strings).decode("utf-8", errors="replace")
            if "v=dmarc1" in txt_str.lower():
                records["has_dmarc"] = True
                records["dmarc_record"] = txt_str
                break
        if records["has_dmarc"]:
            records["positive_signals"].append("Enforced DMARC domain protection policy verified")
        else:
            records["security_flags"].append("No DMARC policy defined at _dmarc zone")
            records["dns_security_score"] -= 15.0
    except Exception:
        records["security_flags"].append("Missing DMARC authentication record")
        records["dns_security_score"] -= 15.0

    records["dns_security_score"] = max(0.0, min(100.0, records["dns_security_score"]))
    return records

if __name__ == "__main__":
    test_domain = "google.com"
    print(f"Testing DNS probe on: {test_domain}")
    import pprint
    pprint.pprint(inspect_dns_records(test_domain))