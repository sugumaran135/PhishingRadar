import unittest
from app import app, analyze_single_url

class TestCheckPhishDecoupledSuite(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def _test_url_case(self, test_num, category, target_url):
        print(f"\n=================== TEST {test_num}: {category} ===================")
        print(f"Target URL: {target_url}")
        
        result = analyze_single_url(target_url)

        print(f"URL ML Probability:    {result['url_model_probability']}%")
        print(f"HTML ML Probability:   {result['html_model_probability'] if result['html_model_probability'] is not None else 'N/A'}")
        print(f"Fused ML Probability:  {result['fused_ml_probability']}%")
        print(f"Final Risk Score:      {result['final_risk_score']}%")
        print(f"Final Classification:  {result['classification']} ({result['risk_level']})")
        print(f"Assessment Source:     {result['assessment_source']}")
        print(f"Web Content Fetched:   {result['content_fetch_success']}")
        print(f"Web Status Note:       {result['web_content_status_note']}")
        print(f"Security Risk Score:   {result['security_risk_score']}")
        print(f"Feature Breakdown:     {result['url_features']}")
        print(f"Model Explanation:     {result['model_explanation']}")

        return result

    # --- Target Benchmark URLs ---
    def test_01_rockstar_games(self):
        res = self._test_url_case(1, "ROCKSTAR GAMES", "https://www.rockstargames.com/")
        self.assertIn('url_model_probability', res)
        self.assertIn('html_model_probability', res)

    def test_02_leetcode(self):
        res = self._test_url_case(2, "LEETCODE", "https://leetcode.com/")
        self.assertIn('url_model_probability', res)

    def test_03_neetcode(self):
        res = self._test_url_case(3, "NEETCODE", "https://neetcode.io/")
        self.assertIn('url_model_probability', res)

    def test_04_harvard_university(self):
        res = self._test_url_case(4, "HARVARD UNIVERSITY", "https://www.harvard.edu/")
        self.assertIn('url_model_probability', res)

    def test_05_wikipedia(self):
        res = self._test_url_case(5, "WIKIPEDIA", "https://www.wikipedia.org/")
        self.assertEqual(res['classification'], 'safe')

    def test_06_python_org(self):
        res = self._test_url_case(6, "PYTHON ORG", "https://python.org/")
        self.assertIn('url_model_probability', res)

    def test_07_github(self):
        res = self._test_url_case(7, "GITHUB", "https://github.com/")
        self.assertEqual(res['classification'], 'safe')

    def test_08_apple(self):
        res = self._test_url_case(8, "APPLE", "https://www.apple.com/")
        self.assertEqual(res['classification'], 'safe')

    def test_09_cloudflare(self):
        res = self._test_url_case(9, "CLOUDFLARE", "https://www.cloudflare.com/")
        self.assertEqual(res['classification'], 'safe')

    # --- Phishing Benchmark URLs ---
    def test_10_synthetic_phishing_url(self):
        res = self._test_url_case(10, "SYNTHETIC PHISHING URL", "http://account-update-security-check.xyz/login.php")
        self.assertEqual(res['classification'], 'phishing')
        self.assertGreaterEqual(res['final_risk_score'], 75.0)

    def test_11_ip_based_phishing_url(self):
        res = self._test_url_case(11, "IP-BASED PHISHING URL", "http://192.168.1.100/verify/account.html")
        self.assertEqual(res['classification'], 'phishing')
        self.assertGreaterEqual(res['final_risk_score'], 70.0)

    def test_12_at_symbol_phishing_url(self):
        res = self._test_url_case(12, "@-SYMBOL PHISHING URL", "https://secure-login.paypal.com@evil-attacker.xyz/verify")
        self.assertEqual(res['classification'], 'phishing')
        self.assertGreaterEqual(res['final_risk_score'], 75.0)

    def test_13_typosquatting_url(self):
        res = self._test_url_case(13, "TYPOSQUATTING PHISHING URL", "http://paypa1-security-update.xyz/login")
        self.assertEqual(res['classification'], 'phishing')
        self.assertGreaterEqual(res['final_risk_score'], 75.0)

if __name__ == '__main__':
    unittest.main()
