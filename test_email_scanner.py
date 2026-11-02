import unittest
import sys
import io
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from app import app

class TestCheckPhishSOCRegressionSuite(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def _run_email_test(self, test_num, category, sender, body, expected_verdict, max_threat=None, min_threat=None):
        payload = {'sender': sender, 'email_text': body}
        res = self.client.post('/scan-email', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        email_ml = data.get('email_ml_probability')
        url_ml = data.get('url_ml_probability')
        final_score = data.get('final_threat_score')
        result = data.get('result')
        risk_level = data.get('risk_level')

        print(f"\n=================== TEST {test_num}: {category} ===================")
        print(f"Sender:               {sender}")
        print(f"Body Preview:         {repr(body[:60])}...")
        print(f"Email ML Probability: {email_ml}%" if email_ml is not None else "Email ML Probability: None")
        print(f"URL ML Probability:   {url_ml}%" if url_ml is not None else "URL ML Probability:   N/A")
        print(f"Final Threat Score:   {final_score}%")
        print(f"Classification:       {result.upper()} ({risk_level})")
        print(f"Expected Verdict:     {expected_verdict.upper()}")

        if expected_verdict == 'safe':
            self.assertEqual(result, 'safe', f"Expected safe for {category}, got {result} ({final_score}%)")
            if max_threat is not None:
                self.assertLess(final_score, max_threat)
        elif expected_verdict == 'phishing':
            self.assertEqual(result, 'phishing', f"Expected phishing for {category}, got {result} ({final_score}%)")
            if min_threat is not None:
                self.assertGreaterEqual(final_score, min_threat)

        return data

    # --- CATEGORY A: BENIGN CONVERSATIONAL EMAILS ---
    def test_01_personal_conversational_sug(self):
        self._run_email_test(1, "BENIGN CONVERSATIONAL EMAIL (sug@gmail.com)",
                             "sug@gmail.com",
                             "hi how are you",
                             expected_verdict="safe",
                             max_threat=30.0)

    def test_02_personal_meeting_plan(self):
        self._run_email_test(2, "BENIGN PERSONAL MEETING",
                             "alex@workplace.example",
                             "Hey team, thanks for your message. Let's catch up tomorrow at 2 PM for coffee and discuss the sprint roadmap.",
                             expected_verdict="safe",
                             max_threat=30.0)

    def test_03_personal_congratulations(self):
        self._run_email_test(3, "BENIGN CONGRATULATIONS",
                             "morgan@university.example",
                             "Congratulations on your graduation and the new job offer! Wishing you all the best in your career.",
                             expected_verdict="safe",
                             max_threat=30.0)

    # --- CATEGORY B: LEGITIMATE NEWSLETTERS ---
    def test_04_tech_newsletter(self):
        self._run_email_test(4, "LEGITIMATE TECH NEWSLETTER",
                             "newsletter@tech-digest.example",
                             "Welcome to Python Weekly issue #142.\nHighlights:\n- Optimizing database query performance\n- Modern async microservices\nRead more at our website. To manage preferences, visit your settings.",
                             expected_verdict="safe",
                             max_threat=30.0)

    def test_05_career_job_recommendation(self):
        self._run_email_test(5, "LEGITIMATE CAREER NOTIFICATION",
                             "alerts@job-network.example",
                             "Here are 3 new job opportunities matching your profile:\n1. Cloud Security Architect\n2. Backend Python Developer\nView these on the portal. No action is required if you are not currently searching.",
                             expected_verdict="safe",
                             max_threat=30.0)

    # --- CATEGORY C: LEGITIMATE TRANSACTIONAL EMAILS ---
    def test_06_order_shipping_notification(self):
        self._run_email_test(6, "LEGITIMATE SHIPPING NOTIFICATION",
                             "orders@retailer.example",
                             "Your order has shipped. No action is required.\nTracking Number: TRK-98412-EXP\nEstimated Delivery: 2 business days.",
                             expected_verdict="safe",
                             max_threat=30.0)

    def test_07_payment_receipt(self):
        self._run_email_test(7, "LEGITIMATE INVOICE / RECEIPT",
                             "billing@cloud-service.example",
                             "Thank you for your payment of $19.00. Your monthly subscription invoice INV-2026-4412 is confirmed and paid. No action is required.",
                             expected_verdict="safe",
                             max_threat=30.0)

    # --- CATEGORY D: LEGITIMATE SECURITY NOTICES (CONTAINS "PASSWORD" / "SECURITY") ---
    def test_08_password_changed_confirmation(self):
        self._run_email_test(8, "LEGITIMATE PASSWORD CHANGE NOTICE",
                             "security@service.example",
                             "Security Update: The password for your account was successfully changed.\nIf you initiated this change, no further action is required. If not, please contact our official support desk.",
                             expected_verdict="safe",
                             max_threat=30.0)

    # --- CATEGORY E-P: PHISHING, CREDENTIAL HARVESTING & THREAT ATTACKS ---
    def test_09_credential_harvesting_no_url(self):
        self._run_email_test(9, "CREDENTIAL HARVESTING (NO URL)",
                             "recruitment@career-opportunity.example",
                             "Visit the application portal and submit your password and OTP immediately.",
                             expected_verdict="phishing",
                             min_threat=60.0)

    def test_10_account_suspension_threat(self):
        self._run_email_test(10, "ACCOUNT SUSPENSION THREAT",
                             "security@account-warning.example",
                             "Your account will be suspended today unless you verify your password and OTP.",
                             expected_verdict="phishing",
                             min_threat=60.0)

    def test_11_urgent_credential_harvesting_with_link(self):
        self._run_email_test(11, "URGENT PHISHING WITH MALICIOUS LINK",
                             "support@account-recovery.example",
                             "Your account access has been restricted due to unauthorized activity.\nEnter your username, password, and OTP here immediately to restore access:\nhttp://account-recovery-portal.example/login",
                             expected_verdict="phishing",
                             min_threat=75.0)

    def test_12_fake_payment_failed_scam(self):
        self._run_email_test(12, "FAKE PAYMENT & BILLING PHISHING",
                             "billing@payment-alert.example",
                             "Your recent transaction failed. Enter your banking credentials, card details, and PIN immediately to avoid account cancellation:\nhttp://billing-update-center.example/verify",
                             expected_verdict="phishing",
                             min_threat=75.0)

if __name__ == '__main__':
    unittest.main()
