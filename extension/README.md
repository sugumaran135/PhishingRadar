# 🛡️ Phishing Radar - Browser Extension (Chrome / Edge / Brave)

Real-time AI Threat Intelligence, Malicious URL Detection, and Email Phishing Scanner extension powered by the **CheckPhish Multi-Vector XGBoost & NLP Engine**.

---

## 🚀 How to Install in Your Browser

### Step 1: Open Extension Management Page
- **Google Chrome**: Go to `chrome://extensions/`
- **Microsoft Edge**: Go to `edge://extensions/`
- **Brave Browser**: Go to `brave://extensions/`

### Step 2: Enable Developer Mode
- Turn on the **Developer mode** toggle switch in the top-right corner.

### Step 3: Load Unpacked Extension
1. Click the **"Load unpacked"** button in the top-left.
2. Select the `extension` folder inside this project directory:
   `C:\Users\SUGUMARAN S\Downloads\Mini Project\Cryptography\extension`
3. The **Phishing Radar** extension icon will now appear in your browser toolbar!

---

## ⚡ How to Use

1. **Start the Flask Backend**:
   Make sure `app.py` is running in your terminal:
   ```bash
   python app.py
   ```
   *(Backend runs on `http://localhost:5000`)*

2. **Scan Current Active Webpage**:
   - Navigate to any website in your browser.
   - Click the **Phishing Radar** shield icon in your browser toolbar.
   - Click **"Analyze Current Page"**.
   - View the live **Threat Score**, **Risk Badge**, **Decision Confidence**, **GenAI SOC Threat Briefing**, and **Telemetry Breakdown**.

3. **Scan Custom URLs or Emails**:
   - Switch to the **"Custom URL"** tab to paste and evaluate external links.
   - Switch to the **"Email"** tab to paste email body text/headers and run the TF-IDF + Logistic Regression NLP forensic classifier.
