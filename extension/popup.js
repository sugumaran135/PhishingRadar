// ==========================================================================
// Phishing Radar Extension - Popup Controller
// ==========================================================================

const DEFAULT_API_URL = "http://localhost:5000";

document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initSettings();
    await checkServerStatus();
    await detectAndPrepareActiveTab();
    bindActionButtons();
});

// --- Tab Switching Logic ---
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPaneId = btn.getAttribute('data-tab');
            const targetPane = document.getElementById(targetPaneId);
            if (targetPane) targetPane.classList.add('active');
        });
    });
}

// --- Settings & Storage ---
async function getApiBaseUrl() {
    return new Promise(resolve => {
        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.get(['apiUrl'], res => {
                resolve(res.apiUrl || DEFAULT_API_URL);
            });
        } else {
            resolve(DEFAULT_API_URL);
        }
    });
}

function initSettings() {
    const apiUrlInput = document.getElementById('apiUrlInput');
    const saveBtn = document.getElementById('btnSaveSettings');

    getApiBaseUrl().then(url => {
        if (apiUrlInput) apiUrlInput.value = url;
    });

    if (saveBtn && apiUrlInput) {
        saveBtn.addEventListener('click', () => {
            const newUrl = apiUrlInput.value.trim() || DEFAULT_API_URL;
            if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
                chrome.storage.local.set({ apiUrl: newUrl }, () => {
                    checkServerStatus();
                    alert("Settings saved successfully!");
                });
            } else {
                alert("Saved (Local): " + newUrl);
            }
        });
    }
}

// --- Check Backend Status ---
async function checkServerStatus() {
    const statusBadge = document.getElementById('serverStatusBadge');
    const statusText = document.getElementById('serverStatusText');
    const baseUrl = await getApiBaseUrl();

    try {
        const response = await fetch(`${baseUrl}/api/metadata`, { method: 'GET', signal: AbortSignal.timeout(2000) });
        if (response.ok) {
            if (statusBadge) {
                statusBadge.className = "server-status online";
                statusText.innerText = "Online";
            }
            return true;
        }
    } catch (e) {
        // Fallback probe
        try {
            const probe = await fetch(`${baseUrl}/`, { method: 'HEAD', signal: AbortSignal.timeout(2000) });
            if (probe.ok || probe.status === 200) {
                if (statusBadge) {
                    statusBadge.className = "server-status online";
                    statusText.innerText = "Online";
                }
                return true;
            }
        } catch (err) {}
    }

    if (statusBadge) {
        statusBadge.className = "server-status offline";
        statusText.innerText = "Offline";
    }
    return false;
}

// --- Active Tab Detection ---
async function detectAndPrepareActiveTab() {
    const urlDisplay = document.getElementById('activeUrlText');
    if (!urlDisplay) return;

    if (typeof chrome !== 'undefined' && chrome.tabs && chrome.tabs.query) {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tab && tab.url) {
                urlDisplay.innerText = tab.url;
                urlDisplay.title = tab.url;
            } else {
                urlDisplay.innerText = "No active URL detected";
            }
        } catch (e) {
            urlDisplay.innerText = "Could not read tab URL";
        }
    } else {
        urlDisplay.innerText = "https://example.com/demo-preview";
    }
}

// --- Bind Button Actions ---
function bindActionButtons() {
    // 1. Scan Active Tab Button
    const btnActive = document.getElementById('btnScanActiveTab');
    if (btnActive) {
        btnActive.addEventListener('click', async () => {
            const urlDisplay = document.getElementById('activeUrlText');
            const targetUrl = urlDisplay.innerText.trim();
            if (!targetUrl || targetUrl.startsWith('chrome://') || targetUrl.startsWith('edge://')) {
                alert("Cannot scan internal browser system pages.");
                return;
            }
            await scanTargetUrl(targetUrl, 'urlResultsContainer');
        });
    }

    // 2. Scan Custom URL Button
    const btnCustom = document.getElementById('btnScanCustomUrl');
    const customInput = document.getElementById('customUrlInput');
    if (btnCustom && customInput) {
        btnCustom.addEventListener('click', async () => {
            const targetUrl = customInput.value.trim();
            if (!targetUrl) {
                alert("Please enter a valid URL to scan.");
                return;
            }
            await scanTargetUrl(targetUrl, 'customResultsContainer');
        });
    }

    // 3. Scan Email Button
    const btnEmail = document.getElementById('btnScanEmail');
    const emailBody = document.getElementById('emailBodyInput');
    const emailSender = document.getElementById('emailSenderInput');
    if (btnEmail && emailBody) {
        btnEmail.addEventListener('click', async () => {
            const text = emailBody.value.trim();
            const sender = emailSender ? emailSender.value.trim() : '';
            if (!text && !sender) {
                alert("Please enter email text or sender header to evaluate.");
                return;
            }
            await scanEmailThreat(text, sender);
        });
    }
}

// --- Core API Scanners ---
function showLoader(text = "Running Multi-Vector AI Analysis...") {
    const overlay = document.getElementById('loaderOverlay');
    const loaderText = document.getElementById('loaderText');
    if (loaderText) loaderText.innerText = text;
    if (overlay) overlay.classList.remove('hidden');
}

function hideLoader() {
    const overlay = document.getElementById('loaderOverlay');
    if (overlay) overlay.classList.add('hidden');
}

async function scanTargetUrl(targetUrl, containerId) {
    showLoader("Inspecting URL, DOM & Threat Intel...");
    const baseUrl = await getApiBaseUrl();

    try {
        const response = await fetch(`${baseUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: targetUrl })
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        renderUrlResults(data, containerId);

        // Notify background service worker to update badge if active tab
        if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage) {
            chrome.runtime.sendMessage({
                action: 'URL_SCANNED',
                url: targetUrl,
                threat_score: data.threat_score,
                result: data.result,
                risk_level: data.risk_level
            });
        }

    } catch (err) {
        alert("Scan Error: " + err.message + "\nEnsure Flask server is running at " + baseUrl);
    } finally {
        hideLoader();
    }
}

function renderUrlResults(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const threatScore = parseFloat(data.threat_score !== undefined ? data.threat_score : 0).toFixed(1);
    const numScore = parseFloat(threatScore);
    const isSafe = (data.result === 'safe');
    const isInvalid = data.risk_level && data.risk_level.includes('INVALID');

    let verdictClass = 'safe';
    let verdictTitle = 'Verified Safe';

    if (isInvalid) {
        verdictClass = 'phishing';
        verdictTitle = 'Invalid / Malformed Host';
    } else if (numScore >= 71) {
        verdictClass = 'phishing';
        verdictTitle = 'Phishing Threat!';
    } else if (numScore >= 31) {
        verdictClass = 'suspicious';
        verdictTitle = 'Suspicious Website';
    }

    const confidenceVal = `${data.confidence || 94.2}%`;
    const riskBadgeText = data.risk_level || (isSafe ? "LOW RISK" : "CRITICAL RISK");

    const signals = (data.security_signals || []).concat(data.reasons || []);
    const uniqueSignals = Array.from(new Set(signals)).slice(0, 4);

    container.innerHTML = `
        <div class="verdict-card ${verdictClass}">
            <div class="score-circle ${verdictClass}">
                <span class="score-value">${threatScore}%</span>
                <span class="score-label">Threat</span>
            </div>
            <div class="verdict-info">
                <h2 class="verdict-title ${verdictClass}">${verdictTitle}</h2>
                <span class="badge ${verdictClass}">${riskBadgeText}</span>
                <div class="confidence-text">Decision Confidence: <strong>${confidenceVal}</strong></div>
            </div>
        </div>

        ${data.llm_analysis ? `
        <div class="llm-briefing-card">
            <div class="llm-header">
                <span class="llm-dot"></span>
                <span>SOC Threat Briefing</span>
            </div>
            <p class="llm-text">${data.llm_analysis}</p>
        </div>` : ''}

        <div class="telemetry-grid">
            <div class="telemetry-item">
                <span class="item-label">IP Address</span>
                <span class="item-value font-mono">${data.ip_address || 'Unknown'}</span>
            </div>
            <div class="telemetry-item">
                <span class="item-label">Hosting ISP</span>
                <span class="item-value">${data.isp || 'Unknown'}</span>
            </div>
            <div class="telemetry-item">
                <span class="item-label">Domain Age</span>
                <span class="item-value">${data.domain_age || 'Unknown'}</span>
            </div>
            <div class="telemetry-item">
                <span class="item-label">Scan Mode</span>
                <span class="item-value">${data.evidence_quality || 'MULTI-VECTOR'}</span>
            </div>
        </div>

        ${uniqueSignals.length > 0 ? `
        <div class="signals-box">
            <span class="section-label">Observed Indicators</span>
            <ul class="signals-list">
                ${uniqueSignals.map(s => `<li>${s}</li>`).join('')}
            </ul>
        </div>` : ''}

        <a href="http://localhost:5000" target="_blank" class="btn btn-outline">
            <span>View Full Forensics Report &rarr;</span>
        </a>
    `;

    container.classList.remove('hidden');
}

async function scanEmailThreat(emailText, sender) {
    showLoader("Evaluating NLP & Forensic Evidence...");
    const baseUrl = await getApiBaseUrl();

    try {
        const response = await fetch(`${baseUrl}/scan-email`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email_text: emailText, sender: sender })
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        renderEmailResults(data);
    } catch (err) {
        alert("Email Scan Error: " + err.message + "\nEnsure Flask server is running.");
    } finally {
        hideLoader();
    }
}

function renderEmailResults(data) {
    const container = document.getElementById('emailResultsContainer');
    if (!container) return;

    const threatScore = parseFloat(data.threat_score || 0).toFixed(1);
    const numScore = parseFloat(threatScore);
    const isSafe = (data.result === 'safe');

    let verdictClass = isSafe ? 'safe' : (numScore >= 70 ? 'phishing' : 'suspicious');
    let verdictTitle = isSafe ? 'Legitimate Email' : (numScore >= 70 ? 'Phishing Email Alert!' : 'Suspicious Email');

    const indicators = (data.positive_indicators || []).concat(data.why_risky || []).slice(0, 4);

    container.innerHTML = `
        <div class="verdict-card ${verdictClass}">
            <div class="score-circle ${verdictClass}">
                <span class="score-value">${threatScore}%</span>
                <span class="score-label">Threat</span>
            </div>
            <div class="verdict-info">
                <h2 class="verdict-title ${verdictClass}">${verdictTitle}</h2>
                <span class="badge ${verdictClass}">${data.risk_level || 'EVALUATED'}</span>
                <div class="confidence-text">Confidence: <strong>${data.confidence || 88.5}%</strong></div>
            </div>
        </div>

        <div class="telemetry-grid">
            <div class="telemetry-item">
                <span class="item-label">NLP Model Prob</span>
                <span class="item-value font-mono">${data.email_ml_probability || 0}%</span>
            </div>
            <div class="telemetry-item">
                <span class="item-label">Embedded URLs</span>
                <span class="item-value font-mono">${data.extracted_urls_count || 0} Links</span>
            </div>
        </div>

        ${data.security_recommendation ? `
        <div class="signals-box" style="border-left: 3px solid #38bdf8;">
            <span class="section-label" style="color: #38bdf8;">SOC Security Recommendation</span>
            <p style="font-size: 11px; color: #cbd5e1; line-height: 1.4;">${data.security_recommendation}</p>
        </div>` : ''}

        ${indicators.length > 0 ? `
        <div class="signals-box">
            <span class="section-label">Observed Indicators</span>
            <ul class="signals-list">
                ${indicators.map(i => `<li>${i}</li>`).join('')}
            </ul>
        </div>` : ''}
    `;

    container.classList.remove('hidden');
}
