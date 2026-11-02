// ==========================================================================
// Phishing Radar Extension - In-Page Content Script
// ==========================================================================

// Listen for threat alerts from extension popup or background worker
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'SHOW_THREAT_BANNER') {
        renderThreatBanner(request.score, request.reasons);
        sendResponse({ status: 'BANNER_SHOWN' });
    }
    return true;
});

function renderThreatBanner(score, reasons = []) {
    if (document.getElementById('phishing-radar-warning-banner')) return;

    const banner = document.createElement('div');
    banner.id = 'phishing-radar-warning-banner';
    banner.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999999999;
        background: linear-gradient(90deg, #881337 0%, #4c0519 100%);
        color: #ffe4e6;
        border-bottom: 2px solid #f43f5e;
        padding: 12px 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    `;

    banner.innerHTML = `
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 20px;">🚨</span>
            <div>
                <strong style="font-size: 13px; color: #fff;">PHISHING RADAR WARNING: HIGH RISK DESTINATION (${score}% Threat Score)</strong>
                <p style="font-size: 11px; margin-top: 2px; color: #fecdd3;">This website exhibits characteristics of credential harvesting or brand impersonation. Do not enter passwords or sensitive personal data.</p>
            </div>
        </div>
        <button id="close-phish-banner-btn" style="
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.3);
            color: #fff;
            padding: 4px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 11px;
            font-weight: bold;
        ">Dismiss</button>
    `;

    document.body.prepend(banner);

    const closeBtn = document.getElementById('close-phish-banner-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            banner.remove();
        });
    }
}
