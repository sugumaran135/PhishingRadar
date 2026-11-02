// ==========================================================================
// Phishing Radar Extension - Background Service Worker
// ==========================================================================

chrome.runtime.onInstalled.addListener(() => {
    console.log("🛡️ Phishing Radar Extension Installed.");
    chrome.action.setBadgeBackgroundColor({ color: "#3b82f6" });
    chrome.action.setBadgeText({ text: "" });
});

// Listen for messages from popup or content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'URL_SCANNED') {
        updateBadge(message.result, message.threat_score);
        sendResponse({ status: 'BADGE_UPDATED' });
    }
    return true;
});

function updateBadge(result, threatScore) {
    const score = parseFloat(threatScore || 0);

    if (result === 'phishing' || score >= 70) {
        chrome.action.setBadgeBackgroundColor({ color: "#ef4444" });
        chrome.action.setBadgeText({ text: "WARN" });
    } else if (result === 'suspicious' || score >= 30) {
        chrome.action.setBadgeBackgroundColor({ color: "#f59e0b" });
        chrome.action.setBadgeText({ text: "SUSP" });
    } else {
        chrome.action.setBadgeBackgroundColor({ color: "#10b981" });
        chrome.action.setBadgeText({ text: "OK" });
    }
}
