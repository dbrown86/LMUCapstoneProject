"""
CSS styles for the dashboard.
Extracted from alternate_dashboard.py for modular architecture.
"""

def get_css_styles() -> str:
    """
    Get the complete CSS styles for the dashboard.
    
    Returns:
        str: CSS styles as HTML string
    """
    return """
<style>
    /* Background colors are now handled by Streamlit's native theme configuration in .streamlit/config.toml */
    /* No need to override - Streamlit will apply the theme automatically */
    
    /* Sidebar styling - DO NOT CHANGE */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%) !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: white; }
    
    /* Hide Streamlit's default page navigation immediately - highest priority */
    [data-testid="stSidebarNav"] {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important;}
    section[data-testid="stSidebarNav"] {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important;}
    nav[data-testid="stSidebarNav"] {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important;}
    .css-1544g2n {display: none !important; visibility: hidden !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important;}
    
    /* Hide any Streamlit default navigation elements */
    [data-testid="stSidebar"] [class*="css-"]:first-child:not([class*="ua-title"]) {
        display: none !important;
    }
    
    .logo-container {
        text-align: center;
        padding: 20px 0 30px 0;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .logo-text {
        font-size: 32px;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .logo-subtext {
        font-size: 14px;
        color: #e3f2fd;
        margin-top: 5px;
    }
    
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-icon { font-size: 36px; margin-bottom: 10px; }
    .metric-value { font-size: 32px; font-weight: bold; margin: 10px 0; }
    .metric-label {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-delta {
        font-size: 12px;
        margin-top: 8px;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    
    .filter-header {
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
    }
    
    /* Prevent text wrapping in sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] [data-baseweb="radio"] label,
    [data-testid="stSidebar"] [data-baseweb="radio"] label span,
    [data-testid="stSidebar"] [data-baseweb="radio"] [class*="label"],
    [data-testid="stSidebar"] [class*="radio"] label {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        display: inline-block !important;
    }
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        white-space: nowrap !important;
        overflow: visible !important;
    }
    /* Ensure radio button items don't wrap */
    [data-testid="stSidebar"] [data-baseweb="radio"] > div,
    [data-testid="stSidebar"] .stRadio > div {
        white-space: nowrap !important;
    }
    
    /* Align page titles with radio buttons and remove spacing between items */
    [data-testid="stSidebar"] [data-baseweb="radio"] > div,
    [data-testid="stSidebar"] .stRadio > div {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
    /* Align radio button labels with their indicators */
    [data-testid="stSidebar"] [data-baseweb="radio"] > div > label,
    [data-testid="stSidebar"] .stRadio > div > label {
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        line-height: 1.4 !important;
        cursor: pointer !important;
    }
    /* Align radio button indicator */
    [data-testid="stSidebar"] [data-baseweb="radio"] input[type="radio"],
    [data-testid="stSidebar"] .stRadio input[type="radio"],
    [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"] {
        margin: 0 !important;
        margin-right: 6px !important;
        vertical-align: middle !important;
        align-self: center !important;
        flex-shrink: 0 !important;
    }
    /* Align text content next to radio button */
    [data-testid="stSidebar"] [data-baseweb="radio"] label > span,
    [data-testid="stSidebar"] .stRadio label > span,
    [data-testid="stSidebar"] [data-baseweb="radio"] label > *:not(input),
    [data-testid="stSidebar"] .stRadio label > *:not(input) {
        vertical-align: middle !important;
        align-self: center !important;
        line-height: 1.4 !important;
    }
    /* Remove spacing between radio button rows */
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:not(:last-child),
    [data-testid="stSidebar"] .stRadio > div:not(:last-child) {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Target all radio button containers in sidebar */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    [data-testid="stSidebar"] .stRadio {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1976d2;
        margin: 15px 0;
    }
    
    .chart-container {
        background: transparent;
        padding: 0 20px 20px 20px;
        border-radius: 12px;
        box-shadow: none;
        margin: 0 0 15px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Do not blanket-hide Streamlit alerts; deprecations filtered via JS below */
    
    .page-title {
        font-size: 36px;
        font-weight: bold;
        color: #1e3c72;
        margin: 0 0 10px 0;
    }
    .page-subtitle {
        font-size: 18px;
        color: #666;
        margin: 0 0 30px 0;
    }
</style>
<script>
(function() {
    // Background colors are now handled by Streamlit's native theme configuration
    // No JavaScript needed for background color - Streamlit handles it automatically
    
    function hideDeprecationWarnings() {
        // More comprehensive selector to catch all warning elements
        const selectors = [
            '[data-testid="stWarning"]',
            '[data-testid="stAlert"]',
            '.stAlert',
            '.element-container',
            '[class*="alert"]',
            '[class*="warning"]',
            '[role="alert"]',
            'div[class*="stMarkdownContainer"]',
            'div[class*="stException"]',
            '[class*="stException"]',
            'div[class*="alert-warning"]',
            '[class*="alert-warning"]'
        ];
        
        // Keywords to match in warning text
        const warningKeywords = [
            'deprecated',
            'keyword arguments',
            'use config instead',
            'will be removed in a future release',
            'plotly',
            'streamlit'
        ];
        
        selectors.forEach(function(selector) {
            try {
                document.querySelectorAll(selector).forEach(function(el) {
                    const text = (el.textContent || el.innerText || '').toLowerCase();
                    const shouldHide = warningKeywords.some(function(keyword) {
                        return text.includes(keyword);
                    });
                    
                    if (shouldHide) {
                        // Aggressively hide the element
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                        el.style.height = '0';
                        el.style.margin = '0';
                        el.style.padding = '0';
                        el.style.overflow = 'hidden';
                        el.style.opacity = '0';
                        el.style.position = 'absolute';
                        el.setAttribute('data-hidden', 'true');
                        
                        // Also try to remove from DOM or hide parent
                        try {
                            if (el.parentNode && el.parentNode.style) {
                                const parentText = (el.parentNode.textContent || el.parentNode.innerText || '').toLowerCase();
                                if (warningKeywords.some(function(kw) { return parentText.includes(kw); })) {
                                    el.parentNode.style.display = 'none';
                                    el.parentNode.style.visibility = 'hidden';
                                }
                            }
                        } catch(e) {
                            // Ignore errors when manipulating parent
                        }
                    }
                });
            } catch(e) {
                // Silently ignore selector errors
            }
        });
    }
    
    // Run immediately
    hideDeprecationWarnings();
    
    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideDeprecationWarnings);
    } else {
        hideDeprecationWarnings();
    }
    
    // Observe for new elements with more aggressive settings
    const observer = new MutationObserver(function(mutations) {
        hideDeprecationWarnings();
    });
    
    if (document.body) {
        observer.observe(document.body, { 
            childList: true, 
            subtree: true,
            attributes: true,
            attributeFilter: ['class', 'data-testid', 'data-hidden']
        });
    }
    
    // Also observe the main container
    const mainObserver = new MutationObserver(function(mutations) {
        hideDeprecationWarnings();
    });
    
    setTimeout(function() {
        const mainContainer = document.querySelector('[data-testid="stAppViewContainer"]') || 
                             document.querySelector('main') || 
                             document.body;
        if (mainContainer) {
            mainObserver.observe(mainContainer, {
                childList: true,
                subtree: true,
                attributes: true
            });
        }
    }, 100);
    
    // Run periodically to catch any missed warnings
    setInterval(hideDeprecationWarnings, 500);
    
    // Also intercept console warnings if possible
    if (window.console && console.warn) {
        const originalWarn = console.warn;
        console.warn = function(...args) {
            const message = args.join(' ').toLowerCase();
            if (message.includes('deprecated') || 
                message.includes('keyword arguments') || 
                message.includes('use config instead') ||
                message.includes('plotly') && message.includes('config')) {
                // Suppress this warning
                return;
            }
            originalWarn.apply(console, args);
        };
    }
    
    // Intercept console.error for deprecation messages
    if (window.console && console.error) {
        const originalError = console.error;
        console.error = function(...args) {
            const message = args.join(' ').toLowerCase();
            if (message.includes('deprecated') && 
                (message.includes('keyword arguments') || message.includes('config'))) {
                // Suppress this error
                return;
            }
            originalError.apply(console, args);
        };
    }
})();
</script>
"""

