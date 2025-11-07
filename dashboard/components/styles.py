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
    .main { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: white; }
    
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
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide Streamlit's automatic page navigation */
    [data-testid="stSidebarNav"] {display: none;}
    section[data-testid="stSidebarNav"] {display: none;}
    .css-1544g2n {display: none;}  /* Legacy selector for page nav */
    
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
    function hideDeprecationWarnings() {
        document.querySelectorAll('[data-testid="stWarning"], [data-testid="stAlert"], .stAlert, .element-container').forEach(function(el) {
            const text = (el.textContent || el.innerText || '').toLowerCase();
            if (text.includes('deprecated') || text.includes('keyword arguments') || text.includes('use config instead')) {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
                el.style.height = '0';
                el.style.margin = '0';
                el.style.padding = '0';
            }
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideDeprecationWarnings);
    } else {
        hideDeprecationWarnings();
    }
    const observer = new MutationObserver(hideDeprecationWarnings);
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

