"""
Page modules for the dashboard.
Each page is extracted from alternate_dashboard.py for modular architecture.
"""

from .performance import render as render_performance
from .features import render as render_features
from .donor_insights import render as render_donor_insights
from .predictions import render as render_predictions
from .take_action import render as render_take_action
from .business_impact import render as render_business_impact
from .model_comparison import render as render_model_comparison
from .dashboard import render as render_dashboard

__all__ = [
    'render_performance',
    'render_features',
    'render_donor_insights',
    'render_predictions',
    'render_take_action',
    'render_business_impact',
    'render_model_comparison',
    'render_dashboard'
]

# All primary pages are now available via wrappers; internals will be migrated incrementally

