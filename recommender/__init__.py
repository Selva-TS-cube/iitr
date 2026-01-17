"""
Netflix Recommendation System
A content-based recommendation engine using TF-IDF similarity
"""

from .engine import RecommendationEngine, get_engine

__version__ = '1.0.0'
__all__ = ['RecommendationEngine', 'get_engine']
