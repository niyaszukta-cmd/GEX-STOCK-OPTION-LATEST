# ============================================================================
# NYZTrade UNIFIED GEX/DEX Dashboard - INDEX + STOCK OPTIONS
# Features: Weekly/Monthly Options | VANNA & CHARM | Gamma Flip Zones | Smart Caching
# Supports: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY + 30 F&O Stocks
# ============================================================================

# Standard imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import requests
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import warnings
import hashlib
import json
import os
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="NYZTrade Unified - GEX & VANNA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Hide GitHub link */
    header[data-testid="stHeader"] a[href*="github"] {
        display: none !important;
    }
    
    button[kind="header"][data-testid="baseButton-header"] svg {
        display: none !important;
    }
    
    a[aria-label*="GitHub"],
    a[aria-label*="github"],
    a[href*="github.com"] {
        display: none !important;
    }
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a2332;
        --bg-card-hover: #232f42;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-yellow: #f59e0b;
        --accent-cyan: #06b6d4;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: #2d3748;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0f172a 50%, var(--bg-primary) 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .sub-title {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 8px;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: var(--bg-card-hover);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card.positive { border-left: 4px solid var(--accent-green); }
    .metric-card.negative { border-left: 4px solid var(--accent-red); }
    .metric-card.neutral { border-left: 4px solid var(--accent-yellow); }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-value.positive { color: var(--accent-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.neutral { color: var(--accent-yellow); }
    
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin-top: 8px;
        color: var(--text-secondary);
    }
    
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .signal-badge.bullish {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .signal-badge.bearish {
        background: rgba(239, 68, 68, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .signal-badge.volatile {
        background: rgba(245, 158, 11, 0.15);
        color: var(--accent-yellow);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .cached-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-red);
        border-radius: 50%;
        animation: blink 1.5s ease-in-out infinite;
    }
    
    .cached-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
    }
    
    .index-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(139, 92, 246, 0.2);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 12px;
        color: #a78bfa;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .stock-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(6, 182, 212, 0.2);
        border: 1px solid rgba(6, 182, 212, 0.4);
        border-radius: 12px;
        color: #22d3ee;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DhanConfig:
    client_id: str = "1100480354"
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxMTQ5OTMwLCJhcHBfaWQiOiJjOTNkM2UwOSIsImlhdCI6MTc3MTA2MzUzMCwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.jXpg-lS5ejJF7J1qCQnD5wv9zmCoXZuBAyE6k15DZyic56mMUUK7Jsko2tYUTvAKJvJEsJLGW5n45UUEuF7SJg"

# INDEX SECURITY IDs
DHAN_INDEX_SECURITY_IDS = {
    "NIFTY": 13, "BANKNIFTY": 25, "FINNIFTY": 27, "MIDCPNIFTY": 442
}

# STOCK SECURITY IDs
DHAN_STOCK_SECURITY_IDS = {
    "RELIANCE": 2885, "TCS": 11536, "HDFCBANK": 1333, "INFY": 1594,
    "ICICIBANK": 4963, "SBIN": 3045, "BHARTIARTL": 1195, "ITC": 1660,
    "KOTAKBANK": 1922, "LT": 2980, "AXISBANK": 5900, "HINDUNILVR": 1394,
    "WIPRO": 3787, "MARUTI": 10999, "BAJFINANCE": 317, "HCLTECH": 7229,
    "ASIANPAINT": 157, "TITAN": 3506, "ULTRACEMCO": 11532, "SUNPHARMA": 3351,
    "TATAMOTORS": 3456, "TATASTEEL": 3499, "TECHM": 13538, "POWERGRID": 2752,
    "NTPC": 11630, "ONGC": 2475, "M&M": 2031, "BAJAJFINSV": 16675,
    "ADANIPORTS": 3718, "COALINDIA": 20374,
}

# INDEX CONFIG
INDEX_CONFIG = {
    "NIFTY": {"contract_size": 25, "strike_interval": 50, "type": "INDEX"},
    "BANKNIFTY": {"contract_size": 15, "strike_interval": 100, "type": "INDEX"},
    "FINNIFTY": {"contract_size": 40, "strike_interval": 50, "type": "INDEX"},
    "MIDCPNIFTY": {"contract_size": 75, "strike_interval": 25, "type": "INDEX"},
}

# STOCK CONFIG
STOCK_CONFIG = {
    "RELIANCE": {"lot_size": 250, "strike_interval": 10, "type": "STOCK"},
    "TCS": {"lot_size": 150, "strike_interval": 25, "type": "STOCK"},
    "HDFCBANK": {"lot_size": 550, "strike_interval": 10, "type": "STOCK"},
    "INFY": {"lot_size": 300, "strike_interval": 25, "type": "STOCK"},
    "ICICIBANK": {"lot_size": 550, "strike_interval": 10, "type": "STOCK"},
    "SBIN": {"lot_size": 1500, "strike_interval": 5, "type": "STOCK"},
    "BHARTIARTL": {"lot_size": 410, "strike_interval": 10, "type": "STOCK"},
    "ITC": {"lot_size": 1600, "strike_interval": 5, "type": "STOCK"},
    "KOTAKBANK": {"lot_size": 400, "strike_interval": 25, "type": "STOCK"},
    "LT": {"lot_size": 300, "strike_interval": 25, "type": "STOCK"},
    "AXISBANK": {"lot_size": 600, "strike_interval": 10, "type": "STOCK"},
    "HINDUNILVR": {"lot_size": 300, "strike_interval": 25, "type": "STOCK"},
    "WIPRO": {"lot_size": 1200, "strike_interval": 5, "type": "STOCK"},
    "MARUTI": {"lot_size": 75, "strike_interval": 50, "type": "STOCK"},
    "BAJFINANCE": {"lot_size": 125, "strike_interval": 50, "type": "STOCK"},
    "HCLTECH": {"lot_size": 350, "strike_interval": 25, "type": "STOCK"},
    "ASIANPAINT": {"lot_size": 300, "strike_interval": 25, "type": "STOCK"},
    "TITAN": {"lot_size": 300, "strike_interval": 25, "type": "STOCK"},
    "ULTRACEMCO": {"lot_size": 100, "strike_interval": 50, "type": "STOCK"},
    "SUNPHARMA": {"lot_size": 400, "strike_interval": 25, "type": "STOCK"},
    "TATAMOTORS": {"lot_size": 1250, "strike_interval": 5, "type": "STOCK"},
    "TATASTEEL": {"lot_size": 900, "strike_interval": 5, "type": "STOCK"},
    "TECHM": {"lot_size": 400, "strike_interval": 25, "type": "STOCK"},
    "POWERGRID": {"lot_size": 1800, "strike_interval": 5, "type": "STOCK"},
    "NTPC": {"lot_size": 2250, "strike_interval": 5, "type": "STOCK"},
    "ONGC": {"lot_size": 2475, "strike_interval": 5, "type": "STOCK"},
    "M&M": {"lot_size": 300, "strike_interval": 25, "type": "STOCK"},
    "BAJAJFINSV": {"lot_size": 500, "strike_interval": 10, "type": "STOCK"},
    "ADANIPORTS": {"lot_size": 250, "strike_interval": 25, "type": "STOCK"},
    "COALINDIA": {"lot_size": 2040, "strike_interval": 5, "type": "STOCK"},
}

# Combined config
SYMBOL_CONFIG = {**INDEX_CONFIG, **STOCK_CONFIG}

# Stock categories for easier selection
STOCK_CATEGORIES = {
    "Banking & Finance": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "BAJAJFINSV"],
    "IT & Technology": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "Energy & Power": ["RELIANCE", "ONGC", "POWERGRID", "NTPC", "COALINDIA"],
    "Auto & Industrial": ["MARUTI", "TATAMOTORS", "M&M", "LT"],
    "FMCG & Consumer": ["HINDUNILVR", "ITC", "ASIANPAINT", "TITAN"],
    "Others": ["SUNPHARMA", "TATASTEEL", "BHARTIARTL", "ADANIPORTS", "ULTRACEMCO"]
}

# Indian timezone
IST = pytz.timezone('Asia/Kolkata')

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# ============================================================================
# CACHE MANAGER - Smart Caching System
# ============================================================================

class CacheManager:
    """
    Smart caching system that:
    - Caches historical (non-current day) data permanently
    - Keeps current trading day data live with incremental updates
    """
    
    def __init__(self):
        self.cache_dir = "/tmp/nyztrade_unified_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, symbol: str, date: str, strikes: List[str], 
                           interval: str, expiry_code: int, expiry_flag: str, instrument_type: str) -> str:
        """Generate unique cache key for a data request"""
        key_data = f"{symbol}_{date}_{sorted(strikes)}_{interval}_{expiry_code}_{expiry_flag}_{instrument_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _get_meta_path(self, cache_key: str) -> str:
        """Get file path for cache metadata"""
        return os.path.join(self.cache_dir, f"{cache_key}_meta.json")
    
    def is_current_trading_day(self, target_date: str) -> bool:
        """Check if target date is current trading day"""
        now_ist = datetime.now(IST)
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()
        return target_dt == now_ist.date()
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        now_ist = datetime.now(IST)
        market_open = now_ist.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
        market_close = now_ist.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
        return market_open <= now_ist <= market_close
    
    def get_cached_data(self, symbol: str, date: str, strikes: List[str], 
                        interval: str, expiry_code: int, expiry_flag: str, 
                        instrument_type: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[datetime]]:
        """Get cached data if available."""
        cache_key = self._generate_cache_key(symbol, date, strikes, interval, expiry_code, expiry_flag, instrument_type)
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_meta_path(cache_key)
        
        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return None, None, None
        
        try:
            df = pd.read_pickle(cache_path)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            if len(df) > 0 and 'timestamp' in df.columns:
                last_timestamp = df['timestamp'].max()
                if isinstance(last_timestamp, str):
                    last_timestamp = pd.to_datetime(last_timestamp)
            else:
                last_timestamp = None
            
            return df, meta, last_timestamp
        except Exception as e:
            st.warning(f"Cache read error: {e}")
            return None, None, None
    
    def save_to_cache(self, df: pd.DataFrame, meta: Dict, symbol: str, date: str, 
                      strikes: List[str], interval: str, expiry_code: int, expiry_flag: str,
                      instrument_type: str):
        """Save data to cache"""
        cache_key = self._generate_cache_key(symbol, date, strikes, interval, expiry_code, expiry_flag, instrument_type)
        cache_path = self._get_cache_path(cache_key)
        meta_path = self._get_meta_path(cache_key)
        
        try:
            df.to_pickle(cache_path)
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
        except Exception as e:
            st.warning(f"Cache write error: {e}")
    
    def merge_incremental_data(self, cached_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with cached data, avoiding duplicates"""
        if cached_df is None or len(cached_df) == 0:
            return new_df
        
        if new_df is None or len(new_df) == 0:
            return cached_df
        
        combined = pd.concat([cached_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp', 'strike'], keep='last')
        combined = combined.sort_values(['timestamp', 'strike']).reset_index(drop=True)
        
        return combined
    
    def clear_cache(self, symbol: str = None, date: str = None):
        """Clear cache - optionally for specific symbol/date"""
        try:
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if symbol is None and date is None:
                    os.remove(file_path)
                elif file.startswith(f"{symbol}_{date}"):
                    os.remove(file_path)
        except Exception as e:
            st.warning(f"Cache clear error: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            files = os.listdir(self.cache_dir)
            pkl_files = [f for f in files if f.endswith('.pkl')]
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in pkl_files)
            return {
                'num_entries': len(pkl_files),
                'total_size_mb': total_size / (1024 * 1024)
            }
        except:
            return {'num_entries': 0, 'total_size_mb': 0}

# Initialize cache manager
cache_manager = CacheManager()

# ============================================================================
# BLACK-SCHOLES CALCULATOR (ENHANCED WITH VANNA & CHARM)
# ============================================================================

class BlackScholesCalculator:
    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def calculate_d2(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
        except:
            return 0
    
    @staticmethod
    def calculate_call_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1)
        except:
            return 0
    
    @staticmethod
    def calculate_put_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1) - 1
        except:
            return 0
    
    @staticmethod
    def calculate_vanna(S, K, T, r, sigma):
        """Calculate Vanna (dDelta/dVol)"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator.calculate_d2(S, K, T, r, sigma)
            vanna = -norm.pdf(d1) * d2 / sigma
            return vanna
        except:
            return 0
    
    @staticmethod
    def calculate_charm(S, K, T, r, sigma, option_type='call'):
        """Calculate Charm (Delta Decay)"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator.calculate_d2(S, K, T, r, sigma)
            charm = -norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
            return charm
        except:
            return 0

# ============================================================================
# GAMMA FLIP ZONE CALCULATOR
# ============================================================================

def identify_gamma_flip_zones(df: pd.DataFrame, spot_price: float) -> List[Dict]:
    """
    Identifies gamma flip zones where GEX crosses zero.
    Returns list of flip zones with strike levels and direction indicators.
    """
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    flip_zones = []
    
    for i in range(len(df_sorted) - 1):
        current_gex = df_sorted.iloc[i]['net_gex']
        next_gex = df_sorted.iloc[i + 1]['net_gex']
        current_strike = df_sorted.iloc[i]['strike']
        next_strike = df_sorted.iloc[i + 1]['strike']
        
        if (current_gex > 0 and next_gex < 0) or (current_gex < 0 and next_gex > 0):
            flip_strike = current_strike + (next_strike - current_strike) * (abs(current_gex) / (abs(current_gex) + abs(next_gex)))
            
            if spot_price < flip_strike:
                if current_gex > 0:
                    direction = "upward"
                    arrow = "↑"
                    color = "#ef4444"
                else:
                    direction = "downward"
                    arrow = "↓"
                    color = "#10b981"
            else:
                if current_gex < 0:
                    direction = "downward"
                    arrow = "↓"
                    color = "#10b981"
                else:
                    direction = "upward"
                    arrow = "↑"
                    color = "#ef4444"
            
            flip_zones.append({
                'strike': flip_strike,
                'lower_strike': current_strike,
                'upper_strike': next_strike,
                'lower_gex': current_gex,
                'upper_gex': next_gex,
                'direction': direction,
                'arrow': arrow,
                'color': color,
                'flip_type': 'Positive→Negative' if current_gex > 0 else 'Negative→Positive'
            })
    
    return flip_zones

# ============================================================================
# UNIFIED DATA FETCHER (INDEX + STOCK OPTIONS)
# ============================================================================

class UnifiedOptionsFetcher:
    def __init__(self, config: DhanConfig):
        self.config = config
        self.headers = {
            'access-token': config.access_token,
            'client-id': config.client_id,
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.dhan.co/v2"
        self.bs_calc = BlackScholesCalculator()
        self.risk_free_rate = 0.07
    
    def get_instrument_type(self, symbol: str) -> str:
        """Determine if symbol is INDEX or STOCK"""
        if symbol in DHAN_INDEX_SECURITY_IDS:
            return "INDEX"
        elif symbol in DHAN_STOCK_SECURITY_IDS:
            return "STOCK"
        else:
            return "UNKNOWN"
    
    def get_security_id(self, symbol: str) -> int:
        """Get security ID for symbol"""
        if symbol in DHAN_INDEX_SECURITY_IDS:
            return DHAN_INDEX_SECURITY_IDS[symbol]
        elif symbol in DHAN_STOCK_SECURITY_IDS:
            return DHAN_STOCK_SECURITY_IDS[symbol]
        else:
            return None
    
    def get_contract_size(self, symbol: str) -> int:
        """Get contract/lot size for symbol"""
        config = SYMBOL_CONFIG.get(symbol, {})
        return config.get('contract_size', config.get('lot_size', 50))
    
    def fetch_rolling_data(self, symbol: str, from_date: str, to_date: str, 
                          strike_type: str = "ATM", option_type: str = "CALL", 
                          interval: str = "60", expiry_code: int = 1, expiry_flag: str = "WEEK"):
        """Fetch historical rolling options data for both INDEX and STOCK"""
        try:
            security_id = self.get_security_id(symbol)
            if security_id is None:
                return None
            
            instrument_type = self.get_instrument_type(symbol)
            instrument = "OPTIDX" if instrument_type == "INDEX" else "OPTSTK"
            
            payload = {
                "exchangeSegment": "NSE_FNO",
                "interval": interval,
                "securityId": security_id,
                "instrument": instrument,
                "expiryFlag": expiry_flag,
                "expiryCode": expiry_code,
                "strike": strike_type,
                "drvOptionType": option_type,
                "requiredData": ["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"],
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = requests.post(
                f"{self.base_url}/charts/rollingoption",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('data', {})
            return None
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def process_historical_data(self, symbol: str, target_date: str, strikes: List[str], 
                               interval: str = "60", expiry_code: int = 1, expiry_flag: str = "WEEK",
                               from_timestamp: datetime = None, incremental: bool = False):
        """Process historical data with VANNA and CHARM for both INDEX and STOCK options."""
        
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        if incremental and from_timestamp:
            from_date = from_timestamp.strftime('%Y-%m-%d')
        else:
            from_date = (target_dt - timedelta(days=2)).strftime('%Y-%m-%d')
        
        to_date = (target_dt + timedelta(days=2)).strftime('%Y-%m-%d')
        
        instrument_type = self.get_instrument_type(symbol)
        contract_size = self.get_contract_size(symbol)
        
        # Determine time to expiry based on expiry type
        if expiry_flag == "WEEK":
            time_to_expiry = 7 / 365
        else:  # MONTH
            time_to_expiry = 30 / 365
        
        # Determine scaling factor (Billions for Index, Crores for Stocks)
        scaling_factor = 1e9 if instrument_type == "INDEX" else 1e7
        unit_label = "B" if instrument_type == "INDEX" else "Cr"
        
        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(strikes) * 2
        current_step = 0
        
        for strike_type in strikes:
            mode_text = "Incremental update" if incremental else "Full fetch"
            status_text.text(f"[{mode_text}] Fetching {symbol} {strike_type} ({expiry_flag} Expiry {expiry_code})...")
            
            call_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "CALL", 
                                                interval, expiry_code, expiry_flag)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(0.3)
            
            put_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "PUT", 
                                               interval, expiry_code, expiry_flag)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(0.3)
            
            if not call_data or not put_data:
                continue
            
            ce_data = call_data.get('ce', {})
            pe_data = put_data.get('pe', {})
            
            if not ce_data:
                continue
            
            timestamps = ce_data.get('timestamp', [])
            
            for i, ts in enumerate(timestamps):
                try:
                    dt_utc = datetime.fromtimestamp(ts, tz=pytz.UTC)
                    dt_ist = dt_utc.astimezone(IST)
                    
                    if dt_ist.date() != target_dt.date():
                        continue
                    
                    if incremental and from_timestamp and dt_ist <= from_timestamp:
                        continue
                    
                    spot_price = ce_data.get('spot', [0])[i] if i < len(ce_data.get('spot', [])) else 0
                    strike_price = ce_data.get('strike', [0])[i] if i < len(ce_data.get('strike', [])) else 0
                    
                    if spot_price == 0 or strike_price == 0:
                        continue
                    
                    call_oi = ce_data.get('oi', [0])[i] if i < len(ce_data.get('oi', [])) else 0
                    put_oi = pe_data.get('oi', [0])[i] if i < len(pe_data.get('oi', [])) else 0
                    call_volume = ce_data.get('volume', [0])[i] if i < len(ce_data.get('volume', [])) else 0
                    put_volume = pe_data.get('volume', [0])[i] if i < len(pe_data.get('volume', [])) else 0
                    call_iv = ce_data.get('iv', [15])[i] if i < len(ce_data.get('iv', [])) else 15
                    put_iv = pe_data.get('iv', [15])[i] if i < len(pe_data.get('iv', [])) else 15
                    
                    call_iv_dec = call_iv / 100 if call_iv > 1 else call_iv
                    put_iv_dec = put_iv / 100 if put_iv > 1 else put_iv
                    
                    # Standard Greeks
                    call_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    call_delta = self.bs_calc.calculate_call_delta(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_delta = self.bs_calc.calculate_put_delta(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    
                    # VANNA and CHARM
                    call_vanna = self.bs_calc.calculate_vanna(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_vanna = self.bs_calc.calculate_vanna(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    call_charm = self.bs_calc.calculate_charm(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec, 'call')
                    put_charm = self.bs_calc.calculate_charm(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec, 'put')
                    
                    # Calculate exposures
                    call_gex = (call_oi * call_gamma * spot_price**2 * contract_size) / scaling_factor
                    put_gex = -(put_oi * put_gamma * spot_price**2 * contract_size) / scaling_factor
                    call_dex = (call_oi * call_delta * spot_price * contract_size) / scaling_factor
                    put_dex = (put_oi * put_delta * spot_price * contract_size) / scaling_factor
                    
                    call_vanna_exp = (call_oi * call_vanna * spot_price * contract_size) / scaling_factor
                    put_vanna_exp = (put_oi * put_vanna * spot_price * contract_size) / scaling_factor
                    call_charm_exp = (call_oi * call_charm * spot_price * contract_size) / scaling_factor
                    put_charm_exp = (put_oi * put_charm * spot_price * contract_size) / scaling_factor
                    
                    all_data.append({
                        'timestamp': dt_ist,
                        'time': dt_ist.strftime('%H:%M IST'),
                        'spot_price': spot_price,
                        'strike': strike_price,
                        'strike_type': strike_type,
                        'call_oi': call_oi,
                        'put_oi': put_oi,
                        'call_volume': call_volume,
                        'put_volume': put_volume,
                        'total_volume': call_volume + put_volume,
                        'call_iv': call_iv,
                        'put_iv': put_iv,
                        'call_gex': call_gex,
                        'put_gex': put_gex,
                        'net_gex': call_gex + put_gex,
                        'call_dex': call_dex,
                        'put_dex': put_dex,
                        'net_dex': call_dex + put_dex,
                        'call_vanna': call_vanna_exp,
                        'put_vanna': put_vanna_exp,
                        'net_vanna': call_vanna_exp + put_vanna_exp,
                        'call_charm': call_charm_exp,
                        'put_charm': put_charm_exp,
                        'net_charm': call_charm_exp + put_charm_exp,
                    })
                    
                except Exception as e:
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            return None, None
        
        df = pd.DataFrame(all_data)
        df = df.sort_values(['strike', 'timestamp']).reset_index(drop=True)
        
        # Calculate flows
        df['call_gex_flow'] = 0.0
        df['put_gex_flow'] = 0.0
        df['net_gex_flow'] = 0.0
        df['call_dex_flow'] = 0.0
        df['put_dex_flow'] = 0.0
        df['net_dex_flow'] = 0.0
        df['call_oi_change'] = 0.0
        df['put_oi_change'] = 0.0
        df['call_oi_gex'] = 0.0
        df['put_oi_gex'] = 0.0
        df['net_oi_gex'] = 0.0
        
        try:
            for strike in df['strike'].unique():
                strike_mask = df['strike'] == strike
                strike_data = df[strike_mask].copy()
                
                if len(strike_data) > 1:
                    df.loc[strike_mask, 'call_gex_flow'] = strike_data['call_gex'].diff().fillna(0)
                    df.loc[strike_mask, 'put_gex_flow'] = strike_data['put_gex'].diff().fillna(0)
                    df.loc[strike_mask, 'net_gex_flow'] = strike_data['net_gex'].diff().fillna(0)
                    df.loc[strike_mask, 'call_dex_flow'] = strike_data['call_dex'].diff().fillna(0)
                    df.loc[strike_mask, 'put_dex_flow'] = strike_data['put_dex'].diff().fillna(0)
                    df.loc[strike_mask, 'net_dex_flow'] = strike_data['net_dex'].diff().fillna(0)
                    
                    df.loc[strike_mask, 'call_oi_change'] = strike_data['call_oi'].diff().fillna(0)
                    df.loc[strike_mask, 'put_oi_change'] = strike_data['put_oi'].diff().fillna(0)
            
            # Calculate OI-based GEX
            for strike in df['strike'].unique():
                strike_mask = df['strike'] == strike
                strike_data = df[strike_mask].copy()
                
                for idx in strike_data.index:
                    try:
                        row = df.loc[idx]
                        
                        spot = row['spot_price']
                        strike_price = row['strike']
                        call_iv_dec = row['call_iv'] / 100 if row['call_iv'] > 1 else row['call_iv']
                        put_iv_dec = row['put_iv'] / 100 if row['put_iv'] > 1 else row['put_iv']
                        
                        call_gamma = self.bs_calc.calculate_gamma(spot, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                        put_gamma = self.bs_calc.calculate_gamma(spot, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                        
                        call_oi_change = row['call_oi_change']
                        put_oi_change = row['put_oi_change']
                        
                        call_oi_gex_val = (call_oi_change * call_gamma * spot**2 * contract_size) / scaling_factor
                        put_oi_gex_val = -(put_oi_change * put_gamma * spot**2 * contract_size) / scaling_factor
                        
                        df.loc[idx, 'call_oi_gex'] = call_oi_gex_val
                        df.loc[idx, 'put_oi_gex'] = put_oi_gex_val
                        df.loc[idx, 'net_oi_gex'] = call_oi_gex_val + put_oi_gex_val
                    except Exception as e:
                        continue
        except Exception as e:
            pass
        
        # Calculate hedging pressure
        max_gex = df['net_gex'].abs().max()
        df['hedging_pressure'] = (df['net_gex'] / max_gex * 100) if max_gex > 0 else 0
        
        # Get latest data point for metadata
        latest = df.sort_values('timestamp').iloc[-1]
        spot_prices = df['spot_price'].unique()
        spot_variation = (spot_prices.max() - spot_prices.min()) / spot_prices.mean() * 100
        
        meta = {
            'symbol': symbol,
            'instrument_type': instrument_type,
            'date': target_date,
            'spot_price': latest['spot_price'],
            'spot_price_min': spot_prices.min(),
            'spot_price_max': spot_prices.max(),
            'spot_variation_pct': spot_variation,
            'total_records': len(df),
            'time_range': f"{df['time'].min()} - {df['time'].max()}",
            'strikes_count': df['strike'].nunique(),
            'interval': f"{interval} minutes" if interval != "1" else "1 minute",
            'expiry_code': expiry_code,
            'expiry_flag': expiry_flag,
            'contract_size': contract_size,
            'unit_label': unit_label,
            'fetch_time': datetime.now(IST).strftime('%H:%M:%S IST'),
            'is_incremental': incremental
        }
        
        return df, meta

# ============================================================================
# SMART DATA FETCHER - Handles caching logic
# ============================================================================

def fetch_data_with_smart_cache(symbol: str, target_date: str, strikes: List[str], 
                                 interval: str, expiry_code: int, expiry_flag: str,
                                 force_refresh: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[Dict], str]:
    """Smart data fetching with caching for both INDEX and STOCK options."""
    fetcher = UnifiedOptionsFetcher(DhanConfig())
    instrument_type = fetcher.get_instrument_type(symbol)
    
    is_current_day = cache_manager.is_current_trading_day(target_date)
    is_market_open = cache_manager.is_market_hours()
    
    cached_df, cached_meta, last_timestamp = cache_manager.get_cached_data(
        symbol, target_date, strikes, interval, expiry_code, expiry_flag, instrument_type
    )
    
    # CASE 1: Historical data (not current trading day)
    if not is_current_day:
        if cached_df is not None and not force_refresh:
            cached_meta['fetch_mode'] = 'cached'
            cached_meta['fetch_time'] = datetime.now(IST).strftime('%H:%M:%S IST')
            return cached_df, cached_meta, 'cached'
        else:
            df, meta = fetcher.process_historical_data(
                symbol, target_date, strikes, interval, expiry_code, expiry_flag
            )
            if df is not None:
                cache_manager.save_to_cache(df, meta, symbol, target_date, strikes, interval, expiry_code, expiry_flag, instrument_type)
            return df, meta, 'full_fetch'
    
    # CASE 2: Current trading day
    else:
        if not is_market_open and cached_df is not None and not force_refresh:
            cached_meta['fetch_mode'] = 'cached'
            cached_meta['fetch_time'] = datetime.now(IST).strftime('%H:%M:%S IST')
            return cached_df, cached_meta, 'cached'
        
        if cached_df is not None and last_timestamp is not None and not force_refresh:
            new_df, new_meta = fetcher.process_historical_data(
                symbol, target_date, strikes, interval, expiry_code, expiry_flag,
                from_timestamp=last_timestamp, incremental=True
            )
            
            if new_df is not None and len(new_df) > 0:
                merged_df = cache_manager.merge_incremental_data(cached_df, new_df)
                
                merged_meta = new_meta.copy()
                merged_meta['total_records'] = len(merged_df)
                merged_meta['time_range'] = f"{merged_df['time'].min()} - {merged_df['time'].max()}"
                merged_meta['fetch_mode'] = 'incremental'
                merged_meta['new_records'] = len(new_df)
                
                cache_manager.save_to_cache(merged_df, merged_meta, symbol, target_date, strikes, interval, expiry_code, expiry_flag, instrument_type)
                
                return merged_df, merged_meta, 'incremental'
            else:
                cached_meta['fetch_mode'] = 'cached'
                cached_meta['fetch_time'] = datetime.now(IST).strftime('%H:%M:%S IST')
                return cached_df, cached_meta, 'cached'
        
        df, meta = fetcher.process_historical_data(
            symbol, target_date, strikes, interval, expiry_code, expiry_flag
        )
        if df is not None:
            cache_manager.save_to_cache(df, meta, symbol, target_date, strikes, interval, expiry_code, expiry_flag, instrument_type)
        return df, meta, 'full_fetch'

# ============================================================================
# ENHANCED OVERLAY VISUALIZATIONS
# ============================================================================

def create_enhanced_vanna_overlay_chart(df: pd.DataFrame, spot_price: float, unit_label: str = "B") -> go.Figure:
    """Enhanced VANNA Overlay: Original vs Enhanced OI VANNA with Greeks & Volume adjustments"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    required_cols = ['net_vanna', 'call_oi_change', 'put_oi_change', 'total_volume', 'call_iv', 'put_iv']
    for col in required_cols:
        if col not in df_sorted.columns:
            df_sorted[col] = 0.0
        df_sorted[col] = df_sorted[col].fillna(0)
    
    df_sorted['enhanced_oi_vanna'] = 0.0
    
    bs_calc = BlackScholesCalculator()
    risk_free_rate = 0.07
    contract_size = 25  # Default
    
    try:
        total_volume_at_time = df_sorted['total_volume'].sum()
        
        for idx, row in df_sorted.iterrows():
            try:
                spot = row.get('spot_price', spot_price)
                strike = row['strike']
                call_oi_change = row['call_oi_change']
                put_oi_change = row['put_oi_change']
                volume = row['total_volume']
                call_iv = row['call_iv']
                put_iv = row['put_iv']
                
                if spot <= 0 or strike <= 0:
                    continue
                
                time_to_expiry = 7 / 365
                
                call_iv_dec = call_iv / 100 if call_iv > 1 else call_iv
                put_iv_dec = put_iv / 100 if put_iv > 1 else put_iv
                
                call_vanna_base = bs_calc.calculate_vanna(spot, strike, time_to_expiry, risk_free_rate, call_iv_dec)
                put_vanna_base = bs_calc.calculate_vanna(spot, strike, time_to_expiry, risk_free_rate, put_iv_dec)
                
                volume_weight = 1.0
                if total_volume_at_time > 0:
                    volume_weight = 1 + (volume / total_volume_at_time)
                
                avg_iv = (call_iv_dec + put_iv_dec) / 2
                iv_adjustment = 1 + (avg_iv * 3)
                
                distance_pct = abs(strike - spot) / spot
                distance_weight = 1 / (1 + distance_pct * 1.5)
                
                vanna_multiplier = 2.0
                
                call_vanna_enhanced = call_vanna_base * volume_weight * iv_adjustment * distance_weight * vanna_multiplier
                put_vanna_enhanced = put_vanna_base * volume_weight * iv_adjustment * distance_weight * vanna_multiplier
                
                scaling = 1e9 if unit_label == "B" else 1e7
                call_oi_vanna_enhanced = (call_oi_change * call_vanna_enhanced * spot * contract_size) / scaling
                put_oi_vanna_enhanced = (put_oi_change * put_vanna_enhanced * spot * contract_size) / scaling
                
                df_sorted.loc[idx, 'enhanced_oi_vanna'] = call_oi_vanna_enhanced + put_oi_vanna_enhanced
                
            except Exception as e:
                continue
    except Exception as e:
        st.warning(f"Error calculating enhanced OI VANNA: {e}")
    
    vanna_sum = abs(df_sorted['net_vanna'].sum())
    enhanced_oi_vanna_sum = abs(df_sorted['enhanced_oi_vanna'].sum())
    has_vanna_data = vanna_sum > 0.000001
    has_enhanced_oi_vanna_data = enhanced_oi_vanna_sum > 0.000001
    
    max_vanna = df_sorted['net_vanna'].abs().max()
    max_enhanced_oi_vanna = df_sorted['enhanced_oi_vanna'].abs().max()
    
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    
    if not has_vanna_data:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="❌ No VANNA Data Found<br>Check data source",
            showarrow=False,
            bgcolor="rgba(255,0,0,0.3)",
            bordercolor="red",
            borderwidth=2,
            font=dict(color="white", size=16),
            align="center"
        )
    else:
        original_vanna_colors = ['#06b6d4' if x > 0 else '#0891b2' for x in df_sorted['net_vanna']]
        fig.add_trace(go.Bar(
            y=df_sorted['strike'],
            x=df_sorted['net_vanna'],
            orientation='h',
            marker=dict(
                color=original_vanna_colors,
                opacity=0.6,
                line=dict(width=0)
            ),
            name=f'Original VANNA - Max: {max_vanna:.4f}{unit_label}',
            hovertemplate=f'Strike: %{{y:,.0f}}<br>Original VANNA: %{{x:.4f}}{unit_label}<extra></extra>'
        ))
        
        if has_enhanced_oi_vanna_data:
            enhanced_vanna_colors = ['#ec4899' if x > 0 else '#be185d' for x in df_sorted['enhanced_oi_vanna']]
            fig.add_trace(go.Bar(
                y=df_sorted['strike'],
                x=df_sorted['enhanced_oi_vanna'],
                orientation='h',
                marker=dict(
                    color=enhanced_vanna_colors,
                    opacity=0.85,
                    line=dict(color='white', width=1)
                ),
                name=f'Enhanced OI VANNA (Greeks+Vol) - Max: {max_enhanced_oi_vanna:.4f}{unit_label}',
                hovertemplate=f'Strike: %{{y:,.0f}}<br>Enhanced OI VANNA: %{{x:.4f}}{unit_label}<extra></extra>'
            ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="white", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white", family="Arial Black")))
    
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=2)
    
    for zone in flip_zones:
        fig.add_hline(
            y=zone['strike'],
            line_dash="dot",
            line_color=zone['color'],
            line_width=1,
            opacity=0.3,
            annotation_text=f"🔄 {zone['strike']:,.0f}",
            annotation_position="left",
            annotation=dict(
                font=dict(size=9, color=zone['color']),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor=zone['color'],
                borderwidth=1
            )
        )
    
    fig.update_layout(
        title=dict(
            text="<b>🌊 Enhanced VANNA Overlay: Original vs Enhanced OI VANNA</b><br><sub>Cyan/Teal = All effects | Pink/Magenta = OI changes with Volume+IV+Distance+VANNA adjustments | ✏️ Use toolbar to draw</sub>", 
            font=dict(size=18, color='white')
        ),
        xaxis_title=f"VANNA (dDelta/dVol) [{unit_label}]",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        barmode='overlay',
        bargap=0.15,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='white', size=11),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=1
        ),
        hovermode='closest',
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=100, b=80),
        dragmode='drawline',
        newshape=dict(
            line=dict(color='#f59e0b', width=2),
            fillcolor='rgba(245, 158, 11, 0.2)'
        )
    )
    
    fig.update_layout(
        modebar_add=[
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        modebar=dict(
            bgcolor='rgba(0,0,0,0.5)',
            color='#94a3b8',
            activecolor='#f59e0b'
        )
    )
    
    return fig

def create_enhanced_gex_overlay_chart(df: pd.DataFrame, spot_price: float, unit_label: str = "B") -> go.Figure:
    """Enhanced GEX Overlay: Original vs Enhanced OI GEX with Greeks & Volume adjustments"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    required_cols = ['net_gex', 'call_oi_change', 'put_oi_change', 'total_volume', 'call_iv', 'put_iv']
    for col in required_cols:
        if col not in df_sorted.columns:
            df_sorted[col] = 0.0
        df_sorted[col] = df_sorted[col].fillna(0)
    
    df_sorted['enhanced_oi_gex'] = 0.0
    
    bs_calc = BlackScholesCalculator()
    risk_free_rate = 0.07
    contract_size = 25  # Default
    
    try:
        total_volume_at_time = df_sorted['total_volume'].sum()
        
        for idx, row in df_sorted.iterrows():
            try:
                spot = row.get('spot_price', spot_price)
                strike = row['strike']
                call_oi_change = row['call_oi_change']
                put_oi_change = row['put_oi_change']
                volume = row['total_volume']
                call_iv = row['call_iv']
                put_iv = row['put_iv']
                
                if spot <= 0 or strike <= 0:
                    continue
                
                time_to_expiry = 7 / 365
                
                call_iv_dec = call_iv / 100 if call_iv > 1 else call_iv
                put_iv_dec = put_iv / 100 if put_iv > 1 else put_iv
                
                call_gamma_base = bs_calc.calculate_gamma(spot, strike, time_to_expiry, risk_free_rate, call_iv_dec)
                put_gamma_base = bs_calc.calculate_gamma(spot, strike, time_to_expiry, risk_free_rate, put_iv_dec)
                
                volume_weight = 1.0
                if total_volume_at_time > 0:
                    volume_weight = 1 + (volume / total_volume_at_time)
                
                avg_iv = (call_iv_dec + put_iv_dec) / 2
                iv_adjustment = 1 + (avg_iv * 2)
                
                distance_pct = abs(strike - spot) / spot
                distance_weight = 1 / (1 + distance_pct * 2)
                
                greeks_multiplier = 1.5
                
                call_gamma_enhanced = call_gamma_base * volume_weight * iv_adjustment * distance_weight * greeks_multiplier
                put_gamma_enhanced = put_gamma_base * volume_weight * iv_adjustment * distance_weight * greeks_multiplier
                
                scaling = 1e9 if unit_label == "B" else 1e7
                call_oi_gex_enhanced = (call_oi_change * call_gamma_enhanced * spot**2 * contract_size) / scaling
                put_oi_gex_enhanced = -(put_oi_change * put_gamma_enhanced * spot**2 * contract_size) / scaling
                
                df_sorted.loc[idx, 'enhanced_oi_gex'] = call_oi_gex_enhanced + put_oi_gex_enhanced
                
            except Exception as e:
                continue
    except Exception as e:
        st.warning(f"Error calculating enhanced OI GEX: {e}")
    
    gex_sum = abs(df_sorted['net_gex'].sum())
    enhanced_oi_gex_sum = abs(df_sorted['enhanced_oi_gex'].sum())
    has_gex_data = gex_sum > 0.000001
    has_enhanced_oi_data = enhanced_oi_gex_sum > 0.000001
    
    max_gex = df_sorted['net_gex'].abs().max()
    max_enhanced_oi_gex = df_sorted['enhanced_oi_gex'].abs().max()
    
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    
    if not has_gex_data:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="❌ No GEX Data Found<br>Check data source",
            showarrow=False,
            bgcolor="rgba(255,0,0,0.3)",
            bordercolor="red",
            borderwidth=2,
            font=dict(color="white", size=16),
            align="center"
        )
    else:
        original_colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex']]
        fig.add_trace(go.Bar(
            y=df_sorted['strike'],
            x=df_sorted['net_gex'],
            orientation='h',
            marker=dict(
                color=original_colors,
                opacity=0.6,
                line=dict(width=0)
            ),
            name=f'Original GEX - Max: {max_gex:.4f}{unit_label}',
            hovertemplate=f'Strike: %{{y:,.0f}}<br>Original GEX: %{{x:.4f}}{unit_label}<extra></extra>'
        ))
        
        if has_enhanced_oi_data:
            enhanced_colors = ['#8b5cf6' if x > 0 else '#f59e0b' for x in df_sorted['enhanced_oi_gex']]
            fig.add_trace(go.Bar(
                y=df_sorted['strike'],
                x=df_sorted['enhanced_oi_gex'],
                orientation='h',
                marker=dict(
                    color=enhanced_colors,
                    opacity=0.85,
                    line=dict(color='white', width=1)
                ),
                name=f'Enhanced OI GEX (Greeks+Vol) - Max: {max_enhanced_oi_gex:.4f}{unit_label}',
                hovertemplate=f'Strike: %{{y:,.0f}}<br>Enhanced OI GEX: %{{x:.4f}}{unit_label}<extra></extra>'
            ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="white", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white", family="Arial Black")))
    
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=2)
    
    for zone in flip_zones:
        fig.add_hline(
            y=zone['strike'],
            line_dash="dot",
            line_color=zone['color'],
            line_width=2,
            annotation_text=f"🔄 {zone['strike']:,.0f}",
            annotation_position="left",
            annotation=dict(
                font=dict(size=10, color=zone['color']),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=zone['color'],
                borderwidth=1
            )
        )
        
        fig.add_hrect(
            y0=zone['lower_strike'],
            y1=zone['upper_strike'],
            fillcolor=zone['color'],
            opacity=0.05,
            line_width=0
        )
    
    fig.update_layout(
        title=dict(
            text="<b>🚀 Enhanced GEX Overlay: Original vs Enhanced OI GEX</b><br><sub>Green/Red = All effects | Purple/Gold = OI changes with Greeks+Volume+IV+Distance adjustments | ✏️ Use toolbar to draw</sub>", 
            font=dict(size=18, color='white')
        ),
        xaxis_title=f"GEX (₹ {unit_label})",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        barmode='overlay',
        bargap=0.15,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='white', size=11),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=1
        ),
        hovermode='closest',
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=100, b=80),
        dragmode='drawline',
        newshape=dict(
            line=dict(color='#f59e0b', width=2),
            fillcolor='rgba(245, 158, 11, 0.2)'
        )
    )
    
    fig.update_layout(
        modebar_add=[
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        modebar=dict(
            bgcolor='rgba(0,0,0,0.5)',
            color='#94a3b8',
            activecolor='#f59e0b'
        )
    )
    
    return fig

# ============================================================================
# STANDARD VISUALIZATION FUNCTIONS
# ============================================================================

def create_separate_gex_chart(df: pd.DataFrame, spot_price: float, unit_label: str = "B") -> go.Figure:
    """GEX chart with Gamma Flip Zones"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex']]
    
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_gex'],
        orientation='h',
        marker_color=colors,
        name='Net GEX',
        hovertemplate=f'Strike: %{{y:,.0f}}<br>Net GEX: %{{x:.4f}}{unit_label}<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    for zone in flip_zones:
        fig.add_hline(
            y=zone['strike'],
            line_dash="dot",
            line_color=zone['color'],
            line_width=2,
            annotation_text=f"🔄 Flip {zone['arrow']} {zone['strike']:,.0f}",
            annotation_position="left",
            annotation=dict(
                font=dict(size=10, color=zone['color']),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=zone['color'],
                borderwidth=1
            )
        )
        
        fig.add_hrect(
            y0=zone['lower_strike'],
            y1=zone['upper_strike'],
            fillcolor=zone['color'],
            opacity=0.1,
            line_width=0,
            annotation_text=zone['arrow'],
            annotation_position="right",
            annotation=dict(font=dict(size=16, color=zone['color']))
        )
    
    fig.update_layout(
        title=dict(text="<b>🎯 Gamma Exposure (GEX) with Flip Zones</b><br><sub>✏️ Use toolbar to draw on chart</sub>", font=dict(size=18, color='white')),
        xaxis_title=f"GEX (₹ {unit_label})",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80),
        dragmode='drawline',
        newshape=dict(
            line=dict(color='#f59e0b', width=2),
            fillcolor='rgba(245, 158, 11, 0.2)'
        )
    )
    
    fig.update_layout(
        modebar_add=[
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        modebar=dict(
            bgcolor='rgba(0,0,0,0.5)',
            color='#94a3b8',
            activecolor='#f59e0b'
        )
    )
    
    return fig

def create_standard_vanna_chart(df: pd.DataFrame, spot_price: float, unit_label: str = "B") -> go.Figure:
    """Standard VANNA Exposure chart"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    colors_call = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['call_vanna']]
    colors_put = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['put_vanna']]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📈 Call VANNA", "📉 Put VANNA"),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['call_vanna'],
        orientation='h',
        marker=dict(color=colors_call),
        name='Call VANNA',
        hovertemplate=f'Strike: %{{y:,.0f}}<br>Call VANNA: %{{x:.4f}}{unit_label}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['put_vanna'],
        orientation='h',
        marker=dict(color=colors_put),
        name='Put VANNA',
        hovertemplate=f'Strike: %{{y:,.0f}}<br>Put VANNA: %{{x:.4f}}{unit_label}<extra></extra>'
    ), row=1, col=2)
    
    for col in [1, 2]:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                      annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                      annotation=dict(font=dict(size=10, color="white")), row=1, col=col)
    
    fig.update_layout(
        title=dict(text="<b>🌊 VANNA Exposure (dDelta/dVol)</b><br><sub>✏️ Use toolbar to draw on chart</sub>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=80, r=80, t=100, b=80),
        dragmode='drawline',
        newshape=dict(
            line=dict(color='#f59e0b', width=2),
            fillcolor='rgba(245, 158, 11, 0.2)'
        )
    )
    
    fig.update_layout(
        modebar_add=[
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        modebar=dict(
            bgcolor='rgba(0,0,0,0.5)',
            color='#94a3b8',
            activecolor='#f59e0b'
        )
    )
    
    fig.update_xaxes(title_text=f"VANNA (₹ {unit_label})", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    fig.update_yaxes(title_text="Strike Price", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    
    return fig

def create_dex_chart(df: pd.DataFrame, spot_price: float, unit_label: str = "B") -> go.Figure:
    """DEX chart"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_dex']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_dex'],
        orientation='h',
        marker_color=colors,
        name='Net DEX',
        hovertemplate=f'Strike: %{{y:,.0f}}<br>Net DEX: %{{x:.4f}}{unit_label}<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right")
    
    fig.update_layout(
        title=dict(text="<b>📊 Delta Exposure (DEX)</b>", font=dict(size=18, color='white')),
        xaxis_title=f"DEX (₹ {unit_label})",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        dragmode='drawline',
        newshape=dict(line=dict(color='#f59e0b', width=2))
    )
    
    fig.update_layout(
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
    )
    
    return fig

def create_oi_distribution_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """OI Distribution chart"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['call_oi'],
        orientation='h',
        name='Call OI',
        marker_color='#10b981',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=-df_sorted['put_oi'],
        orientation='h',
        name='Put OI',
        marker_color='#ef4444',
        opacity=0.7
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2)
    
    fig.update_layout(
        title=dict(text="<b>📋 Open Interest Distribution</b>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        barmode='overlay',
        xaxis_title="Open Interest (Contracts)",
        yaxis_title="Strike Price",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def create_intraday_timeline(df: pd.DataFrame, unit_label: str = "B") -> go.Figure:
    """Create intraday timeline of total GEX and DEX"""
    timeline_df = df.groupby('timestamp').agg({
        'net_gex': 'sum',
        'net_dex': 'sum',
        'spot_price': 'first'
    }).reset_index()
    
    timeline_df = timeline_df.sort_values('timestamp')
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Total Net GEX Over Time', 'Total Net DEX Over Time', 'Spot Price Movement'),
        vertical_spacing=0.1,
        row_heights=[0.35, 0.35, 0.3]
    )
    
    gex_colors = ['#10b981' if x > 0 else '#ef4444' for x in timeline_df['net_gex']]
    fig.add_trace(
        go.Bar(
            x=timeline_df['timestamp'],
            y=timeline_df['net_gex'],
            marker_color=gex_colors,
            name='Net GEX',
            hovertemplate=f'%{{x|%H:%M}}<br>GEX: %{{y:.4f}}{unit_label}<extra></extra>'
        ),
        row=1, col=1
    )
    
    dex_colors = ['#10b981' if x > 0 else '#ef4444' for x in timeline_df['net_dex']]
    fig.add_trace(
        go.Bar(
            x=timeline_df['timestamp'],
            y=timeline_df['net_dex'],
            marker_color=dex_colors,
            name='Net DEX',
            hovertemplate=f'%{{x|%H:%M}}<br>DEX: %{{y:.4f}}{unit_label}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timeline_df['timestamp'],
            y=timeline_df['spot_price'],
            mode='lines+markers',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4),
            name='Spot Price',
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='%{x|%H:%M}<br>Spot: ₹%{y:,.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=dict(text="<b>📈 Intraday Evolution</b>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=900,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time (IST)", row=3, col=1)
    fig.update_yaxes(title_text=f"GEX (₹{unit_label})", row=1, col=1)
    fig.update_yaxes(title_text=f"DEX (₹{unit_label})", row=2, col=1)
    fig.update_yaxes(title_text="Spot Price (₹)", row=3, col=1)
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header with status indicator
    current_time = datetime.now(IST).strftime('%H:%M:%S IST')
    is_market_open = cache_manager.is_market_hours()
    
    market_status = "🟢 MARKET OPEN" if is_market_open else "🔴 MARKET CLOSED"
    market_color = "#10b981" if is_market_open else "#ef4444"
    
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="main-title">📊 NYZTrade UNIFIED Dashboard</h1>
                <p class="sub-title">INDEX + STOCK Options | GEX/DEX/VANNA/CHARM | Smart Caching</p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <div class="live-indicator">
                    <div class="live-dot"></div>
                    <span style="color: #ef4444; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">{current_time}</span>
                </div>
                <div style="padding: 6px 14px; background: rgba({market_color.replace('#', '')[:2]}, {market_color.replace('#', '')[2:4]}, {market_color.replace('#', '')[4:]}, 0.1); border: 1px solid {market_color}30; border-radius: 20px;">
                    <span style="color: {market_color}; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">{market_status}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Instrument Type Selection
        instrument_type = st.radio(
            "📈 Instrument Type",
            ["Index Options", "Stock Options"],
            index=0,
            horizontal=True
        )
        
        st.markdown("---")
        
        if instrument_type == "Index Options":
            # INDEX Selection
            symbol = st.selectbox(
                "🎯 Select Index", 
                options=list(DHAN_INDEX_SECURITY_IDS.keys()), 
                index=0
            )
            
            config = INDEX_CONFIG.get(symbol, INDEX_CONFIG["NIFTY"])
            st.markdown(f"""
            <div class="index-badge">
                📊 INDEX | Lot: {config['contract_size']} | Strike: ₹{config['strike_interval']}
            </div>
            """, unsafe_allow_html=True)
            
            # Default expiry for indices
            default_expiry_type = "Weekly"
            
        else:
            # STOCK Selection
            st.markdown("#### 📂 Stock Category")
            category = st.selectbox(
                "Category",
                options=list(STOCK_CATEGORIES.keys()),
                index=0
            )
            
            symbol = st.selectbox(
                "🎯 Select Stock",
                options=STOCK_CATEGORIES[category],
                index=0
            )
            
            config = STOCK_CONFIG.get(symbol, {"lot_size": 500, "strike_interval": 10})
            st.markdown(f"""
            <div class="stock-badge">
                📈 STOCK | Lot: {config['lot_size']} | Strike: ₹{config['strike_interval']}
            </div>
            """, unsafe_allow_html=True)
            
            # Default expiry for stocks
            default_expiry_type = "Monthly"
        
        st.markdown("---")
        
        # Date Selection
        target_date = st.date_input(
            "📅 Select Date", 
            value=datetime.now(),
            max_value=datetime.now()
        ).strftime('%Y-%m-%d')
        
        # Show if current day or historical
        is_current_day = cache_manager.is_current_trading_day(target_date)
        if is_current_day:
            st.info("📡 **LIVE MODE**: Data will update incrementally")
        else:
            st.success("📦 **HISTORICAL**: Data will be cached after first fetch")
        
        st.markdown("---")
        
        # Expiry Selection
        expiry_type = st.selectbox(
            "📆 Expiry Type", 
            ["Weekly", "Monthly"], 
            index=0 if default_expiry_type == "Weekly" else 1
        )
        expiry_flag = "WEEK" if expiry_type == "Weekly" else "MONTH"
        expiry_code = st.selectbox(
            "Expiry Code", 
            [1, 2, 3], 
            index=0,
            format_func=lambda x: {1: "Current Expiry", 2: "Next Expiry", 3: "Far Expiry"}[x]
        )
        
        st.markdown("---")
        
        # Strike Selection
        strikes = st.multiselect(
            "⚡ Select Strikes",
            ["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3", "ATM+4", "ATM-4", "ATM+5", "ATM-5", 
             "ATM+6", "ATM-6", "ATM+7", "ATM-7", "ATM+8", "ATM-8", "ATM+9", "ATM-9", "ATM+10", "ATM-10"],
            default=["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3"]
        )
        
        interval = st.selectbox(
            "⏱️ Interval", 
            options=["1", "5", "15", "60"], 
            index=1, 
            format_func=lambda x: f"{x} minute" if x == "1" else f"{x} minutes"
        )
        
        st.markdown("---")
        st.markdown("### 🔄 Live Controls")
        
        # Auto-refresh toggle
        auto_refresh_enabled = is_current_day and is_market_open
        
        if auto_refresh_enabled:
            auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", value=False, key="auto_refresh_checkbox")
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (seconds)", min_value=10, max_value=300, value=60, step=10)
                st.info(f"⏱️ Incremental update every {refresh_interval}s")
            else:
                refresh_interval = 60
        else:
            auto_refresh = False
            refresh_interval = 60
            if not is_current_day:
                st.info("ℹ️ Auto-refresh disabled for historical data")
            elif not is_market_open:
                st.info("ℹ️ Auto-refresh disabled (market closed)")
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            fetch_button = st.button("🚀 Fetch Data", use_container_width=True, type="primary")
        with col2:
            if is_current_day:
                refresh_button = st.button("🔄 Update", use_container_width=True)
            else:
                refresh_button = st.button("🔄 Refresh", use_container_width=True)
        
        force_refresh = st.checkbox(
            "🔥 Force Full Refresh", 
            value=False, 
            help="Ignore cache and fetch all data fresh"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Cache Status")
        
        cache_stats = cache_manager.get_cache_stats()
        st.markdown(f"""
        - **Cached entries**: {cache_stats['num_entries']}
        - **Total size**: {cache_stats['total_size_mb']:.2f} MB
        """)
        
        if st.button("🗑️ Clear All Cache", use_container_width=True):
            cache_manager.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ✏️ Drawing Tools")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #94a3b8;">
        Hover on any chart to see toolbar:
        <ul style="margin: 4px 0; padding-left: 16px;">
        <li>📏 <b>Line</b> - Draw straight lines</li>
        <li>✍️ <b>Open Path</b> - Freehand drawing</li>
        <li>🔷 <b>Closed Path</b> - Draw polygons</li>
        <li>⭕ <b>Circle</b> - Draw circles/ellipses</li>
        <li>⬜ <b>Rectangle</b> - Draw rectangles</li>
        <li>🧹 <b>Eraser</b> - Remove drawings</li>
        </ul>
        <i>Drawings are session-only</i>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = None
    
    # Handle fetch/refresh
    if fetch_button or refresh_button:
        st.session_state.fetch_config = {
            'symbol': symbol,
            'target_date': target_date,
            'strikes': strikes,
            'interval': interval,
            'expiry_code': expiry_code,
            'expiry_flag': expiry_flag,
            'force_refresh': force_refresh,
            'instrument_type': instrument_type
        }
        st.session_state.data_fetched = False
        st.session_state.last_refresh_time = datetime.now()
    
    # Auto-refresh logic
    if auto_refresh and auto_refresh_enabled:
        if st.session_state.last_refresh_time is None:
            st.session_state.last_refresh_time = datetime.now()
        
        elapsed = (datetime.now() - st.session_state.last_refresh_time).total_seconds()
        remaining = max(0, int(refresh_interval - elapsed))
        
        if remaining > 0:
            st.sidebar.success(f"⏳ Next update in: **{remaining}s**")
        else:
            st.sidebar.warning("🔄 Updating...")
        
        if elapsed >= refresh_interval and hasattr(st.session_state, 'fetch_config'):
            st.session_state.fetch_config['force_refresh'] = False
            st.session_state.data_fetched = False
            st.session_state.last_refresh_time = datetime.now()
        
        time.sleep(1)
        st.rerun()
    
    # Main content
    if fetch_button or refresh_button or (hasattr(st.session_state, 'fetch_config') and st.session_state.get('data_fetched', False)):
        if hasattr(st.session_state, 'fetch_config'):
            config = st.session_state.fetch_config
            symbol = config['symbol']
            target_date = config['target_date']
            strikes = config['strikes']
            interval = config['interval']
            expiry_code = config.get('expiry_code', 1)
            expiry_flag = config.get('expiry_flag', 'WEEK')
            force_refresh = config.get('force_refresh', False)
        
        if not strikes:
            st.error("❌ Please select at least one strike")
            return
        
        # Check if we need to fetch data
        need_to_fetch = (
            not st.session_state.get('data_fetched', False) or 
            'df_data' not in st.session_state or 
            fetch_button or 
            refresh_button
        )
        
        if need_to_fetch:
            try:
                df, meta, fetch_mode = fetch_data_with_smart_cache(
                    symbol, target_date, strikes, interval, expiry_code, expiry_flag, force_refresh
                )
                
                if df is None or len(df) == 0:
                    st.error("❌ No data available for the selected date/time.")
                    return
                
                st.session_state.df_data = df
                st.session_state.meta_data = meta
                st.session_state.fetch_mode = fetch_mode
                st.session_state.data_fetched = True
                
                if not auto_refresh:
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
        
        # Retrieve from session state
        df = st.session_state.df_data
        meta = st.session_state.meta_data
        fetch_mode = st.session_state.get('fetch_mode', 'unknown')
        
        # Get unit label
        unit_label = meta.get('unit_label', 'B')
        instrument_type_label = meta.get('instrument_type', 'INDEX')
        
        all_timestamps = sorted(df['timestamp'].unique())
        
        # Time selector
        if 'timestamp_idx' not in st.session_state:
            st.session_state.timestamp_idx = len(all_timestamps) - 1
        
        selected_timestamp_idx = st.slider(
            "⏱️ Select Time Point",
            min_value=0,
            max_value=len(all_timestamps) - 1,
            value=min(st.session_state.timestamp_idx, len(all_timestamps) - 1)
        )
        
        selected_timestamp = all_timestamps[selected_timestamp_idx]
        
        # Filter data for selected timestamp
        df_selected = df[df['timestamp'] == selected_timestamp].copy()
        spot_price = df_selected['spot_price'].iloc[0] if len(df_selected) > 0 else 0
        
        # Status message with fetch mode
        fetch_time = meta.get('fetch_time', 'Unknown')
        
        # Create status badge based on fetch mode
        if fetch_mode == 'cached':
            mode_badge = "📦 CACHED"
            mode_color = "#10b981"
        elif fetch_mode == 'incremental':
            new_records = meta.get('new_records', 0)
            mode_badge = f"📡 INCREMENTAL (+{new_records} new)"
            mode_color = "#06b6d4"
        else:
            mode_badge = "🚀 FULL FETCH"
            mode_color = "#8b5cf6"
        
        # Instrument badge
        if instrument_type_label == "INDEX":
            inst_badge = '<span class="index-badge">📊 INDEX</span>'
        else:
            inst_badge = '<span class="stock-badge">📈 STOCK</span>'
        
        st.markdown(f"""
        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap;">
            {inst_badge}
            <span style="padding: 6px 12px; background: {mode_color}20; border: 1px solid {mode_color}40; border-radius: 8px; color: {mode_color}; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">{mode_badge}</span>
            <span style="color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">{meta.get('symbol', symbol)} | Time: {selected_timestamp.strftime('%H:%M:%S IST')} | Spot: ₹{spot_price:,.2f} | Records: {meta.get('total_records', 0)}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        net_gex = df_selected['net_gex'].sum()
        net_dex = df_selected['net_dex'].sum()
        net_vanna = df_selected['net_vanna'].sum()
        flip_zones = identify_gamma_flip_zones(df_selected, spot_price)
        
        with col1:
            gex_color = "positive" if net_gex > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card {gex_color}">
                <div class="metric-label">NET GEX</div>
                <div class="metric-value {gex_color}">{net_gex:.4f}{unit_label}</div>
                <div class="metric-delta">{'🟢 Bullish' if net_gex > 0 else '🔴 Bearish'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            dex_color = "positive" if net_dex > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card {dex_color}">
                <div class="metric-label">NET DEX</div>
                <div class="metric-value {dex_color}">{net_dex:.4f}{unit_label}</div>
                <div class="metric-delta">{'📈 Long Bias' if net_dex > 0 else '📉 Short Bias'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            vanna_color = "positive" if net_vanna > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card {vanna_color}">
                <div class="metric-label">NET VANNA</div>
                <div class="metric-value {vanna_color}">{net_vanna:.4f}{unit_label}</div>
                <div class="metric-delta">Vol Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card neutral">
                <div class="metric-label">SPOT PRICE</div>
                <div class="metric-value">₹{spot_price:,.2f}</div>
                <div class="metric-delta">{meta.get('symbol', symbol)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card neutral">
                <div class="metric-label">FLIP ZONES</div>
                <div class="metric-value">{len(flip_zones)}</div>
                <div class="metric-delta">Gamma Crossovers</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main tabs
        tabs = st.tabs([
            "📈 Intraday Timeline",
            "🎯 Standard GEX", 
            "🚀 Enhanced GEX Overlay",
            "🌊 Standard VANNA",
            "🌊 Enhanced VANNA Overlay",
            "📊 DEX Analysis",
            "📋 OI Distribution",
            "📁 Data Table"
        ])
        
        with tabs[0]:
            st.markdown("### 📈 Intraday Evolution")
            st.plotly_chart(create_intraday_timeline(df, unit_label), use_container_width=True)
        
        with tabs[1]:
            st.markdown("### 🎯 Standard Gamma Exposure (GEX)")
            st.plotly_chart(create_separate_gex_chart(df_selected, spot_price, unit_label), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                positive_gex = df_selected[df_selected['net_gex'] > 0]['net_gex'].sum()
                st.metric("Positive GEX", f"{positive_gex:.4f}{unit_label}")
            with col2:
                negative_gex = df_selected[df_selected['net_gex'] < 0]['net_gex'].sum()
                st.metric("Negative GEX", f"{negative_gex:.4f}{unit_label}")
        
        with tabs[2]:
            st.markdown("### 🚀 Enhanced GEX Overlay")
            st.plotly_chart(create_enhanced_gex_overlay_chart(df_selected, spot_price, unit_label), use_container_width=True)
            
            if 'enhanced_oi_gex' in df_selected.columns:
                enhanced_oi_gex_sum = abs(df_selected['enhanced_oi_gex'].sum())
                
                if enhanced_oi_gex_sum > 0.001:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        original_total = df_selected['net_gex'].sum()
                        st.metric("Original GEX Total", f"{original_total:.4f}{unit_label}")
                    
                    with col2:
                        enhanced_total = df_selected['enhanced_oi_gex'].sum()
                        st.metric("Enhanced OI GEX Total", f"{enhanced_total:.4f}{unit_label}")
                    
                    with col3:
                        if abs(original_total) > 0.001:
                            enhancement_ratio = enhanced_total / original_total
                            st.metric("Enhancement Ratio", f"{enhancement_ratio:.2f}x")
        
        with tabs[3]:
            st.markdown("### 🌊 Standard VANNA Exposure")
            st.plotly_chart(create_standard_vanna_chart(df_selected, spot_price, unit_label), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                call_vanna_total = df_selected['call_vanna'].sum()
                st.metric("Call VANNA Total", f"{call_vanna_total:.4f}{unit_label}")
            with col2:
                put_vanna_total = df_selected['put_vanna'].sum()
                st.metric("Put VANNA Total", f"{put_vanna_total:.4f}{unit_label}")
            with col3:
                net_vanna_total = df_selected['net_vanna'].sum()
                st.metric("Net VANNA Total", f"{net_vanna_total:.4f}{unit_label}")
        
        with tabs[4]:
            st.markdown("### 🌊 Enhanced VANNA Overlay")
            st.plotly_chart(create_enhanced_vanna_overlay_chart(df_selected, spot_price, unit_label), use_container_width=True)
            
            if 'enhanced_oi_vanna' in df_selected.columns:
                enhanced_oi_vanna_sum = abs(df_selected['enhanced_oi_vanna'].sum())
                
                if enhanced_oi_vanna_sum > 0.001:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        original_vanna_total = df_selected['net_vanna'].sum()
                        st.metric("Original VANNA Total", f"{original_vanna_total:.4f}{unit_label}")
                    
                    with col2:
                        enhanced_vanna_total = df_selected['enhanced_oi_vanna'].sum()
                        st.metric("Enhanced OI VANNA Total", f"{enhanced_vanna_total:.4f}{unit_label}")
                    
                    with col3:
                        if abs(original_vanna_total) > 0.001:
                            vanna_enhancement_ratio = enhanced_vanna_total / original_vanna_total
                            st.metric("VANNA Enhancement Ratio", f"{vanna_enhancement_ratio:.2f}x")
        
        with tabs[5]:
            st.markdown("### 📊 Delta Exposure (DEX)")
            st.plotly_chart(create_dex_chart(df_selected, spot_price, unit_label), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                call_dex_total = df_selected['call_dex'].sum()
                st.metric("Call DEX", f"{call_dex_total:.4f}{unit_label}")
            with col2:
                put_dex_total = df_selected['put_dex'].sum()
                st.metric("Put DEX", f"{put_dex_total:.4f}{unit_label}")
        
        with tabs[6]:
            st.markdown("### 📋 Open Interest Distribution")
            st.plotly_chart(create_oi_distribution_chart(df_selected, spot_price), use_container_width=True)
        
        with tabs[7]:
            st.markdown("### 📁 Data Summary")
            
            st.markdown(f"""
            **Symbol**: {meta.get('symbol', symbol)} ({meta.get('instrument_type', 'INDEX')})  
            **Fetch Mode**: {fetch_mode.upper()}  
            **Last Fetch**: {fetch_time}  
            **Total Records**: {meta.get('total_records', 0)}  
            **Time Range**: {meta.get('time_range', 'N/A')}  
            **Contract Size**: {meta.get('contract_size', 'N/A')}  
            **Unit**: ₹ {unit_label}
            """)
            
            display_cols = ['strike', 'net_gex', 'net_dex', 'net_vanna', 'total_volume', 'call_oi', 'put_oi', 'call_iv', 'put_iv']
            available_cols = [col for col in display_cols if col in df_selected.columns]
            st.dataframe(df_selected[available_cols], use_container_width=True, height=400)
            
            # Download button
            csv = df_selected.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f"nyztrade_{meta.get('symbol', symbol)}_{target_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        # Initial instructions
        st.info("""
        👋 **Welcome to NYZTrade UNIFIED Dashboard!**
        
        ### 🎯 Supports Both INDEX and STOCK Options:
        
        **📊 Index Options**
        - NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
        - Weekly & Monthly expiries
        - Values in ₹ Billions
        
        **📈 Stock Options (30 F&O Stocks)**
        - Banking: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK...
        - IT: TCS, INFY, WIPRO, HCLTECH, TECHM
        - Energy: RELIANCE, ONGC, POWERGRID, NTPC...
        - Auto: MARUTI, TATAMOTORS, M&M, LT
        - FMCG: HINDUNILVR, ITC, ASIANPAINT, TITAN
        - And more!
        - Monthly & Weekly expiries
        - Values in ₹ Crores
        
        ### 🔄 Smart Caching Features:
        
        **📦 Historical Data (Past Days)**
        - Fetched once and cached permanently
        - Instant loading on subsequent visits
        
        **📡 Current Trading Day (Live)**
        - Incremental updates fetch only NEW data
        - Auto-refresh available during market hours
        
        ### 📈 Available Charts:
        - **Intraday Timeline**: GEX/DEX evolution over time
        - **Standard GEX**: With Gamma Flip Zones
        - **Enhanced GEX Overlay**: OI-based with Greeks adjustments
        - **Standard VANNA**: Call/Put breakdown
        - **Enhanced VANNA Overlay**: With Volume/IV adjustments
        - **DEX Analysis**: Delta Exposure
        - **OI Distribution**: Call vs Put Open Interest
        
        **Click "🚀 Fetch Data" to begin!**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
            NYZTrade Unified GEX/DEX Dashboard | INDEX + STOCK Options<br>
            Smart Caching | VANNA/CHARM | Gamma Flip Zones | Drawing Tools
        </p>
        <p style="font-size: 0.75rem; margin-top: 8px;">
            ⚠️ For educational and research purposes only. Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
