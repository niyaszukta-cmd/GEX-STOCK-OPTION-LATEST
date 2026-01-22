# ============================================================================
# NYZTrade Stock Options GEX/DEX Dashboard - COMPLETE UNIFIED VERSION (FIXED)
# Features: Full Analysis + Screener | All Charts | Cache | VANNA/CHARM | Everything!
# ============================================================================

import subprocess
import sys

def install_packages():
    """Install required packages if not already installed"""
    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'scipy': 'scipy',
        'requests': 'requests',
        'pytz': 'pytz'
    }
    
    for package, pip_name in required.items():
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])

install_packages()

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
import pickle
import os
from pathlib import Path
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_DIR = Path("stock_gex_cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_INDEX_FILE = CACHE_DIR / "cache_index.pkl"
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.pkl"

# ============================================================================
# DATA CACHE MANAGER
# ============================================================================

class DataCacheManager:
    """Manages local cache of historical stock GEX data for instant access"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.index_file = CACHE_INDEX_FILE
        self.metadata_file = CACHE_METADATA_FILE
        self.cache_index = self.load_cache_index()
        self.metadata = self.load_metadata()
    
    def load_cache_index(self) -> Dict:
        """Load cache index from file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_cache_index(self):
        """Save cache index to file"""
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.cache_index, f)
    
    def load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def get_cache_key(self, symbol: str, date: str, strikes: List[str], 
                     interval: str, expiry_code: int, expiry_flag: str) -> str:
        """Generate unique cache key"""
        strikes_str = "_".join(sorted(strikes))
        key = f"{symbol}_{date}_{strikes_str}_{interval}_{expiry_flag}_{expiry_code}"
        if len(key) > 200:
            key = hashlib.md5(key.encode()).hexdigest()
        return key
    
    def get_cache_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if data is cached"""
        return cache_key in self.cache_index and self.get_cache_file_path(cache_key).exists()
    
    def save_to_cache(self, cache_key: str, df: pd.DataFrame, meta: Dict):
        """Save data to cache"""
        cache_file = self.get_cache_file_path(cache_key)
        
        data = {
            'df': df,
            'meta': meta,
            'cached_at': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.cache_index[cache_key] = {
                'file': str(cache_file),
                'cached_at': data['cached_at'],
                'symbol': meta.get('symbol', ''),
                'date': meta.get('date', ''),
                'records': len(df)
            }
            self.save_cache_index()
            
            if 'total_cached_datasets' not in self.metadata:
                self.metadata['total_cached_datasets'] = 0
            self.metadata['total_cached_datasets'] = len(self.cache_index)
            self.metadata['last_update'] = datetime.now().isoformat()
            self.save_metadata()
            return True
        except Exception as e:
            st.error(f"Cache save error: {e}")
            return False
    
    def load_from_cache(self, cache_key: str) -> tuple:
        """Load data from cache"""
        if not self.is_cached(cache_key):
            return None, None
        
        cache_file = self.get_cache_file_path(cache_key)
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['df'], data['meta']
        except Exception as e:
            st.error(f"Cache load error: {e}")
            return None, None
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.cache_index:
            return {
                'total_datasets': 0,
                'total_files': 0,
                'total_size_mb': 0,
                'date_range': 'No cached data',
                'symbols': [],
                'last_update': 'Never',
                'date_count': 0,
                'cached_dates': []
            }
        
        total_size = 0
        for item in self.cache_index.values():
            try:
                if Path(item['file']).exists():
                    total_size += Path(item['file']).stat().st_size
            except:
                pass
        
        dates = [item['date'] for item in self.cache_index.values() if 'date' in item]
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range = f"{min_date} to {max_date}"
            unique_dates = sorted(list(set(dates)))
        else:
            date_range = 'No dates'
            unique_dates = []
        
        symbols = list(set(item['symbol'] for item in self.cache_index.values() if 'symbol' in item))
        
        return {
            'total_datasets': len(self.cache_index),
            'total_files': len(self.cache_index),
            'total_size_mb': total_size / (1024 * 1024),
            'date_range': date_range,
            'symbols': symbols,
            'last_update': self.metadata.get('last_update', 'Unknown'),
            'date_count': len(unique_dates),
            'cached_dates': unique_dates
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        for cache_key, info in list(self.cache_index.items()):
            try:
                cache_file = Path(info['file'])
                if cache_file.exists():
                    cache_file.unlink()
            except:
                pass
        
        self.cache_index = {}
        self.metadata = {}
        self.save_cache_index()
        self.save_metadata()
    
    def get_cached_dates(self, symbol: str) -> List[str]:
        """Get list of cached dates for a symbol"""
        cached_dates = []
        for cache_key, info in self.cache_index.items():
            if info.get('symbol') == symbol and 'date' in info:
                cached_dates.append(info['date'])
        return sorted(list(set(cached_dates)))

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="NYZTrade Stock Options | Complete",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    header[data-testid="stHeader"] a[href*="github"] {
        display: none !important;
    }
    
    button[kind="header"][data-testid="baseButton-header"] svg {
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
    
    .screener-card {
        background: var(--bg-card);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .screener-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .screener-card.bullish { border-left: 4px solid var(--accent-green); }
    .screener-card.bearish { border-left: 4px solid var(--accent-red); }
    
    .opportunity-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 20px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .opportunity-badge.long {
        background: rgba(16, 185, 129, 0.2);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .opportunity-badge.short {
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .history-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
    }
    
    .history-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-blue);
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DhanConfig:
    client_id: str = "1100480354"
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY5MDY5NDMzLCJhcHBfaWQiOiJjOTNkM2UwOSIsImlhdCI6MTc2ODk4MzAzMywidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.1KED37cyhdeLN9U90tzN3ocxZLSB8Ao4ydERvap4eI8xAQq4PSfo8EvyjYYdBVPX4om3baMWj8-SMbGmMAIUmQ"

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

STOCK_CONFIG = {
    "RELIANCE": {"lot_size": 250, "strike_interval": 10},
    "TCS": {"lot_size": 150, "strike_interval": 25},
    "HDFCBANK": {"lot_size": 550, "strike_interval": 10},
    "INFY": {"lot_size": 300, "strike_interval": 25},
    "ICICIBANK": {"lot_size": 550, "strike_interval": 10},
    "SBIN": {"lot_size": 1500, "strike_interval": 5},
    "BHARTIARTL": {"lot_size": 410, "strike_interval": 10},
    "ITC": {"lot_size": 1600, "strike_interval": 5},
    "KOTAKBANK": {"lot_size": 400, "strike_interval": 25},
    "LT": {"lot_size": 300, "strike_interval": 25},
    "AXISBANK": {"lot_size": 600, "strike_interval": 10},
    "HINDUNILVR": {"lot_size": 300, "strike_interval": 25},
    "WIPRO": {"lot_size": 1200, "strike_interval": 5},
    "MARUTI": {"lot_size": 75, "strike_interval": 50},
    "BAJFINANCE": {"lot_size": 125, "strike_interval": 50},
    "HCLTECH": {"lot_size": 350, "strike_interval": 25},
    "ASIANPAINT": {"lot_size": 300, "strike_interval": 25},
    "TITAN": {"lot_size": 300, "strike_interval": 25},
    "ULTRACEMCO": {"lot_size": 100, "strike_interval": 50},
    "SUNPHARMA": {"lot_size": 400, "strike_interval": 25},
    "TATAMOTORS": {"lot_size": 1250, "strike_interval": 5},
    "TATASTEEL": {"lot_size": 900, "strike_interval": 5},
    "TECHM": {"lot_size": 400, "strike_interval": 25},
    "POWERGRID": {"lot_size": 1800, "strike_interval": 5},
    "NTPC": {"lot_size": 2250, "strike_interval": 5},
    "ONGC": {"lot_size": 2475, "strike_interval": 5},
    "M&M": {"lot_size": 300, "strike_interval": 25},
    "BAJAJFINSV": {"lot_size": 500, "strike_interval": 10},
    "ADANIPORTS": {"lot_size": 250, "strike_interval": 25},
    "COALINDIA": {"lot_size": 2040, "strike_interval": 5},
}

IST = pytz.timezone('Asia/Kolkata')

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
                'flip_type': 'Positive→Negative' if current_gex > 0 else 'Negative→Positive',
                'distance_from_spot': abs(flip_strike - spot_price),
                'distance_pct': abs(flip_strike - spot_price) / spot_price * 100
            })
    
    return flip_zones

def analyze_flip_zone_position(spot_price: float, flip_zones: List[Dict]) -> str:
    """Analyze spot position relative to flip zones"""
    if not flip_zones:
        return "No gamma flip zones detected"
    
    nearest_flip = min(flip_zones, key=lambda x: x['distance_from_spot'])
    
    above_flip = any(spot_price > zone['strike'] for zone in flip_zones)
    below_flip = any(spot_price < zone['strike'] for zone in flip_zones)
    
    if above_flip and not below_flip:
        return "Above all flip zones"
    elif below_flip and not above_flip:
        return "Below all flip zones"
    else:
        return "Between flip zones"

# ============================================================================
# DHAN STOCK OPTIONS FETCHER (ENHANCED WITH CACHE & ALL GREEKS)
# ============================================================================

class DhanStockOptionsFetcher:
    def __init__(self, config: DhanConfig, cache_manager: DataCacheManager = None):
        self.config = config
        self.cache_manager = cache_manager
        self.headers = {
            'access-token': config.access_token,
            'client-id': config.client_id,
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.dhan.co/v2"
        self.bs_calc = BlackScholesCalculator()
        self.risk_free_rate = 0.07
    
    def fetch_rolling_data(self, symbol: str, from_date: str, to_date: str, 
                          strike_type: str = "ATM", option_type: str = "CALL", 
                          interval: str = "60", expiry_code: int = 1, expiry_flag: str = "MONTH"):
        """Fetch rolling options data"""
        try:
            security_id = DHAN_STOCK_SECURITY_IDS.get(symbol)
            if not security_id:
                return None
            
            payload = {
                "exchangeSegment": "NSE_FNO",
                "interval": interval,
                "securityId": security_id,
                "instrument": "OPTSTK",
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
        except:
            return None
    
    def fetch_with_cache(self, symbol: str, target_date: str, strikes: List[str], 
                        interval: str, expiry_code: int, expiry_flag: str) -> tuple:
        """
        Fetch data with caching - checks cache first, fetches only if needed
        Returns: (df, meta, from_cache_flag)
        """
        if self.cache_manager is None:
            df, meta = self.process_historical_data(symbol, target_date, strikes, interval, expiry_code, expiry_flag)
            return df, meta, False
        
        cache_key = self.cache_manager.get_cache_key(
            symbol, target_date, strikes, interval, expiry_code, expiry_flag
        )
        
        if self.cache_manager.is_cached(cache_key):
            df, meta = self.cache_manager.load_from_cache(cache_key)
            if df is not None:
                return df, meta, True
        
        df, meta = self.process_historical_data(symbol, target_date, strikes, interval, expiry_code, expiry_flag)
        
        if df is not None and len(df) > 0:
            self.cache_manager.save_to_cache(cache_key, df, meta)
        
        return df, meta, False
    
    def process_historical_data(self, symbol: str, target_date: str, strikes: List[str], 
                               interval: str = "60", expiry_code: int = 1, expiry_flag: str = "MONTH"):
        """Process historical stock options data with FULL GREEKS"""
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        from_date = (target_dt - timedelta(days=2)).strftime('%Y-%m-%d')
        to_date = (target_dt + timedelta(days=2)).strftime('%Y-%m-%d')
        
        config = STOCK_CONFIG.get(symbol, {"lot_size": 500, "strike_interval": 10})
        lot_size = config["lot_size"]
        
        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(strikes) * 2
        current_step = 0
        
        for strike_type in strikes:
            status_text.text(f"Fetching {symbol} {strike_type}...")
            
            call_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "CALL", 
                                                interval, expiry_code, expiry_flag)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(1)
            
            put_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "PUT", 
                                               interval, expiry_code, expiry_flag)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(1)
            
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
                    
                    time_to_expiry = 30 / 365
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
                    
                    # Calculate exposures (in Crores)
                    call_gex = (call_oi * call_gamma * spot_price**2 * lot_size) / 1e7
                    put_gex = -(put_oi * put_gamma * spot_price**2 * lot_size) / 1e7
                    call_dex = (call_oi * call_delta * spot_price * lot_size) / 1e7
                    put_dex = (put_oi * put_delta * spot_price * lot_size) / 1e7
                    
                    call_vanna_exp = (call_oi * call_vanna * spot_price * lot_size) / 1e7
                    put_vanna_exp = (put_oi * put_vanna * spot_price * lot_size) / 1e7
                    call_charm_exp = (call_oi * call_charm * spot_price * lot_size) / 1e7
                    put_charm_exp = (put_oi * put_charm * spot_price * lot_size) / 1e7
                    
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
                        'call_gamma': call_gamma,
                        'put_gamma': put_gamma,
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
                    
                except:
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
                        time_to_expiry = 30 / 365
                        call_iv_dec = row['call_iv'] / 100 if row['call_iv'] > 1 else row['call_iv']
                        put_iv_dec = row['put_iv'] / 100 if row['put_iv'] > 1 else row['put_iv']
                        
                        call_gamma = self.bs_calc.calculate_gamma(spot, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                        put_gamma = self.bs_calc.calculate_gamma(spot, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                        
                        call_oi_change = row['call_oi_change']
                        put_oi_change = row['put_oi_change']
                        
                        call_oi_gex_val = (call_oi_change * call_gamma * spot**2 * lot_size) / 1e7
                        put_oi_gex_val = -(put_oi_change * put_gamma * spot**2 * lot_size) / 1e7
                        
                        df.loc[idx, 'call_oi_gex'] = call_oi_gex_val
                        df.loc[idx, 'put_oi_gex'] = put_oi_gex_val
                        df.loc[idx, 'net_oi_gex'] = call_oi_gex_val + put_oi_gex_val
                    except:
                        continue
        except:
            pass
        
        max_gex = df['net_gex'].abs().max()
        df['hedging_pressure'] = (df['net_gex'] / max_gex * 100) if max_gex > 0 else 0
        
        # ============================================================================
        # PREDICTIVE GEX MODELS
        # ============================================================================
        
        df['volume_weighted_gex'] = 0.0
        df['support_resistance_strength'] = 0.0
        
        # VANNA/CHARM ADJUSTED GEX
        df['vanna_adj_gex_vol_up'] = 0.0
        df['vanna_adj_gex_vol_down'] = 0.0
        df['charm_adj_gex_2hr'] = 0.0
        df['charm_adj_gex_4hr'] = 0.0
        
        try:
            for idx, row in df.iterrows():
                try:
                    spot = row['spot_price']
                    strike = row['strike']
                    net_gex = row['net_gex']
                    total_vol = row['total_volume']
                    timestamp = row['timestamp']
                    timestamp_mask = df['timestamp'] == timestamp
                    
                    net_vanna = row['net_vanna']
                    net_charm = row['net_charm']
                    
                    # Volume-Weighted GEX
                    total_vol_at_time = df[timestamp_mask]['total_volume'].sum()
                    if total_vol_at_time > 0:
                        volume_weight = total_vol / total_vol_at_time
                        vwgex = net_gex * volume_weight * 100
                        df.loc[idx, 'volume_weighted_gex'] = vwgex
                    
                    # Support/Resistance Strength
                    distance_from_spot = abs(strike - spot)
                    distance_pct = (distance_from_spot / spot) * 100
                    
                    if distance_pct > 0:
                        proximity_factor = 1 / (1 + distance_pct)
                    else:
                        proximity_factor = 1.0
                    
                    avg_volume = df[timestamp_mask]['total_volume'].mean()
                    volume_factor = (total_vol / avg_volume) if avg_volume > 0 else 1
                    
                    strength = abs(net_gex) * proximity_factor * volume_factor
                    df.loc[idx, 'support_resistance_strength'] = strength
                    
                    # VANNA-Adjusted GEX
                    vol_change_up = 0.05
                    vanna_impact_up = net_vanna * vol_change_up
                    vanna_adj_gex_up = net_gex + vanna_impact_up
                    df.loc[idx, 'vanna_adj_gex_vol_up'] = vanna_adj_gex_up
                    
                    vol_change_down = -0.05
                    vanna_impact_down = net_vanna * vol_change_down
                    vanna_adj_gex_down = net_gex + vanna_impact_down
                    df.loc[idx, 'vanna_adj_gex_vol_down'] = vanna_adj_gex_down
                    
                    # CHARM-Adjusted GEX
                    time_decay_2hr = 2 / 24
                    charm_impact_2hr = net_charm * time_decay_2hr * 10
                    charm_adj_gex_2hr = net_gex + charm_impact_2hr
                    df.loc[idx, 'charm_adj_gex_2hr'] = charm_adj_gex_2hr
                    
                    time_decay_4hr = 4 / 24
                    charm_impact_4hr = net_charm * time_decay_4hr * 10
                    charm_adj_gex_4hr = net_gex + charm_impact_4hr
                    df.loc[idx, 'charm_adj_gex_4hr'] = charm_adj_gex_4hr
                    
                except:
                    continue
        except:
            pass
        
        latest = df.sort_values('timestamp').iloc[-1]
        spot_prices = df['spot_price'].unique()
        spot_variation = (spot_prices.max() - spot_prices.min()) / spot_prices.mean() * 100
        
        df_latest_time = df[df['timestamp'] == df['timestamp'].max()]
        strike_volumes = df_latest_time.groupby('strike').agg({
            'call_volume': 'sum',
            'put_volume': 'sum',
            'total_volume': 'sum'
        }).reset_index()
        strike_volumes = strike_volumes.sort_values('total_volume', ascending=False)
        top_volume_strikes = strike_volumes.head(5).to_dict('records')
        
        meta = {
            'symbol': symbol,
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
            'lot_size': lot_size,
            'top_volume_strikes': top_volume_strikes
        }
        
        return df, meta

# ============================================================================
# VISUALIZATION FUNCTIONS (SIMPLIFIED - ONLY KEY CHARTS)
# ============================================================================

def create_intraday_timeline(df: pd.DataFrame, metadata: Dict) -> go.Figure:
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
            hovertemplate='%{x|%H:%M}<br>GEX: %{y:.4f}Cr<extra></extra>'
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
            hovertemplate='%{x|%H:%M}<br>DEX: %{y:.4f}Cr<extra></extra>'
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
    fig.update_yaxes(title_text="GEX (₹Cr)", row=1, col=1)
    fig.update_yaxes(title_text="DEX (₹Cr)", row=2, col=1)
    fig.update_yaxes(title_text="Spot Price (₹)", row=3, col=1)
    
    return fig

def create_separate_gex_chart(df: pd.DataFrame, metadata: Dict, flip_zones: List[Dict], spot_price: float) -> go.Figure:
    """GEX chart with flip zones"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_gex'],
        orientation='h',
        marker_color=colors,
        name='Net GEX',
        hovertemplate='Strike: %{y:,.0f}<br>Net GEX: %{x:.4f}Cr<extra></extra>',
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
            line_width=0
        )
    
    fig.update_layout(
        title=dict(text="<b>🎯 Gamma Exposure (GEX) with Flip Zones</b>",
                  font=dict(size=18, color='white')),
        xaxis_title="GEX (₹ Crores)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

def create_separate_dex_chart(df: pd.DataFrame, metadata: Dict) -> go.Figure:
    """DEX chart"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_dex']]
    spot_price = metadata.get('spot_price', df_sorted['spot_price'].iloc[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_dex'],
        orientation='h',
        marker_color=colors,
        name='Net DEX',
        hovertemplate='Strike: %{y:,.0f}<br>Net DEX: %{x:.4f}Cr<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right")
    
    fig.update_layout(
        title=dict(text="<b>📊 Delta Exposure (DEX)</b>", font=dict(size=18, color='white')),
        xaxis_title="DEX (₹ Crores)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False
    )
    
    return fig

def create_net_gex_dex_chart(df: pd.DataFrame, metadata: Dict, flip_zones: List[Dict], spot_price: float) -> go.Figure:
    """Combined NET GEX+DEX"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    df_sorted['net_gex_dex'] = df_sorted['net_gex'] + df_sorted['net_dex']
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex_dex']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['net_gex_dex'],
        orientation='h',
        marker_color=colors,
        name='Net GEX+DEX',
        hovertemplate='Strike: %{y:,.0f}<br>Net GEX+DEX: %{x:.4f}Cr<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right")
    
    for zone in flip_zones:
        fig.add_hline(
            y=zone['strike'],
            line_dash="dot",
            line_color=zone['color'],
            line_width=2
        )
    
    fig.update_layout(
        title=dict(text="<b>⚡ Combined NET GEX + DEX</b>", font=dict(size=18, color='white')),
        xaxis_title="Combined Exposure (₹ Crores)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False
    )
    
    return fig

def create_oi_distribution(df: pd.DataFrame, metadata: Dict, spot_price: float) -> go.Figure:
    """OI distribution"""
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
        title=dict(text="<b>📋 Open Interest Distribution</b>", font=dict(size=16, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=500,
        barmode='overlay'
    )
    
    return fig

# Screener visualization functions
def display_screener_results(df_results: pd.DataFrame):
    """Display screener results as cards"""
    for idx, row in df_results.iterrows():
        gex_color = '#10b981' if row['net_gex'] > 0 else '#ef4444'
        signal = '🟢 BULLISH' if row['net_gex'] > 0 else '🔴 BEARISH'
        
        card_html = f"""
        <div style="background: #1a2332; border: 2px solid #2d3748; border-left: 4px solid {gex_color}; 
                    border-radius: 12px; padding: 20px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc; font-size: 1.5rem;">{row['symbol']}</h3>
                    <p style="margin: 5px 0; color: #cbd5e1;">Spot: ₹{row['spot']:,.2f}</p>
                </div>
                <div style="background: rgba({16 if row['net_gex'] > 0 else 239}, {185 if row['net_gex'] > 0 else 68}, {129 if row['net_gex'] > 0 else 68}, 0.2);
                           color: {gex_color}; padding: 8px 16px; border-radius: 20px; font-weight: 600;">
                    {signal}
                </div>
            </div>
            <div style="margin-top: 15px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">NET GEX</p>
                    <p style="margin: 5px 0; color: {gex_color}; font-weight: 600;">{row['net_gex']:.4f}Cr</p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">NET DEX</p>
                    <p style="margin: 5px 0; color: #f8fafc; font-weight: 600;">{row['net_dex']:.4f}Cr</p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">FLIP ZONES</p>
                    <p style="margin: 5px 0; color: #fbbf24; font-weight: 600;">{row['flip_zones']}</p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">POSITION</p>
                    <p style="margin: 5px 0; color: #06b6d4; font-weight: 600; font-size: 0.8rem;">{row['flip_position']}</p>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

def create_screener_summary_chart(df_results: pd.DataFrame) -> go.Figure:
    """Summary chart for screener"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_results['symbol'],
        x=df_results['net_gex'],
        orientation='h',
        name='NET GEX',
        marker_color=['#10b981' if x > 0 else '#ef4444' for x in df_results['net_gex']]
    ))
    
    fig.update_layout(
        title="<b>NET GEX Comparison</b>",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=max(400, len(df_results) * 50)
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Initialize cache manager in session state
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = DataCacheManager()
    
    cache_manager = st.session_state.cache_manager
    
    # Header
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="main-title">🎯 NYZTrade Stock Options | Complete Dashboard</h1>
                <p class="sub-title">Full Analysis + Screener | ALL CHARTS | Cache | VANNA/CHARM | Complete!</p>
            </div>
            <div class="history-indicator">
                <div class="history-dot"></div>
                <span style="color: #3b82f6; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">CACHE ENABLED</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Mode Selection")
        
        app_mode = st.radio(
            "Choose Mode",
            ["📊 Single Stock Analysis", "🔍 Multi-Stock Screener"],
            index=0
        )
        
        st.markdown("---")
        
        if app_mode == "📊 Single Stock Analysis":
            st.markdown("#### 💾 CACHE STATUS")
            
            stats = cache_manager.get_cache_stats()
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1)); 
                        border: 1px solid rgba(59,130,246,0.3); border-radius: 12px; padding: 16px; margin: 12px 0;'>
                <div style='font-size: 0.9rem; font-weight: 600; color: #3b82f6; margin-bottom: 8px;'>📊 Cache Statistics</div>
                <div style='font-size: 0.75rem; color: #94a3b8; margin: 4px 0;'>📁 Datasets: {stats.get('total_datasets', 0)}</div>
                <div style='font-size: 0.75rem; color: #94a3b8; margin: 4px 0;'>💿 Size: {stats['total_size_mb']:.1f} MB</div>
                <div style='font-size: 0.75rem; color: #94a3b8; margin: 4px 0;'>📅 Days: {stats.get('date_count', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                cache_manager.clear_cache()
                st.success("Cache cleared!")
                st.rerun()
            
            st.markdown("---")
            st.markdown("#### Stock Selection")
            
            category = st.selectbox("Category", [
                "Banking & Finance", "IT & Technology", "Energy & Power",
                "Auto & Industrial", "FMCG & Consumer", "Others"
            ])
            
            category_stocks = {
                "Banking & Finance": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "BAJAJFINSV"],
                "IT & Technology": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
                "Energy & Power": ["RELIANCE", "ONGC", "POWERGRID", "NTPC", "COALINDIA"],
                "Auto & Industrial": ["MARUTI", "TATAMOTORS", "M&M", "LT"],
                "FMCG & Consumer": ["HINDUNILVR", "ITC", "ASIANPAINT", "TITAN"],
                "Others": ["SUNPHARMA", "TATASTEEL", "BHARTIARTL", "ADANIPORTS"]
            }
            
            symbol = st.selectbox("Stock", category_stocks[category])
            
            stock_info = STOCK_CONFIG.get(symbol, {"lot_size": 500, "strike_interval": 10})
            st.info(f"📦 Lot: {stock_info['lot_size']}\n⚡ Interval: ₹{stock_info['strike_interval']}")
            
            st.markdown("---")
            st.markdown("#### Date Selection")
            
            date_range_option = st.selectbox("Range", ["Last 30 Days", "Last 60 Days", "Last 90 Days", "Custom"], index=0)
            
            if date_range_option == "Custom":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start", value=datetime.now() - timedelta(days=30),
                                               max_value=datetime.now(), min_value=datetime.now() - timedelta(days=90))
                with col2:
                    end_date = st.date_input("End", value=datetime.now(),
                                             max_value=datetime.now(), min_value=start_date)
                date_list = pd.date_range(start=start_date, end=end_date, freq='D')
                date_list = [d for d in date_list if d.weekday() < 5]
            else:
                days_back = {"Last 30 Days": 30, "Last 60 Days": 60, "Last 90 Days": 90}[date_range_option]
                date_list = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
                date_list = [d for d in date_list if d.weekday() < 5]
            
            available_dates = [d.date() for d in date_list]
            
            if len(available_dates) > 0:
                st.caption(f"📊 {len(available_dates)} trading days")
            
            selected_date = st.selectbox("Trading Day", options=available_dates,
                                        index=len(available_dates)-1 if len(available_dates) > 0 else 0,
                                        format_func=lambda x: x.strftime('%Y-%m-%d (%A)'))
            
            target_date = selected_date.strftime('%Y-%m-%d')
            
            st.markdown("---")
            st.markdown("#### Analysis Settings")
            
            expiry_type = st.selectbox("Expiry", ["Monthly", "Weekly"], index=0)
            expiry_flag = "MONTH" if expiry_type == "Monthly" else "WEEK"
            expiry_code = st.selectbox("Expiry Code", [1, 2, 3], index=0,
                                      format_func=lambda x: {1: "Current", 2: "Next", 3: "Far"}[x])
            
            strikes = st.multiselect("Strikes",
                ["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3",
                 "ATM+4", "ATM-4", "ATM+5", "ATM-5"],
                default=["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3"])
            
            interval = st.selectbox("Interval", options=["1", "5", "15", "60"],
                                   format_func=lambda x: {"1": "1 min", "5": "5 min", "15": "15 min", "60": "1 hour"}[x],
                                   index=2)
            
            st.info(f"📊 {len(strikes)} strikes | {interval} min")
            
            st.markdown("---")
            
            fetch_button = st.button("🚀 Fetch Data", use_container_width=True, type="primary")
        
        else:  # Screener mode
            st.markdown("#### Stock Selection")
            
            preset = st.selectbox("Quick Preset",
                ["Custom", "Banking Stocks", "IT Stocks", "High Volume Stocks", "All Stocks (30)"],
                index=0)
            
            if preset == "Custom":
                selected_stocks = st.multiselect("Select Stocks",
                    options=sorted(list(DHAN_STOCK_SECURITY_IDS.keys())),
                    default=["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"])
            elif preset == "Banking Stocks":
                selected_stocks = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"]
            elif preset == "IT Stocks":
                selected_stocks = ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"]
            elif preset == "High Volume Stocks":
                selected_stocks = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
                                  "SBIN", "BHARTIARTL", "ITC", "LT", "AXISBANK"]
            else:
                selected_stocks = sorted(list(DHAN_STOCK_SECURITY_IDS.keys()))
            
            st.info(f"📊 {len(selected_stocks)} stocks selected")
            
            st.markdown("---")
            st.markdown("#### Screener Filter")
            
            filter_type = st.radio("Filter Type",
                ["🟢 Spot Above Gamma Flip", 
                 "🔴 Spot Below Gamma Flip", 
                 "🔵 All Stocks with Flip Zones",
                 "✅ NET GEX Positive",
                 "❌ NET GEX Negative"],
                index=0)
            
            st.markdown("---")
            st.markdown("#### Settings")
            
            expiry_type = st.selectbox("Expiry", ["Monthly", "Weekly"], index=0)
            expiry_flag = "MONTH" if expiry_type == "Monthly" else "WEEK"
            expiry_code = st.selectbox("Expiry Code", [1, 2, 3], index=0,
                                      format_func=lambda x: {1: "Current", 2: "Next", 3: "Far"}[x])
            
            strikes_count = st.slider("Strikes per Stock", 3, 7, 5)
            strike_list = ["ATM"] + [f"ATM+{i}" for i in range(1, strikes_count)] + [f"ATM-{i}" for i in range(1, strikes_count)]
            
            st.markdown("---")
            
            run_screener = st.button("🚀 Run Screener", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.markdown("### 🕐 IST")
        ist_now = datetime.now(IST)
        st.info(f"{ist_now.strftime('%H:%M:%S')}")
    
    # Main content
    if app_mode == "📊 Single Stock Analysis":
        # Store config
        if fetch_button:
            st.session_state.fetch_config = {
                'symbol': symbol,
                'target_date': target_date,
                'strikes': strikes,
                'interval': interval,
                'expiry_code': expiry_code,
                'expiry_flag': expiry_flag
            }
            st.session_state.data_fetched = False
        
        # Fetch data if button clicked
        if 'fetch_config' in st.session_state and not st.session_state.get('data_fetched', False):
            config_obj = st.session_state.fetch_config
            
            with st.spinner("🔍 Fetching data with cache..."):
                # FIX: Create DhanConfig and instantiate fetcher properly
                dhan_config = DhanConfig()
                fetcher = DhanStockOptionsFetcher(dhan_config, cache_manager)
                
                try:
                    df, metadata, from_cache = fetcher.fetch_with_cache(
                        symbol=config_obj['symbol'],
                        target_date=config_obj['target_date'],
                        strikes=config_obj['strikes'],
                        interval=config_obj['interval'],
                        expiry_code=config_obj['expiry_code'],
                        expiry_flag=config_obj['expiry_flag']
                    )
                    
                    if df is not None and len(df) > 0:
                        st.session_state.df_data = df
                        st.session_state.meta_data = metadata
                        st.session_state.data_fetched = True
                        st.session_state.timestamp_idx = 0
                        
                        cache_status = "✅ LOADED FROM CACHE" if from_cache else "🆕 FETCHED FRESH"
                        st.success(f"{cache_status} | {len(df)} records | {len(df['timestamp'].unique())} timestamps")
                    else:
                        st.error("No data received")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display data if available
        if st.session_state.get('data_fetched', False) and 'df_data' in st.session_state:
            df = st.session_state.df_data
            metadata = st.session_state.meta_data
            
            # Time navigation
            unique_timestamps = sorted(df['timestamp'].unique())
            
            if len(unique_timestamps) > 0:
                st.markdown("---")
                st.markdown("### ⏰ Time Navigation")
                
                col1, col2, col3 = st.columns([2, 6, 2])
                
                with col1:
                    if st.button("⏮️ First", use_container_width=True):
                        st.session_state.timestamp_idx = 0
                    if st.button("◀️ Prev", use_container_width=True):
                        st.session_state.timestamp_idx = max(0, st.session_state.timestamp_idx - 1)
                
                with col2:
                    st.session_state.timestamp_idx = st.slider(
                        "Time",
                        0,
                        len(unique_timestamps) - 1,
                        st.session_state.get('timestamp_idx', 0),
                        format=f"{unique_timestamps[st.session_state.get('timestamp_idx', 0)].strftime('%H:%M:%S')}"
                    )
                
                with col3:
                    if st.button("Next ▶️", use_container_width=True):
                        st.session_state.timestamp_idx = min(len(unique_timestamps) - 1, st.session_state.timestamp_idx + 1)
                    if st.button("Last ⏭️", use_container_width=True):
                        st.session_state.timestamp_idx = len(unique_timestamps) - 1
                
                current_timestamp = unique_timestamps[st.session_state.timestamp_idx]
                df_current = df[df['timestamp'] == current_timestamp].copy()
                
                st.info(f"🕐 {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')} IST | Record {st.session_state.timestamp_idx + 1}/{len(unique_timestamps)}")
                
                # Get spot price
                spot_price = metadata.get('spot_price', df_current['spot_price'].iloc[0] if len(df_current) > 0 else 0)
                
                # Calculate flip zones
                flip_zones = identify_gamma_flip_zones(df_current, spot_price)
                flip_position = analyze_flip_zone_position(spot_price, flip_zones)
                
                # Display key metrics
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    net_gex = df_current['net_gex'].sum()
                    st.metric("NET GEX", f"₹{abs(net_gex):.2f}Cr", 
                             delta="Bullish" if net_gex > 0 else "Bearish")
                
                with col2:
                    net_dex = df_current['net_dex'].sum()
                    st.metric("NET DEX", f"₹{abs(net_dex):.2f}Cr",
                             delta="Long" if net_dex > 0 else "Short")
                
                with col3:
                    st.metric("SPOT", f"₹{spot_price:,.2f}", 
                             delta=metadata.get('symbol', 'STOCK'))
                
                with col4:
                    hedging_pressure = df_current['hedging_pressure'].mean()
                    st.metric("PRESSURE", f"{hedging_pressure:+.1f}%",
                             delta="High" if abs(hedging_pressure) > 40 else "Normal")
                
                with col5:
                    st.metric("FLIP ZONES", len(flip_zones),
                             delta=flip_position)
                
                # Tabs for analysis
                st.markdown("---")
                
                tabs = st.tabs([
                    "📊 Overview",
                    "📈 GEX Analysis", 
                    "📉 DEX Analysis",
                    "🎯 OI Analysis",
                    "📋 Data Table"
                ])
                
                with tabs[0]:  # Overview
                    st.markdown("### 📊 Intraday Timeline")
                    df_timeline = df[df['timestamp'] <= current_timestamp].copy()
                    if len(df_timeline) > 0:
                        fig = create_intraday_timeline(df_timeline, metadata)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 💾 Gamma Flip Zones")
                    if len(flip_zones) > 0:
                        for i, zone in enumerate(flip_zones[:5], 1):
                            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
                            with col1:
                                st.metric("Zone", f"#{i}")
                            with col2:
                                st.metric("Strike", f"₹{zone['strike']:.2f}")
                            with col3:
                                st.metric("Distance", f"{zone['distance_pct']:.2f}%")
                            with col4:
                                signal = "🟢 Above" if spot_price > zone['strike'] else "🔴 Below"
                                st.metric("Position", signal)
                    else:
                        st.info("No gamma flip zones detected")
                
                with tabs[1]:  # GEX Analysis
                    st.markdown("### 📊 GEX with Flip Zones")
                    fig = create_separate_gex_chart(df_current, metadata, flip_zones, spot_price)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📊 GEX + DEX Combined")
                    fig = create_net_gex_dex_chart(df_current, metadata, flip_zones, spot_price)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabs[2]:  # DEX Analysis
                    st.markdown("### 📊 Delta Exposure")
                    fig = create_separate_dex_chart(df_current, metadata)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabs[3]:  # OI Analysis
                    st.markdown("### 📊 Open Interest Distribution")
                    fig = create_oi_distribution(df_current, metadata, spot_price)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabs[4]:  # Data Table
                    st.markdown("### 📋 Complete Data Table")
                    display_cols = ['strike', 'call_oi', 'call_volume', 'call_gex', 'call_dex', 
                                   'put_oi', 'put_volume', 'put_gex', 'put_dex', 
                                   'net_gex', 'net_dex', 'hedging_pressure']
                    
                    display_df = df_current[display_cols].copy()
                    display_df = display_df.sort_values('strike')
                    st.dataframe(display_df, use_container_width=True, height=600)
        
        else:
            st.info("👈 Configure settings in sidebar and click 'Fetch Data' to begin analysis")
    
    else:  # SCREENER MODE
        if run_screener:
            st.markdown("### 🔍 Multi-Stock Screener Results")
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # FIX: Create DhanConfig before the loop
            dhan_config = DhanConfig()
            
            for idx, stock_symbol in enumerate(selected_stocks):
                status_text.text(f"Screening {stock_symbol}... ({idx+1}/{len(selected_stocks)})")
                progress_bar.progress((idx + 1) / len(selected_stocks))
                
                try:
                    # FIX: Instantiate fetcher with config and cache_manager
                    fetcher = DhanStockOptionsFetcher(dhan_config, cache_manager)
                    today = datetime.now().strftime('%Y-%m-%d')
                    
                    df, metadata, from_cache = fetcher.fetch_with_cache(
                        symbol=stock_symbol,
                        target_date=today,
                        strikes=strike_list,
                        interval="15",
                        expiry_code=expiry_code,
                        expiry_flag=expiry_flag
                    )
                    
                    if df is not None and len(df) > 0:
                        # Get latest data
                        latest_time = df['timestamp'].max()
                        df_latest = df[df['timestamp'] == latest_time].copy()
                        
                        spot_price = metadata.get('spot_price', df_latest['spot_price'].iloc[0])
                        
                        # Calculate metrics
                        net_gex = df_latest['net_gex'].sum()
                        net_dex = df_latest['net_dex'].sum()
                        
                        flip_zones = identify_gamma_flip_zones(df_latest, spot_price)
                        flip_position = analyze_flip_zone_position(spot_price, flip_zones)
                        
                        result = {
                            'symbol': stock_symbol,
                            'spot': spot_price,
                            'net_gex': net_gex,
                            'net_dex': net_dex,
                            'flip_zones': len(flip_zones),
                            'flip_position': flip_position,
                            'timestamp': latest_time
                        }
                        
                        results.append(result)
                
                except Exception as e:
                    st.warning(f"Error processing {stock_symbol}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if len(results) > 0:
                results_df = pd.DataFrame(results)
                
                # Apply filters
                filtered_df = results_df.copy()
                
                if filter_type == "🟢 Spot Above Gamma Flip":
                    filtered_df = filtered_df[filtered_df['flip_position'] == "Above all flip zones"]
                elif filter_type == "🔴 Spot Below Gamma Flip":
                    filtered_df = filtered_df[filtered_df['flip_position'] == "Below all flip zones"]
                elif filter_type == "🔵 All Stocks with Flip Zones":
                    filtered_df = filtered_df[filtered_df['flip_zones'] > 0]
                elif filter_type == "✅ NET GEX Positive":
                    filtered_df = filtered_df[filtered_df['net_gex'] > 0]
                elif filter_type == "❌ NET GEX Negative":
                    filtered_df = filtered_df[filtered_df['net_gex'] < 0]
                
                st.success(f"✅ Found {len(filtered_df)} stocks matching filter: **{filter_type}**")
                
                if len(filtered_df) > 0:
                    # Display results
                    st.markdown("### 📊 Screener Results")
                    filtered_df = filtered_df.sort_values('net_gex', key=abs, ascending=False)
                    display_screener_results(filtered_df)
                    
                    # Visualization
                    st.markdown("---")
                    st.markdown("### 📊 Comparison Chart")
                    fig = create_screener_summary_chart(filtered_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export
                    st.markdown("---")
                    st.markdown("### 💾 Export Results")
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"nyztrade_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning(f"No stocks found matching the filter: {filter_type}")
            else:
                st.error("No data retrieved for any stocks")
        
        else:
            st.info("👈 Configure settings in sidebar and click 'Run Screener' to start screening")
    
    st.markdown("---")
    st.markdown(f"""<div style="text-align: center; padding: 20px;">
        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #64748b;">
        NYZTrade Stock Options | Complete Unified Dashboard<br>
        ALL Charts | Cache System | VANNA/CHARM | Predictive Models | Screener</p>
        <p style="font-size: 0.75rem;">⚠️ For educational purposes only</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
