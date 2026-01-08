# ============================================================================
# NYZTrade Stock Options GEX/DEX Dashboard - UNIFIED WITH SCREENER
# Features: Full Analysis + Multi-Stock Screener | All-in-One
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="NYZTrade Stock Options | Unified",
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
    
    :root {
        --bg-primary: #0a0e17;
        --bg-card: #1a2332;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-blue: #3b82f6;
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
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card.positive { border-left: 4px solid var(--accent-green); }
    .metric-card.negative { border-left: 4px solid var(--accent-red); }
    .metric-card.neutral { border-left: 4px solid var(--accent-yellow); }
    
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DhanConfig:
    client_id: str = "1100480354"
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY3OTgwMzg1LCJhcHBfaWQiOiJlZDMwMzI5NCIsImlhdCI6MTc2Nzg5Mzk4NSwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.7M7pbTIGcShLApqITPCMOqsD9WynmCf1f3NjGkiu5yKMpeHLEHNLFgnL1Ip-ueuE37WMi2qxQ-cCV82W4w7QGA"
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
# BLACK-SCHOLES & GREEKS
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
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator.calculate_d2(S, K, T, r, sigma)
            return -norm.pdf(d1) * d2 / sigma
        except:
            return 0
    
    @staticmethod
    def calculate_charm(S, K, T, r, sigma, option_type='call'):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            d2 = BlackScholesCalculator.calculate_d2(S, K, T, r, sigma)
            return -norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
        except:
            return 0

# ============================================================================
# GAMMA FLIP ZONE DETECTION
# ============================================================================

def identify_gamma_flip_zones(df: pd.DataFrame, spot_price: float) -> List[Dict]:
    """Identifies gamma flip zones where GEX crosses zero"""
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
                direction = "upward" if current_gex > 0 else "downward"
                arrow = "↑" if current_gex > 0 else "↓"
                color = "#ef4444" if current_gex > 0 else "#10b981"
            else:
                direction = "downward" if current_gex < 0 else "upward"
                arrow = "↓" if current_gex < 0 else "↑"
                color = "#10b981" if current_gex < 0 else "#ef4444"
            
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

def analyze_flip_zone_position(spot_price: float, flip_zones: List[Dict]) -> Dict:
    """Analyze spot position relative to flip zones"""
    if not flip_zones:
        return {
            'has_flip_zones': False,
            'position': 'no_flip_zones',
            'signal': 'neutral',
            'nearest_flip': None,
            'description': 'No gamma flip zones detected'
        }
    
    nearest_flip = min(flip_zones, key=lambda x: x['distance_from_spot'])
    
    above_flip = any(spot_price > zone['strike'] for zone in flip_zones)
    below_flip = any(spot_price < zone['strike'] for zone in flip_zones)
    
    if above_flip and not below_flip:
        position = 'above_all'
        signal = 'bullish_continuation' if nearest_flip['direction'] == 'upward' else 'bearish_reversal'
    elif below_flip and not above_flip:
        position = 'below_all'
        signal = 'bearish_continuation' if nearest_flip['direction'] == 'downward' else 'bullish_reversal'
    else:
        position = 'between'
        signal = 'range_bound'
    
    return {
        'has_flip_zones': True,
        'position': position,
        'signal': signal,
        'nearest_flip': nearest_flip,
        'all_flips': flip_zones,
        'flip_count': len(flip_zones)
    }

# ============================================================================
# DATA FETCHER (UNIFIED)
# ============================================================================

class DhanStockOptionsFetcher:
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
    
    def process_historical_data(self, symbol: str, target_date: str, strikes: List[str], 
                               interval: str = "60", expiry_code: int = 1, expiry_flag: str = "MONTH"):
        """Process historical stock options data"""
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
                    
                    call_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    call_delta = self.bs_calc.calculate_call_delta(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_delta = self.bs_calc.calculate_put_delta(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    
                    call_vanna = self.bs_calc.calculate_vanna(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec)
                    put_vanna = self.bs_calc.calculate_vanna(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec)
                    call_charm = self.bs_calc.calculate_charm(spot_price, strike_price, time_to_expiry, self.risk_free_rate, call_iv_dec, 'call')
                    put_charm = self.bs_calc.calculate_charm(spot_price, strike_price, time_to_expiry, self.risk_free_rate, put_iv_dec, 'put')
                    
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
        
        max_gex = df['net_gex'].abs().max()
        df['hedging_pressure'] = (df['net_gex'] / max_gex * 100) if max_gex > 0 else 0
        
        latest = df.sort_values('timestamp').iloc[-1]
        spot_prices = df['spot_price'].unique()
        spot_variation = (spot_prices.max() - spot_prices.min()) / spot_prices.mean() * 100
        
        # Find high volume strikes from latest timestamp data
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
    
    def screen_single_stock(self, symbol: str, strikes: List[str] = None,
                           expiry_code: int = 1, expiry_flag: str = "MONTH") -> Dict:
        """Screen a single stock for gamma flip zones"""
        if strikes is None:
            strikes = ["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2"]
        
        today = datetime.now()
        from_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        config = STOCK_CONFIG.get(symbol, {"lot_size": 500, "strike_interval": 10})
        lot_size = config["lot_size"]
        
        all_data = []
        strike_volumes = []  # Track individual strike volumes
        
        for strike_type in strikes:
            call_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "CALL", 
                                                "60", expiry_code, expiry_flag)
            time.sleep(0.5)
            
            put_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "PUT", 
                                               "60", expiry_code, expiry_flag)
            time.sleep(0.5)
            
            if not call_data or not put_data:
                continue
            
            ce_data = call_data.get('ce', {})
            pe_data = put_data.get('pe', {})
            
            if not ce_data:
                continue
            
            try:
                spot_price = ce_data.get('spot', [0])[-1] if ce_data.get('spot') else 0
                strike_price = ce_data.get('strike', [0])[-1] if ce_data.get('strike') else 0
                
                if spot_price == 0 or strike_price == 0:
                    continue
                
                call_oi = ce_data.get('oi', [0])[-1] if ce_data.get('oi') else 0
                put_oi = pe_data.get('oi', [0])[-1] if pe_data.get('oi') else 0
                call_volume = ce_data.get('volume', [0])[-1] if ce_data.get('volume') else 0
                put_volume = pe_data.get('volume', [0])[-1] if pe_data.get('volume') else 0
                call_iv = ce_data.get('iv', [15])[-1] if ce_data.get('iv') else 15
                put_iv = pe_data.get('iv', [15])[-1] if pe_data.get('iv') else 15
                
                time_to_expiry = 30 / 365
                call_iv_dec = call_iv / 100 if call_iv > 1 else call_iv
                put_iv_dec = put_iv / 100 if put_iv > 1 else put_iv
                
                call_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, 
                                                         self.risk_free_rate, call_iv_dec)
                put_gamma = self.bs_calc.calculate_gamma(spot_price, strike_price, time_to_expiry, 
                                                        self.risk_free_rate, put_iv_dec)
                call_delta = self.bs_calc.calculate_call_delta(spot_price, strike_price, time_to_expiry, 
                                                               self.risk_free_rate, call_iv_dec)
                put_delta = self.bs_calc.calculate_put_delta(spot_price, strike_price, time_to_expiry, 
                                                             self.risk_free_rate, put_iv_dec)
                
                call_gex = (call_oi * call_gamma * spot_price**2 * lot_size) / 1e7
                put_gex = -(put_oi * put_gamma * spot_price**2 * lot_size) / 1e7
                call_dex = (call_oi * call_delta * spot_price * lot_size) / 1e7
                put_dex = (put_oi * put_delta * spot_price * lot_size) / 1e7
                
                # Track strike volumes and actual gamma values
                strike_volumes.append({
                    'strike': strike_price,
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'total_volume': call_volume + put_volume,
                    'call_gamma': call_gamma,  # Actual BS gamma (e.g., 0.05)
                    'put_gamma': put_gamma,    # Actual BS gamma (e.g., 0.03)
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'moneyness': 'OTM_CALL' if strike_price > spot_price else ('OTM_PUT' if strike_price < spot_price else 'ATM')
                })
                
                all_data.append({
                    'spot_price': spot_price,
                    'strike': strike_price,
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'call_volume': call_volume,
                    'put_volume': put_volume,
                    'call_gamma': call_gamma,
                    'put_gamma': put_gamma,
                    'call_gex': call_gex,
                    'put_gex': put_gex,
                    'net_gex': call_gex + put_gex,
                    'call_dex': call_dex,
                    'put_dex': put_dex,
                    'net_dex': call_dex + put_dex,
                })
            except:
                continue
        
        if not all_data:
            return {'success': False, 'symbol': symbol, 'error': 'No data available'}
        
        df = pd.DataFrame(all_data)
        spot_price = df['spot_price'].iloc[0]
        
        flip_zones = identify_gamma_flip_zones(df, spot_price)
        flip_analysis = analyze_flip_zone_position(spot_price, flip_zones)
        
        total_gex = df['net_gex'].sum()
        total_dex = df['net_dex'].sum()
        pcr = df['put_oi'].sum() / df['call_oi'].sum() if df['call_oi'].sum() > 0 else 1
        total_call_volume = df['call_volume'].sum()
        total_put_volume = df['put_volume'].sum()
        total_volume = total_call_volume + total_put_volume
        volume_pcr = total_put_volume / total_call_volume if total_call_volume > 0 else 1
        
        # Find high volume strikes
        if strike_volumes:
            strike_df = pd.DataFrame(strike_volumes)
            top_volume_strikes = strike_df.nlargest(3, 'total_volume')[['strike', 'total_volume']].to_dict('records')
            
            # Calculate gamma differential (OTM calls vs OTM puts)
            # Using actual Black-Scholes gamma values (not multiplied by OI)
            otm_calls = strike_df[strike_df['moneyness'] == 'OTM_CALL']
            otm_puts = strike_df[strike_df['moneyness'] == 'OTM_PUT']
            
            # Sum of actual BS gamma values across OTM strikes
            total_otm_call_gamma = otm_calls['call_gamma'].sum() if len(otm_calls) > 0 else 0
            total_otm_put_gamma = otm_puts['put_gamma'].sum() if len(otm_puts) > 0 else 0
            gamma_differential = total_otm_call_gamma - total_otm_put_gamma
            
            # Normalize by number of strikes for fair comparison
            gamma_diff_normalized = gamma_differential
        else:
            top_volume_strikes = []
            gamma_differential = 0
            gamma_diff_normalized = 0
        
        return {
            'success': True,
            'symbol': symbol,
            'spot_price': spot_price,
            'total_gex': total_gex,
            'total_dex': total_dex,
            'pcr': pcr,
            'total_volume': total_volume,
            'call_volume': total_call_volume,
            'put_volume': total_put_volume,
            'volume_pcr': volume_pcr,
            'top_volume_strikes': top_volume_strikes,
            'gamma_differential': gamma_differential,
            'gamma_diff_normalized': gamma_diff_normalized,
            'flip_zones': flip_zones,
            'flip_analysis': flip_analysis,
            'strikes_analyzed': len(df),
            'timestamp': datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')
        }
    
    def screen_multiple_stocks(self, symbols: List[str], strikes: List[str] = None,
                               expiry_code: int = 1, expiry_flag: str = "MONTH") -> pd.DataFrame:
        """Screen multiple stocks"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Screening {symbol}... ({i+1}/{len(symbols)})")
            
            result = self.screen_single_stock(symbol, strikes, expiry_code, expiry_flag)
            
            if result['success']:
                results.append(result)
            
            progress_bar.progress((i + 1) / len(symbols))
            time.sleep(1)
        
        progress_bar.empty()
        status_text.empty()
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION FUNCTIONS (Keeping all existing charts)
# ============================================================================

def create_intraday_timeline(df: pd.DataFrame, selected_timestamp) -> go.Figure:
    """Create intraday timeline"""
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
        go.Bar(x=timeline_df['timestamp'], y=timeline_df['net_gex'], marker_color=gex_colors,
               name='Net GEX', hovertemplate='%{x|%H:%M}<br>GEX: %{y:.4f}Cr<extra></extra>'),
        row=1, col=1
    )
    
    dex_colors = ['#10b981' if x > 0 else '#ef4444' for x in timeline_df['net_dex']]
    fig.add_trace(
        go.Bar(x=timeline_df['timestamp'], y=timeline_df['net_dex'], marker_color=dex_colors,
               name='Net DEX', hovertemplate='%{x|%H:%M}<br>DEX: %{y:.4f}Cr<extra></extra>'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=timeline_df['timestamp'], y=timeline_df['spot_price'],
                  mode='lines+markers', line=dict(color='#3b82f6', width=2),
                  marker=dict(size=4), name='Spot Price', fill='tozeroy',
                  fillcolor='rgba(59, 130, 246, 0.1)',
                  hovertemplate='%{x|%H:%M}<br>Spot: ₹%{y:,.2f}<extra></extra>'),
        row=3, col=1
    )
    
    fig.add_vline(x=selected_timestamp, line_dash="dash", line_color="#f59e0b", line_width=3, row=1, col=1)
    fig.add_vline(x=selected_timestamp, line_dash="dash", line_color="#f59e0b", line_width=3, row=2, col=1)
    fig.add_vline(x=selected_timestamp, line_dash="dash", line_color="#f59e0b", line_width=3, row=3, col=1)
    
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

def create_separate_gex_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """GEX chart with flip zones"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex']]
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['net_gex'], orientation='h',
                        marker_color=colors, name='Net GEX',
                        hovertemplate='Strike: %{y:,.0f}<br>Net GEX: %{x:.4f}Cr<extra></extra>',
                        showlegend=False))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    for zone in flip_zones:
        fig.add_hline(y=zone['strike'], line_dash="dot", line_color=zone['color'], line_width=2,
                      annotation_text=f"🔄 Flip {zone['arrow']} {zone['strike']:,.0f}",
                      annotation_position="left",
                      annotation=dict(font=dict(size=10, color=zone['color']),
                                    bgcolor='rgba(0,0,0,0.7)', bordercolor=zone['color'], borderwidth=1))
        
        fig.add_hrect(y0=zone['lower_strike'], y1=zone['upper_strike'],
                     fillcolor=zone['color'], opacity=0.1, line_width=0,
                     annotation_text=zone['arrow'], annotation_position="right",
                     annotation=dict(font=dict(size=16, color=zone['color'])))
    
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
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_separate_dex_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """DEX chart"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_dex']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['net_dex'], orientation='h',
                        marker_color=colors, name='Net DEX',
                        hovertemplate='Strike: %{y:,.0f}<br>Net DEX: %{x:.4f}Cr<extra></extra>',
                        showlegend=False))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    fig.update_layout(
        title=dict(text="<b>📊 Delta Exposure (DEX)</b>", font=dict(size=18, color='white')),
        xaxis_title="DEX (₹ Crores)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_net_gex_dex_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """Combined NET GEX+DEX"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    df_sorted['net_gex_dex'] = df_sorted['net_gex'] + df_sorted['net_dex']
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex_dex']]
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['net_gex_dex'], orientation='h',
                        marker_color=colors, name='Net GEX+DEX',
                        hovertemplate='Strike: %{y:,.0f}<br>Net GEX+DEX: %{x:.4f}Cr<extra></extra>',
                        showlegend=False))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    for zone in flip_zones:
        fig.add_hline(y=zone['strike'], line_dash="dot", line_color=zone['color'], line_width=2,
                      annotation_text=f"🔄 Flip {zone['arrow']} {zone['strike']:,.0f}",
                      annotation_position="left",
                      annotation=dict(font=dict(size=10, color=zone['color']),
                                    bgcolor='rgba(0,0,0,0.7)', bordercolor=zone['color'], borderwidth=1))
        
        fig.add_hrect(y0=zone['lower_strike'], y1=zone['upper_strike'],
                     fillcolor=zone['color'], opacity=0.1, line_width=0,
                     annotation_text=zone['arrow'], annotation_position="right",
                     annotation=dict(font=dict(size=16, color=zone['color'])))
    
    fig.update_layout(
        title=dict(text="<b>⚡ Combined NET GEX + DEX with Flip Zones</b>",
                  font=dict(size=18, color='white')),
        xaxis_title="Combined Exposure (₹ Crores)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_hedging_pressure_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """Hedging pressure"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['hedging_pressure'], orientation='h',
                        marker=dict(color=df_sorted['hedging_pressure'], colorscale='RdYlGn',
                                  showscale=True,
                                  colorbar=dict(title=dict(text='Pressure %', font=dict(color='white', size=12)),
                                              tickfont=dict(color='white'), x=1.02, len=0.7, thickness=20),
                                  cmin=-100, cmax=100),
                        hovertemplate='Strike: %{y:,.0f}<br>Pressure: %{x:.1f}%<extra></extra>',
                        showlegend=False))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
    
    for zone in flip_zones:
        fig.add_hline(y=zone['strike'], line_dash="dot", line_color=zone['color'], line_width=2,
                      annotation_text=f"🔄 Flip {zone['arrow']} {zone['strike']:,.0f}",
                      annotation_position="left",
                      annotation=dict(font=dict(size=10, color=zone['color']),
                                    bgcolor='rgba(0,0,0,0.7)', bordercolor=zone['color'], borderwidth=1))
        
        fig.add_hrect(y0=zone['lower_strike'], y1=zone['upper_strike'],
                     fillcolor=zone['color'], opacity=0.1, line_width=0,
                     annotation_text=zone['arrow'], annotation_position="right",
                     annotation=dict(font=dict(size=16, color=zone['color'])))
    
    fig.update_layout(
        title=dict(text="<b>🎪 Hedging Pressure with Flip Zones</b>", font=dict(size=18, color='white')),
        xaxis_title="Hedging Pressure (%)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True,
                  zeroline=True, zerolinecolor='rgba(128,128,128,0.5)',
                  zerolinewidth=2, range=[-110, 110]),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=120, t=80, b=80)
    )
    
    return fig

def create_oi_distribution(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """OI distribution"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['call_oi'], orientation='h',
                        name='Call OI', marker_color='#10b981', opacity=0.7,
                        hovertemplate='Strike: %{y:,.0f}<br>Call OI: %{x:,.0f}<extra></extra>'))
    
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=-df_sorted['put_oi'], orientation='h',
                        name='Put OI', marker_color='#ef4444', opacity=0.7,
                        hovertemplate='Strike: %{y:,.0f}<br>Put OI: %{customdata:,.0f}<extra></extra>',
                        customdata=df_sorted['put_oi']))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    fig.add_vline(x=0, line_dash="dot", line_color="white", line_width=1)
    
    fig.update_layout(
        title=dict(text="<b>📋 Open Interest Distribution</b>", font=dict(size=16, color='white')),
        xaxis_title="Open Interest (Calls +ve | Puts -ve)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=500,
        barmode='overlay',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='white')),
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True,
                  zeroline=True, zerolinecolor='rgba(255,255,255,0.3)', zerolinewidth=2),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_vanna_exposure_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """VANNA exposure"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    colors_call = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['call_vanna']]
    colors_put = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['put_vanna']]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("📈 Call VANNA", "📉 Put VANNA"),
                       horizontal_spacing=0.12)
    
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['call_vanna'], orientation='h',
                        marker=dict(color=colors_call), name='Call VANNA',
                        hovertemplate='Strike: %{y:,.0f}<br>Call VANNA: %{x:.4f}Cr<extra></extra>'),
                 row=1, col=1)
    
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['put_vanna'], orientation='h',
                        marker=dict(color=colors_put), name='Put VANNA',
                        hovertemplate='Strike: %{y:,.0f}<br>Put VANNA: %{x:.4f}Cr<extra></extra>'),
                 row=1, col=2)
    
    for col in [1, 2]:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                      annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                      annotation=dict(font=dict(size=10, color="white")), row=1, col=col)
    
    fig.update_layout(
        title=dict(text="<b>🌊 VANNA Exposure (dDelta/dVol)</b>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    fig.update_xaxes(title_text="VANNA (₹ Crores)", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    fig.update_yaxes(title_text="Strike Price", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    
    return fig

def create_charm_exposure_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """CHARM exposure"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    colors_call = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['call_charm']]
    colors_put = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['put_charm']]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("📈 Call CHARM", "📉 Put CHARM"),
                       horizontal_spacing=0.12)
    
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['call_charm'], orientation='h',
                        marker=dict(color=colors_call), name='Call CHARM',
                        hovertemplate='Strike: %{y:,.0f}<br>Call CHARM: %{x:.4f}Cr<extra></extra>'),
                 row=1, col=1)
    
    fig.add_trace(go.Bar(y=df_sorted['strike'], x=df_sorted['put_charm'], orientation='h',
                        marker=dict(color=colors_put), name='Put CHARM',
                        hovertemplate='Strike: %{y:,.0f}<br>Put CHARM: %{x:.4f}Cr<extra></extra>'),
                 row=1, col=2)
    
    for col in [1, 2]:
        fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                      annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                      annotation=dict(font=dict(size=10, color="white")), row=1, col=col)
    
    fig.update_layout(
        title=dict(text="<b>⏰ CHARM Exposure (Delta Decay)</b>", font=dict(size=18, color='white')),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    fig.update_xaxes(title_text="CHARM (₹ Crores)", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    fig.update_yaxes(title_text="Strike Price", gridcolor='rgba(128,128,128,0.2)', showgrid=True)
    
    return fig

# ============================================================================
# SCREENER DISPLAY FUNCTIONS
# ============================================================================

def display_screener_results(df_results: pd.DataFrame, filter_type: str):
    """Display screener results"""
    if len(df_results) == 0:
        st.warning("No stocks match the screening criteria")
        return
    
    st.success(f"✅ Found {len(df_results)} stocks matching criteria")
    
    for idx, row in df_results.iterrows():
        flip_analysis = row['flip_analysis']
        
        # Determine card styling based on filter type
        if filter_type in ['above', 'positive_gex', 'positive_gamma_diff']:
            card_class = 'bullish'
            signal_badge = 'long'
            signal_text = '🟢 LONG OPPORTUNITY'
        elif filter_type in ['below', 'negative_gex', 'negative_gamma_diff']:
            card_class = 'bearish'
            signal_badge = 'short'
            signal_text = '🔴 SHORT OPPORTUNITY'
        elif filter_type in ['high_volume', 'high_call_volume']:
            card_class = 'bullish'
            signal_badge = 'long'
            signal_text = '📊 HIGH VOLUME'
        elif filter_type == 'high_put_volume':
            card_class = 'bearish'
            signal_badge = 'short'
            signal_text = '📉 HIGH PUT VOLUME'
        else:
            card_class = 'neutral'
            signal_badge = 'long' if row.get('total_gex', 0) > 0 else 'short'
            signal_text = '🔵 FLIP ZONE DETECTED'
        
        if flip_analysis['has_flip_zones'] and flip_analysis['nearest_flip']:
            nearest = flip_analysis['nearest_flip']
            flip_info = f"Flip @ ₹{nearest['strike']:,.2f} ({nearest['distance_pct']:.2f}% away) {nearest['arrow']}"
        else:
            flip_info = "No flip zones"
        
        # Get volume data
        total_volume = row.get('total_volume', 0)
        call_volume = row.get('call_volume', 0)
        put_volume = row.get('put_volume', 0)
        volume_pcr = row.get('volume_pcr', 1)
        
        # Get high volume strikes
        top_strikes = row.get('top_volume_strikes', [])
        
        # Get gamma differential
        gamma_diff = row.get('gamma_differential', 0)
        gamma_diff_label = 'OTM Call γ > Put γ' if gamma_diff > 0 else 'OTM Put γ > Call γ'
        
        # Determine colors
        gex_color = '#10b981' if row['total_gex'] > 0 else '#ef4444'
        dex_color = '#10b981' if row['total_dex'] > 0 else '#ef4444'
        gamma_diff_color = '#10b981' if gamma_diff > 0 else '#ef4444'
        
        # Build strikes display text
        strikes_text = ""
        if top_strikes and len(top_strikes) > 0:
            strikes_text = " | ".join([f"₹{s['strike']:,.0f} ({s['total_volume']:,.0f})" for s in top_strikes[:3]])
        
        # Create card HTML
        card_html = f"""
        <div class="screener-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc; font-size: 1.5rem;">{row['symbol']}</h3>
                    <p style="margin: 5px 0; color: #cbd5e1;">Spot: ₹{row['spot_price']:,.2f}</p>
                </div>
                <div class="opportunity-badge {signal_badge}">{signal_text}</div>
            </div>
            <div style="margin-top: 15px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">NET GEX</p>
                    <p style="margin: 5px 0; color: {gex_color}; font-weight: 600;">
                        {row['total_gex']:.4f}Cr
                    </p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">NET DEX</p>
                    <p style="margin: 5px 0; color: {dex_color}; font-weight: 600;">
                        {row['total_dex']:.4f}Cr
                    </p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">OI P/C RATIO</p>
                    <p style="margin: 5px 0; color: #f8fafc; font-weight: 600;">{row['pcr']:.2f}</p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">FLIP ZONES</p>
                    <p style="margin: 5px 0; color: #fbbf24; font-weight: 600;">
                        {flip_analysis['flip_count'] if flip_analysis['has_flip_zones'] else 0}
                    </p>
                </div>
            </div>
            <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding-top: 10px; border-top: 1px solid #374151;">
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">TOTAL VOLUME</p>
                    <p style="margin: 5px 0; color: #06b6d4; font-weight: 600;">{total_volume:,.0f}</p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">CALL VOLUME</p>
                    <p style="margin: 5px 0; color: #10b981; font-weight: 600;">{call_volume:,.0f}</p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">PUT VOLUME</p>
                    <p style="margin: 5px 0; color: #ef4444; font-weight: 600;">{put_volume:,.0f}</p>
                </div>
                <div>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">VOL P/C RATIO</p>
                    <p style="margin: 5px 0; color: #f8fafc; font-weight: 600;">{volume_pcr:.2f}</p>
                </div>
            </div>"""
        
        # Add high volume strikes if available
        if strikes_text:
            card_html += f"""
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #374151;">
                <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">📍 HIGH VOLUME STRIKES</p>
                <p style="margin: 5px 0; color: #06b6d4; font-weight: 600; font-size: 0.9rem;">
                    {strikes_text}
                </p>
            </div>"""
        
        # Add gamma differential and footer
        card_html += f"""
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #374151;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">⚡ GAMMA DIFFERENTIAL</p>
                        <p style="margin: 5px 0; color: {gamma_diff_color}; font-weight: 600; font-size: 0.85rem;">
                            {gamma_diff:.6f} ({gamma_diff_label})
                        </p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #cbd5e1; font-size: 0.85rem;">
                            🔄 {flip_info}
                        </p>
                    </div>
                </div>
                <p style="margin: 5px 0 0 0; color: #9ca3af; font-size: 0.75rem;">
                    {row['timestamp']} | {row['strikes_analyzed']} strikes analyzed
                </p>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)

def create_screener_summary_chart(df_results: pd.DataFrame) -> go.Figure:
    """Summary chart for screener"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_results['symbol'],
        x=df_results['total_gex'],
        orientation='h',
        name='NET GEX',
        marker_color=['#10b981' if x > 0 else '#ef4444' for x in df_results['total_gex']],
        hovertemplate='%{y}<br>GEX: %{x:.4f}Cr<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>NET GEX Comparison</b>",
        xaxis_title="NET GEX (₹ Crores)",
        yaxis_title="Stock",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=max(400, len(df_results) * 50),
        showlegend=False
    )
    
    return fig

def create_volume_comparison_chart(df_results: pd.DataFrame) -> go.Figure:
    """Volume comparison chart for screener"""
    fig = go.Figure()
    
    # Call volume (positive)
    fig.add_trace(go.Bar(
        y=df_results['symbol'],
        x=df_results['call_volume'],
        orientation='h',
        name='Call Volume',
        marker_color='#10b981',
        hovertemplate='%{y}<br>Call Vol: %{x:,.0f}<extra></extra>'
    ))
    
    # Put volume (negative for visual separation)
    fig.add_trace(go.Bar(
        y=df_results['symbol'],
        x=-df_results['put_volume'],
        orientation='h',
        name='Put Volume',
        marker_color='#ef4444',
        hovertemplate='%{y}<br>Put Vol: %{customdata:,.0f}<extra></extra>',
        customdata=df_results['put_volume']
    ))
    
    fig.update_layout(
        title="<b>Call vs Put Volume Comparison</b>",
        xaxis_title="Volume (Calls +ve | Puts -ve)",
        yaxis_title="Stock",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=max(400, len(df_results) * 50),
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def create_total_volume_chart(df_results: pd.DataFrame) -> go.Figure:
    """Total volume chart for screener"""
    df_sorted = df_results.sort_values('total_volume', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['symbol'],
        x=df_sorted['total_volume'],
        orientation='h',
        name='Total Volume',
        marker_color='#3b82f6',
        hovertemplate='%{y}<br>Total Vol: %{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Total Options Volume</b>",
        xaxis_title="Total Volume",
        yaxis_title="Stock",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=max(400, len(df_sorted) * 50),
        showlegend=False
    )
    
    return fig

def create_gamma_differential_chart(df_results: pd.DataFrame) -> go.Figure:
    """Gamma differential comparison chart"""
    df_sorted = df_results.sort_values('gamma_diff_normalized', ascending=True)
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['gamma_diff_normalized']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['symbol'],
        x=df_sorted['gamma_diff_normalized'],
        orientation='h',
        name='Gamma Differential',
        marker_color=colors,
        hovertemplate='%{y}<br>γ Diff: %{x:.6f}<br>%{customdata}<extra></extra>',
        customdata=[f"OTM Call γ {'>' if x > 0 else '<'} OTM Put γ" for x in df_sorted['gamma_diff_normalized']]
    ))
    
    fig.add_vline(x=0, line_dash="dot", line_color="white", line_width=2)
    
    fig.update_layout(
        title="<b>⚡ Gamma Differential (OTM Calls vs OTM Puts)</b>",
        xaxis_title="Normalized Gamma Differential (γ_calls - γ_puts) / Spot",
        yaxis_title="Stock",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=max(400, len(df_sorted) * 50),
        showlegend=False
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION - UNIFIED
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🎯 NYZTrade Stock Options | Unified Dashboard
        </h1>
        <p style="margin: 5px 0 0 0; color: var(--text-secondary); font-size: 0.9rem;">
            Full Analysis + Multi-Stock Screener | VANNA & CHARM | Gamma Flip Zones | All-in-One
        </p>
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
        
        else:
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
                 "❌ NET GEX Negative",
                 "📊 High Volume (Top 10)",
                 "📈 High Call Volume",
                 "📉 High Put Volume",
                 "⚡ Positive Gamma Differential",
                 "⚡ Negative Gamma Differential"],
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
        
        # Main analysis (reuse all existing code from original dashboard)
        if fetch_button or (hasattr(st.session_state, 'fetch_config') and st.session_state.get('data_fetched', False)):
            if hasattr(st.session_state, 'fetch_config'):
                config = st.session_state.fetch_config
                symbol = config['symbol']
                target_date = config['target_date']
                strikes = config['strikes']
                interval = config['interval']
                expiry_code = config.get('expiry_code', 1)
                expiry_flag = config.get('expiry_flag', 'MONTH')
            
            if not strikes:
                st.error("❌ Please select at least one strike")
                return
            
            if not st.session_state.get('data_fetched', False) or 'df_data' not in st.session_state:
                st.markdown(f"""
                <div class="metric-card neutral" style="margin: 20px 0;">
                    <div style="color: #3b82f6; font-size: 1.2rem; font-weight: 600;">
                        Fetching {symbol} | {target_date} | {interval} min
                    </div>
                    <div style="margin-top: 10px; color: var(--text-secondary);">Please wait 1-3 minutes...</div>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    fetcher = DhanStockOptionsFetcher(DhanConfig())
                    df, meta = fetcher.process_historical_data(symbol, target_date, strikes, interval, expiry_code, expiry_flag)
                    
                    if df is None or len(df) == 0:
                        st.error("❌ No data available")
                        return
                    
                    st.session_state.df_data = df
                    st.session_state.meta_data = meta
                    st.session_state.data_fetched = True
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    return
            
            # [Continue with ALL the existing dashboard visualization code]
            # This includes time navigation, metrics display, and all 9 tabs
            # (Keeping all the existing code from the original dashboard)
            
            df = st.session_state.df_data
            meta = st.session_state.meta_data
            
            all_timestamps = sorted(df['timestamp'].unique())
            
            st.success(f"✅ Data fetched | {len(df):,} records")
            
            st.markdown("---")
            st.markdown("### ⏱️ Time Navigation")
            
            control_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
            
            with control_cols[0]:
                if st.button("⏮️ First", use_container_width=True):
                    st.session_state.timestamp_idx = 0
            with control_cols[1]:
                if st.button("◀️ Prev", use_container_width=True):
                    current = st.session_state.get('timestamp_idx', len(all_timestamps) - 1)
                    st.session_state.timestamp_idx = max(0, current - 1)
            with control_cols[2]:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.timestamp_idx = len(all_timestamps) - 1
            with control_cols[3]:
                if st.button("▶️ Next", use_container_width=True):
                    current = st.session_state.get('timestamp_idx', len(all_timestamps) - 1)
                    st.session_state.timestamp_idx = min(len(all_timestamps) - 1, current + 1)
            with control_cols[4]:
                if st.button("⏭️ Last", use_container_width=True):
                    st.session_state.timestamp_idx = len(all_timestamps) - 1
            with control_cols[5]:
                if st.button("⏰ 9:30", use_container_width=True):
                    morning_times = [i for i, ts in enumerate(all_timestamps) if ts.hour == 9 and ts.minute >= 30]
                    if morning_times:
                        st.session_state.timestamp_idx = morning_times[0]
            with control_cols[6]:
                if st.button("⏰ 12:00", use_container_width=True):
                    noon_times = [i for i, ts in enumerate(all_timestamps) if ts.hour == 12]
                    if noon_times:
                        st.session_state.timestamp_idx = noon_times[0]
            with control_cols[7]:
                if st.button("⏰ 3:15", use_container_width=True):
                    close_times = [i for i, ts in enumerate(all_timestamps) if ts.hour == 15 and ts.minute >= 15]
                    if close_times:
                        st.session_state.timestamp_idx = close_times[0]
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"""<div class="metric-card neutral" style="padding: 15px;">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Start</div>
                    <div style="color: var(--text-primary); font-size: 1.2rem; font-weight: 600;">{all_timestamps[0].strftime('%H:%M')}</div>
                </div>""", unsafe_allow_html=True)
            
            with col2:
                if 'timestamp_idx' not in st.session_state:
                    st.session_state.timestamp_idx = len(all_timestamps) - 1
                
                selected_timestamp_idx = st.slider("🎯 Navigate", min_value=0, max_value=len(all_timestamps) - 1,
                                                   value=st.session_state.timestamp_idx, format="", key="time_slider")
                
                st.session_state.timestamp_idx = selected_timestamp_idx
                selected_timestamp = all_timestamps[selected_timestamp_idx]
                
                progress = (selected_timestamp_idx + 1) / len(all_timestamps)
                st.progress(progress)
                
                st.info(f"📍 **{selected_timestamp.strftime('%H:%M:%S IST')}** | Point {selected_timestamp_idx + 1} of {len(all_timestamps)}")
            
            with col3:
                st.markdown(f"""<div class="metric-card neutral" style="padding: 15px;">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">End</div>
                    <div style="color: var(--text-primary); font-size: 1.2rem; font-weight: 600;">{all_timestamps[-1].strftime('%H:%M')}</div>
                </div>""", unsafe_allow_html=True)
            
            df_selected = df[df['timestamp'] == selected_timestamp].copy()
            
            if len(df_selected) == 0:
                closest_idx = min(range(len(all_timestamps)),
                                 key=lambda i: abs((all_timestamps[i] - selected_timestamp).total_seconds()))
                df_selected = df[df['timestamp'] == all_timestamps[closest_idx]].copy()
            
            df_latest = df_selected
            spot_price = df_latest['spot_price'].iloc[0] if len(df_latest) > 0 else 0
            
            total_gex = df_latest['net_gex'].sum()
            total_dex = df_latest['net_dex'].sum()
            total_net = total_gex + total_dex
            total_call_oi = df_latest['call_oi'].sum()
            total_put_oi = df_latest['put_oi'].sum()
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
            
            flip_zones = identify_gamma_flip_zones(df_latest, spot_price)
            
            # Quick Summary Banner
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); 
                        border: 2px solid var(--border-color); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <h3 style="margin: 0 0 15px 0; color: var(--accent-cyan);">🎯 Quick Analysis Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            
            quick_cols = st.columns(3)
            
            with quick_cols[0]:
                # Top 3 high volume strikes
                if meta.get('top_volume_strikes'):
                    top_3 = meta['top_volume_strikes'][:3]
                    strikes_str = " → ".join([f"₹{s['strike']:,.0f}" for s in top_3])
                    st.markdown(f"""
                    <div class="metric-card neutral" style="background: rgba(6, 182, 212, 0.1);">
                        <div style="color: var(--text-muted); font-size: 0.8rem;">📍 TOP VOLUME STRIKES</div>
                        <div style="color: var(--accent-cyan); font-size: 1.1rem; font-weight: 700; margin: 8px 0;">
                            {strikes_str}
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.75rem;">
                            Key price levels where action is happening
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with quick_cols[1]:
                # Gamma differential quick view - using actual BS gamma values
                otm_calls_quick = df_latest[df_latest['strike'] > spot_price]
                otm_puts_quick = df_latest[df_latest['strike'] < spot_price]
                call_gamma_quick = otm_calls_quick['call_gamma'].sum() if len(otm_calls_quick) > 0 else 0
                put_gamma_quick = otm_puts_quick['put_gamma'].sum() if len(otm_puts_quick) > 0 else 0
                gamma_diff_quick = call_gamma_quick - put_gamma_quick
                
                gamma_color_quick = "var(--accent-green)" if gamma_diff_quick > 0 else "var(--accent-red)"
                gamma_signal_quick = "OTM Call γ Dominance" if gamma_diff_quick > 0 else "OTM Put γ Dominance"
                gamma_arrow = "↑" if gamma_diff_quick > 0 else "↓"
                
                st.markdown(f"""
                <div class="metric-card {'positive' if gamma_diff_quick > 0 else 'negative'}">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">⚡ GAMMA DIFFERENTIAL</div>
                    <div style="color: {gamma_color_quick}; font-size: 1.3rem; font-weight: 700; margin: 8px 0;">
                        {gamma_diff_quick:.6f} {gamma_arrow}
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem;">
                        {gamma_signal_quick} (BS Γ)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with quick_cols[2]:
                # Trading recommendation
                if gamma_diff_quick > 0 and total_gex < 0:
                    setup_text = "🚀 BULLISH SQUEEZE SETUP"
                    setup_color = "var(--accent-green)"
                    setup_desc = "Positive γ diff + Negative GEX = Amplified upside"
                elif gamma_diff_quick < 0 and total_gex < 0:
                    setup_text = "💥 BEARISH PRESSURE SETUP"
                    setup_color = "var(--accent-red)"
                    setup_desc = "Negative γ diff + Negative GEX = Amplified downside"
                elif total_gex > 0:
                    setup_text = "📊 RANGE-BOUND SETUP"
                    setup_color = "var(--accent-yellow)"
                    setup_desc = "Positive GEX = Suppression expected"
                else:
                    setup_text = "⚖️ BALANCED SETUP"
                    setup_color = "var(--accent-cyan)"
                    setup_desc = "Mixed signals, watch for direction"
                
                st.markdown(f"""
                <div class="metric-card neutral" style="border-left: 4px solid {setup_color};">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">🎯 MARKET SETUP</div>
                    <div style="color: {setup_color}; font-size: 1.1rem; font-weight: 700; margin: 8px 0;">
                        {setup_text}
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem;">
                        {setup_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Overview")
            
            if len(flip_zones) > 0:
                flip_info = " | ".join([f"🔄 {z['strike']:,.0f} {z['arrow']}" for z in flip_zones[:3]])
                st.info(f"**Gamma Flip Zones**: {flip_info}")
            
            cols = st.columns(6)
            
            with cols[0]:
                st.markdown(f"""<div class="metric-card neutral">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Stock & Date</div>
                    <div style="color: var(--text-primary); font-size: 1.2rem; font-weight: 600;">{symbol}</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 5px;">{target_date}</div>
                </div>""", unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""<div class="metric-card neutral">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Spot Price</div>
                    <div style="color: var(--text-primary); font-size: 1.5rem; font-weight: 700;">₹{spot_price:,.2f}</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 5px;">{selected_timestamp.strftime('%H:%M IST')}</div>
                </div>""", unsafe_allow_html=True)
            
            with cols[2]:
                gex_class = "positive" if total_gex > 0 else "negative"
                st.markdown(f"""<div class="metric-card {gex_class}">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Total NET GEX</div>
                    <div style="color: {'var(--accent-green)' if total_gex > 0 else 'var(--accent-red)'}; font-size: 1.5rem; font-weight: 700;">{total_gex:.4f}Cr</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 5px;">{'Suppression' if total_gex > 0 else 'Amplification'}</div>
                </div>""", unsafe_allow_html=True)
            
            with cols[3]:
                dex_class = "positive" if total_dex > 0 else "negative"
                st.markdown(f"""<div class="metric-card {dex_class}">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Total NET DEX</div>
                    <div style="color: {'var(--accent-green)' if total_dex > 0 else 'var(--accent-red)'}; font-size: 1.5rem; font-weight: 700;">{total_dex:.4f}Cr</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 5px;">{'Bullish' if total_dex > 0 else 'Bearish'}</div>
                </div>""", unsafe_allow_html=True)
            
            with cols[4]:
                net_class = "positive" if total_net > 0 else "negative"
                st.markdown(f"""<div class="metric-card {net_class}">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">GEX + DEX</div>
                    <div style="color: {'var(--accent-green)' if total_net > 0 else 'var(--accent-red)'}; font-size: 1.5rem; font-weight: 700;">{total_net:.4f}Cr</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 5px;">Combined</div>
                </div>""", unsafe_allow_html=True)
            
            with cols[5]:
                pcr_class = "positive" if pcr > 1 else "negative"
                st.markdown(f"""<div class="metric-card {pcr_class}">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Put/Call Ratio</div>
                    <div style="color: {'var(--accent-green)' if pcr > 1 else 'var(--accent-red)'}; font-size: 1.5rem; font-weight: 700;">{pcr:.2f}</div>
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 5px;">{'Bearish' if pcr > 1.2 else 'Bullish' if pcr < 0.8 else 'Neutral'}</div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # High Volume Strikes Display
            if meta.get('top_volume_strikes'):
                st.markdown("### 📍 High Volume Strikes")
                top_strikes = meta['top_volume_strikes'][:5]
                
                strike_cols = st.columns(min(5, len(top_strikes)))
                for idx, strike_data in enumerate(top_strikes):
                    with strike_cols[idx]:
                        strike_price = strike_data['strike']
                        strike_vol = strike_data['total_volume']
                        call_vol = strike_data['call_volume']
                        put_vol = strike_data['put_volume']
                        
                        # Determine if OTM call, OTM put, or ATM
                        if abs(strike_price - spot_price) < (spot_price * 0.01):
                            strike_type = "ATM"
                            color = "var(--accent-yellow)"
                        elif strike_price > spot_price:
                            strike_type = "OTM Call"
                            color = "var(--accent-green)"
                        else:
                            strike_type = "OTM Put"
                            color = "var(--accent-red)"
                        
                        st.markdown(f"""
                        <div class="metric-card neutral" style="border-left: 4px solid {color};">
                            <div style="color: var(--text-muted); font-size: 0.75rem;">{strike_type}</div>
                            <div style="color: {color}; font-size: 1.3rem; font-weight: 700;">₹{strike_price:,.0f}</div>
                            <div style="color: var(--text-primary); font-size: 0.9rem; margin-top: 5px; font-weight: 600;">{strike_vol:,.0f} vol</div>
                            <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 3px;">
                                C: {call_vol:,.0f} | P: {put_vol:,.0f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.caption("💡 **High volume strikes often act as support/resistance levels**")
            
            # Gamma Differential Analysis
            st.markdown("### ⚡ Gamma Differential Analysis")
            
            # Calculate gamma differential for current view using actual BS gamma values
            df_current = df_latest.copy()
            otm_calls = df_current[df_current['strike'] > spot_price]
            otm_puts = df_current[df_current['strike'] < spot_price]
            
            # Sum of actual Black-Scholes gamma values (not multiplied by OI)
            total_otm_call_gamma = otm_calls['call_gamma'].sum() if len(otm_calls) > 0 else 0
            total_otm_put_gamma = otm_puts['put_gamma'].sum() if len(otm_puts) > 0 else 0
            gamma_differential = total_otm_call_gamma - total_otm_put_gamma
            gamma_diff_normalized = gamma_differential
            
            gamma_cols = st.columns(4)
            
            with gamma_cols[0]:
                st.markdown(f"""
                <div class="metric-card neutral" style="border-left: 4px solid var(--accent-green);">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">OTM Call Gamma</div>
                    <div style="color: var(--accent-green); font-size: 1.2rem; font-weight: 700;">{total_otm_call_gamma:.6f}</div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 3px;">BS Gamma (Sum)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with gamma_cols[1]:
                st.markdown(f"""
                <div class="metric-card neutral" style="border-left: 4px solid var(--accent-red);">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">OTM Put Gamma</div>
                    <div style="color: var(--accent-red); font-size: 1.2rem; font-weight: 700;">{total_otm_put_gamma:.6f}</div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 3px;">BS Gamma (Sum)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with gamma_cols[2]:
                gamma_diff_color = "var(--accent-green)" if gamma_differential > 0 else "var(--accent-red)"
                gamma_diff_signal = "Bullish Pressure" if gamma_differential > 0 else "Bearish Pressure"
                st.markdown(f"""
                <div class="metric-card {'positive' if gamma_differential > 0 else 'negative'}">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Gamma Differential</div>
                    <div style="color: {gamma_diff_color}; font-size: 1.2rem; font-weight: 700;">{gamma_differential:.6f}</div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 3px;">{gamma_diff_signal}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with gamma_cols[3]:
                # Magnitude indicator based on absolute gamma values
                abs_diff = abs(gamma_diff_normalized)
                if abs_diff > 0.15:
                    magnitude = "VERY HIGH"
                    mag_color = "var(--accent-yellow)"
                elif abs_diff > 0.05:
                    magnitude = "MODERATE"
                    mag_color = "var(--accent-cyan)"
                else:
                    magnitude = "LOW"
                    mag_color = "var(--text-muted)"
                
                st.markdown(f"""
                <div class="metric-card neutral" style="border-left: 4px solid {mag_color};">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Magnitude</div>
                    <div style="color: {mag_color}; font-size: 1.2rem; font-weight: 700;">{magnitude}</div>
                    <div style="color: var(--text-secondary); font-size: 0.75rem; margin-top: 3px;">Differential: {gamma_diff_normalized:.6f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretation
            if gamma_differential > 0:
                if abs_diff > 0.15:
                    interpretation = "🚀 **STRONG BULLISH GAMMA SETUP**: OTM call strikes have significantly higher Black-Scholes gamma values, indicating strong curvature sensitivity to upward moves."
                elif abs_diff > 0.05:
                    interpretation = "🟢 **MODERATE BULLISH PRESSURE**: OTM call gamma (BS values) dominance means higher rate of delta change on upward price moves."
                else:
                    interpretation = "📊 **SLIGHT BULLISH BIAS**: Small positive gamma differential based on Black-Scholes values."
            else:
                if abs_diff > 0.15:
                    interpretation = "💥 **STRONG BEARISH GAMMA SETUP**: OTM put strikes have significantly higher Black-Scholes gamma values, indicating strong curvature sensitivity to downward moves."
                elif abs_diff > 0.05:
                    interpretation = "🔴 **MODERATE BEARISH PRESSURE**: OTM put gamma (BS values) dominance means higher rate of delta change on downward price moves."
                else:
                    interpretation = "📊 **SLIGHT BEARISH BIAS**: Small negative gamma differential based on Black-Scholes values."
            
            st.info(interpretation)
            st.caption("💡 **Gamma Differential** = Sum of OTM Call BS Gamma - Sum of OTM Put BS Gamma. These are actual Black-Scholes gamma values (like 0.05, 0.03), NOT multiplied by OI or spot. Shows which direction has higher gamma curvature.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            cols = st.columns(5)
            with cols[0]:
                gex_signal = "🟢 GEX SUPPRESSION" if total_gex > 0 else "🔴 GEX AMPLIFICATION"
                gex_badge = "bullish" if total_gex > 0 else "bearish"
                st.markdown(f'<div class="signal-badge {gex_badge}">{gex_signal}</div>', unsafe_allow_html=True)
            with cols[1]:
                dex_signal = "🟢 DEX BULLISH" if total_dex > 0 else "🔴 DEX BEARISH"
                dex_badge = "bullish" if total_dex > 0 else "bearish"
                st.markdown(f'<div class="signal-badge {dex_badge}">{dex_signal}</div>', unsafe_allow_html=True)
            with cols[2]:
                net_signal = "🟢 NET POSITIVE" if total_net > 0 else "🔴 NET NEGATIVE"
                net_badge = "bullish" if total_net > 0 else "bearish"
                st.markdown(f'<div class="signal-badge {net_badge}">{net_signal}</div>', unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f'<div class="signal-badge volatile">📊 {len(df_latest)} Strikes</div>', unsafe_allow_html=True)
            with cols[4]:
                if len(flip_zones) > 0:
                    st.markdown(f'<div class="signal-badge volatile">🔄 {len(flip_zones)} Flip Zones</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            tabs = st.tabs(["🎯 GEX", "📊 DEX", "⚡ NET", "🎪 Hedge",
                           "🌊 VANNA", "⏰ CHARM", "⚡ Γ DIFF", "📈 Timeline", "📋 Data"])
            
            with tabs[0]:
                st.markdown("### 🎯 Gamma Exposure")
                st.plotly_chart(create_separate_gex_chart(df_latest, spot_price), use_container_width=True)
            
            with tabs[1]:
                st.markdown("### 📊 Delta Exposure")
                st.plotly_chart(create_separate_dex_chart(df_latest, spot_price), use_container_width=True)
            
            with tabs[2]:
                st.markdown("### ⚡ Combined NET GEX + DEX")
                st.plotly_chart(create_net_gex_dex_chart(df_latest, spot_price), use_container_width=True)
            
            with tabs[3]:
                st.markdown("### 🎪 Hedging Pressure")
                st.plotly_chart(create_hedging_pressure_chart(df_latest, spot_price), use_container_width=True)
            
            with tabs[4]:
                st.markdown("### 🌊 VANNA Exposure")
                st.plotly_chart(create_vanna_exposure_chart(df_latest, spot_price), use_container_width=True)
                st.markdown("**VANNA**: Delta sensitivity to volatility")
            
            with tabs[5]:
                st.markdown("### ⏰ CHARM Exposure")
                st.plotly_chart(create_charm_exposure_chart(df_latest, spot_price), use_container_width=True)
                st.markdown("**CHARM**: Delta decay over time")
            
            with tabs[6]:
                st.markdown("### ⚡ Gamma Differential Analysis")
                
                # Separate OTM calls and puts
                df_otm_calls = df_latest[df_latest['strike'] > spot_price].copy()
                df_otm_puts = df_latest[df_latest['strike'] < spot_price].copy()
                
                # Use actual Black-Scholes gamma values (not multiplied by OI)
                df_otm_calls['bs_call_gamma'] = df_otm_calls['call_gamma']
                df_otm_puts['bs_put_gamma'] = df_otm_puts['put_gamma']
                
                # Create comparison chart
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("📈 OTM Call BS Gamma", "📉 OTM Put BS Gamma"),
                    horizontal_spacing=0.15
                )
                
                # OTM Calls
                fig.add_trace(
                    go.Bar(
                        y=df_otm_calls['strike'],
                        x=df_otm_calls['bs_call_gamma'],
                        orientation='h',
                        marker_color='#10b981',
                        name='OTM Call Γ',
                        hovertemplate='Strike: %{y:,.0f}<br>BS Gamma: %{x:.6f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # OTM Puts
                fig.add_trace(
                    go.Bar(
                        y=df_otm_puts['strike'],
                        x=df_otm_puts['bs_put_gamma'],
                        orientation='h',
                        marker_color='#ef4444',
                        name='OTM Put Γ',
                        hovertemplate='Strike: %{y:,.0f}<br>BS Gamma: %{x:.6f}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                # Add spot line
                fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                             annotation_text=f"Spot: {spot_price:,.2f}",
                             row=1, col=1)
                fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=2,
                             annotation_text=f"Spot: {spot_price:,.2f}",
                             row=1, col=2)
                
                fig.update_layout(
                    title="<b>⚡ OTM Call vs Put Black-Scholes Gamma Comparison</b>",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26,35,50,0.8)',
                    height=700,
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="BS Gamma Value", row=1, col=1)
                fig.update_xaxes(title_text="BS Gamma Value", row=1, col=2)
                fig.update_yaxes(title_text="Strike Price", row=1, col=1)
                fig.update_yaxes(title_text="Strike Price", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### 📊 Gamma Distribution Summary")
                
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    st.metric(
                        "Total OTM Call Gamma",
                        f"{total_otm_call_gamma:.6f}",
                        delta="Bullish" if total_otm_call_gamma > total_otm_put_gamma else None
                    )
                
                with summary_cols[1]:
                    st.metric(
                        "Total OTM Put Gamma",
                        f"{total_otm_put_gamma:.6f}",
                        delta="Bearish" if total_otm_put_gamma > total_otm_call_gamma else None
                    )
                
                with summary_cols[2]:
                    st.metric(
                        "Net Differential",
                        f"{gamma_differential:.6f}",
                        delta=f"BS Gamma Diff"
                    )
                
                # Top gamma strikes
                st.markdown("#### 🎯 Top BS Gamma Strikes")
                
                top_call_gamma = df_otm_calls.nlargest(3, 'bs_call_gamma')[['strike', 'bs_call_gamma', 'call_oi']]
                top_put_gamma = df_otm_puts.nlargest(3, 'bs_put_gamma')[['strike', 'bs_put_gamma', 'put_oi']]
                
                gamma_strike_cols = st.columns(2)
                
                with gamma_strike_cols[0]:
                    st.markdown("**📈 Highest OTM Call BS Gamma Strikes:**")
                    for idx, row in top_call_gamma.iterrows():
                        st.markdown(f"- **₹{row['strike']:,.0f}**: {row['bs_call_gamma']:.6f} γ ({row['call_oi']:,.0f} OI)")
                
                with gamma_strike_cols[1]:
                    st.markdown("**📉 Highest OTM Put BS Gamma Strikes:**")
                    for idx, row in top_put_gamma.iterrows():
                        st.markdown(f"- **₹{row['strike']:,.0f}**: {row['bs_put_gamma']:.6f} γ ({row['put_oi']:,.0f} OI)")
                
                # Explanation
                st.markdown("---")
                st.markdown("""
                **💡 Understanding Black-Scholes Gamma Differential:**
                
                - **Positive Differential** (Call γ > Put γ): OTM calls have higher gamma values, meaning faster delta changes on upward price moves
                - **Negative Differential** (Put γ > Call γ): OTM puts have higher gamma values, meaning faster delta changes on downward price moves
                - **High Magnitude**: Large differentials suggest asymmetric options positioning
                - **Values**: These are actual Black-Scholes gamma values (e.g., 0.05, 0.03) - NOT multiplied by OI or spot
                
                Gamma measures the rate of change of delta. Higher gamma = more sensitive to price changes.
                """)
            
            with tabs[7]:
                st.markdown("### 📈 Intraday Evolution")
                st.plotly_chart(create_intraday_timeline(df, selected_timestamp), use_container_width=True)
            
            with tabs[8]:
                st.markdown("### 📋 Open Interest")
                st.plotly_chart(create_oi_distribution(df_latest, spot_price), use_container_width=True)
                
                st.markdown("### 📊 Data Table")
                display_df = df_latest[['strike', 'call_oi', 'put_oi', 'net_gex', 'net_dex']].copy()
                display_df['net_gex'] = display_df['net_gex'].apply(lambda x: f"{x:.4f}Cr")
                display_df['net_dex'] = display_df['net_dex'].apply(lambda x: f"{x:.4f}Cr")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", data=csv,
                                  file_name=f"{symbol}_GEX_DEX_{target_date}.csv", mime="text/csv")
        
        else:
            st.info("""
            👋 **Welcome to Single Stock Analysis!**
            
            Select a stock, configure settings, and click "Fetch Data" to begin deep analysis.
            
            **Features:**
            - 🎯 GEX/DEX/VANNA/CHARM Greeks
            - 🔄 Gamma Flip Zone Detection
            - 📈 Intraday Timeline Navigation
            - ⏱️ 1-min to 1-hour intervals
            """)
    
    else:
        # Screener mode
        if run_screener:
            if not selected_stocks:
                st.error("Please select at least one stock")
                return
            
            st.markdown("### 🔍 Running Screener...")
            
            screener = DhanStockOptionsFetcher(DhanConfig())
            df_results = screener.screen_multiple_stocks(selected_stocks, strike_list, expiry_code, expiry_flag)
            
            if len(df_results) == 0:
                st.error("No data available")
                return
            
            # Apply filters based on selected type
            if filter_type == "🟢 Spot Above Gamma Flip":
                filtered_df = df_results[
                    df_results['flip_analysis'].apply(
                        lambda x: x['has_flip_zones'] and x['position'] in ['above_all', 'between']
                    )
                ]
                filter_key = 'above'
                st.info("**Filter**: Stocks trading ABOVE gamma flip zones (bullish continuation/momentum)")
                
            elif filter_type == "🔴 Spot Below Gamma Flip":
                filtered_df = df_results[
                    df_results['flip_analysis'].apply(
                        lambda x: x['has_flip_zones'] and x['position'] in ['below_all', 'between']
                    )
                ]
                filter_key = 'below'
                st.info("**Filter**: Stocks trading BELOW gamma flip zones (bearish continuation/momentum)")
                
            elif filter_type == "🔵 All Stocks with Flip Zones":
                filtered_df = df_results[df_results['flip_analysis'].apply(lambda x: x['has_flip_zones'])]
                filter_key = 'all'
                st.info("**Filter**: All stocks with detected gamma flip zones")
                
            elif filter_type == "✅ NET GEX Positive":
                filtered_df = df_results[df_results['total_gex'] > 0]
                filter_key = 'positive_gex'
                st.info("**Filter**: Stocks with POSITIVE NET GEX (dealer suppression → lower volatility expected)")
                
            elif filter_type == "❌ NET GEX Negative":
                filtered_df = df_results[df_results['total_gex'] < 0]
                filter_key = 'negative_gex'
                st.info("**Filter**: Stocks with NEGATIVE NET GEX (dealer amplification → higher volatility expected)")
                
            elif filter_type == "📊 High Volume (Top 10)":
                filtered_df = df_results.nlargest(min(10, len(df_results)), 'total_volume')
                filter_key = 'high_volume'
                st.info("**Filter**: Top 10 stocks by total options volume (high activity)")
                
            elif filter_type == "📈 High Call Volume":
                filtered_df = df_results.nlargest(min(10, len(df_results)), 'call_volume')
                filter_key = 'high_call_volume'
                st.info("**Filter**: Top 10 stocks by call volume (bullish options activity)")
                
            elif filter_type == "📉 High Put Volume":
                filtered_df = df_results.nlargest(min(10, len(df_results)), 'put_volume')
                filter_key = 'high_put_volume'
                st.info("**Filter**: Top 10 stocks by put volume (bearish options activity)")
                
            elif filter_type == "⚡ Positive Gamma Differential":
                filtered_df = df_results[df_results['gamma_differential'] > 0].nlargest(
                    min(10, len(df_results[df_results['gamma_differential'] > 0])), 
                    'gamma_diff_normalized'
                )
                filter_key = 'positive_gamma_diff'
                st.info("**Filter**: OTM Call Gamma > OTM Put Gamma (bullish gamma imbalance → potential gamma squeeze upward)")
                st.caption("💡 **Trading Implication**: Heavy OTM call gamma suggests dealers will buy stock on rallies, amplifying upward moves")
                
            elif filter_type == "⚡ Negative Gamma Differential":
                filtered_df = df_results[df_results['gamma_differential'] < 0].nsmallest(
                    min(10, len(df_results[df_results['gamma_differential'] < 0])), 
                    'gamma_diff_normalized'
                )
                filter_key = 'negative_gamma_diff'
                st.info("**Filter**: OTM Put Gamma > OTM Call Gamma (bearish gamma imbalance → potential gamma squeeze downward)")
                st.caption("💡 **Trading Implication**: Heavy OTM put gamma suggests dealers will sell stock on declines, amplifying downward moves")
                
            else:
                filtered_df = df_results
                filter_key = 'all'
            
            st.session_state.screener_results = filtered_df
            st.session_state.filter_type = filter_key
            
            st.markdown("---")
            
            if len(filtered_df) > 0:
                st.markdown("### 📊 Screener Results")
                display_screener_results(filtered_df, filter_key)
                
                st.markdown("---")
                st.markdown("### 📈 Visual Comparison")
                
                chart_tabs = st.tabs(["🎯 NET GEX", "📊 Volume Split", "📈 Total Volume", "⚡ Gamma Differential"])
                
                with chart_tabs[0]:
                    st.plotly_chart(create_screener_summary_chart(filtered_df), use_container_width=True)
                    st.caption("**NET GEX**: Positive (green) = dealer suppression | Negative (red) = dealer amplification")
                
                with chart_tabs[1]:
                    st.plotly_chart(create_volume_comparison_chart(filtered_df), use_container_width=True)
                    st.caption("**Volume Split**: Call volume (green) vs Put volume (red)")
                
                with chart_tabs[2]:
                    st.plotly_chart(create_total_volume_chart(filtered_df), use_container_width=True)
                    st.caption("**Total Volume**: Combined call + put options activity")
                
                with chart_tabs[3]:
                    st.plotly_chart(create_gamma_differential_chart(filtered_df), use_container_width=True)
                    st.caption("**Gamma Differential**: Positive (green) = OTM call gamma dominance (bullish squeeze potential) | Negative (red) = OTM put gamma dominance (bearish pressure potential)")
                    st.caption("💡 **Key Insight**: Large gamma imbalances suggest potential for rapid moves as dealers hedge their gamma exposure")
                
                st.markdown("---")
                export_data = filtered_df[['symbol', 'spot_price', 'total_gex', 'total_dex', 'pcr', 
                                           'total_volume', 'call_volume', 'put_volume', 'volume_pcr',
                                           'gamma_differential', 'gamma_diff_normalized',
                                           'timestamp']].copy()
                export_data['flip_zones_count'] = filtered_df['flip_analysis'].apply(lambda x: x['flip_count'] if x['has_flip_zones'] else 0)
                
                # Add high volume strikes summary
                export_data['top_strike_1'] = filtered_df['top_volume_strikes'].apply(
                    lambda x: f"{x[0]['strike']} ({x[0]['total_volume']})" if x and len(x) > 0 else ""
                )
                export_data['top_strike_2'] = filtered_df['top_volume_strikes'].apply(
                    lambda x: f"{x[1]['strike']} ({x[1]['total_volume']})" if x and len(x) > 1 else ""
                )
                export_data['top_strike_3'] = filtered_df['top_volume_strikes'].apply(
                    lambda x: f"{x[2]['strike']} ({x[2]['total_volume']})" if x and len(x) > 2 else ""
                )
                
                csv = export_data.to_csv(index=False)
                st.download_button("📥 Download Results (CSV)", data=csv,
                                  file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                  mime="text/csv")
            else:
                st.warning("No stocks match the selected filter criteria")
        
        else:
            st.info("""
            👋 **Welcome to Multi-Stock Screener!**
            
            **How to use:**
            1. Select stocks (presets or custom)
            2. Choose filter type
            3. Configure settings
            4. Click "Run Screener"
            
            **Filters Available:**
            
            **Gamma Flip Zone Filters:**
            - 🟢 **Above Flip**: Bullish continuation setups (dealers amplify upward moves)
            - 🔴 **Below Flip**: Bearish continuation setups (dealers amplify downward moves)
            - 🔵 **All Flip Zones**: All stocks with detected flip zones
            
            **NET GEX Filters:**
            - ✅ **NET GEX Positive**: Dealer suppression → Lower volatility, mean reversion
            - ❌ **NET GEX Negative**: Dealer amplification → Higher volatility, momentum
            
            **Volume Filters:**
            - 📊 **High Volume (Top 10)**: Highest total options activity
            - 📈 **High Call Volume**: Highest call buying (bullish sentiment)
            - 📉 **High Put Volume**: Highest put buying (bearish sentiment)
            
            **Gamma Differential Filters (NEW!):**
            - ⚡ **Positive Gamma Differential**: OTM call gamma > OTM put gamma → Bullish gamma squeeze potential
            - ⚡ **Negative Gamma Differential**: OTM put gamma > OTM call gamma → Bearish gamma pressure
            
            **NEW Features:**
            - 📍 **High Volume Strikes**: See which specific strikes have the most activity
            - ⚡ **Gamma Differential**: Absolute gamma comparison (not GEX) between OTM calls and puts
            - 📊 **4 Comparison Charts**: GEX, Volume, Total Volume, and Gamma Differential
            """)
    
    st.markdown("---")
    st.markdown(f"""<div style="text-align: center; padding: 20px;">
        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #64748b;">
        NYZTrade Stock Options | Unified Dashboard<br>
        Single Stock Analysis + Multi-Stock Screener | All-in-One</p>
        <p style="font-size: 0.75rem;">⚠️ For educational purposes only</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
