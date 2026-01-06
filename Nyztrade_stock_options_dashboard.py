# ============================================================================
# NYZTrade Historical Stock Options GEX/DEX Dashboard - ENHANCED VERSION
# Features: Stock Options | Weekly/Monthly | VANNA & CHARM | Gamma Flip | 1-min
# ============================================================================

import subprocess
import sys

# Auto-install required packages
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

# Install packages first
install_packages()

# Now import everything
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
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="NYZTrade Stock Options GEX/DEX",
    page_icon="📈",
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
# CONFIGURATION - POPULAR STOCK OPTIONS
# ============================================================================

@dataclass
class DhanConfig:
    client_id: str = "1100480354"
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY3Nzc1Mzk3LCJhcHBfaWQiOiJlZDMwMzI5NCIsImlhdCI6MTc2NzY4ODk5NywidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.BeP4GfNg5r1JO2L7Zr15I6wPF523BCJyHlilBZc0dq_Yg4MNN-PHWpxF1VjWd2LatSUbB7lwkJ1GaYuflLbqAQ"

# Popular Stock Options with their Dhan Security IDs
DHAN_STOCK_SECURITY_IDS = {
    # High Volume Stocks
    "RELIANCE": 2885,
    "TCS": 11536,
    "HDFCBANK": 1333,
    "INFY": 1594,
    "ICICIBANK": 4963,
    "SBIN": 3045,
    "BHARTIARTL": 1195,
    "ITC": 1660,
    "KOTAKBANK": 1922,
    "LT": 2980,
    "AXISBANK": 5900,
    "HINDUNILVR": 1394,
    "WIPRO": 3787,
    "MARUTI": 10999,
    "BAJFINANCE": 317,
    "HCLTECH": 7229,
    "ASIANPAINT": 157,
    "TITAN": 3506,
    "ULTRACEMCO": 11532,
    "SUNPHARMA": 3351,
    "TATAMOTORS": 3456,
    "TATASTEEL": 3499,
    "TECHM": 13538,
    "POWERGRID": 2752,
    "NTPC": 11630,
    "ONGC": 2475,
    "M&M": 2031,
    "BAJAJFINSV": 16675,
    "ADANIPORTS": 3718,
    "COALINDIA": 20374,
}

# Stock-specific configurations (lot sizes and typical strike intervals)
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

# Indian timezone
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
        
        # Check if GEX crosses zero between these strikes
        if (current_gex > 0 and next_gex < 0) or (current_gex < 0 and next_gex > 0):
            # Interpolate the exact flip strike
            flip_strike = current_strike + (next_strike - current_strike) * (abs(current_gex) / (abs(current_gex) + abs(next_gex)))
            
            # Determine flip direction based on spot position
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
# DHAN ROLLING API FETCHER FOR STOCKS (ENHANCED)
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
        """Fetch historical rolling stock options data"""
        try:
            security_id = DHAN_STOCK_SECURITY_IDS.get(symbol)
            if not security_id:
                st.error(f"Security ID not found for {symbol}")
                return None
            
            payload = {
                "exchangeSegment": "NSE_FNO",
                "interval": interval,
                "securityId": security_id,
                "instrument": "OPTSTK",  # Stock options
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
            else:
                st.warning(f"API returned status {response.status_code} for {symbol} {strike_type} {option_type}")
            return None
        except Exception as e:
            st.error(f"API Error for {symbol}: {str(e)}")
            return None
    
    def process_historical_data(self, symbol: str, target_date: str, strikes: List[str], 
                               interval: str = "60", expiry_code: int = 1, expiry_flag: str = "MONTH"):
        """Process historical stock options data with VANNA and CHARM"""
        
        # Convert to datetime
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
            status_text.text(f"Fetching {symbol} {strike_type} ({expiry_flag} Expiry {expiry_code})...")
            
            # Fetch CALL data
            call_data = self.fetch_rolling_data(symbol, from_date, to_date, strike_type, "CALL", 
                                                interval, expiry_code, expiry_flag)
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            time.sleep(1)
            
            # Fetch PUT data
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
                    # Convert timestamp to IST
                    dt_utc = datetime.fromtimestamp(ts, tz=pytz.UTC)
                    dt_ist = dt_utc.astimezone(IST)
                    
                    # Filter for target date
                    if dt_ist.date() != target_dt.date():
                        continue
                    
                    spot_price = ce_data.get('spot', [0])[i] if i < len(ce_data.get('spot', [])) else 0
                    strike_price = ce_data.get('strike', [0])[i] if i < len(ce_data.get('strike', [])) else 0
                    
                    if spot_price == 0 or strike_price == 0:
                        continue
                    
                    # Get OI and other data
                    call_oi = ce_data.get('oi', [0])[i] if i < len(ce_data.get('oi', [])) else 0
                    put_oi = pe_data.get('oi', [0])[i] if i < len(pe_data.get('oi', [])) else 0
                    call_volume = ce_data.get('volume', [0])[i] if i < len(ce_data.get('volume', [])) else 0
                    put_volume = pe_data.get('volume', [0])[i] if i < len(pe_data.get('volume', [])) else 0
                    call_iv = ce_data.get('iv', [15])[i] if i < len(ce_data.get('iv', [])) else 15
                    put_iv = pe_data.get('iv', [15])[i] if i < len(pe_data.get('iv', [])) else 15
                    
                    # Calculate Greeks
                    time_to_expiry = 30 / 365  # Approximate for monthly
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
                    
                    # Calculate exposures (in crores)
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
        
        # Sort by timestamp for flow calculations
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
        
        # Calculate hedging pressure
        max_gex = df['net_gex'].abs().max()
        df['hedging_pressure'] = (df['net_gex'] / max_gex * 100) if max_gex > 0 else 0
        
        # Get latest data point for metadata
        latest = df.sort_values('timestamp').iloc[-1]
        spot_prices = df['spot_price'].unique()
        spot_variation = (spot_prices.max() - spot_prices.min()) / spot_prices.mean() * 100
        
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
            'lot_size': lot_size
        }
        
        return df, meta

# ============================================================================
# VISUALIZATION FUNCTIONS (Same as index dashboard but adapted for stocks)
# ============================================================================

def create_intraday_timeline(df: pd.DataFrame, selected_timestamp) -> go.Figure:
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
    
    # GEX timeline
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
    
    # DEX timeline
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
    
    # Spot price
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
    
    # Add vertical line at selected timestamp
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
        hovertemplate='Strike: %{y:,.0f}<br>Net GEX: %{x:.4f}Cr<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    # Add gamma flip zones
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
        title=dict(text="<b>🎯 Gamma Exposure (GEX) with Flip Zones</b>", font=dict(size=18, color='white')),
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
    """NET GEX+DEX chart with Flip Zones"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    df_sorted['net_gex_dex'] = df_sorted['net_gex'] + df_sorted['net_dex']
    colors = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['net_gex_dex']]
    
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
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
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    # Add gamma flip zones
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
        title=dict(text="<b>⚡ Combined NET GEX + DEX with Flip Zones</b>", font=dict(size=18, color='white')),
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
    """Hedging pressure chart with Flip Zones"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    flip_zones = identify_gamma_flip_zones(df_sorted, spot_price)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['hedging_pressure'],
        orientation='h',
        marker=dict(
            color=df_sorted['hedging_pressure'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title=dict(text='Pressure %', font=dict(color='white', size=12)),
                tickfont=dict(color='white'),
                x=1.02,
                len=0.7,
                thickness=20
            ),
            cmin=-100,
            cmax=100
        ),
        hovertemplate='Strike: %{y:,.0f}<br>Pressure: %{x:.1f}%<extra></extra>',
        showlegend=False
    ))
    
    fig.add_hline(y=spot_price, line_dash="dash", line_color="#06b6d4", line_width=3,
                  annotation_text=f"Spot: {spot_price:,.2f}", annotation_position="top right",
                  annotation=dict(font=dict(size=12, color="white")))
    
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
    
    # Add gamma flip zones
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
        title=dict(text="<b>🎪 Hedging Pressure with Flip Zones</b>", font=dict(size=18, color='white')),
        xaxis_title="Hedging Pressure (%)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,35,50,0.8)',
        height=700,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.5)',
            zerolinewidth=2,
            range=[-110, 110]
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=120, t=80, b=80)
    )
    
    return fig

def create_oi_distribution(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """OI Distribution chart"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['call_oi'],
        orientation='h',
        name='Call OI',
        marker_color='#10b981',
        opacity=0.7,
        hovertemplate='Strike: %{y:,.0f}<br>Call OI: %{x:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=-df_sorted['put_oi'],
        orientation='h',
        name='Put OI',
        marker_color='#ef4444',
        opacity=0.7,
        hovertemplate='Strike: %{y:,.0f}<br>Put OI: %{customdata:,.0f}<extra></extra>',
        customdata=df_sorted['put_oi']
    ))
    
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
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)', 
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, autorange=True),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_vanna_exposure_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
    """VANNA Exposure chart"""
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
        hovertemplate='Strike: %{y:,.0f}<br>Call VANNA: %{x:.4f}Cr<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['put_vanna'],
        orientation='h',
        marker=dict(color=colors_put),
        name='Put VANNA',
        hovertemplate='Strike: %{y:,.0f}<br>Put VANNA: %{x:.4f}Cr<extra></extra>'
    ), row=1, col=2)
    
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
    """CHARM Exposure chart"""
    df_sorted = df.sort_values('strike').reset_index(drop=True)
    
    colors_call = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['call_charm']]
    colors_put = ['#10b981' if x > 0 else '#ef4444' for x in df_sorted['put_charm']]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📈 Call CHARM", "📉 Put CHARM"),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['call_charm'],
        orientation='h',
        marker=dict(color=colors_call),
        name='Call CHARM',
        hovertemplate='Strike: %{y:,.0f}<br>Call CHARM: %{x:.4f}Cr<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        y=df_sorted['strike'],
        x=df_sorted['put_charm'],
        orientation='h',
        marker=dict(color=colors_put),
        name='Put CHARM',
        hovertemplate='Strike: %{y:,.0f}<br>Put CHARM: %{x:.4f}Cr<extra></extra>'
    ), row=1, col=2)
    
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
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="main-title">📈 NYZTrade Stock Options GEX/DEX Dashboard</h1>
                <p class="sub-title">Historical Stock Options Greeks | Weekly/Monthly | VANNA & CHARM | Gamma Flip Zones | IST</p>
            </div>
            <div class="history-indicator">
                <div class="history-dot"></div>
                <span style="color: #3b82f6; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">STOCK OPTIONS</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Stock selection with categories
        st.markdown("#### 📊 Select Stock")
        
        stock_categories = {
            "Banking & Finance": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "BAJAJFINSV"],
            "IT & Technology": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
            "Energy & Power": ["RELIANCE", "ONGC", "POWERGRID", "NTPC", "COALINDIA"],
            "Auto & Industrial": ["MARUTI", "TATAMOTORS", "M&M", "LT"],
            "FMCG & Consumer": ["HINDUNILVR", "ITC", "ASIANPAINT", "TITAN"],
            "Pharma & Chemicals": ["SUNPHARMA"],
            "Metals & Mining": ["TATASTEEL"],
            "Telecom & Infra": ["BHARTIARTL", "ADANIPORTS"]
        }
        
        category = st.selectbox("Select Category", list(stock_categories.keys()))
        symbol = st.selectbox("Select Stock", stock_categories[category])
        
        # Display stock info
        stock_info = STOCK_CONFIG.get(symbol, {"lot_size": 500, "strike_interval": 10})
        st.info(f"📦 Lot Size: {stock_info['lot_size']}\n\n⚡ Strike Interval: ₹{stock_info['strike_interval']}")
        
        st.markdown("---")
        st.markdown("### 📅 Historical Date Selection")
        
        date_range_option = st.selectbox(
            "Select Date Range",
            ["Last 30 Days", "Last 60 Days", "Last 90 Days", "Custom Range"],
            index=0
        )
        
        if date_range_option == "Custom Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=30),
                    max_value=datetime.now(),
                    min_value=datetime.now() - timedelta(days=90)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now(),
                    min_value=start_date
                )
            
            date_list = pd.date_range(start=start_date, end=end_date, freq='D')
            date_list = [d for d in date_list if d.weekday() < 5]
            
        else:
            days_back = {"Last 30 Days": 30, "Last 60 Days": 60, "Last 90 Days": 90}[date_range_option]
            date_list = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            date_list = [d for d in date_list if d.weekday() < 5]
        
        available_dates = [d.date() for d in date_list]
        
        if len(available_dates) > 0:
            st.caption(f"📊 {len(available_dates)} trading days available")
        
        selected_date = st.selectbox(
            "Select Trading Day",
            options=available_dates,
            index=len(available_dates)-1 if len(available_dates) > 0 else 0,
            format_func=lambda x: x.strftime('%Y-%m-%d (%A)')
        )
        
        target_date = selected_date.strftime('%Y-%m-%d')
        
        st.markdown("---")
        st.markdown("### 📆 Expiry Selection")
        
        expiry_type = st.selectbox("Expiry Type", ["Monthly", "Weekly"], index=0)
        expiry_flag = "MONTH" if expiry_type == "Monthly" else "WEEK"
        
        expiry_option = st.selectbox(
            "Select Expiry",
            ["Current (Nearest)", "Next", "Far"],
            index=0
        )
        
        expiry_code = {"Current (Nearest)": 1, "Next": 2, "Far": 3}[expiry_option]
        
        st.info(f"📌 {expiry_type} | {expiry_option}")
        
        st.markdown("---")
        st.markdown("### 🎯 Strike Selection")
        
        strikes = st.multiselect(
            "Select Strikes",
            ["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3", 
             "ATM+4", "ATM-4", "ATM+5", "ATM-5"],
            default=["ATM", "ATM+1", "ATM-1", "ATM+2", "ATM-2", "ATM+3", "ATM-3"]
        )
        
        st.markdown("---")
        st.markdown("### ⏱️ Time Interval")
        
        interval = st.selectbox(
            "Select Interval",
            options=["1", "5", "15", "60"],
            format_func=lambda x: {"1": "1 minute", "5": "5 minutes", "15": "15 minutes", "60": "1 hour"}[x],
            index=2  # Default to 15 minutes
        )
        
        st.info(f"📊 {len(strikes)} strikes | {interval} min")
        
        st.markdown("---")
        
        fetch_button = st.button("🚀 Fetch Stock Options Data", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.markdown("### 🕐 Indian Standard Time")
        ist_now = datetime.now(IST)
        st.info(f"IST: {ist_now.strftime('%H:%M:%S')}")
    
    # Store configuration
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
    
    # Main content
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
                <div class="metric-label">Fetching Stock Options Data</div>
                <div class="metric-value" style="color: #3b82f6; font-size: 1.2rem;">
                    {symbol} | {target_date} | {interval} min | {expiry_flag} {expiry_code}
                </div>
                <div class="metric-delta">This may take 1-3 minutes...</div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                fetcher = DhanStockOptionsFetcher(DhanConfig())
                df, meta = fetcher.process_historical_data(symbol, target_date, strikes, interval, expiry_code, expiry_flag)
                
                if df is None or len(df) == 0:
                    st.error("❌ No data available. Please check if it was a trading day or try different settings.")
                    return
                
                st.session_state.df_data = df
                st.session_state.meta_data = meta
                st.session_state.data_fetched = True
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
        
        # Retrieve from session state
        df = st.session_state.df_data
        meta = st.session_state.meta_data
        
        all_timestamps = sorted(df['timestamp'].unique())
        
        st.success(f"✅ Data fetched | {len(df):,} records | Lot Size: {meta['lot_size']}")
        
        st.markdown("---")
        st.markdown("### ⏱️ Time Navigation")
        
        # Time controls (same as index dashboard)
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
                <div class="metric-label">Start</div>
                <div class="metric-value" style="font-size: 1.2rem;">{all_timestamps[0].strftime('%H:%M')}</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            if 'timestamp_idx' not in st.session_state:
                st.session_state.timestamp_idx = len(all_timestamps) - 1
            
            selected_timestamp_idx = st.slider(
                "🎯 Navigate through time",
                min_value=0,
                max_value=len(all_timestamps) - 1,
                value=st.session_state.timestamp_idx,
                format="",
                key="time_slider"
            )
            
            st.session_state.timestamp_idx = selected_timestamp_idx
            selected_timestamp = all_timestamps[selected_timestamp_idx]
            
            progress = (selected_timestamp_idx + 1) / len(all_timestamps)
            st.progress(progress)
            
            st.info(f"📍 **{selected_timestamp.strftime('%H:%M:%S IST')}** | Point {selected_timestamp_idx + 1} of {len(all_timestamps)}")
        
        with col3:
            st.markdown(f"""<div class="metric-card neutral" style="padding: 15px;">
                <div class="metric-label">End</div>
                <div class="metric-value" style="font-size: 1.2rem;">{all_timestamps[-1].strftime('%H:%M')}</div>
            </div>""", unsafe_allow_html=True)
        
        # Filter data
        df_selected = df[df['timestamp'] == selected_timestamp].copy()
        
        if len(df_selected) == 0:
            closest_idx = min(range(len(all_timestamps)), 
                             key=lambda i: abs((all_timestamps[i] - selected_timestamp).total_seconds()))
            df_selected = df[df['timestamp'] == all_timestamps[closest_idx]].copy()
        
        df_latest = df_selected
        spot_price = df_latest['spot_price'].iloc[0] if len(df_latest) > 0 else 0
        
        # Calculate metrics
        total_gex = df_latest['net_gex'].sum()
        total_dex = df_latest['net_dex'].sum()
        total_net = total_gex + total_dex
        total_call_oi = df_latest['call_oi'].sum()
        total_put_oi = df_latest['put_oi'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
        
        flip_zones = identify_gamma_flip_zones(df_latest, spot_price)
        
        # Display metrics
        st.markdown("### 📊 Stock Options Overview")
        
        if len(flip_zones) > 0:
            flip_info = " | ".join([f"🔄 {z['strike']:,.0f} {z['arrow']}" for z in flip_zones[:3]])
            st.info(f"🎯 **Gamma Flip Zones**: {flip_info}")
        
        cols = st.columns(6)
        
        with cols[0]:
            st.markdown(f"""<div class="metric-card neutral">
                <div class="metric-label">Stock & Date</div>
                <div class="metric-value" style="font-size: 1.2rem;">{symbol}</div>
                <div class="metric-delta">{target_date}</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""<div class="metric-card neutral">
                <div class="metric-label">Spot Price</div>
                <div class="metric-value">₹{spot_price:,.2f}</div>
                <div class="metric-delta">{selected_timestamp.strftime('%H:%M IST')}</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[2]:
            gex_class = "positive" if total_gex > 0 else "negative"
            st.markdown(f"""<div class="metric-card {gex_class}">
                <div class="metric-label">Total NET GEX</div>
                <div class="metric-value {gex_class}">{total_gex:.4f}Cr</div>
                <div class="metric-delta">{'Suppression' if total_gex > 0 else 'Amplification'}</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[3]:
            dex_class = "positive" if total_dex > 0 else "negative"
            st.markdown(f"""<div class="metric-card {dex_class}">
                <div class="metric-label">Total NET DEX</div>
                <div class="metric-value {dex_class}">{total_dex:.4f}Cr</div>
                <div class="metric-delta">{'Bullish' if total_dex > 0 else 'Bearish'}</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[4]:
            net_class = "positive" if total_net > 0 else "negative"
            st.markdown(f"""<div class="metric-card {net_class}">
                <div class="metric-label">GEX + DEX</div>
                <div class="metric-value {net_class}">{total_net:.4f}Cr</div>
                <div class="metric-delta">Combined</div>
            </div>""", unsafe_allow_html=True)
        
        with cols[5]:
            pcr_class = "positive" if pcr > 1 else "negative"
            st.markdown(f"""<div class="metric-card {pcr_class}">
                <div class="metric-label">Put/Call Ratio</div>
                <div class="metric-value {pcr_class}">{pcr:.2f}</div>
                <div class="metric-delta">{'Bearish' if pcr > 1.2 else 'Bullish' if pcr < 0.8 else 'Neutral'}</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Signal badges
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
        
        # Tabs for visualizations
        tabs = st.tabs(["🎯 GEX", "📊 DEX", "⚡ NET", "🎪 Hedge", 
                        "🌊 VANNA", "⏰ CHARM",
                        "📈 Timeline", "📋 Data"])
        
        with tabs[0]:
            st.markdown("### 🎯 Gamma Exposure with Flip Zones")
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
            
            st.markdown("""
            **VANNA**: Delta sensitivity to volatility changes
            - Critical for hedging in volatile markets
            - Positive VANNA: Delta increases with volatility
            """)
        
        with tabs[5]:
            st.markdown("### ⏰ CHARM Exposure")
            st.plotly_chart(create_charm_exposure_chart(df_latest, spot_price), use_container_width=True)
            
            st.markdown("""
            **CHARM**: Delta decay over time
            - Shows time decay effects on delta
            - Important near expiration
            """)
        
        with tabs[6]:
            st.markdown("### 📈 Intraday Evolution")
            st.plotly_chart(create_intraday_timeline(df, selected_timestamp), use_container_width=True)
        
        with tabs[7]:
            st.markdown("### 📋 Open Interest")
            st.plotly_chart(create_oi_distribution(df_latest, spot_price), use_container_width=True)
            
            st.markdown("### 📊 Data Table")
            display_df = df_latest[['strike', 'call_oi', 'put_oi', 'net_gex', 'net_dex']].copy()
            display_df['net_gex'] = display_df['net_gex'].apply(lambda x: f"{x:.4f}Cr")
            display_df['net_dex'] = display_df['net_dex'].apply(lambda x: f"{x:.4f}Cr")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"{symbol}_GEX_DEX_{target_date}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("""
        👋 **Welcome to NYZTrade Stock Options Dashboard!**
        
        **Features:**
        - 📈 30 Popular Stock Options
        - 🌊 VANNA & CHARM Greeks
        - 🔄 Gamma Flip Zones
        - ⏱️ 1-min to 1-hour intervals
        - 📊 Historical data up to 90 days
        
        **How to use:**
        1. Select stock category and symbol
        2. Choose date and expiry
        3. Select strikes (ATM ±5)
        4. Pick interval
        5. Click "Fetch Data"
        6. Navigate through time
        
        **Available Stocks:**
        - Banking: HDFCBANK, ICICIBANK, SBIN, etc.
        - IT: TCS, INFY, WIPRO, etc.
        - Others: RELIANCE, MARUTI, LT, etc.
        
        ⚠️ Educational purposes only
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""<div style="text-align: center; padding: 20px;">
        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #64748b;">
        NYZTrade Stock Options GEX/DEX Dashboard | Powered by Dhan API<br>
        VANNA & CHARM | Gamma Flip Zones | 30 Stocks Available</p>
        <p style="font-size: 0.75rem;">⚠️ For educational purposes only</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
