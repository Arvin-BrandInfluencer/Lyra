# ================================================
# FILE: app.py (FINAL VERSION - FIXES JSON SERIALIZATION ERROR)
# ================================================
import os
import sys
import traceback
import math
import pandas as pd
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from loguru import logger
from typing import Dict, Any

# --- 1. GLOBAL CONFIGURATION & INITIALIZATION ---
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", colorize=True)
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
if not url or not key:
    logger.critical("Supabase URL and Key must be set in the .env file.")
    sys.exit(1)
try:
    supabase: Client = create_client(url, key)
    logger.success("Successfully connected to Supabase.")
except Exception as e:
    logger.critical(f"Failed to connect to Supabase. Check credentials. Error: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# --- Shared Constants ---
CAMPAIGN_VIEW_NAME = 'all_influencer_campaigns'
TARGET_VIEW_NAME = 'all_market_targets'

NORDIC_COUNTRIES = ['Sweden', 'Norway', 'Denmark']
HARDCODED_RATES = {"EUR": 1.0, "GBP": 0.85, "SEK": 11.30, "NOK": 11.50, "DKK": 7.46}
MONTH_ORDER = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# --- 2. SHARED HELPER FUNCTIONS ---
def convert_to_eur(amount, currency):
    if amount is None: return 0.0
    rate = HARDCODED_RATES.get(str(currency).upper(), 1.0)
    return float(amount) / rate if rate != 0 else 0.0

# --- 3. ENDPOINT DEFINITIONS ---
@app.route('/api/influencer/query', methods=['POST'])
def handle_influencer_query():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        source = payload.get("source")
        logger.info(f"Routing request for source: '{source}'")

        if source == "dashboard":
            result = _get_dashboard_data_from_view(payload)
        elif source == "influencer_analytics":
            result = _get_analytics_data_from_view(payload)
        else:
            result = {"error": f"Invalid 'source'. Must be 'dashboard' or 'influencer_analytics'."}
        
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"})

def _get_dashboard_data_from_view(payload: Dict[str, Any]):
    logger.info("Starting 'get_dashboard_data' processing from view.")
    try:
        filters = payload.get("filters", {})
        market_filter, year_filter = filters.get("market"), filters.get("year")
        
        query = supabase.from_(TARGET_VIEW_NAME).select('*')
        if year_filter and year_filter != "All": query = query.eq('year', int(year_filter))
        if market_filter and market_filter != "All":
            if market_filter == "Nordics": query = query.in_('region', NORDIC_COUNTRIES)
            else: query = query.eq('region', market_filter)
        
        res = query.execute()
        all_data = res.data if res.data else []
        if not all_data: return {"kpi_summary": {}, "monthly_detail": []}

        df = pd.DataFrame(all_data)
        numeric_cols = ['year', 'target_budget_clean', 'actual_spend_clean', 'target_conversions_clean', 'actual_conversions_clean']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if market_filter == "Nordics":
            df['target_budget_eur'] = df.apply(lambda row: convert_to_eur(row['target_budget_clean'], row['currency']), axis=1)
            df['actual_spend_eur'] = df.apply(lambda row: convert_to_eur(row['actual_spend_clean'], row['currency']), axis=1)
            monthly_agg = df.groupby('month').agg(
                target_budget_clean=('target_budget_eur', 'sum'), actual_spend_clean=('actual_spend_eur', 'sum'),
                target_conversions_clean=('target_conversions_clean', 'sum'), actual_conversions_clean=('actual_conversions_clean', 'sum')
            ).reset_index()
            monthly_agg['region'], monthly_agg['currency'] = 'Nordics', 'EUR'
            df = monthly_agg

        kpi = {
            'target_budget': int(df['target_budget_clean'].sum()), 
            'actual_spend': int(df['actual_spend_clean'].sum()),
            'target_conversions': int(df['target_conversions_clean'].sum()), 
            'actual_conversions': int(df['actual_conversions_clean'].sum())
        }
        kpi['actual_cac'] = float(kpi['actual_spend'] / kpi['actual_conversions']) if kpi['actual_conversions'] else 0.0
        
        df.fillna(0, inplace=True); df.replace([float('inf'), -float('inf')], 0, inplace=True)
        df['month_order'] = df['month'].map(MONTH_ORDER)
        df = df.sort_values('month_order').drop(columns=['month_order'])
        
        return {"source": "dashboard", "kpi_summary": kpi, "monthly_detail": df.to_dict(orient='records')}
    except Exception as e:
        logger.error(f"Dashboard query from view failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Dashboard query failed: {str(e)}"}

def _get_analytics_data_from_view(payload: Dict[str, Any]):
    logger.info("Starting analytics request directly from view.")
    try:
        filters = payload.get("filters", {})
        query = supabase.from_(CAMPAIGN_VIEW_NAME).select("*")

        if influencer_name := filters.get("influencer_name"): query = query.ilike('influencer_name', f'%{influencer_name.strip()}%')
        if year_str := filters.get("year", "All"):
            if year_str != "All": query = query.eq('year', int(year_str))
        if market := filters.get("market", "All"):
            if market != "All":
                markets_to_filter = NORDIC_COUNTRIES if market == "Nordics" else [market]
                query = query.in_('market', markets_to_filter)
        if month := filters.get("month", "All"):
            if month != "All": query = query.eq('month', month)

        response = query.execute()
        if not response.data:
            logger.warning(f"No data found in view matching filters: {filters}")
            return {"items": [], "count": 0}
            
        logger.success(f"Fetched {len(response.data)} filtered records from view.")
        df = pd.DataFrame(response.data)
        
        numeric_cols = ['total_budget_clean', 'actual_conversions_clean', 'views_clean', 'views', 'clicks_clean', 'clicks', 'ctr_clean', 'cvr_clean']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['views'] = df['views_clean'].where(df['views_clean'] > 0, df['views'])
        df['clicks'] = df['clicks_clean'].where(df['clicks_clean'] > 0, df['clicks'])

        view = payload.get("view", "summary")
        if "influencer_name" in filters: return _influencer_process_profile(df, filters.get("influencer_name"))
        if view == "summary": return _influencer_process_summary(df, payload)
        if view == "discovery_tiers": return _influencer_process_discovery_tiers(df, payload)
        if view == "monthly_breakdown": return _influencer_process_monthly_breakdown(df)
        
        return {"error": f"Invalid view '{view}'."}
    except Exception as e:
        logger.error(f"Analytics request from view failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Influencer Analytics query failed: {str(e)}"}

# --- Data Processing Helper Functions ---
def _influencer_process_summary(df: pd.DataFrame, payload: dict):
    grouped = df.groupby('influencer_name').apply(lambda x: pd.Series({
        # CORRECTED: Wrap all pandas aggregations in standard Python types
        'campaign_count': int(len(x)),
        'total_conversions': int(x['actual_conversions_clean'].sum()),
        'total_views': int(x['views'].sum()),
        'total_clicks': int(x['clicks'].sum()),
        'markets': list(x['market'].unique()),
        'assets': list(x['asset'].dropna().unique()),
        'total_spend_eur': float(sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in x.iterrows())),
        'avg_ctr': float(x[x['ctr_clean'] > 0]['ctr_clean'].mean()),
        'avg_cvr': float(x[x['cvr_clean'] > 0]['cvr_clean'].mean())
    })).reset_index()
    
    grouped['effective_cac_eur'] = (grouped['total_spend_eur'] / grouped['total_conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)
    grouped.fillna({'avg_ctr': 0, 'avg_cvr': 0}, inplace=True)
    
    if sort_config := payload.get("sort"):
        grouped = grouped.sort_values(by=sort_config.get("by", "total_spend_eur"), ascending=sort_config.get("order", "desc") == "asc")
        
    return {"source": "influencer_summary", "count": len(grouped), "items": grouped.to_dict(orient='records')}

def _influencer_process_discovery_tiers(df: pd.DataFrame, payload: dict):
    summary_result = _influencer_process_summary(df, {})
    if not summary_result.get("items"): return {"gold": [], "silver": [], "bronze": []}
    grouped = pd.DataFrame(summary_result["items"])
    if grouped.empty: return {"gold": [], "silver": [], "bronze": []}
    
    zero_cac = grouped[grouped['effective_cac_eur'] <= 0]
    ranked = grouped[grouped['effective_cac_eur'] > 0].sort_values(by='effective_cac_eur', ascending=True)
    count = len(ranked)
    top_third_index, mid_third_index = math.ceil(count / 3), math.ceil(count * 2 / 3)
    gold_df, silver_df = ranked.iloc[:top_third_index], ranked.iloc[top_third_index:mid_third_index]
    bronze_df = pd.concat([ranked.iloc[mid_third_index:], zero_cac])
    
    all_tiers = {"gold": gold_df.to_dict(orient='records'), "silver": silver_df.to_dict(orient='records'), "bronze": bronze_df.to_dict(orient='records')}
    
    if requested_tier := payload.get("filters", {}).get("tier"):
        if requested_tier in all_tiers:
            return {"source": "discovery_tier_specific", "tier": requested_tier, "items": all_tiers[requested_tier]}
            
    return {"source": "discovery_tiers", **all_tiers}

def _influencer_process_monthly_breakdown(df: pd.DataFrame):
    if df.empty or 'month' not in df.columns: return {"monthly_data": []}
    df = df.dropna(subset=['month'])
    results = []
    
    for month_name, month_df in df.groupby('month'):
        # CORRECTED: Wrap all pandas aggregations in standard Python types
        total_spend_eur = float(sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in month_df.iterrows()))
        total_conversions = int(month_df['actual_conversions_clean'].sum())
        
        summary = {
            'total_spend_eur': total_spend_eur,
            'total_conversions': total_conversions,
            'avg_cac_eur': total_spend_eur / total_conversions if total_conversions else 0.0,
            'influencer_count': int(month_df['influencer_name'].nunique())
        }
        
        details = month_df[['influencer_name', 'market', 'currency', 'total_budget_clean', 'actual_conversions_clean']].rename(columns={'total_budget_clean': 'budget_local', 'actual_conversions_clean': 'conversions'})
        details['cac_local'] = (details['budget_local'] / details['conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)
        results.append({'month': month_name, 'summary': summary, 'details': details.to_dict(orient='records')})
        
    results.sort(key=lambda x: MONTH_ORDER.get(x['month'], 99))
    return {"source": "monthly_breakdown", "monthly_data": results}

def _influencer_process_profile(df: pd.DataFrame, influencer_name: str):
    influencer_df = df.copy()
    influencer_df['cac_local'] = (influencer_df['total_budget_clean'] / influencer_df['actual_conversions_clean']).fillna(0).replace([float('inf'), -float('inf')], 0)
    influencer_df['ctr'] = (influencer_df['clicks'] / influencer_df['views']).fillna(0).replace([float('inf'), -float('inf')], 0)
    
    if 'month' in influencer_df.columns:
        influencer_df['month_order'] = influencer_df['month'].map(MONTH_ORDER)
        influencer_df = influencer_df.sort_values(by=['year', 'month_order'])
        
    return {"source": "influencer_detail", "campaigns": influencer_df.to_dict(orient='records')}

# --- 4. HEALTH CHECK & SERVER EXECUTION ---
@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Unified Brand Influence Query API is running."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Unified Flask API server on host 0.0.0.0 and port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
