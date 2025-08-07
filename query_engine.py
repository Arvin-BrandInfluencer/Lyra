# arvin-brandinfluencer-api/query_engine.py

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
import math
import traceback

# --- Initialization & Configuration ---
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# --- CONFIGURATION (Unchanged) ---
MARKET_TARGET_TABLES = [
    'uk_target', 'france_target', 'sweden_target', 'norway_target', 'denmark_target'
]
INFLUENCER_CAMPAIGN_TABLES = [
    'france_2023_influencers', 'france_2024_influencers', 'france_2025_influencers',
    'uk_2023_influencers', 'uk_2024_influencers', 'uk_2025_influencers',
    'nordics_2025_influencers'
]

# --- Helper Functions (Unchanged) ---
HARDCODED_RATES = { "EUR": 1.0, "GBP": 0.85, "SEK": 11.30, "NOK": 11.50, "DKK": 7.46 }
NORDIC_COUNTRIES = ['Sweden', 'Norway', 'Denmark']
MONTH_ORDER = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def convert_to_eur(amount, currency):
    rate = HARDCODED_RATES.get(str(currency).upper(), 1.0)
    return amount / rate if rate != 0 else 0

def get_market_from_table_name(table_name):
    if 'uk' in table_name: return 'UK'
    if 'france' in table_name: return 'France'
    if 'sweden' in table_name: return 'Sweden'
    if 'norway' in table_name: return 'Norway'
    if 'denmark' in table_name: return 'Denmark'
    if 'nordics' in table_name: return 'Nordics'
    return 'Unknown'
    
# --- Main Query Router (Unchanged) ---
def execute_query(payload: dict):
    source = payload.get("source")
    if source == "dashboard":
        return get_dashboard_data(payload)
    elif source == "influencer_analytics":
        return get_influencer_analytics_data(payload)
    else:
        return {"error": f"Invalid 'source'. Must be 'dashboard' or 'influencer_analytics'."}

# --- Data Fetching & Processing Functions ---

def get_dashboard_data(payload: dict):
    # This function remains unchanged as the new features apply to influencer analytics
    try:
        filters = payload.get("filters", {})
        all_data = []
        for table in MARKET_TARGET_TABLES:
            res = supabase.from_(table).select('*').execute()
            for row in res.data:
                row['table_source'] = table
            all_data.extend(res.data)
        if not all_data: return {"kpi_summary": {}, "monthly_detail": []}
        df = pd.DataFrame(all_data)
        df['region'] = df['table_source'].apply(get_market_from_table_name)
        df['currency'] = df['region'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'})
        numeric_cols = ['year', 'target_budget_clean', 'actual_spend_clean', 'target_conversions_clean', 'actual_conversions_clean']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if market := filters.get("market"):
            df = df[df['region'] == market]
        if year := filters.get("year"):
            df = df[df['year'] == int(year)]
        kpi = {
            'target_budget': df['target_budget_clean'].sum(),
            'actual_spend': df['actual_spend_clean'].sum(),
            'target_conversions': df['target_conversions_clean'].sum(),
            'actual_conversions': df['actual_conversions_clean'].sum(),
        }
        kpi['actual_cac'] = kpi['actual_spend'] / kpi['actual_conversions'] if kpi['actual_conversions'] else 0
        df['target_cac'] = df['target_budget_clean'] / df['target_conversions_clean']
        df['actual_cac'] = df['actual_spend_clean'] / df['actual_conversions_clean']
        df.fillna(0, inplace=True)
        df.replace([float('inf'), -float('inf')], 0, inplace=True)
        return { "source": "dashboard", "kpi_summary": kpi, "monthly_detail": df.to_dict(orient='records') }
    except Exception as e:
        return {"error": f"Dashboard query failed: {str(e)}"}

def get_influencer_analytics_data(payload: dict):
    """
    Handles all advanced analytics for influencer data.
    """
    try:
        # Step 1: Fetch and Standardize data (Unchanged)
        all_campaigns = []
        for table in INFLUENCER_CAMPAIGN_TABLES:
            try: table_year = int(table.split('_')[1])
            except (IndexError, ValueError): table_year = 0 
            res = supabase.from_(table).select('*').execute()
            if res.data:
                for row in res.data:
                    row['table_source'] = table
                    if 'year' not in row or not row['year']: row['year'] = table_year
                all_campaigns.extend(res.data)
        if not all_campaigns: return { "items": [], "count": 0 }
        df = pd.DataFrame(all_campaigns)
        if df.empty: return {"count": 0, "items": []}
        
        # Step 2: Standardize and clean columns (Unchanged)
        df['market'] = df.apply(lambda row: row.get('market') or get_market_from_table_name(row.get('table_source', '')), axis=1)
        df['currency'] = df['market'].map({'UK': 'GBP', 'France': 'EUR', 'Sweden': 'SEK', 'Norway': 'NOK', 'Denmark': 'DKK'}).fillna('EUR')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        numeric_cols = ['total_budget_clean', 'actual_conversions_clean', 'views_clean', 'views', 'clicks_clean', 'clicks', 'ctr_clean', 'cvr_clean']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['views'] = df['views_clean'].where(df['views_clean'] > 0, df['views'])
        df['clicks'] = df['clicks_clean'].where(df['clicks_clean'] > 0, df['clicks'])

        # --- NEW LOGIC STARTS HERE ---
        
        # Step 3: Aggregate data first to prepare for advanced filtering
        grouped = df.groupby('influencer_name').apply(lambda x: pd.Series({
            'campaign_count': len(x),
            'total_conversions': x['actual_conversions_clean'].sum(),
            'total_views': x['views'].sum(),
            'total_clicks': x['clicks'].sum(),
            'markets': list(x['market'].unique()),
            'assets': list(x['asset'].dropna().unique()),
            'total_spend_eur': sum(convert_to_eur(row['total_budget_clean'], row['currency']) for _, row in x.iterrows())
        })).reset_index()
        grouped['effective_cac_eur'] = (grouped['total_spend_eur'] / grouped['total_conversions']).fillna(0).replace([float('inf'), -float('inf')], 0)

        # Step 4: Apply Advanced Filters on the aggregated data
        if filters := payload.get("filters"):
            for key, value in filters.items():
                if isinstance(value, dict): # Advanced filter e.g., {"gt": 100}
                    for op, num in value.items():
                        if op == 'gt': grouped = grouped[grouped[key] > num]
                        if op == 'lt': grouped = grouped[grouped[key] < num]
                        if op == 'gte': grouped = grouped[grouped[key] >= num]
                        if op == 'lte': grouped = grouped[grouped[key] <= num]
                else: # Simple equality filter
                    if key == 'market':
                        markets_to_filter = NORDIC_COUNTRIES if value == "Nordics" else [value]
                        # This filter is tricky post-aggregation. We check if the market is in the list of markets for the influencer.
                        grouped = grouped[grouped['markets'].apply(lambda markets_list: any(m in markets_to_filter for m in markets_list))]
                    # Note: Filtering by year must happen before aggregation if needed, or added to the grouped df. For simplicity, we assume year is a pre-filter.
        
        # Step 5: Handle Statistical Summary requests
        if payload.get("summary_stats"):
            if grouped.empty: return {"summary": {}, "count": 0}
            summary = {
                'count': int(grouped.shape[0]),
                'avg_spend_eur': grouped['total_spend_eur'].mean(),
                'median_spend_eur': grouped['total_spend_eur'].median(),
                'total_spend_eur': grouped['total_spend_eur'].sum(),
                'avg_conversions': grouped['total_conversions'].mean(),
                'median_conversions': grouped['total_conversions'].median(),
                'total_conversions': grouped['total_conversions'].sum(),
                'avg_cac_eur': grouped[grouped['effective_cac_eur'] > 0]['effective_cac_eur'].mean(),
                'median_cac_eur': grouped[grouped['effective_cac_eur'] > 0]['effective_cac_eur'].median(),
            }
            return {"source": "influencer_summary_stats", "summary": {k: (v if pd.notna(v) else 0) for k, v in summary.items()}}

        # Step 6: Apply Sorting and Top N logic
        sort_config = payload.get("sort")
        top_n = payload.get("top_n")

        # Define default sorting for "best" performers
        sort_defaults = {
            'total_spend_eur': False, # descending
            'total_conversions': False, # descending
            'effective_cac_eur': True # ascending
        }
        
        sort_by = sort_config.get("by") if sort_config else 'effective_cac_eur'
        # Default to descending unless it's a cost metric
        default_ascending = sort_defaults.get(sort_by, False)

        if top_n and isinstance(top_n, int):
            limit = abs(top_n)
            # Negative top_n means "worst", so we reverse the default sort order
            is_ascending = default_ascending if top_n > 0 else not default_ascending
            
            # Special case for CAC where worst is descending
            if sort_by == 'effective_cac_eur':
                df_to_sort = grouped[grouped['effective_cac_eur'] > 0]
            else:
                df_to_sort = grouped

            sorted_df = df_to_sort.sort_values(by=sort_by, ascending=is_ascending)
            final_df = sorted_df.head(limit)
        elif sort_config:
            is_ascending = sort_config.get("order", "asc") == "asc"
            final_df = grouped.sort_values(by=sort_by, ascending=is_ascending)
        else:
            final_df = grouped

        return { "source": "influencer_summary", "count": len(final_df), "items": final_df.to_dict(orient='records') }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": f"Influencer Analytics query failed: {str(e)}"}
