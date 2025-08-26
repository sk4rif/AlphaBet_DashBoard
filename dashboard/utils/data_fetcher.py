import datetime
import pandas as pd
import numpy as np
from pymongo import MongoClient
import streamlit as st
from ..config import MONGO_URI, DB_NAME, HISTORY_COLLECTION, TIME_ZONE, REFRESH_INTERVAL

def normalize_dates(start_date, end_date):
    now = datetime.datetime.now(TIME_ZONE)
    if end_date is None:
        end_dt = now
    else:
        end_dt = datetime.datetime.combine(end_date, datetime.time(23, 59, 59), tzinfo=TIME_ZONE)
    if start_date is None:
        start_dt = end_dt - datetime.timedelta(days=7)
    else:
        start_dt = datetime.datetime.combine(start_date, datetime.time(0, 0), tzinfo=TIME_ZONE)
    return start_dt, end_dt

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_data(start_date=None, end_date=None, tab_type="main"):
    try:
        client = MongoClient(MONGO_URI)
        col = client[DB_NAME][HISTORY_COLLECTION]
        start_dt, end_dt = normalize_dates(start_date, end_date)
        fmt = lambda dt: dt.strftime('%Y%m%d_%H%M%S')
        
        # Calculate 24 hours ago from now
        now = datetime.datetime.now(TIME_ZONE)
        twenty_four_hours_ago = now - datetime.timedelta(hours=24)
        
        if tab_type in ["questions", "optimization"]:
            # For Questions and Optimization tabs: only fetch data from past 24 hours
            query_start = max(start_dt, twenty_four_hours_ago)
            query = {"_id": {"$gte": fmt(query_start), "$lte": fmt(end_dt)}}
            docs = list(col.find(query))
        elif tab_type == "main":
            # For Main tab: use aggregation for data older than 24 hours
            recent_query = {"_id": {"$gte": fmt(twenty_four_hours_ago), "$lte": fmt(end_dt)}}
            old_query = {"_id": {"$gte": fmt(start_dt), "$lt": fmt(twenty_four_hours_ago)}}
            
            # Get recent data (last 24 hours) without aggregation
            recent_docs = list(col.find(recent_query))
            
            # Get hourly aggregated data for older than 24 hours
            old_docs = []
            if start_dt < twenty_four_hours_ago:
                old_docs = list(col.aggregate([
                    {"$match": old_query},
                    {"$addFields": {
                        "hour_id": {
                            "$substr": ["$_id", 0, 11]  # Extract YYYYMMDD_HH part
                        }
                    }},
                    {"$group": {
                        "_id": "$hour_id",
                        "order_info": {"$first": "$order_info"},
                        "orders": {"$first": "$orders"},
                        "timestamp": {"$first": "$_id"}
                    }},
                    {"$sort": {"_id": 1}}
                ]))
                # Restore original _id format for consistency
                for doc in old_docs:
                    doc['_id'] = doc['timestamp']
                    del doc['timestamp']
            
            docs = old_docs + recent_docs
        else:
            # Default behavior for other tabs
            query = {"_id": {"$gte": fmt(start_dt), "$lte": fmt(end_dt)}}
            docs = list(col.find(query))
        
        if not docs:
            docs = list(col.aggregate([
                {"$sort": {"_id": -1}},
                {"$limit": 100}
            ]))

        records = []
        for doc in docs:
            ts = pd.to_datetime(doc['_id'], format='%Y%m%d_%H%M%S').tz_localize(TIME_ZONE)
            info = doc.get('order_info', {})
            info['timestamp'] = ts
            records.append(info)
            for order in doc.get('orders', []):
                order['timestamp'] = ts
                records.append(order)

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df.set_index('timestamp', inplace=True)
        if df.empty:
            return df
            
        print("Available columns:", df.columns.tolist())
        
        # Calculate weighted averages for probability and best_ask
        for ts in df.index.unique():
            mask = df.index == ts
            ts_data = df.loc[mask]
            
            # For probability
            if all(col in df.columns for col in ['probability', 'current_weights']):
                weights = ts_data['current_weights']
                prob = ts_data['probability']
                try:
                    weights = pd.to_numeric(weights)
                    prob = pd.to_numeric(prob)
                    df.loc[mask, 'weighted_probability'] = (prob * weights).sum()
                    print(f"\nTimestamp {ts}:")
                    print("Probabilities:", prob.tolist())
                    print("Weights:", weights.tolist())
                    print("Weighted prob:", (prob * weights).sum())
                except Exception as e:
                    print(f"Error calculating weighted probability: {e}")
                    
            # For best_ask
            if all(col in df.columns for col in ['best_ask', 'current_weights']):
                weights = ts_data['current_weights']
                ask = ts_data['best_ask']
                try:
                    weights = pd.to_numeric(weights)
                    ask = pd.to_numeric(ask)
                    weighted_ask = (ask * weights).sum()
                    df.loc[mask, 'weighted_best_ask'] = weighted_ask
                    print("Best asks:", ask.tolist())
                    print("Weighted ask:", weighted_ask)
                    
                    # Calculate difference between probability and best ask
                    if 'weighted_probability' in df.loc[mask].columns:
                        df.loc[mask, 'weighted_difference'] = df.loc[mask, 'weighted_probability'] - weighted_ask
                except Exception as e:
                    print(f"Error calculating weighted best_ask: {e}")

        # Calculate adjustment ratio
        if {'adjustments', 'position_value'}.issubset(df.columns):
            df['Adjustment Ratio'] = df['adjustments'] / df['position_value'].replace(0, np.nan)

        df.sort_index(inplace=True)

        # Filter to active questions
        if 'current_position' in df.columns:
            last_ts = df.index.max()
            active_qs = df.loc[last_ts][
                df.loc[last_ts]['current_position'] != 0]['question']
            df = df[df['question'].isin(active_qs.unique())]

        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_optimization_data():
    """Fetch optimization data with 24-hour filtering at MongoDB level"""
    try:
        from helpers.mongo import MongoClient
        import certifi
        
        # Connect to MongoDB
        client = MongoClient(
            "mongodb+srv://CarlGrimaldi:AlphaBeta21$@alphabetcluster.x7lvc.mongodb.net/",
            tlsCAFile=certifi.where()
        )
        
        db = client["AlphaBet"]
        collection = db["OPTI"]
        
        # Calculate 24 hours ago
        now = datetime.datetime.now(TIME_ZONE)
        twenty_four_hours_ago = now - datetime.timedelta(hours=24)
        
        # Use aggregation to filter history data to last 24 hours
        pipeline = [
            {"$match": {"_id": "RESULTS"}},
            {"$addFields": {
                "filtered_history": {
                    "$filter": {
                        "input": "$history",
                        "cond": {
                            "$gte": [
                                {"$dateFromString": {
                                    "dateString": "$$this.timestamp",
                                    "onError": twenty_four_hours_ago
                                }},
                                twenty_four_hours_ago
                            ]
                        }
                    }
                }
            }},
            {"$project": {
                "history": "$filtered_history",
                "questions": 1,
                "timestamp": 1
            }}
        ]
        
        result = list(collection.aggregate(pipeline))
        return result[0] if result else None
        
    except Exception as e:
        st.error(f"Error fetching optimization data: {e}")
        return None

def fetch_vol_surface(curr):
    from helpers.mongo import pull_data
    raw = pull_data(database=DB_NAME, sub_collection=curr, _id="SVI_IV")
    taus = np.array(raw['Taus'], float)
    strikes = np.array(raw['Strikes'], float)
    iv = np.array(raw['SVI_IV'], float)
    if taus.size and (taus[0] == 0 or np.isnan(iv[0]).all()):
        taus, iv = taus[1:], iv[1:]
    return taus, strikes, iv
