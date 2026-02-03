
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import sys

# Add python-services to path to import modules
sys.path.append(os.path.join(os.getcwd(), 'python-services'))

from aurix.features import FeatureEngine as FeatureEngineer

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def debug_bot():
    print("="*50)
    print(" AURIX BOT DIAGNOSTIC")
    print("="*50)

    # 1. Connect to DB
    db_path = 'data/aurix.db'
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    print(" Database connection established")

    # 2. Load Data
    query_15m = "SELECT * FROM candles WHERE timeframe='15m' ORDER BY open_time"
    df_15m = pd.read_sql_query(query_15m, conn)
    
    query_1h = "SELECT * FROM candles WHERE timeframe='1h' ORDER BY open_time"
    df_1h = pd.read_sql_query(query_1h, conn)
    
    conn.close()

    print(f" Loaded {len(df_15m)} 15m candles, {len(df_1h)} 1h candles")
    
    if len(df_15m) < 100:
        print(" No 15m data found. Collector needs more time.")
        return

    # Fix column types
    for df in [df_15m, df_1h]:
        if not df.empty:
            # Check timestamp unit heuristic
            first_ts = df['open_time'].iloc[0]
            if first_ts > 10000000000: # ms
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            else:
                df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
            df.set_index('open_time', inplace=True)

    # 3. Feature Engineering
    print("\n Generating Features...")
    engine = FeatureEngineer() # Aliased from FeatureEngine
    try:
        df_features = engine.compute_features(df_15m, df_1h)
        print(f"   Features generated. Shape: {df_features.shape}")
        
        # Check for NaNs
        na_count = df_features.isna().sum().sum()
        if na_count > 0:
            print(f"    Warning: {na_count} NaN values found")
            df_features = df_features.dropna()
            print(f"   Cleaned shape: {df_features.shape}")
        
        if len(df_features) == 0:
            print(" No data left after dropping NaNs! Need more history.")
            return

    except Exception as e:
        print(f" Feature Engineering Failed: {e}")
        return

    # 4. Load Model
    model_path = 'models/long_model.pkl'
    if not os.path.exists(model_path):
        print(f" Model not found at {model_path}")
        return
    
    try:
        model = joblib.load(model_path)
        print(f" Model loaded: {type(model)}")
        
        # Check expected features if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = config_features = list(model.feature_names_in_)
            generated_features = list(df_features.columns)
            
            print(f"   Model expects {len(expected_features)} features")
            print(f"   Generated {len(generated_features)} features")
            
            missing = set(expected_features) - set(generated_features)
            extra = set(generated_features) - set(expected_features)
            
            if missing:
                print(f" MISSING FEATURES inside DF: {missing}")
                # Try to fill with 0 to allow prediction for debug
                for col in missing:
                    df_features[col] = 0.0
            
            if extra:
                print(f"   (Extra features generated: {len(extra)})")
                
            # Align columns
            df_features = df_features[expected_features]
            
    except Exception as e:
        print(f" Failed to inspect/load model: {e}")
        return

    # 5. Predict on Last 5 Candles
    print("\n Analyzing Last 5 Candles:")
    print(f"{'Time':<20} | {'Close':<10} | {'Prob (LONG)':<12} | {'Decision (0.60)'}")
    print("-" * 65)

    last_rows = df_features.tail(5)
    
    # Ensure we use the exact same columns as training (if possible, blindly predict for now)
    # The model usually expects specific feature subset. 
    # Let's try predicting with current features.
    
    for idx, row in last_rows.iterrows():
        try:
            # Reshape for prediction
            X_curr = row.to_frame().T
            
            # Predict
            prob = model.predict_proba(X_curr)[0][1] # Probability of Class 1 (Long)
            
            decision = "✅ BUY" if prob >= 0.60 else "❌ WAIT"
            
            print(f"{str(idx):<20} | {row['close']:<10.2f} | {prob:.4f}       | {decision}")
            
        except Exception as e:
            print(f"{str(idx):<20} | ERROR: {e}")

    print("\n" + "="*50)
    print("VERDICT:")
    if len(df_features) < 50:
        print("⚠️ Data quantity is low. Some indicators (RSI/MACD) might be unstable.")
    print("Bot is technically WORKING if you see probabilities above.")
    print("If all Prob < 0.60, the market simply doesn't match the strategy (Wait Mode).")
    print("="*50)

if __name__ == "__main__":
    debug_bot()
