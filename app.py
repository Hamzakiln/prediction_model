from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge
import streamlit as st

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    print("arch package not available, using simple volatility measures")
    ARCH_AVAILABLE = False


warnings.filterwarnings('ignore')


def prepare_X_y(df, exo_cols, target_col):
    X = df[exo_cols].copy()
    X.index = pd.DatetimeIndex(X.index, freq='D')
    y = df[target_col].copy()
    y.index =pd.DatetimeIndex(y.index, freq='D')
    

    return X, y



def create_volatility_features(data, window_sizes=[5, 10, 20, 30]):
    """
    Crée des variables de volatilité incluant GARCH si disponible
    """
    vol_features = pd.DataFrame(index=data.index)
    
    # Variables de volatilité classiques
    for window in window_sizes:
        # Volatilité réalisée (rolling std)
        vol_features[f'realized_vol_{window}d'] = data['SOL_close'].rolling(window).std()
        
        # Volatilité EWMA
        vol_features[f'ewma_vol_{window}d'] = data['SOL_close'].ewm(span=window).std()
        
        # Range-based volatility (si on a OHLC)
        if 'SOL_high' in data.columns and 'SOL_low' in data.columns:
            # Parkinson volatility
            vol_features[f'parkinson_vol_{window}d'] = np.sqrt(
                np.log(data['SOL_high'] / data['SOL_low']).rolling(window).var() / (4 * np.log(2))
            )
    
    # GARCH volatility si disponible
    if ARCH_AVAILABLE and len(data) > 100:
        try:
            # Modèle GARCH(1,1)
            garch_model = arch_model(data['SOL_close'].dropna() * 100, vol='Garch', p=1, q=1)
            garch_fitted = garch_model.fit(disp='off')
            
            # Volatilité conditionnelle GARCH
            garch_vol = garch_fitted.conditional_volatility / 100
            vol_features.loc[garch_vol.index, 'garch_vol'] = garch_vol
            
            # Variance conditionnelle
            vol_features.loc[garch_vol.index, 'garch_var'] = garch_vol ** 2
            
        except Exception as e:
            print(f"GARCH model failed: {e}")
    
    # Variables de volatilité dérivées
    vol_features['vol_ratio_5_20'] = vol_features['realized_vol_5d'] / vol_features['realized_vol_20d']
    vol_features['vol_ratio_10_30'] = vol_features['realized_vol_10d'] / vol_features['realized_vol_30d']
    
    # Volatilité de la volatilité
    vol_features['vol_of_vol'] = vol_features['realized_vol_20d'].rolling(10).std()
    
    return vol_features.fillna(method='ffill').fillna(0)

def create_technical_features(data):
    """
    Crée des indicateurs techniques additionnels
    """
    tech_features = pd.DataFrame(index=data.index)
    
    # RSI
    delta = data['SOL_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    tech_features['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum
    for period in [5, 10, 20]:
        tech_features[f'momentum_{period}d'] = data['SOL_close'] - data['SOL_close'].shift(period)
    
    # Moving averages ratios
    if 'SOL_close' in data.columns:
        ma_5 = data['SOL_close'].rolling(5).mean()
        ma_20 = data['SOL_close'].rolling(20).mean()
        tech_features['ma_ratio_5_20'] = ma_5 / ma_20
    
    return tech_features.fillna(method='ffill').fillna(0)

def prepare_enhanced_features(data, exo_variables, target_var):
    """
    Prépare les features enrichies avec volatilité et indicateurs techniques
    """
    # Features de base
    X_base = data[exo_variables].copy()
    y = data[target_var].copy()
    
    # Features de volatilité
    vol_features = create_volatility_features(data)
    
    # Features techniques
    tech_features = create_technical_features(data)
    
    # Lags additionnels pour certaines variables importantes
    lag_vars = ['SOL_volume', 'SOL_fr', 'ETH_close', 'BTC_close']
    lag_features = pd.DataFrame(index=data.index)
    
    for var in lag_vars:
        if var in data.columns:
            for lag in [1, 2, 3, 5, 7]:
                lag_features[f'{var}_lag_{lag}'] = data[var].shift(lag)
    
    # Combinaison de toutes les features
    X_enhanced = pd.concat([X_base, vol_features, tech_features, lag_features], axis=1)
    
    # Nettoyage
    X_enhanced = X_enhanced.fillna(method='ffill').fillna(0)
    
    # Alignement des indices
    common_idx = X_enhanced.index.intersection(y.index)
    X_enhanced = X_enhanced.loc[common_idx]
    y = y.loc[common_idx]
    
    return X_enhanced, y

def train_random_forest_with_uncertainty(X_train, y_train, n_estimators=200, max_depth=15):
    """
    Entraîne un Random Forest avec estimation d'incertitude
    """
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Random Forest principal
    rf_main = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )
    rf_main.fit(X_train_scaled, y_train)
    
    # Random Forests pour quantiles (pour intervalles de confiance)
    rf_upper = RandomForestRegressor(
        n_estimators=n_estimators//2,
        max_depth=max_depth,
        random_state=123,
        n_jobs=-1
    )
    rf_lower = RandomForestRegressor(
        n_estimators=n_estimators//2,
        max_depth=max_depth,
        random_state=456,
        n_jobs=-1
    )
    
    # Entraînement sur résidus pour estimer l'incertitude
    y_pred_train = rf_main.predict(X_train_scaled)
    residuals = np.abs(y_train - y_pred_train)
    
    rf_upper.fit(X_train_scaled, y_train + 1.645 * residuals)
    rf_lower.fit(X_train_scaled, y_train - 1.645 * residuals)
    
    return rf_main, rf_upper, rf_lower, scaler

def predict_with_confidence(models, X_test):
    """
    Prédiction avec intervalles de confiance
    """
    rf_main, rf_upper, rf_lower, scaler = models
    
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = rf_main.predict(X_test_scaled)
    y_pred_upper = rf_upper.predict(X_test_scaled)
    y_pred_lower = rf_lower.predict(X_test_scaled)
    
    return y_pred, y_pred_upper, y_pred_lower

def log_returns_to_prices(log_returns, initial_price):
    """
    Convertit les log returns en prix
    """
    return initial_price * np.exp(log_returns.cumsum())

def create_future_predictions(models, X_test, data_test, n_future_days=30):
    """
    Crée des prédictions futures avec path de prix
    """
    rf_main, rf_upper, rf_lower, scaler = models
    
    # Dernière date observée
    last_date = X_test.index[-1]
    
    # Création des dates futures
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=n_future_days, freq='D')
    
    # Initialisation
    y_future_pred = []
    y_future_upper = []
    y_future_lower = []
    
    # Dernières valeurs connues pour construire les features futures
    last_values = X_test.iloc[-30:].copy()  # 30 derniers jours pour les lags
    
    for i in range(n_future_days):
        # Pour la prédiction future, on utilise X_test(t-30) comme mentionné
        if i < len(X_test):
            # On prend la valeur à t-30 jours depuis la fin de X_test
            future_features = X_test.iloc[-(30-i)].values.reshape(1, -1)
        else:
            # Pour les jours au-delà, on utilise la dernière valeur disponible
            # et on met à jour certaines features avec les prédictions précédentes
            future_features = X_test.iloc[-1].values.reshape(1, -1)
            
        # Prédiction
        future_features_scaled = scaler.transform(future_features)
        
        pred = rf_main.predict(future_features_scaled)[0]
        pred_upper = rf_upper.predict(future_features_scaled)[0]
        pred_lower = rf_lower.predict(future_features_scaled)[0]
        
        y_future_pred.append(pred)
        y_future_upper.append(pred_upper)
        y_future_lower.append(pred_lower)
    
    return pd.Series(y_future_pred, index=future_dates), \
        pd.Series(y_future_upper, index=future_dates), \
        pd.Series(y_future_lower, index=future_dates)

def load_rewards_data_rew(filepath):
    """
    Charge les données de rewards en filtrant les jours avec rewards > 0
    """
    print("Chargement des données de rewards...")
    rewards_df_rew = pd.read_csv(filepath)
    rewards_df_rew.rename(columns={"sol_revenue":"SOL"},inplace=True)
    
    # Conversion de la colonne DATE
    rewards_df_rew['date'] = pd.to_datetime(rewards_df_rew['date'])
    rewards_df_rew = rewards_df_rew.set_index('date')
    
    # Suppression de la colonne ETH si elle existe
    if 'ETH' in rewards_df_rew.columns:
        rewards_df_rew = rewards_df_rew.drop('ETH', axis=1)
    
    # Tri par date
    rewards_df_rew = rewards_df_rew.sort_index()
    

    print(rewards_df_rew)

    print(f"Données totales: {len(rewards_df_rew)} jours")
    print(f"Jours avec rewards > 0: {(rewards_df_rew['SOL'] > 0).sum()}")
    print(f"Jours avec rewards = 0: {(rewards_df_rew['SOL'] == 0).sum()}")
    
    # Filtrage: garder seulement les jours avec rewards > 0

    rewards_active_rew = rewards_df_rew[rewards_df_rew['SOL'] > 0].copy()
    
    print(f"Données filtrées: {len(rewards_active_rew)} jours avec rewards")
    print(f"Période: {rewards_active_rew.index.min()} à {rewards_active_rew.index.max()}")
    print(f"Reward moyen: {rewards_active_rew['SOL'].mean():.4f} SOL")
    print(f"Reward médian: {rewards_active_rew['SOL'].median():.4f} SOL")
    print(f"Reward std: {rewards_active_rew['SOL'].std():.4f} SOL")
    print(f"Reward CV: {rewards_active_rew['SOL'].std() / rewards_active_rew['SOL'].mean():.2f}")
    
    return rewards_active_rew

def create_advanced_features_rew(rewards_data_rew):
    """
    Crée des features plus sophistiquées pour capturer la volatilité
    """
    features_rew = pd.DataFrame(index=rewards_data_rew.index)
    
    # Variable target
    features_rew['sol_rewards'] = rewards_data_rew['SOL']
    
    # Features temporelles avancées
    features_rew['day_of_week'] = rewards_data_rew.index.dayofweek
    features_rew['day_of_month'] = rewards_data_rew.index.day
    features_rew['month'] = rewards_data_rew.index.month
    features_rew['quarter'] = rewards_data_rew.index.quarter
    features_rew['is_weekend'] = (rewards_data_rew.index.dayofweek >= 5).astype(int)
    features_rew['is_month_end'] = (rewards_data_rew.index.day >= 28).astype(int)
    
    # Encoding cyclique
    features_rew['day_of_week_cos'] = np.cos(2 * np.pi * features_rew['day_of_week'] / 7)
    features_rew['day_of_week_sin'] = np.sin(2 * np.pi * features_rew['day_of_week'] / 7)
    features_rew['month_cos'] = np.cos(2 * np.pi * features_rew['month'] / 12)
    features_rew['month_sin'] = np.sin(2 * np.pi * features_rew['month'] / 12)
    
    # FEATURES DE VOLATILITÉ AMÉLIORÉES
    rewards_series_rew = rewards_data_rew['SOL']
    
    # Moyennes mobiles de différentes fenêtres
    for window in [2, 3, 5, 7, 10, 14, 21, 30]:
        features_rew[f'reward_ma_{window}'] = rewards_series_rew.rolling(window, min_periods=1).mean()
        features_rew[f'reward_std_{window}'] = rewards_series_rew.rolling(window, min_periods=1).std().fillna(0)
        features_rew[f'reward_cv_{window}'] = features_rew[f'reward_std_{window}'] / features_rew[f'reward_ma_{window}']
        
        # Quantiles mobiles pour capturer les extrêmes
        features_rew[f'reward_q25_{window}'] = rewards_series_rew.rolling(window, min_periods=1).quantile(0.25)
        features_rew[f'reward_q75_{window}'] = rewards_series_rew.rolling(window, min_periods=1).quantile(0.75)
        features_rew[f'reward_iqr_{window}'] = features_rew[f'reward_q75_{window}'] - features_rew[f'reward_q25_{window}']
    
    # Lags plus nombreux et patterns
    for lag in [1, 2, 3, 4, 5, 6, 7, 10, 14]:
        features_rew[f'reward_lag_{lag}'] = rewards_series_rew.shift(lag)
        
        # Différences avec lags
        features_rew[f'reward_diff_{lag}'] = rewards_series_rew - rewards_series_rew.shift(lag)
        features_rew[f'reward_pct_change_{lag}'] = rewards_series_rew.pct_change(lag).fillna(0)
    
    # MOMENTUM ET TENDANCES
    for period in [3, 5, 7, 14, 21]:
        # RSI-like indicator
        delta_rew = rewards_series_rew.diff()
        gain_rew = (delta_rew.where(delta_rew > 0, 0)).rolling(period).mean()
        loss_rew = (-delta_rew.where(delta_rew < 0, 0)).rolling(period).mean()
        rs_rew = gain_rew / (loss_rew + 1e-8)
        features_rew[f'rsi_{period}'] = 100 - (100 / (1 + rs_rew))
        
        # Momentum
        features_rew[f'momentum_{period}'] = rewards_series_rew / rewards_series_rew.shift(period) - 1
        
        # Tendance (régression linéaire)
        def calc_trend_rew(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            try:
                slope = np.polyfit(x, series, 1)[0]
                return slope
            except:
                return 0
        
        features_rew[f'trend_{period}'] = rewards_series_rew.rolling(period).apply(calc_trend_rew, raw=True)
    
    # VOLATILITÉ AVANCÉE
    # Volatilité réalisée (Garman-Klass style avec high/low simulés)
    for window in [5, 10, 20]:
        # Simulate intraday volatility from daily data
        high_proxy_rew = rewards_series_rew.rolling(window).max()
        low_proxy_rew = rewards_series_rew.rolling(window).min()
        features_rew[f'realized_vol_{window}'] = np.log(high_proxy_rew / low_proxy_rew)
        
        # Z-score (position dans la distribution)
        mean_roll_rew = rewards_series_rew.rolling(window).mean()
        std_roll_rew = rewards_series_rew.rolling(window).std()
        features_rew[f'zscore_{window}'] = (rewards_series_rew - mean_roll_rew) / (std_roll_rew + 1e-8)
    
    # RATIOS ET INTERACTIONS
    # Ratios avec moyennes mobiles (vérification que les colonnes existent)
    for short, long in [(3, 7), (5, 14), (7, 21), (14, 30)]:
        if f'reward_ma_{short}' in features_rew.columns and f'reward_ma_{long}' in features_rew.columns:
            features_rew[f'ma_ratio_{short}_{long}'] = features_rew[f'reward_ma_{short}'] / (features_rew[f'reward_ma_{long}'] + 1e-8)
        if f'reward_std_{short}' in features_rew.columns and f'reward_std_{long}' in features_rew.columns:
            features_rew[f'vol_ratio_{short}_{long}'] = features_rew[f'reward_std_{short}'] / (features_rew[f'reward_std_{long}'] + 1e-8)
    
    # Position relative dans les distributions récentes
    for window in [10, 20, 30]:
        features_rew[f'percentile_rank_{window}'] = rewards_series_rew.rolling(window).rank(pct=True)
        
        # Distance aux extrêmes
        rolling_min_rew = rewards_series_rew.rolling(window).min()
        rolling_max_rew = rewards_series_rew.rolling(window).max()
        features_rew[f'dist_to_min_{window}'] = (rewards_series_rew - rolling_min_rew) / (rolling_max_rew - rolling_min_rew + 1e-8)
        features_rew[f'dist_to_max_{window}'] = (rolling_max_rew - rewards_series_rew) / (rolling_max_rew - rolling_min_rew + 1e-8)
    
    # PATTERNS AVANCÉS
    # Détection de breakouts (vérification que les colonnes existent)
    for window in [5, 10, 20]:
        if f'reward_ma_{window}' in features_rew.columns and f'reward_std_{window}' in features_rew.columns:
            upper_band_rew = features_rew[f'reward_ma_{window}'] + 2 * features_rew[f'reward_std_{window}']
            lower_band_rew = features_rew[f'reward_ma_{window}'] - 2 * features_rew[f'reward_std_{window}']
            
            features_rew[f'above_upper_band_{window}'] = (rewards_series_rew > upper_band_rew).astype(int)
            features_rew[f'below_lower_band_{window}'] = (rewards_series_rew < lower_band_rew).astype(int)
            features_rew[f'band_position_{window}'] = (rewards_series_rew - lower_band_rew) / (upper_band_rew - lower_band_rew + 1e-8)
    
    # Streaks (séries consécutives) - correction pour gérer les NaN
    diff_series_rew = rewards_series_rew.diff().fillna(0)
    diff_sign_rew = np.sign(diff_series_rew)
    
    # Calcul des streaks positifs
    positive_mask_rew = (diff_sign_rew == 1).astype(int)
    positive_groups_rew = (diff_sign_rew != 1).cumsum()
    features_rew['positive_streak'] = positive_mask_rew.groupby(positive_groups_rew).cumsum()
    
    # Calcul des streaks négatifs
    negative_mask_rew = (diff_sign_rew == -1).astype(int)
    negative_groups_rew = (diff_sign_rew != -1).cumsum()
    features_rew['negative_streak'] = negative_mask_rew.groupby(negative_groups_rew).cumsum()
    
    # Volatility clustering avec gestion des NaN
    returns_rew = rewards_series_rew.pct_change().fillna(0)
    squared_returns_rew = (returns_rew ** 2)
    for window in [5, 10, 20]:
        features_rew[f'vol_cluster_{window}'] = squared_returns_rew.rolling(window, min_periods=1).mean()
    
    # Nettoyage final des NaN et valeurs infinies
    features_rew = features_rew.replace([np.inf, -np.inf], 0)
    features_rew = features_rew.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features_rew

def align_market_data_rew(market_data_rew, rewards_dates_rew):
    """
    Aligne les données de marché avec features plus agressives
    """
    market_aligned_rew = market_data_rew.reindex(rewards_dates_rew, method='ffill')
    features_rew = pd.DataFrame(index=rewards_dates_rew)
    
    market_vars_rew = ['SOL_close', 'ETH_close', 'BTC_close']
    
    for var in market_vars_rew:
        if var in market_aligned_rew.columns:
            series_rew = market_aligned_rew[var]
            
            # Returns de base
            features_rew[var] = series_rew
            
            # Volatilité et momentum plus agressifs
            for window in [2, 3, 5, 7, 14]:
                features_rew[f'{var}_ma{window}'] = series_rew.rolling(window).mean()
                features_rew[f'{var}_vol{window}'] = series_rew.rolling(window).std()
                features_rew[f'{var}_momentum{window}'] = series_rew / series_rew.shift(window) - 1
                
                # Z-score pour market data
                mean_roll_rew = series_rew.rolling(window).mean()
                std_roll_rew = series_rew.rolling(window).std()
                features_rew[f'{var}_zscore{window}'] = (series_rew - mean_roll_rew) / (std_roll_rew + 1e-8)
            
            # Lags plus nombreux
            for lag in [1, 2, 3, 5, 7]:
                features_rew[f'{var}_lag{lag}'] = series_rew.shift(lag)
                features_rew[f'{var}_diff{lag}'] = series_rew - series_rew.shift(lag)
    
    # Corrélations plus sophistiquées
    if 'SOL_close' in market_aligned_rew.columns and 'BTC_close' in market_aligned_rew.columns:
        for window in [5, 10, 20]:
            corr_series_rew = market_aligned_rew['SOL_close'].rolling(window).corr(market_aligned_rew['BTC_close'])
            features_rew[f'sol_btc_corr{window}'] = corr_series_rew.fillna(0)
    
    if 'SOL_close' in market_aligned_rew.columns and 'ETH_close' in market_aligned_rew.columns:
        for window in [5, 10, 20]:
            corr_series_rew = market_aligned_rew['SOL_close'].rolling(window).corr(market_aligned_rew['ETH_close'])
            features_rew[f'sol_eth_corr{window}'] = corr_series_rew.fillna(0)
    
    return features_rew.fillna(method='ffill').fillna(0)

def prepare_data_for_training_rew(rewards_data_rew, market_data_rew=None):
    """
    Prépare les données avec features avancées
    """
    # Features avancées des rewards
    rewards_features_rew = create_advanced_features_rew(rewards_data_rew)
    
    if market_data_rew is not None:
        # Features de marché
        market_features_rew = align_market_data_rew(market_data_rew, rewards_data_rew.index)
        X_rew = pd.concat([rewards_features_rew.drop('sol_rewards', axis=1), market_features_rew], axis=1)
    else:
        X_rew = rewards_features_rew.drop('sol_rewards', axis=1)
    
    y_rew = rewards_features_rew['sol_rewards']
    
    # Nettoyage plus agressif
    X_rew = X_rew.replace([np.inf, -np.inf], 0)
    X_rew = X_rew.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return X_rew, y_rew

def train_advanced_model_rew(X_train_rew, y_train_rew):
    """
    Entraîne un ensemble de modèles plus agressifs
    """
    print("Entraînement des modèles avancés...")
    
    # Utilisation de RobustScaler pour mieux gérer les outliers
    scaler_rew = RobustScaler()
    X_train_scaled_rew = scaler_rew.fit_transform(X_train_rew)
    
    # 1. Gradient Boosting - Plus agressif pour capturer la volatilité
    gb_model_rew = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=3,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # 2. Random Forest optimisé pour volatilité
    rf_model_rew = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,  # Plus profond
        min_samples_split=3,  # Plus agressif
        min_samples_leaf=2,   # Plus agressif
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Ridge avec moins de régularisation
    ridge_model_rew = Ridge(alpha=0.1)
    
    # Entraînement
    gb_model_rew.fit(X_train_scaled_rew, y_train_rew)
    rf_model_rew.fit(X_train_scaled_rew, y_train_rew)
    ridge_model_rew.fit(X_train_scaled_rew, y_train_rew)
    
    return {
        'gb': gb_model_rew,
        'rf': rf_model_rew,
        'ridge': ridge_model_rew,
        'scaler': scaler_rew
    }

def predict_ensemble_rew(models_rew, X_rew, volatility_boost=1.2):
    """
    Prédiction d'ensemble avec boost de volatilité
    """
    X_scaled_rew = models_rew['scaler'].transform(X_rew)
    
    # Prédictions individuelles
    pred_gb_rew = models_rew['gb'].predict(X_scaled_rew)
    pred_rf_rew = models_rew['rf'].predict(X_scaled_rew)
    pred_ridge_rew = models_rew['ridge'].predict(X_scaled_rew)
    
    # Ensemble pondéré (GB plus de poids car meilleur pour volatilité)
    ensemble_pred_rew = 0.5 * pred_gb_rew + 0.3 * pred_rf_rew + 0.2 * pred_ridge_rew
    
    # BOOST DE VOLATILITÉ
    # Calcul de la volatilité locale
    if len(ensemble_pred_rew) > 5:
        local_std_rew = np.std(ensemble_pred_rew[-5:])  # Volatilité des 5 dernières prédictions
        mean_pred_rew = np.mean(ensemble_pred_rew)
        
        # Ajout de bruit corrélé à la volatilité
        noise_rew = np.random.normal(0, local_std_rew * 0.3, len(ensemble_pred_rew))
        ensemble_pred_rew = ensemble_pred_rew + noise_rew
    
    # Amplification de la volatilité
    pred_mean_rew = np.mean(ensemble_pred_rew)
    deviations_rew = ensemble_pred_rew - pred_mean_rew
    ensemble_pred_rew = pred_mean_rew + deviations_rew * volatility_boost
    
    # Les rewards ne peuvent pas être négatifs
    ensemble_pred_rew = np.maximum(ensemble_pred_rew, 0.001)
    
    return ensemble_pred_rew

def predict_future_rewards_advanced_rew(models_rew, X_recent_rew, historical_rewards_rew, n_days=30):
    """
    Prédictions futures avec simulation de volatilité réaliste, rewards générés tous les 2 jours
    """
    predictions_rew = []
    dates_rew = []
    
    last_date_rew = X_recent_rew.index[-1]
    
    # Calcul de stats historiques pour calibration
    hist_mean_rew = historical_rewards_rew.mean()
    hist_std_rew = historical_rewards_rew.std()
    hist_cv_rew = hist_std_rew / hist_mean_rew  # Coefficient de variation historique
    
    # Utilise une fenêtre glissante des features récentes
    recent_window_rew = min(30, len(X_recent_rew))
    recent_features_rew = X_recent_rew.tail(recent_window_rew).copy()
    
    for i in range(n_days):
        future_date_rew = last_date_rew + pd.Timedelta(days=2 * (i + 1))  # <-- tous les deux jours
        dates_rew.append(future_date_rew)
        
        # Sélection des features (rotation avec variation)
        if i < len(recent_features_rew):
            base_features_rew = recent_features_rew.iloc[i].values
        else:
            idx1_rew = i % len(recent_features_rew)
            idx2_rew = (i + 1) % len(recent_features_rew)
            base_features_rew = 0.7 * recent_features_rew.iloc[idx1_rew].values + 0.3 * recent_features_rew.iloc[idx2_rew].values
        
        # Ajout de bruit réaliste aux features
        noise_level_rew = 0.05  # 5% de bruit
        noisy_features_rew = base_features_rew + np.random.normal(0, noise_level_rew * np.abs(base_features_rew))
        
        # Prédiction
        pred_rew = predict_ensemble_rew(models_rew, noisy_features_rew.reshape(1, -1), volatility_boost=1.3)[0]
        
        # Calibration avec l'historique
        if i > 5:
            recent_preds_rew = predictions_rew[-5:]
            pred_std_rew = np.std(recent_preds_rew)
            target_std_rew = hist_std_rew * 0.8
            
            if pred_std_rew < target_std_rew:
                vol_adjustment_rew = np.random.normal(0, target_std_rew * 0.3)
                pred_rew += vol_adjustment_rew
        
        # Application d'une variabilité minimale
        if i > 0:
            last_pred_rew = predictions_rew[-1]
            min_change_rew = last_pred_rew * 0.02
            max_change_rew = last_pred_rew * 0.15
            
            change_rew = pred_rew - last_pred_rew
            if abs(change_rew) < min_change_rew:
                change_rew = min_change_rew * np.sign(np.random.randn())
                pred_rew = last_pred_rew + change_rew
            elif abs(change_rew) > max_change_rew:
                pred_rew = last_pred_rew + max_change_rew * np.sign(change_rew)
        
        # Contraintes finales
        pred_rew = max(pred_rew, hist_mean_rew * 0.1)
        pred_rew = min(pred_rew, hist_mean_rew * 3.0)
        
        predictions_rew.append(pred_rew)
    
    return pd.Series(predictions_rew, index=pd.to_datetime(dates_rew))



def main():
    ####################### Importing data and Stationarity test ##############################################
    data = pd.read_pickle("data_v1.pickle").dropna()
    data_shifted = pd.read_pickle("data_v1_offset.pickle").dropna()


    ####################### Data Split ###############################################################################################################################

    data_train = data[data.index < datetime(2025,1,1)]
    data_test = data[data.index >= datetime(2025,1,1)]

    data_shifted_train = data_shifted[data_shifted.index < datetime(2025,1,1)]
    data_shifted_test = data_shifted[data_shifted.index >= datetime(2025,1,1)]

    ####################### Variables Selection ######################################################################################################################


    kept_exo_variables = ["SOL_volume","SOL_fr","ETH_close","BTC_close"]




    # ==================== EXECUTION PRINCIPALE ====================

    print("=== CREATION DES VARIABLES ENRICHIES ===")

    print("Création des features de volatilité et techniques...")
    X_shifted_train_enhanced, y_shifted_train_enhanced = prepare_enhanced_features(
        data_shifted_train, kept_exo_variables, "SOL_close"
    )

    X_shifted_test_enhanced, y_shifted_test_enhanced = prepare_enhanced_features(
        data_shifted_test, kept_exo_variables, "SOL_close"
    )

    X_test_enhanced, y_test_enhanced = prepare_enhanced_features(
        data_test, kept_exo_variables, "SOL_close"
    )

    print(f"Features enrichies - Train: {X_shifted_train_enhanced.shape}")
    print(f"Features enrichies - Test: {X_shifted_test_enhanced.shape}")
    print(f"Nouvelles features créées: {X_shifted_train_enhanced.shape[1] - len(kept_exo_variables)}")

    print("\n=== ENTRAINEMENT DU MODELE ===")

    models = train_random_forest_with_uncertainty(
        X_shifted_train_enhanced, y_shifted_train_enhanced
    )

    print("Modèle Random Forest entraîné avec estimation d'incertitude")

    print("\n=== PREDICTIONS SUR DONNEES TEST ===")

    y_pred, y_pred_upper, y_pred_lower = predict_with_confidence(
        models, X_shifted_test_enhanced
    )

    mse = mean_squared_error(y_shifted_test_enhanced, y_pred)
    mae = mean_absolute_error(y_shifted_test_enhanced, y_pred)
    r2 = r2_score(y_shifted_test_enhanced, y_pred)

    print(f"Métriques sur données test:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.4f}")

    initial_price = 100


    true_prices = initial_price * np.exp(pd.Series(y_shifted_test_enhanced).cumsum())
    pred_prices = initial_price * np.exp(pd.Series(y_pred, index=y_shifted_test_enhanced.index).cumsum())

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(true_prices.index, true_prices, label='Prix Réels (à partir des log-returns)', linewidth=2)
    plt.plot(pred_prices.index, pred_prices, label='Prix Prédits (à partir des log-returns)', linestyle='--', linewidth=2)

    plt.title('Évolution des Prix reconstruits depuis les Log-Returns (Base 100)')
    plt.xlabel('Date')
    plt.ylabel('Prix (Base 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    print("\n=== PREDICTIONS FUTURES ===")

    # Prédictions futures (y_future_pred)
    y_future_pred, y_future_upper, y_future_lower = create_future_predictions(
        models, X_test_enhanced, data_test, n_future_days=30
    )

    print(f"Prédictions futures créées pour {len(y_future_pred)} jours")
    print(f"Période: {y_future_pred.index[0]} à {y_future_pred.index[-1]}")

    print("\n=== CONVERSION EN PRIX ===")

    # Pour convertir en prix, nous avons besoin du dernier prix observé
    # Assumons que nous avons une série de prix SOL (à adapter selon vos données)
    try:
        # Essayer de récupérer le dernier prix connu
        # Si vous avez les prix dans vos data, utilisez la dernière valeur
        # Sinon, utilisez une valeur de référence (exemple: 100)
        sol_prices = pd.read_csv("sol-usd-max.csv",index_col=0,parse_dates=True)
        sol_prices.rename(columns = {'price':"SOL_price","market_cap":"SOL_mc","total_volume":"SOL_volume"},inplace=True)
        sol_prices.index = sol_prices.index.tz_localize(None)


        last_price = sol_prices["SOL_price"][-1]
        print("-"*50)
        print(last_price)
        
        # Conversion des prédictions futures en prix
        future_prices = log_returns_to_prices(y_future_pred, last_price)
        future_prices_upper = log_returns_to_prices(y_future_upper, last_price)
        future_prices_lower = log_returns_to_prices(y_future_lower, last_price)
        
        print(f"Dernier prix observé: ${last_price:.2f}")
        print(f"Prix prédit dans 30 jours: ${future_prices.iloc[-1]:.2f}")
        print(f"Intervalle de confiance: [${future_prices_lower.iloc[-1]:.2f}, ${future_prices_upper.iloc[-1]:.2f}]")
        
    except Exception as e:
        print(f"Erreur dans la conversion en prix: {e}")
        future_prices = y_future_pred
        future_prices_upper = y_future_upper
        future_prices_lower = y_future_lower

    print("\n=== VISUALISATION ===")


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0,0].scatter(y_shifted_test_enhanced, y_pred, alpha=0.6)
    axes[0,0].plot([y_shifted_test_enhanced.min(), y_shifted_test_enhanced.max()], 
                [y_shifted_test_enhanced.min(), y_shifted_test_enhanced.max()], 'r--')
    axes[0,0].set_xlabel('Valeurs Réelles')
    axes[0,0].set_ylabel('Prédictions')
    axes[0,0].set_title(f'Prédictions vs Réalité (R² = {r2:.3f})')

    residuals = y_shifted_test_enhanced - y_pred
    axes[0,1].scatter(y_pred, residuals, alpha=0.6)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Prédictions')
    axes[0,1].set_ylabel('Résidus')
    axes[0,1].set_title('Analyse des Résidus')


    axes[1,0].plot(y_future_pred.index, y_future_pred, 'b-', label='Prédiction')
    axes[1,0].fill_between(y_future_pred.index, y_future_lower, y_future_upper, 
                        alpha=0.3, label='IC 90%')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Log Returns')
    axes[1,0].set_title('Prédictions Futures - Log Returns')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)


    if 'future_prices' in locals():
        axes[1,1].plot(future_prices.index, future_prices, 'g-', linewidth=2, label='Prix Prédit')
        axes[1,1].fill_between(future_prices.index, future_prices_lower, future_prices_upper, 
                            alpha=0.3, color='green', label='IC 95%')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Prix SOL ($)')
        axes[1,1].set_title('Path de Prix Futur avec Intervalle de Confiance')
        axes[1,1].legend()
        axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    print("\n=== FEATURE IMPORTANCE ===")


    feature_importance = pd.DataFrame({
        'feature': X_shifted_train_enhanced.columns,
        'importance': models[0].feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 15 variables les plus importantes:")
    print(feature_importance.head(15))

    print("\n=== RESULTATS FINAUX ===")
    print(f"Modèle entraîné sur {len(X_shifted_train_enhanced)} observations")
    print(f"Nombre total de features: {X_shifted_train_enhanced.shape[1]}")
    print(f"Performance R²: {r2:.4f}")
    print(f"Prédictions futures générées: {len(y_future_pred)} jours")


    results = {
        'y_pred': pd.Series(y_pred, index=X_shifted_test_enhanced.index),
        'y_pred_upper': pd.Series(y_pred_upper, index=X_shifted_test_enhanced.index),
        'y_pred_lower': pd.Series(y_pred_lower, index=X_shifted_test_enhanced.index),
        'y_future_pred': y_future_pred,
        'y_future_upper': y_future_upper,
        'y_future_lower': y_future_lower,
        'models': models,
        'feature_importance': feature_importance
    }


    if 'future_prices' in locals():
        results['future_prices'] = future_prices
        results['future_prices_upper'] = future_prices_upper
        results['future_prices_lower'] = future_prices_lower

    print("Résultats stockés dans la variable 'results'")
    print("Variables disponibles: y_pred, y_future_pred, future_prices, models, etc.")


    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################




    # ==================== EXECUTION AVANCÉE ====================
    print("=== CHARGEMENT DES DONNEES ===")
    rewards_active_rew = load_rewards_data_rew("rewards data.csv")

    print("\n=== PREPARATION DES DONNEES AVANCÉES ===")
    try:
        X_full_rew, y_full_rew = prepare_data_for_training_rew(rewards_active_rew, data_test_rew)
        print(f"Données préparées avec market data: {X_full_rew.shape}")
    except NameError:
        print("Utilisation des features rewards uniquement (version avancée)")
        X_full_rew, y_full_rew = prepare_data_for_training_rew(rewards_active_rew)

    print(f"Features: {X_full_rew.shape[1]}")
    print(f"Observations: {len(X_full_rew)}")

    print("\n=== DIVISION TRAIN/TEST ===")
    split_idx_rew = int(len(X_full_rew) * 0.8)
    X_train_rew = X_full_rew.iloc[:split_idx_rew]
    X_test_rew = X_full_rew.iloc[split_idx_rew:]
    y_train_rew = y_full_rew.iloc[:split_idx_rew]
    y_test_rew = y_full_rew.iloc[split_idx_rew:]

    print(f"Train: {len(X_train_rew)} observations")
    print(f"Test: {len(X_test_rew)} observations")

    print("\n=== ENTRAINEMENT AVANCÉ ===")
    models_rew = train_advanced_model_rew(X_train_rew, y_train_rew)

    print("\n=== EVALUATION ===")
    y_pred_rew = predict_ensemble_rew(models_rew, X_test_rew)

    # Métriques
    mse_rew = mean_squared_error(y_test_rew, y_pred_rew)
    mae_rew = mean_absolute_error(y_test_rew, y_pred_rew)
    r2_rew = r2_score(y_test_rew, y_pred_rew)

    # Métriques de volatilité
    actual_std_rew = y_test_rew.std()
    pred_std_rew = np.std(y_pred_rew)
    vol_ratio_rew = pred_std_rew / actual_std_rew

    print(f"MSE: {mse_rew:.4f}")
    print(f"MAE: {mae_rew:.4f}")
    print(f"R²: {r2_rew:.4f}")
    print(f"Volatilité réelle: {actual_std_rew:.4f}")
    print(f"Volatilité prédite: {pred_std_rew:.4f}")
    print(f"Ratio volatilité: {vol_ratio_rew:.2f} (target > 0.8)")

    print("\n=== PREDICTIONS FUTURES AVANCÉES ===")
    future_rewards_rew = predict_future_rewards_advanced_rew(models_rew, X_full_rew, rewards_active_rew['SOL'], n_days=16)

    print(f"Période prédite: {future_rewards_rew.index[0]} à {future_rewards_rew.index[-1]}")
    print(f"Reward total prédit (30j): {future_rewards_rew.sum():.4f} SOL")
    print(f"Reward moyen prédit: {future_rewards_rew.mean():.4f} SOL")
    print(f"Reward std prédit: {future_rewards_rew.std():.4f} SOL")
    print(f"CV prédit: {future_rewards_rew.std() / future_rewards_rew.mean():.2f}")

    print(f"\nComparaison historique:")
    print(f"Reward moyen historique: {rewards_active_rew['SOL'].mean():.4f} SOL")
    print(f"Reward std historique: {rewards_active_rew['SOL'].std():.4f} SOL")
    print(f"CV historique: {rewards_active_rew['SOL'].std() / rewards_active_rew['SOL'].mean():.2f}")

    print("\n=== VISUALISATION AVANCÉE ===")
    fig_rew, axes_rew = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Scatter plot amélioré
    axes_rew[0,0].scatter(y_test_rew, y_pred_rew, alpha=0.6, s=50)
    axes_rew[0,0].plot([y_test_rew.min(), y_test_rew.max()], [y_test_rew.min(), y_test_rew.max()], 'r--', linewidth=2)
    axes_rew[0,0].set_xlabel('Rewards Réels')
    axes_rew[0,0].set_ylabel('Rewards Prédits')
    axes_rew[0,0].set_title(f'Prédictions vs Réalité\nR² = {r2_rew:.3f}, Vol Ratio = {vol_ratio_rew:.2f}')
    axes_rew[0,0].grid(True, alpha=0.3)

    # 2. Série temporelle test avec bandes de confiance
    axes_rew[0,1].plot(y_test_rew.index, y_test_rew, 'o-', label='Réel', markersize=4, linewidth=2)
    axes_rew[0,1].plot(y_test_rew.index, y_pred_rew, 's-', label='Prédit', markersize=4, linewidth=2)
    # Bandes de confiance basées sur l'erreur
    error_band_rew = np.std(y_test_rew - y_pred_rew)
    axes_rew[0,1].fill_between(y_test_rew.index, y_pred_rew - error_band_rew, y_pred_rew + error_band_rew, alpha=0.2, color='red')
    axes_rew[0,1].set_xlabel('Date')
    axes_rew[0,1].set_ylabel('Rewards SOL')
    axes_rew[0,1].set_title('Prédictions sur Période Test')
    axes_rew[0,1].legend()
    axes_rew[0,1].tick_params(axis='x', rotation=45)
    axes_rew[0,1].grid(True, alpha=0.3)

    # 3. Prédictions futures avec volatilité
    axes_rew[0,2].plot(future_rewards_rew.index, future_rewards_rew, 'g-', linewidth=2, marker='o', markersize=4)
    # Moyenne mobile pour tendance
    ma_7_rew = future_rewards_rew.rolling(7, min_periods=1).mean()
    axes_rew[0,2].plot(future_rewards_rew.index, ma_7_rew, 'r--', linewidth=2, alpha=0.7, label='MA 7j')
    axes_rew[0,2].set_xlabel('Date')
    axes_rew[0,2].set_ylabel('Rewards SOL Prédits')
    axes_rew[0,2].set_title('Prédictions Futures (30 jours)')
    axes_rew[0,2].legend()
    axes_rew[0,2].tick_params(axis='x', rotation=45)
    axes_rew[0,2].grid(True, alpha=0.3)

    # 4. Distribution comparative
    axes_rew[1,0].hist(rewards_active_rew['SOL'], bins=30, alpha=0.7, label='Historique', color='blue', density=True)
    axes_rew[1,0].hist(future_rewards_rew, bins=20, alpha=0.7, label='Prédictions', color='green', density=True)
    axes_rew[1,0].axvline(rewards_active_rew['SOL'].mean(), color='blue', linestyle='--', label='Moy. Hist.')
    axes_rew[1,0].axvline(future_rewards_rew.mean(), color='green', linestyle='--', label='Moy. Pred.')
    axes_rew[1,0].set_xlabel('Rewards SOL')
    axes_rew[1,0].set_ylabel('Densité')
    axes_rew[1,0].set_title('Distribution des Rewards')
    axes_rew[1,0].legend()
    axes_rew[1,0].grid(True, alpha=0.3)

    # 5. Analyse des résidus
    residuals_rew = y_test_rew - y_pred_rew
    axes_rew[1,1].scatter(y_pred_rew, residuals_rew, alpha=0.6)
    axes_rew[1,1].axhline(y=0, color='r', linestyle='--')
    axes_rew[1,1].set_xlabel('Prédictions')
    axes_rew[1,1].set_ylabel('Résidus')
    axes_rew[1,1].set_title('Analyse des Résidus')
    axes_rew[1,1].grid(True, alpha=0.3)

    # 6. Volatilité mobile
    vol_window_rew = 7
    hist_vol_rew = rewards_active_rew['SOL'].rolling(vol_window_rew).std()
    pred_vol_rew = future_rewards_rew.rolling(vol_window_rew, min_periods=1).std()

    axes_rew[1,2].plot(hist_vol_rew.index[-30:], hist_vol_rew.tail(30), 'b-', label=f'Vol. Hist. {vol_window_rew}j', linewidth=2)
    axes_rew[1,2].plot(pred_vol_rew.index, pred_vol_rew, 'g-', label=f'Vol. Pred. {vol_window_rew}j', linewidth=2)
    axes_rew[1,2].set_xlabel('Date')
    axes_rew[1,2].set_ylabel('Volatilité (Std Dev)')
    axes_rew[1,2].set_title('Évolution de la Volatilité')
    axes_rew[1,2].legend()
    axes_rew[1,2].tick_params(axis='x', rotation=45)
    axes_rew[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Feature importance pour le modèle principal (GB)
    print("\n=== FEATURE IMPORTANCE (Gradient Boosting) ===")
    if hasattr(models_rew['gb'], 'feature_importances_'):
        importance_df_rew = pd.DataFrame({
            'feature': X_full_rew.columns,
            'importance': models_rew['gb'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 20 features les plus importantes:")
        print(importance_df_rew.head(20))

    print(f"\nSTATISTIQUES FINALES:")
    print(f"Amélioration volatilité: {vol_ratio_rew:.1%} de la volatilité réelle capturée")
    print(f"Coefficient de variation prédit: {future_rewards_rew.std() / future_rewards_rew.mean():.2f}")
    print(f"Range prédit: {future_rewards_rew.min():.4f} - {future_rewards_rew.max():.4f} SOL")
    print(f"Écart-type prédictions: {future_rewards_rew.std():.4f} SOL")

    # Sauvegarde
    advanced_results_rew = {
        'models': models_rew,
        'future_predictions': future_rewards_rew,
        'test_predictions': pd.Series(y_pred_rew, index=y_test_rew.index),
        'test_actual': y_test_rew,
        'metrics': {'mse': mse_rew, 'mae': mae_rew, 'r2': r2_rew, 'vol_ratio': vol_ratio_rew},
        'feature_importance': importance_df_rew if 'importance_df_rew' in locals() else None
    }

    print("\nRésultats avancés sauvegardés dans 'advanced_results_rew'")

    return future_prices,future_prices_lower,future_prices_upper,advanced_results_rew,true_prices,pred_prices,models,X_test_enhanced,X_test_enhanced, data_test,y_shifted_test_enhanced,y_pred,X_shifted_train_enhanced,r2,mse_rew, mae_rew, r2_rew,actual_std_rew, pred_std_rew, vol_ratio_rew,models_rew, X_full_rew,y_test_rew, y_pred_rew,rewards_active_rew,future_rewards_rew,importance_df_rew





@st.cache_data
def run_model():
    future_prices,future_prices_lower,future_prices_upper,advanced_results_rew,true_prices,pred_prices,models,X_test_enhanced,X_test_enhanced, data_test,y_shifted_test_enhanced,y_pred,X_shifted_train_enhanced,r2,mse_rew, mae_rew, r2_rew,actual_std_rew, pred_std_rew, vol_ratio_rew,models_rew, X_full_rew,y_test_rew, y_pred_rew,rewards_active_rew,future_rewards_rew,importance_df_rew = main()
    return future_prices,future_prices_lower,future_prices_upper,advanced_results_rew,true_prices,pred_prices,models,X_test_enhanced,X_test_enhanced, data_test,y_shifted_test_enhanced,y_pred,X_shifted_train_enhanced,r2,mse_rew, mae_rew, r2_rew,actual_std_rew, pred_std_rew, vol_ratio_rew,models_rew, X_full_rew,y_test_rew, y_pred_rew,rewards_active_rew,future_rewards_rew,importance_df_rew


st.title("Predictions Model")


# Hedging Parameters Section
st.subheader("Hedging Strategy Configuration")

col1, col2 = st.columns(2)
with col1:
    hedge_coverage = st.selectbox("Hedge Coverage", ["Conservative (50%)", "Aggressive (80%)"])
with col2:
    put_maturity = st.selectbox("Put Option Maturity", [30], index=0)

sol_holdings = 0.0

# Strike selection
strike_selection = st.selectbox("Strike Selection", 
    ["10% OTM (90% of current price)", "5% OTM (95% of current price)", "ATM (100% of current price)"], 
    index=0)

if st.button("Run Model"):
    # Run the main model
    future_prices, future_prices_lower, future_prices_upper, advanced_results_rew, true_prices, pred_prices, models, X_test_enhanced, X_test_enhanced, data_test, y_shifted_test_enhanced, y_pred, X_shifted_train_enhanced, r2, mse_rew, mae_rew, r2_rew, actual_std_rew, pred_std_rew, vol_ratio_rew, models_rew, X_full_rew, y_test_rew, y_pred_rew, rewards_active_rew, future_rewards_rew, importance_df_rew = run_model()
    
    st.success("Model run complete")
    sol_prices = pd.read_csv("sol-usd-max.csv", index_col=0, parse_dates=True)
    sol_prices.rename(columns={'price': "SOL_price"}, inplace=True)
    sol_prices.index = sol_prices.index.tz_localize(None)
    
    current_sol_price = float(sol_prices["SOL_price"].iloc[-1])
    
    # Display key results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actual SOL Price", f"${current_sol_price:.2f}")
    with col2:
        st.metric("Predicted SOL Price (30 days)", f"${future_prices[-1]:.2f}")
    with col3:
        st.metric("Predicted Revenue (30 days)", f"{future_rewards_rew.sum():.2f} SOL")

    # Main prediction charts
    st.subheader("Price & Revenue Predictions")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Price prediction chart
    ax1.plot(future_prices.index, future_prices, label="Predicted Price", color='green', linewidth=2)
    ax1.fill_between(future_prices.index, future_prices_lower, future_prices_upper, 
                     alpha=0.3, color='green', label='95% Confidence Interval')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price SOL ($)')
    ax1.set_title('Future Price Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Revenue prediction chart
    ax2.bar(future_rewards_rew.index, future_rewards_rew, color='skyblue', alpha=0.7, label='Predicted Revenues')
    ax2.plot(future_rewards_rew.index, future_rewards_rew.rolling(7, min_periods=1).mean(), 
             'r--', linewidth=2, alpha=0.8, label='7-day Moving Average')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Revenues (SOL)')
    ax2.set_title('Future Revenue Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

    # === HEDGING ANALYSIS SECTION ===
    st.subheader("SOL Hedging Analysis with GARCH Implied Volatility")
    
    def estimate_garch_volatility(price_series, forecast_horizon=30):
        """Estimate GARCH volatility with proper scaling"""
        try:
            from arch import arch_model
            import warnings
            warnings.filterwarnings('ignore')
            
            # Calculate daily returns (not scaled by 100)
            returns = np.log(price_series / price_series.shift(1)).dropna()
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
            garch_fitted = garch_model.fit(disp='off')
            
            # Forecast volatility
            forecast = garch_fitted.forecast(horizon=forecast_horizon)
            forecasted_variance = forecast.variance.iloc[-1].mean()
            
            # Convert to annualized volatility (already in decimal form)
            annualized_vol = np.sqrt(forecasted_variance * 252)
            
            # Historical volatility for comparison
            historical_vol = returns.std() * np.sqrt(252)
            
            return {
                'garch_vol': annualized_vol,
                'historical_vol': historical_vol,
                'garch_model': garch_fitted,
                'returns': returns
            }
            
        except ImportError:
            st.warning("ARCH package not available. Using historical volatility.")
            returns = np.log(price_series / price_series.shift(1)).dropna()
            historical_vol = returns.std() * np.sqrt(252)
            return {
                'garch_vol': historical_vol,
                'historical_vol': historical_vol,
                'garch_model': None,
                'returns': returns
            }
        except Exception as e:
            st.warning(f"GARCH estimation failed: {e}. Using historical volatility.")
            returns = np.log(price_series / price_series.shift(1)).dropna()
            historical_vol = returns.std() * np.sqrt(252)
            return {
                'garch_vol': historical_vol,
                'historical_vol': historical_vol,
                'garch_model': None,
                'returns': returns
            }

    def black_scholes_put(S, K, T, r, sigma):
        """Calculate Black-Scholes put option price"""
        from scipy.stats import norm
        
        if sigma <= 0 or T <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    def calculate_greeks(S, K, T, r, sigma):
        """Calculate option Greeks"""
        from scipy.stats import norm
        
        if sigma <= 0 or T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Put option Greeks
        delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

    # Load SOL price data for volatility analysis
    try:
        
        # Estimate volatility using recent price history
        recent_prices = sol_prices["SOL_price"].tail(365)  # Last year of data
        vol_estimates = estimate_garch_volatility(recent_prices)
        
        # Display volatility analysis
        st.subheader("Volatility Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current SOL Price", f"${current_sol_price:.2f}")
        with col2:
            st.metric("GARCH Volatility", f"{vol_estimates['garch_vol']:.1%}")
        with col3:
            st.metric("Historical Volatility", f"{vol_estimates['historical_vol']:.1%}")
        
        # Volatility visualization
        with st.expander("GARCH Model Volatility", expanded=False):
            if vol_estimates['garch_model'] is not None:
                fig_vol, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Returns plot
                vol_estimates['returns'].plot(ax=ax1, title='SOL Daily Returns', color='blue', alpha=0.7)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylabel('Returns')
                
                # Conditional volatility
                conditional_vol = vol_estimates['garch_model'].conditional_volatility
                conditional_vol.plot(ax=ax2, title='GARCH Conditional Volatility', color='red', linewidth=2)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylabel('Volatility')
                ax2.set_xlabel('Date')
                
                plt.tight_layout()
                st.pyplot(fig_vol)
        
        # Parse user inputs for hedging strategy
        coverage_pct = 0.5 if "Conservative" in hedge_coverage else 0.8
        
        strike_multiplier = {
            "10% OTM": 0.90,
            "5% OTM": 0.95,
            "ATM": 1.00
        }
        selected_multiplier = next(v for k, v in strike_multiplier.items() if k in strike_selection)
        
        # Calculate hedge parameters
        total_future_revenue = float(future_rewards_rew.sum())
        total_exposure = sol_holdings + total_future_revenue
        hedge_amount = total_exposure * coverage_pct
        sol_holdings = total_exposure
        strike_price = current_sol_price * selected_multiplier
        T = put_maturity / 365  # Convert days to years
        risk_free_rate = 0.05
        
        # Use GARCH volatility with slight adjustment for moneyness
        base_vol = vol_estimates['garch_vol']
        if selected_multiplier < 1.0:  # OTM puts typically have higher implied vol
            implied_vol = base_vol * (1 + 0.05 * (1 - selected_multiplier))
        else:
            implied_vol = base_vol
        
        # Calculate option price and Greeks
        put_premium = black_scholes_put(current_sol_price, strike_price, T, risk_free_rate, implied_vol)
        greeks = calculate_greeks(current_sol_price, strike_price, T, risk_free_rate, implied_vol)
        
        # Hedging strategy summary
        st.subheader("Hedging Strategy Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strike Price", f"${strike_price:.2f}")
        with col2:
            st.metric("Put Premium", f"${put_premium:.2f}")
        with col3:
            st.metric("Contracts Needed", f"{hedge_amount:.0f}")
        with col4:
            st.metric("Total Hedge Cost", f"${put_premium * hedge_amount:,.0f}")
        
        # Risk metrics
        st.subheader("Risk Analysis")
        
        price_lower = float(future_prices_lower.iloc[-1])
        price_upper = float(future_prices_upper.iloc[-1])
        
        max_downside = (current_sol_price - price_lower) * total_exposure
        downside_risk_pct = (current_sol_price - price_lower) / current_sol_price * 100
        hedge_cost_pct = (put_premium * hedge_amount) / (current_sol_price * total_exposure) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Downside Risk (Bear case prediction)", f"${max_downside:,.0f}", f"-{downside_risk_pct:.1f}%")
        with col2:
            st.metric("Total SOL Exposure", f"{total_exposure:,.1f} SOL")
        with col3:
            st.metric("Hedge Cost", f"{hedge_cost_pct:.1f}%")
        
        # Option details table
        st.subheader("Option Details")
        
        option_details = {
            'Parameter': [ 'Current Price','Strike Price', 'Premium', 'Time to Expiry', 'Implied Volatility', 
                         'Risk-free Rate', 'Delta', 'Gamma', 'Theta (daily)', 'Vega'],
            'Value': [f"${current_sol_price:.2f}",f"${strike_price:.2f}",f"${put_premium:.2f}", f"{put_maturity} days", 
                     f"{implied_vol:.1%}", f"{risk_free_rate:.1%}", f"{greeks['delta']:.3f}", 
                     f"{greeks['gamma']:.4f}", f"${greeks['theta']:.2f}", f"{greeks['vega']:.2f}"]
        }
        
        st.table(pd.DataFrame(option_details))
        
        # Scenario analysis
        st.subheader("Scenario Analysis")
        
        scenarios = {
            'Bear Case': price_lower,
            'Current Price': current_sol_price,
            'Predicted Price': future_prices[-1],
            'Bull Case': price_upper
        }
        
        scenario_results = []
        for scenario_name, price in scenarios.items():
            # SOL position PnL
            non_hedged_sol_pnl = (price - current_sol_price) * total_exposure
            
            # Put option payoff
            put_intrinsic = max(strike_price - price, 0)
            put_net_payoff = put_intrinsic - put_premium
            hedge_payoff = put_net_payoff * hedge_amount
            unhedged_payoff = (price - current_sol_price) * (total_exposure - hedge_amount)

            hedge_pnl = unhedged_payoff + hedge_payoff
            
            # Total PnL
            total_pnl = (hedge_pnl - non_hedged_sol_pnl)
            
            # Protection percentage

            scenario_results.append({
                'Scenario': scenario_name,
                'SOL Price': f"${price:.2f}",
                'SOL Unhedged Amount $': f"${non_hedged_sol_pnl:,.0f}",
                'SOL Hedged Amount $': f"${hedge_pnl:,.0f}",
                'SOL PnL $': f"${total_pnl:,.0f}"
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        st.table(scenario_df)
        
        range_pct = np.arange(0,100,1)
        tot_pnls = []

        for pct in range_pct:
            hedge_amount = (pct/100)*total_exposure
            hedge_payoff = put_net_payoff * hedge_amount
            unhedged_payoff = (price - current_sol_price) * (total_exposure - hedge_amount)

            hedge_pnl = unhedged_payoff + hedge_payoff
            
            # Total PnL
            total_pnl = (hedge_pnl - non_hedged_sol_pnl)
            tot_pnls.append(tot_pnls)
        
        tot_pnls_df = pd.DataFrame({"Hedge pct": range_pct,"PnL":tot_pnls})
        st.table(tot_pnls_df)




        # PnL visualization
        st.subheader("Portfolio PnL Analysis")
        
        # Generate price range for analysis
        price_range = np.linspace(price_lower * 0.8, price_upper * 1.2, 100)
        
        # Calculate PnL components
        sol_pnl_range = (current_sol_price-price_range) * total_exposure
        
        # Put option payoffs at different times
        put_payoffs_now = []
        put_payoffs_expiry = []
        
        for p in price_range:
            # Current time value + intrinsic
            current_put_value = black_scholes_put(p, strike_price, T, risk_free_rate, implied_vol)
            put_payoffs_now.append((current_put_value - put_premium) * hedge_amount)
            
            # At expiry (intrinsic value only)
            intrinsic_value = max(strike_price - p, 0)
            put_payoffs_expiry.append((intrinsic_value - put_premium) * hedge_amount)
        
        put_payoffs_now = np.array(put_payoffs_now)
        put_payoffs_expiry = np.array(put_payoffs_expiry)
        
        # Total PnL
        total_pnl_now = sol_pnl_range + put_payoffs_now
        total_pnl_expiry = sol_pnl_range + put_payoffs_expiry
        
        # Create PnL chart
        fig_pnl, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Portfolio PnL
        ax1.plot(price_range, -sol_pnl_range, 'r:', linewidth=2, label='SOL Only (Unhedged)')
        ax1.plot(price_range, total_pnl_now, 'b-', linewidth=2, label='Hedged (Current)')
        ax1.plot(price_range, total_pnl_expiry, 'g--', linewidth=2, label='Hedged (At Expiry)')
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=current_sol_price, color='blue', linestyle=':', alpha=0.7, label='Current Price')
        ax1.axvline(x=price_lower, color='red', linestyle=':', alpha=0.7, label='Bear Case')
        ax1.axvline(x=price_upper, color='green', linestyle=':', alpha=0.7, label='Bull Case')
        
        ax1.set_xlabel('SOL Price ($)')
        ax1.set_ylabel('Portfolio PnL ($)')
        ax1.set_title('Portfolio PnL vs SOL Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Greeks visualization
        delta_values = []
        for p in price_range:
            greeks_temp = calculate_greeks(p, strike_price, T, risk_free_rate, implied_vol)
            delta_values.append(greeks_temp['delta'] * hedge_amount)
        
        ax2.plot(price_range, delta_values, 'purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=current_sol_price, color='blue', linestyle=':', alpha=0.7)
        ax2.set_xlabel('SOL Price ($)')
        ax2.set_ylabel('Portfolio Delta')
        ax2.set_title('Portfolio Delta (Price Sensitivity)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_pnl)
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        break_even = strike_price - put_premium
        max_protection = min(hedge_amount / total_exposure, 1.0) * 100
        
        insights_list = [
            f"**Break-even price:** ${break_even:.2f} (put becomes profitable below this level)",
            f"**Hedge efficiency:** {hedge_cost_pct:.1f}% of portfolio value for {max_protection:.0f}% exposure coverage",
            f"**Maximum protection:** Up to {max_protection:.0f}% of downside risk is hedged",
            f"**Volatility insight:** GARCH model suggests {vol_estimates['garch_vol']:.1%} annual volatility",
            f"**Risk-reward:** Hedge limits downside while preserving upside above ${strike_price:.0f}"
        ]
        
        for insight in insights_list:
            st.write(insight)
    
    except Exception as e:
        st.error(f"Error in hedging analysis: {e}")
        import traceback
        st.text(traceback.format_exc())

    # === DETAILED MODEL RESULTS (EXPANDABLE) ===
    with st.expander("📊 Price Model Details", expanded=False):
        st.write(models)
        
        st.write("**Model Performance:**")
        st.write(f"- R² Score: {r2:.4f}")
        st.write(f"- Training samples: {len(X_shifted_train_enhanced):,}")
        st.write(f"- Features: {X_shifted_train_enhanced.shape[1]}")
        
        # Model backtest visualization
        fig_detail, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Actual vs Predicted prices
        axes[0, 0].plot(true_prices.index, true_prices, label='Actual Prices', linewidth=2)
        axes[0, 0].plot(pred_prices.index, pred_prices, label='Predicted Prices', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Price Reconstruction from Log-Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_shifted_test_enhanced, y_pred, alpha=0.6)
        axes[0, 1].plot([y_shifted_test_enhanced.min(), y_shifted_test_enhanced.max()],
                       [y_shifted_test_enhanced.min(), y_shifted_test_enhanced.max()], 'r--')
        axes[0, 1].set_title(f'Predicted vs Actual (R² = {r2:.3f})')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_shifted_test_enhanced - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residuals Analysis')
        axes[1, 0].set_xlabel('Predictions')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance
        if hasattr(models[0], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_shifted_train_enhanced.columns,
                'importance': models[0].feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_detail)
        
        # Feature importance table
        if hasattr(models[0], 'feature_importances_'):
            st.subheader("Feature Importance")
            feature_df = pd.DataFrame({
                'Feature': X_shifted_train_enhanced.columns,
                'Importance': models[0].feature_importances_
            }).sort_values('Importance', ascending=False)
            st.dataframe(feature_df.head(15))

    with st.expander("Revenue Model Details", expanded=False):
        st.write(models_rew)
        
        st.write("**Revenue Model Performance:**")
        st.write(f"- R² Score: {r2_rew:.4f}")
        st.write(f"- MSE: {mse_rew:.4f}")
        st.write(f"- MAE: {mae_rew:.4f}")
        st.write(f"- Volatility Ratio: {vol_ratio_rew:.2f}")
        
        # Revenue model visualization
        fig_rev, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y_test_rew, y_pred_rew, alpha=0.6)
        axes[0, 0].plot([y_test_rew.min(), y_test_rew.max()],
                       [y_test_rew.min(), y_test_rew.max()], 'r--')
        axes[0, 0].set_title(f'Revenue: Predicted vs Actual (R² = {r2_rew:.3f})')
        axes[0, 0].set_xlabel('Actual Rewards (SOL)')
        axes[0, 0].set_ylabel('Predicted Rewards (SOL)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time series
        axes[0, 1].plot(y_test_rew.index, y_test_rew, 'o-', label='Actual', markersize=4)
        axes[0, 1].plot(y_test_rew.index, y_pred_rew, 's-', label='Predicted', markersize=4)
        axes[0, 1].set_title('Revenue Predictions: Test Period')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution comparison
        axes[1, 0].hist(rewards_active_rew['SOL'], bins=30, alpha=0.7, label='Historical', density=True)
        axes[1, 0].hist(future_rewards_rew, bins=20, alpha=0.7, label='Predicted', density=True)
        axes[1, 0].set_title('Revenue Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals_rev = y_test_rew - y_pred_rew
        axes[1, 1].scatter(y_pred_rew, residuals_rev, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Revenue Model Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_rev)
        
        # Revenue statistics
        st.subheader("Revenue Statistics")
        st.write(f"**Historical average reward:** {rewards_active_rew['SOL'].mean():.4f} SOL")
        st.write(f"**Predicted average reward:** {future_rewards_rew.mean():.4f} SOL")
        st.write(f"**Total predicted reward (30 days):** {future_rewards_rew.sum():.4f} SOL")
        
        # Feature importance for revenue model
        if 'importance_df_rew' in locals() and importance_df_rew is not None:
            st.subheader("Revenue Model Feature Importance")
            st.dataframe(importance_df_rew.head(15))