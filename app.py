# app.py  —  ONE-FILE NFL predictor for Streamlit (no other files needed)
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score

st.set_page_config(page_title="NFL Predictor", page_icon="🏈", layout="wide")
st.title("🏈 NFL Prediction — Win %, Score & Bet Recommendation (1-file app)")

# --- helpers for odds ---
def american_to_prob(odds):
    try:
        o = float(odds)
    except:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)

def prob_to_american(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return - (p / (1 - p)) * 100.0
    else:
        return (1 - p) / p * 100.0

# --- tiny model class (logistic for win%, GBM for scores) ---
DEFAULT_FEATURES = [
    "home_elo","away_elo","home_days_rest","away_days_rest",
    "home_last5_off","home_last5_def","away_last5_off","away_last5_def",
    "is_division_game","is_playoffs"
]

class SportsPredictor:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.clf = None
        self.reg_home = None
        self.reg_away = None
        self.ct = None

    def _prep(self):
        return ColumnTransformer([("num", StandardScaler(), self.feature_cols)], remainder="drop")

    def fit(self, df: pd.DataFrame):
        # sanity check
        for c in ["home_points","away_points","home_win"]:
            if c not in df.columns:
                raise ValueError(f"Missing target column '{c}'")
        for c in self.feature_cols:
            if c not in df.columns:
                raise ValueError(f"Missing feature column '{c}'")

        X = df[self.feature_cols].copy()
        y_cls = df["home_win"].astype(int).values
        y_home = df["home_points"].astype(float).values
        y_away = df["away_points"].astype(float).values

        self.ct = self._prep()
        # classifier + calibration for win%
        base = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_pipe = Pipeline([("prep", self.ct), ("clf", base)])
        self.clf = CalibratedClassifierCV(clf_pipe, method="isotonic", cv=3)

        # score regressors
        self.reg_home = Pipeline([("prep", self.ct), ("reg", GradientBoostingRegressor(random_state=42))])
        self.reg_away = Pipeline([("prep", self.ct), ("reg", GradientBoostingRegressor(random_state=42))])

        # simple holdout just to show metrics
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.25, random_state=42, stratify=y_cls)
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_cls[tr], y_cls[te]
        yhtr, yhte = y_home[tr], y_home[te]
        yatr, yate = y_away[tr], y_away[te]

        self.clf.fit(Xtr, ytr)
        self.reg_home.fit(Xtr, yhtr)
        self.reg_away.fit(Xtr, yatr)

        p = self.clf.predict_proba(Xte)[:,1]
        brier = brier_score_loss(yte, p)
        ph = self.reg_home.predict(Xte); pa = self.reg_away.predict(Xte)
        mae_h = mean_absolute_error(yhte, ph)
        mae_a = mean_absolute_error(yate, pa)
        r2_h = r2_score(yhte, ph)
        r2_a = r2_score(yate, pa)

        return {"brier": brier, "mae_home": mae_h, "mae_away": mae_a, "r2_home": r2_h, "r2_away": r2_a}

    def predict_game(self, row: pd.Series,
                     market_spread=None, market_total=None,
                     home_ml=None, away_ml=None,
                     edge_spread=1.0, edge_total=2.0, ml_edge=0.03):
        if self.clf is None:
            raise RuntimeError("Model not trained.")
        x = row[self.feature_cols].to_frame().T
        p_home = float(self.clf.predict_proba(x)[:,1][0])
        ph = float(self.reg_home.predict(x)[0])
        pa = float(self.reg_away.predict(x)[0])
        total = ph + pa
        model_spread = ph - pa  # home - away

        rec, details = None, {}

        # --- FIXED: spread edge (market_spread is home-relative; market margin = Home−Away = -market_spread)
        if market_spread is not None:
            mkt_margin = -float(market_spread)          # e.g., home +5.5 -> market margin = -5.5
            spread_edge = model_spread - mkt_margin     # + means model is more pro-home than market
            if abs(spread_edge) >= edge_spread:
                if spread_edge >= 0:
                    # Bet the HOME side at the posted home spread
                    rec = f"HOME {market_spread:+.1f}"
                else:
                    # Bet the AWAY side at the opposite spread
                    rec = f"AWAY {-market_spread:+.1f}"
                details["spread_edge_pts"] = float(spread_edge)

        # total edge
        if rec is None and market_total is not None:
            total_edge = total - market_total
            if abs(total_edge) >= edge_total:
                rec = "OVER" if total_edge > 0 else "UNDER"
                details["total_edge_pts"] = float(total_edge)

        # ML edge
        if rec is None and (home_ml is not None or away_ml is not None):
            ih = american_to_prob(home_ml) if home_ml is not None else None
            ia = american_to_prob(away_ml) if away_ml is not None else None
            cand = []
            if ih is not None: cand.append(("HOME ML", p_home - ih))
            if ia is not None: cand.append(("AWAY ML", (1 - p_home) - ia))
            if cand:
                side, edge = max(cand, key=lambda t: t[1])
                if edge >= ml_edge:
                    rec = side
                    details["ml_edge"] = float(edge)
                    details["fair_home_ml"] = float(prob_to_american(p_home))

        if rec is None:
            rec = "HOME ML" if p_home >= 0.55 else "AWAY ML"

        return {
            "home_win_prob": p_home,
            "away_win_prob": 1 - p_home,
            "predicted_home_points": ph,
            "predicted_away_points": pa,
            "predicted_total": total,
            "model_spread_home_minus_away": model_spread,
            "recommendation": rec,
            "details": details
        }

# --- synthetic demo data (until you upload real history) ---
def build_synthetic(n=800):
    rng = np.random.default_rng(42)
    home_strength = rng.normal(1500, 75, n)
    away_strength = rng.normal(1500, 75, n)
    is_playoffs = rng.integers(0, 2, n)
    is_div = rng.integers(0, 2, n)
    home_rest = rng.integers(3, 8, n)
    away_rest = rng.integers(3, 8, n)

    h_off = (home_strength - 1500)*0.08 + rng.normal(0,2,n) + 24
    h_def = (home_strength - 1500)*-0.06 + rng.normal(0,2,n) + 23
    a_off = (away_strength - 1500)*0.08 + rng.normal(0,2,n) + 24
    a_def = (away_strength - 1500)*-0.06 + rng.normal(0,2,n) + 23

    home_mu = 0.015*(home_strength-away_strength) + 0.3*(home_rest-away_rest) + 0.3*is_playoffs + 0.15*is_div + 24
    away_mu = -0.015*(home_strength-away_strength) - 0.2*(home_rest-away_rest) + 0.2*is_playoffs + 0.10*is_div + 23

    home_pts = np.round(home_mu + rng.normal(0,6,n)).clip(0, None)
    away_pts = np.round(away_mu + rng.normal(0,6,n)).clip(0, None)
    home_win = (home_pts > away_pts).astype(int)

    df = pd.DataFrame({
        'home_elo': home_strength,
        'away_elo': away_strength,
        'home_days_rest': home_rest,
        'away_days_rest': away_rest,
        'home_last5_off': h_off,
        'home_last5_def': h_def,
        'away_last5_off': a_off,
        'away_last5_def': a_def,
        'is_division_game': is_div,
        'is_playoffs': is_playoffs,
        'home_points': home_pts.astype(int),
        'away_points': away_pts.astype(int),
        'home_win': home_win.astype(int),
    })
    return df

# --- sidebar: train model ---
st.sidebar.header("⚙️ Train the model")
train_source = st.sidebar.radio("Training source", ["Use demo synthetic data", "Upload NFL history CSV"], index=0)
uploaded_hist = None
if train_source == "Upload NFL history CSV":
    uploaded_hist = st.sidebar.file_uploader("Upload historical NFL CSV", type=["csv"])

edge_spread = st.sidebar.number_input("Spread edge threshold (pts)", value=1.0, step=0.5, format="%.1f")
edge_total  = st.sidebar.number_input("Total edge threshold (pts)",  value=2.0, step=0.5, format="%.1f")
ml_edge     = st.sidebar.number_input("Moneyline edge min (prob)",   value=0.03, step=0.01, format="%.2f")

if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

if st.sidebar.button("Train / Re-train"):
    if train_source == "Use demo synthetic data":
        df = build_synthetic(1200)
    else:
        if uploaded_hist is None:
            st.error("Please upload a CSV.")
            st.stop()
        df = pd.read_csv(uploaded_hist)

    sp = SportsPredictor(feature_cols=DEFAULT_FEATURES)
    try:
        metrics = sp.fit(df)
        st.session_state.predictor = sp
        st.session_state.metrics = metrics
        st.success("Model trained.")
    except Exception as e:
        st.error(f"Training failed: {e}")

if st.session_state.metrics:
    st.sidebar.write({k: round(float(v),4) for k,v in st.session_state.metrics.items()})

st.divider()
tabs = st.tabs(["🕹️ Single Game", "📦 Batch CSV"])

with tabs[0]:
    st.subheader("Single Game Prediction")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        home_elo = st.number_input("Home ELO", value=1540.0, step=5.0)
        home_rest = st.number_input("Home days rest", value=6.0, step=1.0)
        home_off = st.number_input("Home last5 offense (PPG)", value=24.5, step=0.5)
        home_def = st.number_input("Home last5 defense (PPG allowed)", value=20.0, step=0.5)
    with c2:
        away_elo = st.number_input("Away ELO", value=1510.0, step=5.0)
        away_rest = st.number_input("Away days rest", value=6.0, step=1.0)
        away_off = st.number_input("Away last5 offense (PPG)", value=21.5, step=0.5)
        away_def = st.number_input("Away last5 defense (PPG allowed)", value=23.0, step=0.5)
    with c3:
        is_div = st.selectbox("Division game?", ["No", "Yes"])
        is_playoffs = st.selectbox("Playoffs?", ["No", "Yes"])
        market_spread = st.number_input("Market spread (home-relative)", value=-3.5, step=0.5, format="%.1f")
        st.caption("Positive = home underdog, Negative = home favorite.")
        market_total = st.number_input("Market total (O/U)", value=44.5, step=0.5, format="%.1f")
    with c4:
        home_ml = st.text_input("Home ML (American, optional)", value="-165")
        away_ml = st.text_input("Away ML (American, optional)", value="+145")
        st.caption("ML used only if no spread/total edge.")

    if st.button("Predict Game", type="primary"):
        if st.session_state.predictor is None:
            st.warning("Training a quick demo model for you now.")
            sp = SportsPredictor(DEFAULT_FEATURES)
            st.session_state.metrics = sp.fit(build_synthetic(800))
            st.session_state.predictor = sp

        row = pd.Series({
            "home_elo": home_elo, "away_elo": away_elo,
            "home_days_rest": home_rest, "away_days_rest": away_rest,
            "home_last5_off": home_off, "home_last5_def": home_def,
            "away_last5_off": away_off, "away_last5_def": away_def,
            "is_division_game": 1 if is_div=="Yes" else 0,
            "is_playoffs": 1 if is_playoffs=="Yes" else 0,
            "home_points": 0, "away_points": 0, "home_win": 0
        })

        def tf(x):
            try: return float(x)
            except: return None

        pred = st.session_state.predictor.predict_game(
            row,
            market_spread=market_spread,
            market_total=market_total,
            home_ml=tf(home_ml) if home_ml else None,
            away_ml=tf(away_ml) if away_ml else None,
            edge_spread=edge_spread, edge_total=edge_total, ml_edge=ml_edge
        )

        left, right = st.columns(2)
        with left:
            st.metric("Home Win %", f"{pred['home_win_prob']*100:.1f}%")
            st.metric("Away Win %", f"{pred['away_win_prob']*100:.1f}%")
            st.write(f"Model Spread (Home - Away): **{pred['model_spread_home_minus_away']:.2f}**")
            st.write(f"Predicted Total: **{pred['predicted_total']:.2f}**")
        with right:
            st.metric("Pred Home Score", f"{pred['predicted_home_points']:.1f}")
            st.metric("Pred Away Score", f"{pred['predicted_away_points']:.1f}")
            st.success(f"Recommended Play: **{pred['recommendation']}**")
            if pred.get("details"): st.caption(f"Details: {pred['details']}")

with tabs[1]:
    st.subheader("Batch Predictions (CSV)")
    st.markdown("CSV must include at least the feature columns shown below. Market columns optional but recommended.")
    st.code(", ".join(DEFAULT_FEATURES) + ", market_spread, market_total, home_ml, away_ml", language="text")
    upc = st.file_uploader("Upload upcoming games CSV", type=["csv"])
    if st.button("Run Batch"):
        if upc is None:
            st.error("Upload a CSV first.")
        else:
            df_upc = pd.read_csv(upc)
            missing = [c for c in DEFAULT_FEATURES if c not in df_upc.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.session_state.predictor is None:
                    st.warning("Training a quick demo model for you now.")
                    sp = SportsPredictor(DEFAULT_FEATURES)
                    st.session_state.metrics = sp.fit(build_synthetic(800))
                    st.session_state.predictor = sp

                rows = []
                for _, r in df_upc.iterrows():
                    pred = st.session_state.predictor.predict_game(
                        r,
                        market_spread=r.get("market_spread", None),
                        market_total=r.get("market_total", None),
                        home_ml=r.get("home_ml", None),
                        away_ml=r.get("away_ml", None)
                    )
                    rows.append({**r.to_dict(), **{
                        "home_win_prob": pred["home_win_prob"],
                        "pred_home_pts": pred["predicted_home_points"],
                        "pred_away_pts": pred["predicted_away_points"],
                        "pred_total": pred["predicted_total"],
                        "model_spread_home_minus_away": pred["model_spread_home_minus_away"],
                        "recommendation": pred["recommendation"]
                    }})
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                st.download_button("Download predictions.csv", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
