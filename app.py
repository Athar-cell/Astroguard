# app.py  — AstroGuard single-file Streamlit app (all-in-one)
# Save this as app.py and run: streamlit run app.py

import os
import io
import zipfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Optional XGBoost import — gracefully fallback if unavailable
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Optional SGP4 import — for real TLE propagation
try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except Exception:
    SGP4_AVAILABLE = False

# ---------------------------
# Config / Output directories
# ---------------------------
st.set_page_config(layout="wide", page_title="AstroGuard — Simulation & ML", initial_sidebar_state="expanded")
ROOT = Path.cwd()
OUT_DIR = ROOT / "astroguard_outputs"
MODELS_DIR = OUT_DIR / "models"
VIS_DIR = OUT_DIR / "visualizations"
REPORTS_DIR = OUT_DIR / "reports"
for d in [OUT_DIR, MODELS_DIR, VIS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6378.137
np.random.seed(0)

# Developer-provided uploaded PPTX file path (from this session)
PPTX_PATH = "/mnt/data/ASTROGUARD.pptx.pptx"   # <---- uses your uploaded path
PPTX_URL = f"file://{PPTX_PATH}"

# ---------------------------
# Helper functions
# ---------------------------
def safe_get_tles(n=6, timeout=6):
    """
    Download TLEs from Celestrak. If fails, fallback to minimal ISS TLE.
    Returns list of dicts: {'name','line1','line2'}
    """
    urls = [
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"
    ]
    lines = []
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            txt = r.text.splitlines()
            if len(txt) > 10:
                lines = txt
                break
        except Exception:
            continue
    if not lines:
        lines = [
            "ISS (ZARYA)",
            "1 25544U 98067A   20348.54791667  .00001264  00000-0  29634-4 0  9991",
            "2 25544  51.6448  20.6781 0001691  57.3457 302.8509 15.49409478"
        ]
    tles = []
    for i in range(0, len(lines), 3):
        if i+2 < len(lines):
            name = lines[i].strip(); l1 = lines[i+1].strip(); l2 = lines[i+2].strip()
            if l1.startswith("1 ") and l2.startswith("2 "):
                tles.append({"name": name, "line1": l1, "line2": l2})
        if len(tles) >= n:
            break
    return tles

def build_times(start=None, duration_s=1200, step_s=10):
    if start is None:
        start = datetime.utcnow()
    steps = max(1, int(duration_s/step_s) + 1)
    return [start + timedelta(seconds=i*step_s) for i in range(steps)]

def propagate_tle(line1, line2, times):
    """
    If sgp4 available, propagate TLE into ECI positions (km).
    If not available, return an array of NaNs so callers fall back to synthetic.
    """
    if not SGP4_AVAILABLE:
        return np.full((len(times), 3), np.nan)
    sat = Satrec.twoline2rv(line1, line2)
    coords = []
    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            coords.append([np.nan, np.nan, np.nan])
        else:
            coords.append(r)  # r is km ECI
    return np.array(coords)

def circular_orbit_coords(r_km, incl_deg=0, rasc_deg=0, n_steps=200, phase=0):
    """
    Simple synthetic circular orbit in ECI for visualization (not precise physics)
    """
    theta = np.linspace(0, 2*np.pi, n_steps)
    X = r_km * np.cos(theta + phase)
    Y = r_km * np.sin(theta + phase)
    Z = (np.sin(theta*3) * (r_km*0.03))  # small wobble for 3D effect
    # rotate by inclination & RAAN approximations (quick visual)
    inc = np.deg2rad(incl_deg); rasc = np.deg2rad(rasc_deg)
    # apply simple rotation on XZ plane and around Z
    x2 = X * np.cos(inc) - Z * np.sin(inc)
    z2 = X * np.sin(inc) + Z * np.cos(inc)
    # rotate around z (RAAN)
    x3 = x2 * np.cos(rasc) - Y * np.sin(rasc)
    y3 = x2 * np.sin(rasc) + Y * np.cos(rasc)
    return np.vstack([x3, y3, z2]).T

def generate_debris_field(times, n_debris=200, alt_choices=(500,700,900), spread_km=80):
    """
    Generates many synthetic debris tracks (visual + event simulation)
    Return list of objects {'name','pos':np.array(len(times),3)}
    """
    debris = []
    mu = 398600.4418
    t0 = times[0]
    for i in range(n_debris):
        shell = int(np.random.choice(alt_choices))
        alt = np.random.normal(loc=shell, scale=spread_km)
        a = EARTH_RADIUS_KM + max(160, alt)
        T = 2*np.pi * (a**1.5) / (mu**0.5)
        mean_motion = 2*np.pi / max(T, 1.0)
        inc = np.random.uniform(0, 98)
        rasc = np.random.uniform(0, 360)
        argp = np.random.uniform(0, 360)
        phase0 = np.random.uniform(0, 2*np.pi)
        traj = circular_orbit_coords(a, incl_deg=inc, rasc_deg=rasc, n_steps=len(times), phase=phase0)
        # add small jitter & drift
        traj += np.random.normal(0, 0.5, traj.shape)
        debris.append({"name": f"DEB_{i:04d}", "pos": traj})
    return debris

def detect_close_approaches(objects, times, threshold_km=30.0):
    """
    Naive pairwise close approach detector. Returns list of event dicts.
    """
    events = []
    N = len(times); dt = (times[1]-times[0]).total_seconds() if N>1 else 1.0
    M = len(objects)
    for ti in range(N):
        positions = [o["pos"][ti] for o in objects]
        for i in range(M):
            for j in range(i+1, M):
                p1 = positions[i]; p2 = positions[j]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue
                dist = np.linalg.norm(p1 - p2)
                if dist <= threshold_km:
                    # estimate velocity using central difference if possible
                    if 0 < ti < N-1:
                        v1 = (objects[i]["pos"][ti+1] - objects[i]["pos"][ti-1])/(2*dt)
                        v2 = (objects[j]["pos"][ti+1] - objects[j]["pos"][ti-1])/(2*dt)
                    else:
                        v1 = v2 = np.zeros(3)
                    rel_vel = v1 - v2
                    rel_speed = np.linalg.norm(rel_vel)
                    rel_pos = p1 - p2
                    denom = np.dot(rel_vel, rel_vel)
                    tca = 0.0 if denom < 1e-9 else -np.dot(rel_pos, rel_vel)/denom
                    dv_heur = min(300.0, 1200.0/(dist*1000.0 + 1e-6))
                    events.append({
                        "t_idx": ti, "t_utc": times[ti], "A": objects[i]["name"], "B": objects[j]["name"],
                        "dist_km": float(dist), "rel_speed_km_s": float(rel_speed), "tca_s": float(tca), "dv_m_s": float(dv_heur)
                    })
    return events

def ensure_dataset(X, y, min_samples=200):
    """Augment with synthetic rows if too few events exist."""
    if len(X) < min_samples:
        add = max(0, min_samples - len(X))
        syn_X = []
        syn_y = []
        for _ in range(add):
            d = np.random.uniform(0.2, 40.0)
            s = np.random.uniform(0, 5.0)
            tca = np.random.uniform(0, 900)
            syn_X.append([d, s, tca])
            syn_y.append(1 if d < 5.0 else 0)
        if len(X) == 0:
            X = np.array(syn_X)
            y = np.array(syn_y)
        else:
            X = np.vstack([X, syn_X])
            y = np.hstack([y, syn_y])
    return X, y

def train_models_safe(X, y, out_dir=MODELS_DIR):
    """
    Train 7 models safely, return results DataFrame and dict of trained models.
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, out_dir / "astroguard_scaler.pkl")
    if Xs.shape[0] < 10:
        X_train, X_test, y_train, y_test = Xs, Xs, y, y
    else:
        strat = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=strat)

    models = {
        "LogReg": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradBoost": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=1200, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, use_label_encoder=False, eval_metric="logloss", random_state=42)
    else:
        # if xgboost not available, use XGBoost slot as RandomForest fallback for UI consistency
        models["XGBoost"] = RandomForestClassifier(n_estimators=150, random_state=42)

    results = []
    trained = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
            results.append([name, acc, f1, prec, rec, auc])
            trained[name] = model
            joblib.dump(model, out_dir / f"model_{name}.pkl")
        except Exception as e:
            print("Training failed for", name, e)

    # Soft Voting ensemble (use models that have predict_proba)
    estimators = [(n, trained[n]) for n in trained if hasattr(trained[n], "predict_proba")]
    if len(estimators) > 1:
        vc = VotingClassifier(estimators=estimators, voting="soft")
        vc.fit(X_train, y_train)
        joblib.dump(vc, out_dir / "model_Voting.pkl")
        y_pred = vc.predict(X_test)
        y_proba = vc.predict_proba(X_test)[:,1]
        results.append(["Voting", accuracy_score(y_test, y_pred), f1_score(y_test,y_pred), precision_score(y_test,y_pred), recall_score(y_test,y_pred), roc_auc_score(y_test,y_proba)])
        trained["Voting"] = vc

    df_results = pd.DataFrame(results, columns=["Model","Accuracy","F1","Precision","Recall","AUC"]).sort_values(by="F1", ascending=False)
    df_results.to_csv(out_dir / "model_results.csv", index=False)
    return df_results, trained, (X_train, X_test, y_train, y_test)

# ---------------------------
# Plotting helpers
# ---------------------------
def plot_3d_orbit_scene(objects, times, frame_idx=None, title="3D Orbit Preview"):
    """
    Create a 3D Plotly figure showing Earth and object trails up to frame_idx.
    objects: list of dicts {'name','pos':np.array(len(times),3)}
    """
    if frame_idx is None:
        frame_idx = min(len(times)-1, int(len(times)/2))
    # Earth sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0,'#050510'],[1,'#0e1222']], opacity=0.95, showscale=False))

    colors = px.colors.qualitative.Dark24
    # trails and current positions
    for si,o in enumerate(objects):
        traj = o["pos"][:frame_idx+1]
        if np.isnan(traj).all():
            continue
        fig.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode="lines",
                                   line=dict(color=colors[si%len(colors)], width=2), name=o["name"]))
        p = traj[-1]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                   marker=dict(size=4, color=colors[si%len(colors)]), showlegend=False))
    fig.update_layout(template="plotly_dark", title=title, scene=dict(aspectmode="data"), height=720)
    return fig

def plot_model_metrics(df_results):
    fig_acc = px.bar(df_results, x="Model", y="Accuracy", color="Model", template="plotly_dark", title="Model Accuracy")
    fig_f1 = px.bar(df_results, x="Model", y="F1", color="Model", template="plotly_dark", title="Model F1 Score")
    return fig_acc, fig_f1

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown("<div style='display:flex; align-items:center; gap:12px'>"
            "<img src='https://img.icons8.com/color/96/satellite.png' width='48'/>"
            "<h1 style='margin:0;color:#00E5FF'>AstroGuard</h1>"
            "<div style='color:#9aa6b2;margin-left:12px'>Realtime orbit sim · collision detection · ML</div>"
            "</div>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col2:
    if Path(PPTX_PATH).exists():
        st.markdown("**Uploaded PPTX:**")
        st.markdown(f"[Open PPTX]({PPTX_URL})")
    else:
        st.info("No PPTX found at the session path.")

st.sidebar.header("Simulation Controls")
duration_s = st.sidebar.slider("Simulation duration (s)", 600, 3600, 1200, step=200)
step_s = st.sidebar.slider("Timestep (s)", 4, 20, 8, step=2)
n_debris = st.sidebar.slider("Debris count", 50, 800, 200, step=50)
tles_count = st.sidebar.slider("TLEs to load (active)", 0, 12, 6, step=1)
threshold_km = st.sidebar.slider("Close-approach threshold (km)", 5.0, 200.0, 30.0, step=5.0)
min_train_samples = st.sidebar.number_input("Min dataset size for training", min_value=50, max_value=2000, value=300, step=50)

run_sim = st.sidebar.button("▶ Run Simulation")
train_models_btn = st.sidebar.button("⚙ Train Models (7)")
save_orbit_html_btn = st.sidebar.button("Save 3D visualization HTML")
export_zip_btn = st.sidebar.button("📦 Package outputs & ZIP")

# display small tips
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If XGBoost fails to install on deployment, the app will use RandomForest as a fallback.")

# Session state defaults
if "sim" not in st.session_state:
    st.session_state["sim"] = None
if "trained" not in st.session_state:
    st.session_state["trained"] = None
if "df_results" not in st.session_state:
    st.session_state["df_results"] = None

# -------------
# Run simulation
# -------------
if run_sim:
    with st.spinner("Downloading TLEs and running simulation..."):
        times = build_times(duration_s=duration_s, step_s=step_s)
        objects = []
        # Try TLE propagation if user requested any
        if tles_count > 0:
            tles = safe_get_tles(n=tles_count)
            for t in tles:
                pos = propagate_tle(t["line1"], t["line2"], times)
                # if propagation failed or produced NaNs -> fallback to synthetic orbit
                if np.isnan(pos).all():
                    # synthetic orbit at 7000 + offset km
                    idx = len(objects)
                    traj = circular_orbit_coords(7000 + idx*200, incl_deg=idx*10, n_steps=len(times), phase=np.random.uniform(0,2*np.pi))
                    objects.append({"name": t["name"], "pos": traj})
                else:
                    objects.append({"name": t["name"], "pos": pos})
        # generate debris
        debris = generate_debris_field(times, n_debris=n_debris)
        # combine objects (keep satellites first)
        objects = objects + debris
        events = detect_close_approaches(objects, times, threshold_km=threshold_km)
        st.session_state["sim"] = {"times": times, "objects": objects, "events": events}
        st.success(f"Simulation complete — events detected: {len(events)}")

# -------------
# Show sim & visuals
# -------------
if st.session_state["sim"] is not None:
    sim = st.session_state["sim"]
    st.markdown("### Simulation Summary")
    st.write(f"Objects: {len(sim['objects'])}, Timesteps: {len(sim['times'])}, Events: {len(sim['events'])}")
    # show small sample events
    if len(sim["events"]) > 0:
        st.markdown("**Sample events** (first 20)")
        df_ev = pd.DataFrame(sim["events"][:20])
        st.dataframe(df_ev)

    # 3D preview
    st.markdown("### 3D Orbit Preview")
    # choose a frame index slider for preview
    frame_idx = st.slider("Preview frame index", 0, max(0, len(sim["times"])-1), min(10, max(0, len(sim["times"])-1)))
    fig3d = plot_3d_orbit_scene(sim["objects"], sim["times"], frame_idx=frame_idx, title="AstroGuard — Orbit & Debris (Preview)")
    st.plotly_chart(fig3d, use_container_width=True)

    # Save 3D HTML
    if save_orbit_html_btn:
        orbit_file = VIS_DIR / f"orbit_preview_{int(time.time())}.html"
        fig3d.write_html(str(orbit_file), include_plotlyjs="cdn")
        st.success(f"Saved orbit HTML: {orbit_file.name}")
        with open(orbit_file, "rb") as fh:
            st.download_button("⬇ Download orbit HTML", fh, file_name=orbit_file.name, mime="text/html")

# -------------
# Build dataset / Train models
# -------------
if train_models_btn:
    if st.session_state["sim"] is None:
        st.error("Run simulation first.")
    else:
        with st.spinner("Building ML dataset & training models..."):
            sim = st.session_state["sim"]
            events = sim["events"]
            X = []
            y = []
            for e in events:
                X.append([e["dist_km"], e["rel_speed_km_s"], abs(e["tca_s"])])
                y.append(1 if e["dist_km"] < 5.0 else 0)
            X = np.array(X)
            y = np.array(y)
            X, y = ensure_dataset(X, y, min_samples=min_train_samples)
            df_results, trained, splits = train_models_safe(X, y, out_dir=MODELS_DIR)
            st.session_state["trained"] = trained
            st.session_state["df_results"] = df_results
            st.success("Models trained and saved to models folder.")
            st.write(df_results)

# -------------
# Show metrics & download
# -------------
if st.session_state.get("df_results") is not None:
    df_results = st.session_state["df_results"]
    st.markdown("### Model Performance")
    fig_acc, fig_f1 = plot_model_metrics(df_results)
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(fig_acc, use_container_width=True)
        acc_file = VIS_DIR / "model_accuracy.html"; fig_acc.write_html(str(acc_file), include_plotlyjs="cdn")
        with open(acc_file, "rb") as fh:
            st.download_button("⬇ Download Accuracy HTML", fh, file_name=acc_file.name, mime="text/html")
    with colB:
        st.plotly_chart(fig_f1, use_container_width=True)
        f1_file = VIS_DIR / "model_f1.html"; fig_f1.write_html(str(f1_file), include_plotlyjs="cdn")
        with open(f1_file, "rb") as fh:
            st.download_button("⬇ Download F1 HTML", fh, file_name=f1_file.name, mime="text/html")

    # Confusion matrices
    st.markdown("#### Confusion Matrices (Top models)")
    trained = st.session_state.get("trained", {})
    top_models = df_results["Model"].tolist()[:3]
    for m in top_models:
        try:
            model = trained[m]
            # use last splits if available
            X_train, X_test, y_train, y_test = splits
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["Act 0","Act 1"], colorscale="Viridis"))
            fig_cm.update_layout(title=f"Confusion Matrix - {m}", template="plotly_dark", height=360)
            st.plotly_chart(fig_cm, use_container_width=True)
            cm_file = VIS_DIR / f"cm_{m}.html"; fig_cm.write_html(str(cm_file), include_plotlyjs="cdn")
            with open(cm_file, "rb") as fh:
                st.download_button(f"⬇ Download CM - {m}", fh, file_name=cm_file.name, mime="text/html")
        except Exception as e:
            st.warning(f"Could not show CM for {m}: {e}")

# -------------
# Package outputs into ZIP
# -------------
if export_zip_btn:
    with st.spinner("Packaging outputs into zip..."):
        zip_path = ROOT / "AstroGuard_Submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # include models, visuals, reports
            for folder in [MODELS_DIR, VIS_DIR, REPORTS_DIR]:
                for root, _, files in os.walk(folder):
                    for fname in files:
                        full = os.path.join(root, fname)
                        arc = os.path.relpath(full, OUT_DIR)
                        zf.write(full, arc)
            # include PPTX if exists
            if Path(PPTX_PATH).exists():
                zf.write(PPTX_PATH, os.path.join("ppt", os.path.basename(PPTX_PATH)))
        st.success(f"Packaged to {zip_path.name}")
        with open(zip_path, "rb") as fh:
            st.download_button("⬇ Download Submission ZIP", fh, file_name=zip_path.name, mime="application/zip")

# Footer
st.markdown("---")
st.markdown("AstroGuard — built for hackathon demos. Tips: run simulation → train models → export ZIP.")
if not XGBOOST_AVAILABLE:
    st.warning("XGBoost not available in environment — the app used RandomForest fallback for the XGBoost slot.")
