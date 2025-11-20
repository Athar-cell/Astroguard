# app.py - AstroGuard Streamlit (improved UI + downloads)
import os
import zipfile
import io
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sgp4.api import Satrec, jday
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from fpdf import FPDF

st.set_page_config(page_title="AstroGuard", layout="wide", initial_sidebar_state="expanded")

# ------------------------
# Paths & Globals
# ------------------------
ROOT = Path.cwd()
OUT_DIR = ROOT / "astroguard_streamlit_outputs"
MODELS_DIR = OUT_DIR / "models"
VIS_DIR = OUT_DIR / "visualizations"
REPORTS_DIR = OUT_DIR / "reports"
for d in [OUT_DIR, MODELS_DIR, VIS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6378.137
np.random.seed(0)

# Uploaded PPTX path from this session (developer-provided)
PPTX_PATH = "/mnt/data/ASTROGUARD.pptx.pptx"
PPTX_URL = f"file://{PPTX_PATH}"   # developer asked to keep exact path as URL

# ------------------------
# Small UI styling
# ------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #060616; color: #e6eef6; }
    .header { font-size:34px; font-weight:800; color:#00E5FF; letter-spacing:1px; }
    .sub { color:#cbd5e1; margin-top: -8px; }
    .card { background: linear-gradient(90deg, rgba(6,6,22,0.6), rgba(12,12,30,0.6)); padding: 12px; border-radius: 8px; margin-bottom: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Helper functions
# ------------------------
def safe_get_tles(n=6, timeout=6):
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
    tles=[]
    for i in range(0, len(lines), 3):
        if i+2 < len(lines):
            name = lines[i].strip(); l1 = lines[i+1].strip(); l2 = lines[i+2].strip()
            if l1.startswith("1 ") and l2.startswith("2 "):
                tles.append({"name": name, "line1": l1, "line2": l2})
        if len(tles) >= n:
            break
    return tles

def build_times(duration_s=1200, step_s=8):
    start = datetime.utcnow()
    steps = int(duration_s/step_s) + 1
    return [start + timedelta(seconds=i*step_s) for i in range(steps)]

def propagate_tle(line1, line2, times):
    sat = Satrec.twoline2rv(line1, line2)
    coords = []
    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            coords.append([np.nan, np.nan, np.nan])
        else:
            coords.append(r)
    return np.array(coords)

def generate_debris_field(times, n_debris=200, alt_choices=(500,700,900), spread_km=80):
    debris=[]
    mu=398600.4418
    for i in range(n_debris):
        shell = int(np.random.choice(alt_choices))
        alt = np.random.normal(loc=shell, scale=spread_km)
        a = EARTH_RADIUS_KM + max(160, alt)
        T = 2*np.pi * (a**1.5) / (mu**0.5)
        n = 2*np.pi / max(T, 1.0)
        inc = np.deg2rad(np.random.uniform(0,98)); rasc=np.random.uniform(0,2*np.pi); argp=np.random.uniform(0,2*np.pi)
        phase0 = np.random.uniform(0,2*np.pi)
        pos=[]
        for t in times:
            dt=(t-times[0]).total_seconds()
            M = phase0 + n*dt + np.random.normal(0,4e-4)
            x_orb = a*np.cos(M); y_orb = a*np.sin(M)
            ca,sa = np.cos(argp), np.sin(argp)
            ci,si = np.cos(inc), np.sin(inc)
            cr,sr = np.cos(rasc), np.sin(rasc)
            x = (cr*(ca*x_orb - sa*y_orb) - sr*(ci*sa*x_orb + ci*ca*y_orb))
            y = (sr*(ca*x_orb - sa*y_orb) + cr*(ci*sa*x_orb + ci*ca*y_orb))
            z = (si*sa*x_orb + si*ca*y_orb)
            x += np.random.normal(0,0.9); y += np.random.normal(0,0.9); z += np.random.normal(0,0.4)
            pos.append([x,y,z])
        debris.append({"name": f"DEB_{i:04d}", "pos": np.array(pos)})
    return debris

def detect_close_approaches(objects, times, threshold_km=30.0):
    events=[]
    N=len(times); dt=(times[1]-times[0]).total_seconds(); M=len(objects)
    for ti in range(N):
        pos_at_t = [o["pos"][ti] for o in objects]
        for i in range(M):
            for j in range(i+1, M):
                p1=pos_at_t[i]; p2=pos_at_t[j]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)): continue
                dist = np.linalg.norm(p1-p2)
                if dist <= threshold_km:
                    if 0<ti<N-1:
                        v1=(objects[i]["pos"][ti+1]-objects[i]["pos"][ti-1])/(2*dt)
                        v2=(objects[j]["pos"][ti+1]-objects[j]["pos"][ti-1])/(2*dt)
                    else:
                        v1=v2=np.zeros(3)
                    rel_vel = v1 - v2; rel_speed=np.linalg.norm(rel_vel)
                    rel_pos = p1 - p2
                    denom = np.dot(rel_vel, rel_vel)
                    tca = 0.0 if denom < 1e-9 else -np.dot(rel_pos, rel_vel)/denom
                    dv = min(300.0, 1200.0/(dist*1000.0 + 1e-6))
                    events.append({"t_idx":ti,"t_utc":times[ti],"A":objects[i]["name"],"B":objects[j]["name"],
                                   "dist_km":float(dist),"rel_speed_km_s":float(rel_speed),"tca_s":float(tca),"dv_m_s":float(dv)})
    return events

def ensure_dataset(X, y, min_samples=300):
    if len(X) < min_samples:
        add = max(0, min_samples - len(X))
        syn_X=[]; syn_y=[]
        for _ in range(add):
            d = np.random.uniform(0.2,40.0); s=np.random.uniform(0,5.0); tca=np.random.uniform(0,900)
            syn_X.append([d,s,tca]); syn_y.append(1 if d<5.0 else 0)
        if len(X)==0:
            X=np.array(syn_X); y=np.array(syn_y)
        else:
            X=np.vstack([X,syn_X]); y=np.hstack([y,syn_y])
    return X,y

# ------------------------
# Layout: Header + Sidebar
# ------------------------
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    st.image("https://raw.githubusercontent.com/your-repo/astroguard-assets/main/logo_placeholder.png" if False else "https://img.icons8.com/fluency/48/000000/satellite.png", width=58)
with header_col2:
    st.markdown('<div class="header">AstroGuard</div><div class="sub">Realtime simulation • collision detection • ML models • export</div>', unsafe_allow_html=True)

st.sidebar.markdown("## Controls")
duration_s = st.sidebar.slider("Simulation duration (s)", 600, 3600, 1600, step=200)
step_s = st.sidebar.slider("Timestep (s)", 4, 20, 8, step=2)
n_debris = st.sidebar.slider("Debris count", 50, 800, 220, step=50)
threshold_km = st.sidebar.slider("Close-approach threshold (km)", 10.0, 200.0, 30.0, step=5.0)
train_target = st.sidebar.number_input("Min dataset size for training", min_value=50, max_value=2000, value=300, step=50)

run_sim = st.sidebar.button("▶ Run Simulation")
train_btn = st.sidebar.button("⚙ Train Models")
zip_btn = st.sidebar.button("📦 Package & Download ZIP")

st.sidebar.markdown("---")
st.sidebar.markdown("**PPTX (uploaded)**")
if Path(PPTX_PATH).exists():
    st.sidebar.markdown(f"[Open PPTX file]({PPTX_URL})")
else:
    st.sidebar.markdown("_PPTX not found in the environment._")

# ------------------------
# Run Simulation
# ------------------------
if run_sim:
    with st.spinner("Running simulation..."):
        times = build_times(duration_s=duration_s, step_s=step_s)
        tles = safe_get_tles(6)
        sats=[]
        for t in tles:
            pos = propagate_tle(t["line1"], t["line2"], times)
            sats.append({"name": t["name"], "pos": pos})
        debris = generate_debris_field(times, n_debris=n_debris)
        objects = sats + debris
        events = detect_close_approaches(objects, times, threshold_km=threshold_km)
        st.session_state["sim"] = {"times":times,"sats":sats,"debris":debris,"objects":objects,"events":events}
        st.success(f"Simulation finished — events detected: {len(events)}")

# ------------------------
# Simulation Summary & 3D Plot
# ------------------------
if "sim" in st.session_state:
    sim = st.session_state["sim"]
    st.markdown("### Simulation Summary")
    st.write(f"Satellites: {len(sim['sats'])} • Debris: {len(sim['debris'])} • Timesteps: {len(sim['times'])} • Events: {len(sim['events'])}")
    st.markdown("### 3D Orbit Preview")
    # build a static interactive scene (mid frame)
    frame_idx = min(int(len(sim["times"])/2), 20)
    u = np.linspace(0,2*np.pi,40); v=np.linspace(0,np.pi,20)
    x_sph = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y_sph = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z_sph = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x_sph, y=y_sph, z=z_sph, colorscale=[[0,"#050510"],[1,"#0e1222"]], opacity=0.93, showscale=False))
    colors = ["#00E5FF","#FF6EFF","#FFC300","#8DFF57","#FF5733","#9D4EDD"]
    for si, sat in enumerate(sim["sats"]):
        traj = sat["pos"][:frame_idx+1]
        if np.isnan(traj).all(): continue
        fig.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode="lines", line=dict(color=colors[si%len(colors)], width=3), name=sat["name"]))
        p = sat["pos"][frame_idx]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=6,color=colors[si%len(colors)]), showlegend=False))
    # debris traces (sample)
    for deb in sim["debris"][:120]:
        traj = deb["pos"][:frame_idx+1]
        fig.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode="lines", line=dict(color="rgba(255,255,255,0.14)", width=1), showlegend=False))
    # event markers
    for e in sim["events"]:
        if e["t_idx"]==frame_idx:
            try:
                a = next(o for o in sim["objects"] if o["name"]==e["A"])
                b = next(o for o in sim["objects"] if o["name"]==e["B"])
                pa=a["pos"][frame_idx]; pb=b["pos"][frame_idx]; mid=(np.array(pa)+np.array(pb))/2.0
                prob = 0.9 if e["dist_km"]<5 else 0.45
                color = "#FF1744" if prob>0.6 else "#FF9100"
                fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode="markers+text", marker=dict(size=8,color=color), text=[f"{prob:.2f}"], textposition="top center"))
            except StopIteration:
                pass
    fig.update_layout(template="plotly_dark", scene=dict(aspectmode="data"), height=720)
    st.plotly_chart(fig, use_container_width=True)

    # save & download HTML
    save_html = st.button("Save 3D visualization as HTML")
    if save_html:
        orbit_html = VIS_DIR / "orbit_animation_neon.html"
        fig.write_html(str(orbit_html), include_plotlyjs='cdn')
        st.success(f"Saved: {orbit_html.name}")
        with open(orbit_html, "rb") as fh:
            btn = st.download_button(label="⬇ Download 3D HTML", data=fh, file_name=orbit_html.name, mime="text/html")

# ------------------------
# Training
# ------------------------
if train_btn:
    if "sim" not in st.session_state:
        st.warning("Run the simulation first.")
    else:
        sim = st.session_state["sim"]
        events = sim["events"]
        X=[]; y=[]
        for e in events:
            X.append([e["dist_km"], e["rel_speed_km_s"], abs(e["tca_s"])])
            y.append(1 if e["dist_km"]<5.0 else 0)
        X = np.array(X); y = np.array(y)
        X,y = ensure_dataset(X,y,min_samples=train_target)
        scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
        joblib.dump(scaler, MODELS_DIR / "astroguard_scaler.pkl")
        if Xs.shape[0] < 10:
            X_train, X_test, y_train, y_test = Xs, Xs, y, y
        else:
            strat = y if len(np.unique(y))>1 else None
            X_train, X_test, y_train, y_test = train_test_split(Xs,y,test_size=0.25,random_state=42,stratify=strat)

        models = {
            "LogReg": LogisticRegression(max_iter=1200),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "GradBoost": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "NeuralNet": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=1200, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08, use_label_encoder=False, eval_metric="logloss", random_state=42)
        }

        results=[]; trained={}
        with st.spinner("Training models..."):
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else None
                    acc = accuracy_score(y_test, y_pred); f1 = f1_score(y_test, y_pred, zero_division=0)
                    prec = precision_score(y_test, y_pred, zero_division=0); rec = recall_score(y_test, y_pred, zero_division=0)
                    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                    results.append([name, acc, f1, prec, rec, auc])
                    trained[name] = model
                    joblib.dump(model, MODELS_DIR / f"model_{name}.pkl")
                except Exception as e:
                    st.warning(f"Failed {name}: {e}")

            estimators = [(n, trained[n]) for n in trained if hasattr(trained[n],"predict_proba")]
            if len(estimators)>1:
                vc = VotingClassifier(estimators=estimators, voting="soft")
                vc.fit(X_train, y_train)
                joblib.dump(vc, MODELS_DIR / "model_Voting.pkl")
                y_pred = vc.predict(X_test); y_proba = vc.predict_proba(X_test)[:,1]
                results.append(["Voting", accuracy_score(y_test,y_pred), f1_score(y_test,y_pred), precision_score(y_test,y_pred), recall_score(y_test,y_pred), roc_auc_score(y_test,y_proba)])

        df_results = pd.DataFrame(results, columns=["Model","Accuracy","F1","Precision","Recall","AUC"]).sort_values(by="F1", ascending=False)
        df_results.to_csv(MODELS_DIR / "model_results.csv", index=False)
        st.session_state["trained"]=trained
        st.session_state["df_results"]=df_results
        st.success("Training completed.")
        st.write(df_results)

# ------------------------
# Show metrics & allow download for visuals & models
# ------------------------
if "df_results" in st.session_state:
    st.markdown("### Model Metrics")
    df = st.session_state["df_results"]
    col1, col2 = st.columns(2)
    with col1:
        fig_acc = px.bar(df, x="Model", y="Accuracy", color="Model", template="plotly_dark", title="Accuracy")
        st.plotly_chart(fig_acc, use_container_width=True)
        html_acc = VIS_DIR / "model_accuracy.html"; fig_acc.write_html(str(html_acc), include_plotlyjs='cdn')
        with open(html_acc, "rb") as f: st.download_button("⬇ Download Accuracy HTML", f, file_name=html_acc.name, mime="text/html")
    with col2:
        fig_f1 = px.bar(df, x="Model", y="F1", color="Model", template="plotly_dark", title="F1 Scores")
        st.plotly_chart(fig_f1, use_container_width=True)
        html_f1 = VIS_DIR / "model_f1.html"; fig_f1.write_html(str(html_f1), include_plotlyjs='cdn')
        with open(html_f1, "rb") as f: st.download_button("⬇ Download F1 HTML", f, file_name=html_f1.name, mime="text/html")

    st.markdown("#### Confusion Matrices (top 3)")
    top = df["Model"].tolist()[:3]
    for m in top:
        try:
            model = st.session_state["trained"][m]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["Act 0","Act 1"], colorscale="Viridis"))
            fig_cm.update_layout(title=f"CM - {m}", template="plotly_dark", height=360)
            st.plotly_chart(fig_cm, use_container_width=True)
            cm_html = VIS_DIR / f"cm_{m}.html"; fig_cm.write_html(str(cm_html), include_plotlyjs='cdn')
            with open(cm_html, "rb") as f: st.download_button(f"⬇ Download CM {m}", f, file_name=cm_html.name, mime="text/html")
        except Exception as e:
            st.warning(f"Could not render CM for {m}: {e}")

# ------------------------
# Package & Download ZIP
# ------------------------
if zip_btn:
    zip_path = ROOT / "AstroGuard_Streamlit_Submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # include outputs
        for root, _, files in os.walk(OUT_DIR):
            for fname in files:
                fp = os.path.join(root, fname)
                arc = os.path.relpath(fp, OUT_DIR)
                zf.write(fp, arc)
        # include PPTX if present
        if os.path.exists(PPTX_PATH):
            zf.write(PPTX_PATH, os.path.join("ppt", os.path.basename(PPTX_PATH)))
    st.success(f"Packaged to {zip_path.name}")
    with open(zip_path, "rb") as f:
        st.download_button("⬇ Download Submission ZIP", f, file_name=zip_path.name, mime="application/zip")

# Footer
st.markdown("---")
st.markdown("AstroGuard — built for hackathon demos. Use the sidebar: Run Simulation → Train Models → Package ZIP.")
