"""NeuroScan AI - premium Streamlit dashboard.   streamlit run app/app.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import *

st.set_page_config("NeuroScan AI", "🧠", layout="wide")
PAL = {"glioma": "#f43f5e", "meningioma": "#f59e0b", "no_tumor": "#10b981", "pituitary": "#8b5cf6"}
INFO = {"glioma": "Tumour arising from glial cells; often aggressive, needs urgent specialist review.",
        "meningioma": "Usually slow-growing tumour of the meninges; frequently benign.",
        "no_tumor": "No tumour pattern detected in this scan.",
        "pituitary": "Tumour of the pituitary gland; may affect hormone regulation."}
NICE = {"custom_cnn": "Custom CNN", "resnet50": "ResNet50", "mobilenetv2": "MobileNetV2",
        "inceptionv3": "InceptionV3", "efficientnetb0": "EfficientNetB0"}

st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1350px}
.hero{padding:2rem 2.4rem;border-radius:22px;background:radial-gradient(1200px 300px at 0% 0%,#22d3ee33,transparent),
linear-gradient(120deg,#131a30,#1b1447);border:1px solid #ffffff18;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.4rem;background:linear-gradient(90deg,#22d3ee,#a78bfa);-webkit-background-clip:text;color:transparent}
.hero p{margin:.3rem 0 0;color:#aab4cf}
.kpi{padding:1.1rem 1.3rem;border-radius:16px;background:#131a30;border:1px solid #ffffff14;transition:.2s}
.kpi:hover{transform:translateY(-3px);border-color:#22d3ee88}
.kpi b{display:block;font-size:1.8rem;color:#fff}.kpi span{color:#8f9bb8;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase}
.res{padding:1.4rem;border-radius:18px;background:#131a30;border-left:6px solid var(--c)}
.res h2{margin:0;color:var(--c)}
.stTabs [data-baseweb="tab"]{font-weight:600;padding:.6rem 1.1rem}
</style>""", unsafe_allow_html=True)

def rj(p):
    p = Path(p); return json.load(open(p)) if p.exists() else None

def style(f, h=380):
    f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=h, margin=dict(l=10, r=10, t=40, b=10)); return f

def kpi(col, label, val):
    col.markdown(f'<div class="kpi"><span>{label}</span><b>{val}</b></div>', unsafe_allow_html=True)

@st.cache_resource
def get_model(name):
    from tensorflow import keras
    return keras.models.load_model(MODELS / f"{name}.h5", compile=False)

avail = sorted(p.stem for p in MODELS.glob("*.h5"))
best = (rj(REPORTS / "best_model.json") or {}).get("best")
cmp_df = pd.read_csv(REPORTS / "comparison.csv") if (REPORTS / "comparison.csv").exists() else None
stats = rj(REPORTS / "dataset_stats.json")

st.markdown('<div class="hero"><h1>🧠 NeuroScan AI</h1><p>Brain tumour MRI classification · custom CNN vs transfer learning · '
            'explainable predictions with Grad-CAM</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", avail, index=avail.index(best) if best in avail else 0,
                              format_func=lambda n: NICE[n] + (" ⭐" if n == best else "")) if avail else None
    thr = st.slider("Low-confidence warning below", 0.5, 0.95, 0.70)
    st.caption("⚠️ Research/education tool - not a medical device. Not for clinical decisions.")

t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🩻 Diagnose", "📈 Performance", "🏆 Comparison", "🗂 Dataset"])

with t1:
    if cmp_df is None:
        st.info("No results yet. Run `python -m src.train` then refresh (see README).")
    else:
        b = cmp_df.iloc[0]; c = st.columns(5)
        kpi(c[0], "Best model", NICE[b.model]); kpi(c[1], "Test accuracy", f"{b.accuracy:.2%}")
        kpi(c[2], "Macro F1", f"{b.f1_macro:.3f}"); kpi(c[3], "Macro AUC", f"{b.auc_macro:.3f}")
        kpi(c[4], "Latency", f"{b.latency_ms:.1f} ms/img")
        st.markdown("&nbsp;")
        d = cmp_df.assign(name=cmp_df.model.map(NICE))
        f = px.bar(d, x="name", y="accuracy", color="accuracy", color_continuous_scale="Tealgrn", text_auto=".2%",
                   title="Test accuracy by model"); f.update_yaxes(range=[max(0, d.accuracy.min() - .1), 1])
        st.plotly_chart(style(f, 340), use_container_width=True)

with t2:
    if not avail:
        st.warning("No trained .h5 models found in /models. Train first.")
    else:
        L, R = st.columns([1, 1.2])
        up = L.file_uploader("Upload a brain MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up).convert("RGB"); arr = np.asarray(img.resize(IMG), dtype="float32") / 255.0
            L.image(img, caption="Uploaded scan", use_container_width=True)
            m = get_model(model_name); p = m.predict(arr[None], verbose=0)[0]; k = int(p.argmax())
            cls = CLASSES[k]
            R.markdown(f'<div class="res" style="--c:{PAL[cls]}"><h2>{cls.replace("_", " ").title()}</h2>'
                       f'<p>{INFO[cls]}</p></div>', unsafe_allow_html=True)
            if p[k] < thr: R.warning("Low confidence - recommend expert review.")
            g = go.Figure(go.Indicator(mode="gauge+number", value=p[k] * 100, number={"suffix": "%"},
                          title={"text": "Confidence"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": PAL[cls]}}))
            a, b_ = R.columns(2); a.plotly_chart(style(g, 250), use_container_width=True)
            pf = px.bar(x=p, y=[c.replace("_", " ") for c in CLASSES], orientation="h", color=CLASSES,
                        color_discrete_map={c.replace("_", " "): PAL[c] for c in CLASSES}, text=[f"{v:.1%}" for v in p])
            pf.update_layout(showlegend=False, xaxis_range=[0, 1], title="Class probabilities")
            b_.plotly_chart(style(pf, 250), use_container_width=True)
            try:
                import matplotlib; from src.gradcam import gradcam
                h = np.asarray(Image.fromarray((gradcam(m, arr[None], k) * 255).astype("uint8")).resize(IMG)) / 255.0
                ov = 0.55 * arr + 0.45 * matplotlib.colormaps["jet"](h)[..., :3]
                R.image(np.clip(ov, 0, 1), caption="Grad-CAM: regions driving the prediction", use_container_width=True)
            except Exception as e:
                R.caption(f"Grad-CAM unavailable for this model ({type(e).__name__}).")
            R.download_button("⬇ Download probabilities (CSV)", pd.DataFrame({"class": CLASSES, "probability": p}).to_csv(index=False),
                              "prediction.csv")

with t3:
    mt = rj(REPORTS / f"{model_name}_metrics.json") if model_name else None
    if not mt:
        st.info("Run evaluation to see metrics.")
    else:
        c = st.columns(4)
        for col, (lab, key) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision_macro"), ("Recall", "recall_macro"), ("F1", "f1_macro")]):
            kpi(col, lab, f"{mt[key]:.2%}")
        a, b = st.columns(2)
        cm = np.array(mt["confusion"])
        f = px.imshow(cm, x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix",
                      labels=dict(x="Predicted", y="Actual")); a.plotly_chart(style(f), use_container_width=True)
        pc = pd.DataFrame(mt["per_class"]).T.reset_index().melt("index", ["precision", "recall", "f1-score"])
        b.plotly_chart(style(px.bar(pc, x="index", y="value", color="variable", barmode="group", title="Per-class metrics",
                                    labels={"index": "", "value": ""})), use_container_width=True)
        a, b = st.columns(2)
        fr = go.Figure()
        for cl, r in mt["roc"].items():
            fr.add_scatter(x=r["fpr"], y=r["tpr"], name=f'{cl} (AUC {r["auc"]:.3f})', line=dict(color=PAL[cl]))
        fr.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"), showlegend=False)
        a.plotly_chart(style(fr.update_layout(title="ROC curves (one-vs-rest)")), use_container_width=True)
        hs = rj(REPORTS / f"{model_name}_history.json")
        if hs:
            fh = go.Figure()
            for k, col in [("accuracy", "#22d3ee"), ("val_accuracy", "#a78bfa"), ("loss", "#f43f5e"), ("val_loss", "#f59e0b")]:
                fh.add_scatter(y=hs[k], name=k, line=dict(color=col, dash="dot" if "val" in k else "solid"))
            b.plotly_chart(style(fh.update_layout(title="Training history")), use_container_width=True)

with t4:
    if cmp_df is None:
        st.info("Train at least one model to unlock the comparison.")
    else:
        d = cmp_df.assign(name=cmp_df.model.map(NICE))
        st.subheader("Leaderboard")
        st.dataframe(d.drop(columns="model").set_index("name").style.format(
            {c: "{:.3f}" for c in d.columns if c not in ("name", "model", "params")} | {"params": "{:,}"})
            .background_gradient(cmap="Blues", subset=["accuracy", "f1_macro", "auc_macro", "score"]), use_container_width=True)
        a, b = st.columns(2)
        fs = px.scatter(d, x="latency_ms", y="f1_macro", size="size_mb", color="name", title="Accuracy vs speed (bubble = file size MB)")
        a.plotly_chart(style(fs), use_container_width=True)
        fr = go.Figure()
        for _, r in d.iterrows():
            v = [r.accuracy, r.precision_macro, r.recall_macro, r.f1_macro, r.auc_macro]
            fr.add_scatterpolar(r=v + v[:1], theta=["Accuracy", "Precision", "Recall", "F1", "AUC", "Accuracy"], name=r["name"], fill="toself", opacity=.55)
        b.plotly_chart(style(fr.update_layout(title="Metric radar")), use_container_width=True)
        st.success(f"🏆 Recommended for deployment: **{NICE[d.iloc[0].model]}** (70% macro-F1, 15% speed, 15% size).")

with t5:
    if not stats:
        st.info("Run `python -m src.data` to generate dataset statistics.")
    else:
        rows = [{"split": s, "class": c, "images": n} for s, cs in stats["counts"].items() for c, n in cs.items()]
        a, b = st.columns(2)
        a.plotly_chart(style(px.bar(rows, x="split", y="images", color="class", color_discrete_map=PAL, title="Class balance per split")), use_container_width=True)
        tr = stats["counts"]["train"]
        b.plotly_chart(style(px.pie(names=list(tr), values=list(tr.values()), hole=.6, color=list(tr), color_discrete_map=PAL, title="Training distribution")), use_container_width=True)
        st.caption("Most common image resolutions: " + ", ".join(f"{k} ({v})" for k, v in stats["sizes"].items()) + " → all resized to 224×224.")
        if (DATA / "train").exists():
            for cl in CLASSES:
                st.markdown(f"**{cl}**"); cols = st.columns(6)
                for col, p in zip(cols, sorted((DATA / "train" / cl).glob("*"))[:6]): col.image(str(p), use_container_width=True)
