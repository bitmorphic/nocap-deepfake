import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import os, gdown, tempfile
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NoCap — Deepfake Detection",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    background-color: #080810;
    color: #e8e8f0;
    font-family: 'DM Sans', sans-serif;
}
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0; opacity: 0.4;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; }

/* Hero */
.hero-wrap { position: relative; padding: 56px 0 40px 0; text-align: center; overflow: hidden; }
.hero-glow {
    position: absolute; top: -40px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(255,50,90,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-logo { display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 16px; }
.hero-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 5rem; letter-spacing: 6px;
    background: linear-gradient(135deg, #ffffff 0%, #ff2d55 50%, #ff6b35 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;
}
.hero-sub { font-family: 'DM Mono', monospace; font-size: 0.72rem; letter-spacing: 4px; text-transform: uppercase; color: #444; margin-top: 12px; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,45,85,0.08); border: 1px solid rgba(255,45,85,0.2);
    border-radius: 99px; padding: 6px 16px;
    font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #ff2d55; letter-spacing: 2px; margin-top: 20px;
}
.pulse { width: 6px; height: 6px; background: #ff2d55; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }

/* Divider */
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,45,85,0.2), transparent); margin: 28px 0; }

/* Verdict cards */
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,45,85,0.08), rgba(255,45,85,0.03));
    border: 1px solid rgba(255,45,85,0.3); border-radius: 20px; padding: 32px 24px;
    text-align: center; position: relative; overflow: hidden;
}
.verdict-fake::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #ff2d55, transparent);
}
.verdict-real {
    background: linear-gradient(135deg, rgba(0,210,120,0.08), rgba(0,210,120,0.03));
    border: 1px solid rgba(0,210,120,0.3); border-radius: 20px; padding: 32px 24px;
    text-align: center; position: relative; overflow: hidden;
}
.verdict-real::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d278, transparent);
}
.verdict-label { font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 4px; text-transform: uppercase; color: #555; margin-bottom: 10px; }
.verdict-text { font-family: 'Bebas Neue', sans-serif; font-size: 4rem; letter-spacing: 8px; line-height: 1; }
.verdict-fake .verdict-text { color: #ff2d55; }
.verdict-real .verdict-text { color: #00d278; }
.verdict-conf { font-family: 'DM Mono', monospace; font-size: 0.82rem; color: #666; margin-top: 10px; }

/* Score grid */
.score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
.score-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 16px; text-align: center; }
.score-card-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 2px; text-transform: uppercase; color: #444; margin-bottom: 8px; }
.score-card-value { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; letter-spacing: 2px; color: #ff6b35; }
.score-card-sub { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: #333; margin-top: 4px; letter-spacing: 1px; }

/* Section title */
.section-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem; letter-spacing: 4px; color: #555; margin: 24px 0 14px 0; }

/* Gradcam */
.gradcam-caption { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #333; letter-spacing: 2px; text-transform: uppercase; text-align: center; margin-top: 6px; }

/* About cards */
.about-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 22px; margin-bottom: 14px; }
.about-card-title { font-family: 'DM Mono', monospace; font-size: 0.66rem; letter-spacing: 3px; text-transform: uppercase; color: #ff2d55; margin-bottom: 12px; }
.about-card p { font-size: 0.86rem; color: #888; line-height: 1.8; }
.pipeline-row { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.84rem; color: #aaa; }
.pipeline-row:last-child { border-bottom: none; }
.pipeline-num { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #ff2d55; min-width: 22px; }
.metric-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.84rem; }
.metric-row:last-child { border-bottom: none; }
.metric-label { color: #666; font-family: 'DM Mono', monospace; font-size: 0.70rem; }
.metric-value { color: #00d278; font-weight: 600; font-family: 'DM Mono', monospace; }
.tag { display: inline-block; background: rgba(255,45,85,0.08); border: 1px solid rgba(255,45,85,0.15); border-radius: 6px; padding: 3px 10px; font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #ff6b35; margin: 3px; letter-spacing: 1px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid rgba(255,255,255,0.06); gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #444 !important; font-family: 'DM Mono', monospace !important; font-size: 0.70rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; padding: 12px 22px !important; border: none !important; }
.stTabs [aria-selected="true"] { color: #ff2d55 !important; border-bottom: 2px solid #ff2d55 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 28px; }
.stProgress > div > div { background: #ff2d55 !important; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,0.02) !important; border: 1.5px dashed rgba(255,45,85,0.25) !important; border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DRIVE = '/content/drive/MyDrive/NoCap-Deepfake'
MODEL_ID   = "13goF5n1TXIOtaimlNZPQ2s4mAr3uTJCp"

# ── Smart dual-condition threshold ───────────────────────────
# A video is FAKE only when BOTH are true:
# 1. avg score across all faces  > AVG_THRESHOLD
# 2. % of frames above FRAME_THRESHOLD > FAKE_FRAME_RATIO
# This prevents a few spike frames from causing false FAKEs
AVG_THRESHOLD    = 0.90
FRAME_THRESHOLD  = 0.80
FAKE_FRAME_RATIO = 0.55

# ── Model definition ─────────────────────────────────────────
class EfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b4(weights=None)
        in_f = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, 1))
        self.model = base

    def forward(self, x): return self.model(x)

    @property
    def features(self): return self.model.features

@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)
    path = "models/efficientnet_b4_dfdc.pth"
    if not os.path.exists(path):
        dp = f"{BASE_DRIVE}/models/checkpoints/efficientnet_b4_dfdc.pth"
        if os.path.exists(dp):
            import shutil; shutil.copy(dp, path)
        else:
            gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", path, quiet=True)
    net = EfficientNetB4().to(DEVICE)
    net.model.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return net, tfm

@st.cache_resource
def load_mtcnn():
    try:
        from facenet_pytorch import MTCNN
        return MTCNN(image_size=224, margin=20, min_face_size=40,
                     thresholds=[0.6,0.7,0.7], post_process=False, device=DEVICE)
    except: return None

# ── Grad-CAM ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model; self.grads = None; self.acts = None
        model.features[-1].register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        model.features[-1].register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))

    def generate(self, inp):
        self.model.eval()
        out = self.model(inp)
        self.model.zero_grad()
        out[0,0].backward()
        w   = self.grads[0].mean(dim=(1,2))
        cam = F.relu((w[:,None,None] * self.acts[0]).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

def apply_heatmap(cam, img_np):
    cam_arr = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((224,224))) / 255.0
    heatmap = (cm.inferno(cam_arr)[:,:,:3] * 255).astype(np.uint8)
    overlay = (0.55*img_np + 0.45*heatmap).astype(np.uint8)
    return heatmap, overlay

# ── Helpers ──────────────────────────────────────────────────
def crop_face(pil_img, mtcnn):
    if mtcnn is None: return pil_img.resize((224,224))
    try:
        face = mtcnn(pil_img)
        if face is not None:
            fn = face.permute(1,2,0).numpy()
            fn = ((fn - fn.min())/(fn.max()-fn.min()+1e-8)*255).astype(np.uint8)
            return Image.fromarray(fn)
    except: pass
    return pil_img.resize((224,224))

def score_face(face_img, model, transform):
    inp = transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(inp)).item()

def smart_verdict(scores):
    avg        = float(np.mean(scores))
    fake_ratio = float(np.mean([s > FRAME_THRESHOLD for s in scores]))
    is_fake    = avg > AVG_THRESHOLD and fake_ratio > FAKE_FRAME_RATIO
    verdict    = "FAKE" if is_fake else "REAL"
    conf       = avg*100 if is_fake else (1-avg)*100
    return verdict, min(round(conf,1), 99.9), round(avg,4), round(fake_ratio,3)

def extract_frames(video_path, max_frames=20):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); return []
    idxs  = np.linspace(0, total-1, min(max_frames,total), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret: frames.append(Image.fromarray(frame[:,:,::-1].copy()))
    cap.release()
    return frames

def run_gradcam(model, transform, face):
    try:
        gc  = GradCAM(model)
        inp = transform(face).unsqueeze(0).to(DEVICE)
        inp.requires_grad = True
        cam = gc.generate(inp)
        img_np = np.array(face.resize((224,224)))
        return apply_heatmap(cam, img_np)
    except: return None

def render_verdict_card(verdict, confidence, avg_score, fake_ratio, faces_count):
    v_class  = "verdict-fake" if verdict=="FAKE" else "verdict-real"
    icon_svg = ("""<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 4L17 10L24 11L19 16L20 23L14 20L8 23L9 16L4 11L11 10L14 4Z" fill="#ff2d55" opacity="0.9"/>
        </svg>""" if verdict=="FAKE" else
        """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M6 14L12 20L22 8" stroke="#00d278" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""")
    st.markdown(f"""
    <div class="{v_class}">
        <div class="verdict-label">Detection Result</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:12px">
            {icon_svg}<div class="verdict-text">{verdict}</div>
        </div>
        <div class="verdict-conf">{confidence}% confidence &nbsp;·&nbsp; score {avg_score}</div>
    </div>
    <div class="score-grid">
        <div class="score-card">
            <div class="score-card-label">Avg Score</div>
            <div class="score-card-value">{avg_score:.2f}</div>
            <div class="score-card-sub">threshold {AVG_THRESHOLD}</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Fake Frames</div>
            <div class="score-card-value">{round(fake_ratio*100)}%</div>
            <div class="score-card-sub">above {FRAME_THRESHOLD}</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Faces</div>
            <div class="score-card-value">{faces_count}</div>
            <div class="score-card-sub">via MTCNN</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-logo">
        <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
            <rect width="52" height="52" rx="14" fill="url(#lg)"/>
            <path d="M16 20C16 17.8 17.8 16 20 16H32C34.2 16 36 17.8 36 20V32C36 34.2 34.2 36 32 36H20C17.8 36 16 34.2 16 32V20Z" stroke="white" stroke-width="2" fill="none"/>
            <circle cx="26" cy="26" r="5" fill="white" opacity="0.9"/>
            <path d="M22 16L18 12M30 16L34 12" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <path d="M14 22L10 20M14 30L10 32" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <line x1="20" y1="20" x2="32" y2="32" stroke="#ff2d55" stroke-width="2.5" stroke-linecap="round"/>
            <defs><linearGradient id="lg" x1="0" y1="0" x2="52" y2="52">
                <stop offset="0%" stop-color="#ff2d55"/><stop offset="100%" stop-color="#ff6b35"/>
            </linearGradient></defs>
        </svg>
        <div class="hero-title">NOCAP</div>
    </div>
    <div class="hero-sub">AI-Powered Deepfake Detection &nbsp;·&nbsp; EfficientNet-B4 &nbsp;·&nbsp; 2026</div>
    <div style="display:flex;justify-content:center">
        <div class="hero-badge"><div class="pulse"></div>SYSTEM ONLINE &nbsp;·&nbsp; AUC 0.9507</div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["ANALYSE VIDEO", "WEBCAM", "ABOUT"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — VIDEO UPLOAD
# ══════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);border:1.5px dashed rgba(255,45,85,0.25);
                    border-radius:20px;padding:48px 32px;text-align:center;">
            <div style="font-size:1.1rem;font-weight:600;color:#ccc;margin-bottom:8px;">Drop your video here</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:3px;
                        text-transform:uppercase;color:#444;">MP4 · AVI · MOV · MKV &nbsp;·&nbsp; Any resolution</div>
        </div>""", unsafe_allow_html=True)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded.read()); tmp_path = tmp.name

        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            st.video(tmp_path)
        with col2:
            with st.spinner("Loading model..."):
                model, transform = load_model()
                mtcnn = load_mtcnn()

            frames = extract_frames(tmp_path)
            if not frames:
                st.error("No frames extracted from video.")
            else:
                faces  = []
                scores = []
                prog   = st.progress(0, text="Analysing frames...")
                for i, frame in enumerate(frames):
                    face  = crop_face(frame, mtcnn)
                    score = score_face(face, model, transform)
                    faces.append(face); scores.append(score)
                    prog.progress((i+1)/len(frames), text=f"Frame {i+1}/{len(frames)}")
                prog.empty()

                verdict, confidence, avg_score, fake_ratio = smart_verdict(scores)
                render_verdict_card(verdict, confidence, avg_score, fake_ratio, len(faces))

        # Frame chart
        if 'scores' in dir() and scores:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">FRAME-BY-FRAME SCORES</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({
                "Fake Score": [round(s,4) for s in scores],
                "Threshold":  [AVG_THRESHOLD]*len(scores),
            }, index=[f"F{i+1}" for i in range(len(scores))])
            st.line_chart(chart_df, color=["#ff2d55","#333344"])
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">
                Red = per-frame fake score &nbsp;·&nbsp; Gray = threshold ({AVG_THRESHOLD}) &nbsp;·&nbsp; Spikes = suspicious moments
            </div>""", unsafe_allow_html=True)

        # Grad-CAM
        if 'faces' in dir() and faces and 'scores' in dir() and scores:
            gc_imgs = run_gradcam(model, transform, faces[int(np.argmax(scores))])
            if gc_imgs:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">ACTIVATION MAP</div>', unsafe_allow_html=True)
                g1,g2,g3 = st.columns(3, gap="small")
                img_np = np.array(faces[int(np.argmax(scores))].resize((224,224)))
                with g1: st.image(img_np, use_column_width=True); st.markdown('<div class="gradcam-caption">Original Face</div>', unsafe_allow_html=True)
                with g2: st.image(gc_imgs[0], use_column_width=True); st.markdown('<div class="gradcam-caption">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
                with g3: st.image(gc_imgs[1], use_column_width=True); st.markdown('<div class="gradcam-caption">Overlay</div>', unsafe_allow_html=True)

        try: os.unlink(tmp_path)
        except: pass

# ══════════════════════════════════════════════════════════════
# TAB 2 — WEBCAM
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#555;
                letter-spacing:3px;text-transform:uppercase;margin-bottom:20px;">
        Point your camera at a face and click capture
    </div>""", unsafe_allow_html=True)

    cam_col, res_col = st.columns([1,1], gap="large")

    with cam_col:
        webcam_img = st.camera_input("", label_visibility="collapsed")

    with res_col:
        if webcam_img is None:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                        border-radius:16px;padding:60px 24px;text-align:center;margin-top:8px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#333;
                            letter-spacing:3px;text-transform:uppercase;">Waiting for capture...</div>
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Analysing..."):
                model, transform = load_model()
                mtcnn = load_mtcnn()
                pil_img  = Image.open(webcam_img).convert("RGB")
                face     = crop_face(pil_img, mtcnn)
                score    = score_face(face, model, transform)
                is_fake  = score > AVG_THRESHOLD
                verdict  = "FAKE" if is_fake else "REAL"
                conf     = score*100 if is_fake else (1-score)*100
                conf     = min(round(conf,1), 99.9)

            v_class  = "verdict-fake" if verdict=="FAKE" else "verdict-real"
            icon_svg = ("""<svg width="24" height="24" viewBox="0 0 28 28" fill="none">
                <path d="M14 4L17 10L24 11L19 16L20 23L14 20L8 23L9 16L4 11L11 10L14 4Z" fill="#ff2d55" opacity="0.9"/>
                </svg>""" if verdict=="FAKE" else
                """<svg width="24" height="24" viewBox="0 0 28 28" fill="none">
                <path d="M6 14L12 20L22 8" stroke="#00d278" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>""")

            st.markdown(f"""
            <div class="{v_class}" style="margin-top:8px;">
                <div class="verdict-label">Webcam Result</div>
                <div style="display:flex;align-items:center;justify-content:center;gap:10px">
                    {icon_svg}<div class="verdict-text" style="font-size:3rem;">{verdict}</div>
                </div>
                <div class="verdict-conf">{conf}% confidence &nbsp;·&nbsp; raw score {round(score,4)}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider" style="margin:16px 0"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:1rem;margin-bottom:12px;">ACTIVATION MAP</div>', unsafe_allow_html=True)
            gc_imgs = run_gradcam(model, transform, face)
            if gc_imgs:
                w1,w2,w3 = st.columns(3, gap="small")
                img_np = np.array(face.resize((224,224)))
                with w1: st.image(img_np, use_column_width=True); st.markdown('<div class="gradcam-caption">Captured Face</div>', unsafe_allow_html=True)
                with w2: st.image(gc_imgs[0], use_column_width=True); st.markdown('<div class="gradcam-caption">Heatmap</div>', unsafe_allow_html=True)
                with w3: st.image(gc_imgs[1], use_column_width=True); st.markdown('<div class="gradcam-caption">Overlay</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"""
        <div class="about-card">
            <div class="about-card-title">Detection Pipeline</div>
            <div class="pipeline-row"><span class="pipeline-num">01</span>OpenCV — extract 20 evenly spaced frames</div>
            <div class="pipeline-row"><span class="pipeline-num">02</span>MTCNN — detect & crop faces to 224×224</div>
            <div class="pipeline-row"><span class="pipeline-num">03</span>EfficientNet-B4 — score each face (0–1)</div>
            <div class="pipeline-row"><span class="pipeline-num">04</span>Dual threshold — avg &gt; {AVG_THRESHOLD} AND {int(FAKE_FRAME_RATIO*100)}% frames &gt; {FRAME_THRESHOLD}</div>
            <div class="pipeline-row"><span class="pipeline-num">05</span>Grad-CAM — highlight suspicious regions</div>
        </div>
        <div class="about-card">
            <div class="about-card-title">Model Performance</div>
            <div class="metric-row"><span class="metric-label">Val AUC</span><span class="metric-value">0.9507</span></div>
            <div class="metric-row"><span class="metric-label">Val F1</span><span class="metric-value">0.9188</span></div>
            <div class="metric-row"><span class="metric-label">Val Accuracy</span><span class="metric-value">87.44%</span></div>
            <div class="metric-row"><span class="metric-label">Fake Detection</span><span class="metric-value">91%</span></div>
            <div class="metric-row"><span class="metric-label">Real Detection</span><span class="metric-value">78%</span></div>
            <div class="metric-row"><span class="metric-label">Avg Threshold</span><span class="metric-value">{AVG_THRESHOLD}</span></div>
            <div class="metric-row"><span class="metric-label">Frame Threshold</span><span class="metric-value">{FRAME_THRESHOLD}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">Smart Dual-Condition Detection</div>
            <p>
                NoCap uses a dual-condition threshold for robust detection.
                A video is flagged FAKE only when <strong style="color:#e0e0e0">both</strong>
                conditions are true: the average face score exceeds 0.90,
                AND at least 55% of individual frames score above 0.80.<br><br>
                This prevents false positives caused by isolated high-scoring frames
                in real videos. The frame-by-frame chart shows exactly where suspicious
                moments occur in the video timeline.
            </p>
        </div>
        <div class="about-card">
            <div class="about-card-title">Team & Stack</div>
            <p style="margin-bottom:14px">
                <strong style="color:#e0e0e0">Team Quad Logic</strong><br>
                NIT Delhi &nbsp;·&nbsp; B.Tech CSE &nbsp;·&nbsp; 2026
            </p>
            <span class="tag">PyTorch</span>
            <span class="tag">EfficientNet-B4</span>
            <span class="tag">MTCNN</span>
            <span class="tag">Grad-CAM</span>
            <span class="tag">OpenCV</span>
            <span class="tag">DFDC</span>
            <span class="tag">Streamlit</span>
        </div>
        """, unsafe_allow_html=True)
