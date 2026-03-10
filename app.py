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
import os
import gdown
import tempfile

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="NoCap — Deepfake Detection",
    page_icon="https://img.icons8.com/fluency/48/no-camera.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    background-color: #080810;
    color: #e8e8f0;
    font-family: 'DM Sans', sans-serif;
}

/* ── Noise overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; }

/* ── Hero ── */
.hero-wrap {
    position: relative;
    padding: 64px 0 48px 0;
    text-align: center;
    overflow: hidden;
}
.hero-glow {
    position: absolute;
    top: -40px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(255,50,90,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin-bottom: 16px;
}
.logo-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #ff2d55, #ff6b35);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 30px rgba(255,45,85,0.4);
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 6px;
    background: linear-gradient(135deg, #ffffff 0%, #ff2d55 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #444;
    margin-top: 12px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,45,85,0.08);
    border: 1px solid rgba(255,45,85,0.2);
    border-radius: 99px;
    padding: 6px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #ff2d55;
    letter-spacing: 2px;
    margin-top: 20px;
}
.pulse {
    width: 6px; height: 6px;
    background: #ff2d55;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,45,85,0.2), transparent);
    margin: 32px 0;
}

/* ── Upload zone ── */
.upload-zone {
    background: rgba(255,255,255,0.02);
    border: 1.5px dashed rgba(255,45,85,0.25);
    border-radius: 20px;
    padding: 48px 32px;
    text-align: center;
    transition: all 0.3s;
}
.upload-icon {
    font-size: 3rem;
    margin-bottom: 16px;
    opacity: 0.6;
}
.upload-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #ccc;
    margin-bottom: 8px;
}
.upload-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Verdict ── */
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,45,85,0.08), rgba(255,45,85,0.03));
    border: 1px solid rgba(255,45,85,0.3);
    border-radius: 20px;
    padding: 36px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-fake::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff2d55, transparent);
}
.verdict-real {
    background: linear-gradient(135deg, rgba(0,210,120,0.08), rgba(0,210,120,0.03));
    border: 1px solid rgba(0,210,120,0.3);
    border-radius: 20px;
    padding: 36px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-real::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d278, transparent);
}
.verdict-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 12px;
}
.verdict-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 8px;
    line-height: 1;
}
.verdict-fake .verdict-text { color: #ff2d55; }
.verdict-real .verdict-text { color: #00d278; }
.verdict-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #666;
    margin-top: 12px;
}

/* ── Score cards ── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 16px;
}
.score-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
}
.score-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 10px;
}
.score-card-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 2px;
    color: #ff6b35;
}
.score-card-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #333;
    margin-top: 4px;
    letter-spacing: 1px;
}

/* ── Gradcam section ── */
.gradcam-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 4px;
    color: #888;
    margin: 28px 0 16px 0;
}
.gradcam-caption {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #333;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
    margin-top: 8px;
}

/* ── About tab ── */
.about-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.about-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #ff2d55;
    margin-bottom: 14px;
}
.about-card p {
    font-size: 0.88rem;
    color: #888;
    line-height: 1.8;
}
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
    color: #aaa;
}
.pipeline-row:last-child { border-bottom: none; }
.pipeline-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #ff2d55;
    min-width: 22px;
}
.tag {
    display: inline-block;
    background: rgba(255,45,85,0.08);
    border: 1px solid rgba(255,45,85,0.15);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #ff6b35;
    margin: 3px;
    letter-spacing: 1px;
}
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-label { color: #666; font-family: 'DM Mono', monospace; font-size: 0.72rem; }
.metric-value { color: #00d278; font-weight: 600; font-family: 'DM Mono', monospace; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #444 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #ff2d55 !important;
    border-bottom: 2px solid #ff2d55 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 32px; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #ff2d55 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(255,45,85,0.25) !important;
    border-radius: 16px !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: #ff2d55 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DRIVE  = '/content/drive/MyDrive/NoCap-Deepfake'
MODEL_ID    = "13goF5n1TXIOtaimlNZPQ2s4mAr3uTJCp"
THRESHOLD   = 0.94

# ── Model ───────────────────────────────────────────────────
class EfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b4(weights=None)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_features, 1))
        self.model = base

    def forward(self, x):
        return self.model(x)

    @property
    def features(self):
        return self.model.features

@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)
    model_path = "models/efficientnet_b4_dfdc.pth"
    if not os.path.exists(model_path):
        # Try Drive first
        drive_path = f"{BASE_DRIVE}/models/checkpoints/efficientnet_b4_dfdc.pth"
        if os.path.exists(drive_path):
            import shutil
            shutil.copy(drive_path, model_path)
        else:
            gdown.download(
                f"https://drive.google.com/uc?id={MODEL_ID}",
                model_path, quiet=True)
    net = EfficientNetB4().to(DEVICE)
    net.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return net, transform

# ── Grad-CAM ────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = None
        self.acts  = None
        model.features[-1].register_forward_hook(
            lambda m,i,o: setattr(self,'acts',o.detach()))
        model.features[-1].register_full_backward_hook(
            lambda m,gi,go: setattr(self,'grads',go[0].detach()))

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
    cam_pil = Image.fromarray((cam*255).astype(np.uint8)).resize((224,224))
    cam_arr = np.array(cam_pil) / 255.0
    heatmap = (cm.inferno(cam_arr)[:,:,:3] * 255).astype(np.uint8)
    overlay = (0.55*img_np + 0.45*heatmap).astype(np.uint8)
    return heatmap, overlay

# ── Video processing ─────────────────────────────────────────
def extract_frames(video_path, max_frames=20):
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release(); return []
    indices = np.linspace(0, total-1, min(max_frames, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = frame[:,:,::-1].copy()
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

def detect_faces(frames):
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(image_size=224, margin=20, min_face_size=40,
                      thresholds=[0.6,0.7,0.7], post_process=False, device=DEVICE)
        faces = []
        for frame in frames:
            try:
                face = mtcnn(frame)
                if face is not None:
                    face_np = face.permute(1,2,0).numpy()
                    face_np = ((face_np - face_np.min()) /
                               (face_np.max() - face_np.min() + 1e-8) * 255).astype(np.uint8)
                    faces.append(Image.fromarray(face_np))
            except: continue
        return faces if faces else [f.resize((224,224)) for f in frames]
    except:
        return [f.resize((224,224)) for f in frames]

def analyse_video(video_path, model, transform):
    frames = extract_frames(video_path)
    if not frames:
        return None, None, "No frames extracted from video"

    faces = detect_faces(frames)
    if not faces:
        return None, None, "No faces detected in video"

    scores = []
    progress = st.progress(0)
    for i, face in enumerate(faces):
        inp = transform(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            score = torch.sigmoid(model(inp)).item()
        scores.append(score)
        progress.progress((i+1)/len(faces))
    progress.empty()

    avg_score  = float(np.mean(scores))
    # DEBUG — remove after threshold is confirmed
    st.info(f"Debug scores → Min: {min(scores):.3f} | Max: {max(scores):.3f} | Avg: {avg_score:.3f}")
    verdict    = "FAKE" if avg_score > THRESHOLD else "REAL"
    confidence = avg_score*100 if verdict=="FAKE" else (1-avg_score)*100
    confidence = min(round(confidence, 1), 99.9)

    # Grad-CAM on most suspicious face
    gradcam_result = None
    try:
        gradcam  = GradCAM(model)
        best_idx = int(np.argmax(scores))
        best_face = faces[best_idx]
        inp = transform(best_face).unsqueeze(0).to(DEVICE)
        inp.requires_grad = True
        cam  = gradcam.generate(inp)
        img_np = np.array(best_face.resize((224,224)))
        heatmap, overlay = apply_heatmap(cam, img_np)
        gradcam_result = (img_np, heatmap, overlay)
    except: pass

    result = {
        "verdict":    verdict,
        "confidence": confidence,
        "avg_score":  round(avg_score, 4),
        "max_score":  round(float(max(scores)), 4),
        "min_score":  round(float(min(scores)), 4),
        "faces":      len(faces),
        "frames":     len(frames),
    }
    return result, gradcam_result, None

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-logo">
        <svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="52" height="52" rx="14" fill="url(#logoGrad)"/>
            <path d="M16 20C16 17.8 17.8 16 20 16H32C34.2 16 36 17.8 36 20V32C36 34.2 34.2 36 32 36H20C17.8 36 16 34.2 16 32V20Z" stroke="white" stroke-width="2" fill="none"/>
            <circle cx="26" cy="26" r="5" fill="white" opacity="0.9"/>
            <path d="M22 16L18 12M30 16L34 12" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <path d="M14 22L10 20M14 30L10 32" stroke="white" stroke-width="2" stroke-linecap="round"/>
            <line x1="20" y1="20" x2="32" y2="32" stroke="#ff2d55" stroke-width="2.5" stroke-linecap="round"/>
            <defs>
                <linearGradient id="logoGrad" x1="0" y1="0" x2="52" y2="52">
                    <stop offset="0%" stop-color="#ff2d55"/>
                    <stop offset="100%" stop-color="#ff6b35"/>
                </linearGradient>
            </defs>
        </svg>
        <div class="hero-title">NOCAP</div>
    </div>
    <div class="hero-sub">AI-Powered Deepfake Detection &nbsp;·&nbsp; EfficientNet-B4 &nbsp;·&nbsp; 2026</div>
    <div style="display:flex;justify-content:center">
        <div class="hero-badge">
            <div class="pulse"></div>
            SYSTEM ONLINE &nbsp;·&nbsp; AUC 0.9507
        </div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["ANALYSE VIDEO", "ABOUT"])

# ── Tab 1 — Analyse ──────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload video",
        type=["mp4","avi","mov","mkv"],
        label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                    <path d="M24 8L24 32M24 8L16 16M24 8L32 16" stroke="#ff2d55" stroke-width="2.5"
                          stroke-linecap="round" stroke-linejoin="round" opacity="0.6"/>
                    <path d="M8 36H40" stroke="#333" stroke-width="2" stroke-linecap="round"/>
                    <rect x="8" y="36" width="32" height="4" rx="2" fill="#ff2d55" opacity="0.1"/>
                </svg>
            </div>
            <div class="upload-title">Drop your video here</div>
            <div class="upload-sub">MP4 · AVI · MOV · MKV &nbsp;·&nbsp; Any resolution</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.video(tmp_path)

        with col2:
            with st.spinner("Loading model..."):
                model, transform = load_model()

            with st.spinner("Analysing video..."):
                result, gradcam_imgs, error = analyse_video(tmp_path, model, transform)

            if error:
                st.error(f"Analysis failed: {error}")
            elif result:
                # Verdict
                v_class = "verdict-fake" if result['verdict']=="FAKE" else "verdict-real"
                icon_svg = """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                    <path d="M14 4L17 10L24 11L19 16L20 23L14 20L8 23L9 16L4 11L11 10L14 4Z"
                    fill="#ff2d55" opacity="0.9"/>
                </svg>""" if result['verdict']=="FAKE" else """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                    <path d="M6 14L12 20L22 8" stroke="#00d278" stroke-width="3"
                    stroke-linecap="round" stroke-linejoin="round"/>
                </svg>"""

                st.markdown(f"""
                <div class="{v_class}">
                    <div class="verdict-label">Detection Result</div>
                    <div style="display:flex;align-items:center;justify-content:center;gap:12px">
                        {icon_svg}
                        <div class="verdict-text">{result['verdict']}</div>
                    </div>
                    <div class="verdict-conf">{result['confidence']}% confidence &nbsp;·&nbsp;
                    score {result['avg_score']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="score-grid">
                    <div class="score-card">
                        <div class="score-card-label">Avg Score</div>
                        <div class="score-card-value">{result['avg_score']:.2f}</div>
                        <div class="score-card-sub">threshold 0.45</div>
                    </div>
                    <div class="score-card">
                        <div class="score-card-label">Faces Found</div>
                        <div class="score-card-value">{result['faces']}</div>
                        <div class="score-card-sub">via MTCNN</div>
                    </div>
                    <div class="score-card">
                        <div class="score-card-label">Frames</div>
                        <div class="score-card-value">{result['frames']}</div>
                        <div class="score-card-sub">analysed</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Grad-CAM
        if gradcam_imgs is not None:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="gradcam-title">ACTIVATION MAP</div>', unsafe_allow_html=True)
            g1, g2, g3 = st.columns(3, gap="small")
            with g1:
                st.image(gradcam_imgs[0], use_column_width=True)
                st.markdown('<div class="gradcam-caption">Original Face</div>', unsafe_allow_html=True)
            with g2:
                st.image(gradcam_imgs[1], use_column_width=True)
                st.markdown('<div class="gradcam-caption">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            with g3:
                st.image(gradcam_imgs[2], use_column_width=True)
                st.markdown('<div class="gradcam-caption">Overlay — Suspicious Regions</div>', unsafe_allow_html=True)

        try: os.unlink(tmp_path)
        except: pass

# ── Tab 2 — About ────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">Detection Pipeline</div>
            <div class="pipeline-row"><span class="pipeline-num">01</span>OpenCV — extract 20 evenly spaced frames</div>
            <div class="pipeline-row"><span class="pipeline-num">02</span>MTCNN — detect & crop faces to 224×224</div>
            <div class="pipeline-row"><span class="pipeline-num">03</span>EfficientNet-B4 — score each face (0–1)</div>
            <div class="pipeline-row"><span class="pipeline-num">04</span>Average score — compared to threshold 0.45</div>
            <div class="pipeline-row"><span class="pipeline-num">05</span>Grad-CAM — highlight suspicious regions</div>
        </div>
        <div class="about-card">
            <div class="about-card-title">Model Performance</div>
            <div class="metric-row"><span class="metric-label">Val AUC</span><span class="metric-value">0.9507</span></div>
            <div class="metric-row"><span class="metric-label">Val F1</span><span class="metric-value">0.9188</span></div>
            <div class="metric-row"><span class="metric-label">Val Accuracy</span><span class="metric-value">87.44%</span></div>
            <div class="metric-row"><span class="metric-label">Fake Detection</span><span class="metric-value">91%</span></div>
            <div class="metric-row"><span class="metric-label">Real Detection</span><span class="metric-value">78%</span></div>
            <div class="metric-row"><span class="metric-label">Threshold</span><span class="metric-value">0.45</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">Why EfficientNet-B4</div>
            <p>
                EfficientNet-B4 was fine-tuned directly on the DFDC (DeepFake Detection Challenge)
                dataset — 124,647 face crops from real and manipulated videos. Training on the same
                distribution as inference eliminates domain mismatch, which was the core failure of
                our previous ResNet50 + LSTM architecture.<br><br>
                The model learns spatial artifacts introduced by deepfake generation — boundary
                inconsistencies, texture anomalies, and unnatural frequency patterns that are
                invisible to the human eye but detectable by deep convolutional networks.
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
