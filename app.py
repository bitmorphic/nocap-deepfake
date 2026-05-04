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
import os, gdown, tempfile, io
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

/* Risk meter */
.risk-wrap { margin-top: 16px; }
.risk-label-row { font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:3px; text-transform:uppercase; color:#444; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }
.risk-label-name { font-weight:600; }
.risk-bar-bg { background:rgba(255,255,255,0.05); border-radius:99px; height:10px; width:100%; overflow:hidden; }
.risk-bar-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#00d278,#f5a623,#ff2d55); }
.risk-ticks { display:flex; justify-content:space-between; margin-top:5px; }
.risk-tick { font-family:'DM Mono',monospace; font-size:0.52rem; color:#333; letter-spacing:1px; }

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #ff2d55, #ff6b35) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    padding: 10px 24px !important; width: 100% !important;
    box-shadow: 0 0 20px rgba(255,45,85,0.3) !important;
}

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
BASE_DRIVE = '/content/drive/MyDrive/nocap-deepfake'
MODEL_ID   = "1TbdPmcS-VMmp2s1N8ElOwmIAq-18_KlI"

# ── Calibrated score normalisation ──────────────────────────
# Raw model scores are compressed in 0.85-1.0 because DFDC is 79% fake.
# We normalise using empirical anchors so real→~0, fake→~1.
#   REAL_ANCHOR: typical avg score for a known real video
#   FAKE_ANCHOR: typical avg score for a known fake video
# Normalised score = (raw - REAL_ANCHOR) / (FAKE_ANCHOR - REAL_ANCHOR)
# Decision at NORM_THRESHOLD=0.50 on normalised score.
# Guard: 60%+ of frames must also be above FRAME_THRESHOLD (raw).
REAL_ANCHOR      = 0.54
FAKE_ANCHOR      = 0.93
NORM_THRESHOLD   = 0.50
FRAME_THRESHOLD  = 0.73
FAKE_FRAME_RATIO = 0.60

# Risk level scale on normalised 0-1 score
RISK_LEVELS = [
    (0.00, 0.20, "AUTHENTIC",       "#00d278"),
    (0.20, 0.40, "LOW RISK",        "#4ecb71"),
    (0.40, 0.60, "SUSPICIOUS",      "#f5a623"),
    (0.60, 0.80, "HIGH RISK",       "#ff6b35"),
    (0.80, 1.00, "LIKELY DEEPFAKE", "#ff2d55"),
]

def get_risk(norm):
    for lo, hi, label, color in RISK_LEVELS:
        if norm <= hi:
            return label, color
    return RISK_LEVELS[-1][2], RISK_LEVELS[-1][3]

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
    # Fine-tuned Celeb-DF v2 model — different filename forces fresh download
    path = "models/efficientnet_b4_celebdf.pth"
    if not os.path.exists(path):
        # Try Drive first (Colab with mounted Drive)
        dp = f"{BASE_DRIVE}/models/checkpoints/efficientnet_b4_celebdf.pth"
        if os.path.exists(dp):
            import shutil; shutil.copy(dp, path)
        else:
            # Download fine-tuned model from Google Drive
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
        # Resize large frames before MTCNN — high-res frames (1080p+) can confuse MTCNN
        w, h = pil_img.size
        if w > 640:
            pil_small = pil_img.resize((640, int(h * 640 / w)))
        else:
            pil_small = pil_img
        face = mtcnn(pil_small)
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

def normalise_score(raw):
    """Rescale raw model score to 0-1 using empirical anchors.
    Scores at or below REAL_ANCHOR → 0 (definitely real)
    Scores at or above FAKE_ANCHOR → 1 (definitely fake)
    """
    norm = (raw - REAL_ANCHOR) / (FAKE_ANCHOR - REAL_ANCHOR)
    return float(np.clip(norm, 0.0, 1.0))

def smart_verdict(scores):
    avg        = float(np.mean(scores))
    norm_avg   = normalise_score(avg)
    fake_ratio = float(np.mean([s > FRAME_THRESHOLD for s in scores]))
    # FAKE only when normalised score passes threshold AND frame ratio confirms
    is_fake    = norm_avg > NORM_THRESHOLD and fake_ratio > FAKE_FRAME_RATIO
    verdict    = "FAKE" if is_fake else "REAL"
    # Confidence shown as normalised score * 100
    conf       = norm_avg * 100 if is_fake else (1 - norm_avg) * 100
    return verdict, min(round(conf, 1), 99.9), round(avg, 4), round(fake_ratio, 3), round(norm_avg, 3)

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

def render_verdict_card(verdict, confidence, avg_score, fake_ratio, norm_score, faces_count):
    risk_label, risk_color = get_risk(norm_score)
    v_class  = "verdict-fake" if verdict=="FAKE" else "verdict-real"
    icon_svg = ("""<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 4L17 10L24 11L19 16L20 23L14 20L8 23L9 16L4 11L11 10L14 4Z" fill="#ff2d55" opacity="0.9"/>
        </svg>""" if verdict=="FAKE" else
        """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M6 14L12 20L22 8" stroke="#00d278" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""")
    bar_pct = round(norm_score * 100, 1)
    st.markdown(f"""
    <div class="{v_class}">
        <div class="verdict-label">Detection Result</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:12px">
            {icon_svg}<div class="verdict-text">{verdict}</div>
        </div>
        <div class="verdict-conf">{confidence}% confidence &nbsp;·&nbsp; score {avg_score}</div>
    </div>
    <div class="risk-wrap">
        <div class="risk-label-row">
            <span>Risk Level</span>
            <span class="risk-label-name" style="color:{risk_color}">{risk_label}</span>
        </div>
        <div class="risk-bar-bg">
            <div class="risk-bar-fill" style="width:{bar_pct}%"></div>
        </div>
        <div class="risk-ticks">
            <span class="risk-tick" style="color:#00d278">AUTHENTIC</span>
            <span class="risk-tick" style="color:#f5a623">SUSPICIOUS</span>
            <span class="risk-tick" style="color:#ff2d55">DEEPFAKE</span>
        </div>
    </div>
    <div class="score-grid">
        <div class="score-card">
            <div class="score-card-label">Fake Prob</div>
            <div class="score-card-value">{norm_score:.2f}</div>
            <div class="score-card-sub">normalised</div>
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
    return risk_label, risk_color

# ── PDF Report Generator ─────────────────────────────────────
def generate_pdf(filename, verdict, risk_label, risk_color,
                 norm_score, avg_raw, fake_ratio, scores, gradcam_overlay_np):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage, HRFlowable)
    from reportlab.lib.enums import TA_CENTER

    buf  = io.BytesIO()
    W, H = letter
    MW   = W - 1.5*inch
    doc  = SimpleDocTemplate(buf, pagesize=letter,
                              leftMargin=0.75*inch, rightMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle('T', fontName='Helvetica-Bold', fontSize=26,
                              textColor=colors.HexColor('#C0392B'), alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ParagraphStyle('S', fontName='Helvetica', fontSize=9,
                              textColor=colors.HexColor('#888888'), alignment=TA_CENTER, spaceAfter=2)
    h2_s    = ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=12,
                              textColor=colors.HexColor('#1A1A2E'), spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('B', fontName='Helvetica', fontSize=10,
                              textColor=colors.HexColor('#333333'), leading=15, spaceAfter=6)

    # Header
    story.append(Paragraph("NOCAP", title_s))
    story.append(Paragraph("Deepfake Detection Forensic Report", sub_s))
    story.append(Paragraph("EfficientNet-B4  ·  DFDC Dataset  ·  AUC 0.9507", sub_s))
    story.append(HRFlowable(width=MW, thickness=2, color=colors.HexColor('#C0392B'), spaceAfter=12))

    # Meta
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    meta = Table([["File", filename], ["Date", now], ["Model", "EfficientNet-B4 (DFDC fine-tuned)"]],
                 colWidths=[1.4*inch, MW-1.4*inch])
    meta.setStyle(TableStyle([
        ('FONTNAME',  (0,0),(-1,-1), 'Helvetica'),
        ('FONTNAME',  (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1), 9),
        ('TEXTCOLOR', (0,0),(0,-1),  colors.HexColor('#888888')),
        ('TEXTCOLOR', (1,0),(1,-1),  colors.HexColor('#222222')),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#F8F8F8'),colors.white]),
        ('GRID',      (0,0),(-1,-1), 0.5, colors.HexColor('#EEEEEE')),
        ('LEFTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(meta)
    story.append(Spacer(1, 14))

    # Verdict box
    v_c  = colors.HexColor('#C0392B') if verdict=="FAKE" else colors.HexColor('#1E8449')
    v_bg = colors.HexColor('#FEF0EE') if verdict=="FAKE" else colors.HexColor('#EAFAF1')
    vt = Table([[
        Paragraph(f'<font size="20" color="{v_c.hexval()}"><b>{verdict}</b></font>',
                  ParagraphStyle('V', alignment=TA_CENTER)),
        Paragraph(f'<font size="10" color="#555">Risk Level</font><br/>'
                  f'<font size="13" color="{risk_color}"><b>{risk_label}</b></font>',
                  ParagraphStyle('R', alignment=TA_CENTER)),
        Paragraph(f'<font size="10" color="#555">Fake Probability</font><br/>'
                  f'<font size="13" color="#C0392B"><b>{round(norm_score*100,1)}%</b></font>',
                  ParagraphStyle('F', alignment=TA_CENTER)),
        Paragraph(f'<font size="10" color="#555">Fake Frames</font><br/>'
                  f'<font size="13" color="#E67E22"><b>{round(fake_ratio*100)}%</b></font>',
                  ParagraphStyle('FF', alignment=TA_CENTER)),
    ]], colWidths=[MW/4]*4)
    vt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),v_bg),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#DDDDDD')),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
    ]))
    story.append(vt)
    story.append(Spacer(1, 14))

    # Frame chart
    story.append(Paragraph("Frame-by-Frame Fake Probability", h2_s))
    story.append(HRFlowable(width=MW, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))
    norm_scores = [(s - REAL_ANCHOR)/(FAKE_ANCHOR - REAL_ANCHOR) for s in scores]
    norm_scores = [max(0.0, min(1.0, s)) for s in norm_scores]
    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.patch.set_facecolor('#F8F8F8'); ax.set_facecolor('#F8F8F8')
    xs = list(range(1, len(norm_scores)+1))
    ax.plot(xs, norm_scores, color='#C0392B', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=NORM_THRESHOLD, color='#888', linestyle='--', linewidth=1)
    ax.fill_between(xs, norm_scores, NORM_THRESHOLD,
                    where=[s>NORM_THRESHOLD for s in norm_scores], alpha=0.15, color='#C0392B')
    ax.set_ylim(0,1); ax.set_xlim(1, len(norm_scores))
    ax.set_xlabel('Frame', fontsize=8, color='#555'); ax.set_ylabel('Fake Prob', fontsize=8, color='#555')
    ax.tick_params(labelsize=7, colors='#777')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for sp in ax.spines.values(): sp.set_color('#CCC')
    plt.tight_layout(pad=0.5)
    cbuf = io.BytesIO(); plt.savefig(cbuf, format='png', dpi=150, bbox_inches='tight'); plt.close(); cbuf.seek(0)
    story.append(RLImage(cbuf, width=MW, height=2.0*inch))
    story.append(Spacer(1, 14))

    # Grad-CAM
    if gradcam_overlay_np is not None:
        story.append(Paragraph("Grad-CAM Activation Map", h2_s))
        story.append(HRFlowable(width=MW, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=6))
        story.append(Paragraph(
            "The heatmap highlights facial regions that most influenced the model's decision. "
            "Bright regions (yellow/white in inferno colourmap) indicate high activation — "
            "areas associated with deepfake artifacts such as boundary inconsistencies, "
            "texture anomalies, and unnatural frequency patterns.", body_s))
        story.append(Spacer(1,6))
        cam_buf = io.BytesIO()
        Image.fromarray(gradcam_overlay_np).save(cam_buf, format='PNG'); cam_buf.seek(0)
        story.append(RLImage(cam_buf, width=2.0*inch, height=2.0*inch))
        story.append(Spacer(1, 14))

    # Interpretation
    story.append(Paragraph("Interpretation", h2_s))
    story.append(HRFlowable(width=MW, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))
    story.append(Paragraph(
        f"The NoCap AI system analysed this video and returned a verdict of <b>{verdict}</b> "
        f"with a fake probability of {round(norm_score*100,1)}% (risk level: {risk_label}). "
        f"Raw model average score: {round(avg_raw,4)}. "
        f"{round(fake_ratio*100)}% of frames exceeded the individual frame threshold of {FRAME_THRESHOLD}.<br/><br/>"
        f"This report is generated automatically and should be treated as indicative, "
        f"not as definitive legal or forensic evidence.", body_s))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width=MW, thickness=1, color=colors.HexColor('#C0392B'), spaceAfter=6))
    story.append(Paragraph("NoCap Deepfake Detector  ·  EfficientNet-B4  ·  2026",
                             ParagraphStyle('Ft', fontName='Helvetica', fontSize=8,
                                            textColor=colors.HexColor('#AAAAAA'), alignment=TA_CENTER)))
    doc.build(story)
    buf.seek(0); return buf.read()

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

tab1, tab2 = st.tabs(["ANALYSE VIDEO", "ABOUT"])

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

                verdict, confidence, avg_score, fake_ratio, norm_score = smart_verdict(scores)
                risk_label, risk_color = render_verdict_card(verdict, confidence, avg_score, fake_ratio, norm_score, len(faces))

        # Frame chart
        if 'scores' in dir() and scores:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">FRAME-BY-FRAME SCORES</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({
                "Fake Score": [round(s,4) for s in scores],
                "Threshold":  [FAKE_ANCHOR]*len(scores),
            }, index=[f"F{i+1}" for i in range(len(scores))])
            st.line_chart(chart_df, color=["#ff2d55","#333344"])
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#444;letter-spacing:2px;text-transform:uppercase;margin-top:4px;">
                Red = per-frame fake score &nbsp;·&nbsp; Gray = fake anchor ({FAKE_ANCHOR}) &nbsp;·&nbsp; Spikes = suspicious moments
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

        # ── PDF Download ──────────────────────────────────────────
        if 'verdict' in dir() and verdict and 'scores' in dir() and scores:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">FORENSIC REPORT</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'DM Mono',monospace;font-size:0.66rem;color:#555;
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;">
                Download a complete PDF with verdict, frame chart, Grad-CAM &amp; interpretation
            </div>""", unsafe_allow_html=True)
            overlay_np = None
            if 'gc_imgs' in dir() and gc_imgs:
                overlay_np = gc_imgs[1]
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_pdf(
                    filename         = uploaded.name,
                    verdict          = verdict,
                    risk_label       = risk_label,
                    risk_color       = risk_color,
                    norm_score       = norm_score,
                    avg_raw          = avg_score,
                    fake_ratio       = fake_ratio,
                    scores           = scores,
                    gradcam_overlay_np = overlay_np
                )
            st.download_button(
                label     = "DOWNLOAD FORENSIC REPORT  (PDF)",
                data      = pdf_bytes,
                file_name = f"NoCap_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime      = "application/pdf"
            )

        try: os.unlink(tmp_path)
        except: pass

# ══════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"""
        <div class="about-card">
            <div class="about-card-title">Detection Pipeline</div>
            <div class="pipeline-row"><span class="pipeline-num">01</span>OpenCV — extract 20 evenly spaced frames</div>
            <div class="pipeline-row"><span class="pipeline-num">02</span>MTCNN — detect & crop faces to 224×224</div>
            <div class="pipeline-row"><span class="pipeline-num">03</span>EfficientNet-B4 — score each face (0–1)</div>
            <div class="pipeline-row"><span class="pipeline-num">04</span>Score normalisation + {int(FAKE_FRAME_RATIO*100)}% frames above {FRAME_THRESHOLD} guard</div>
            <div class="pipeline-row"><span class="pipeline-num">05</span>Grad-CAM — highlight suspicious regions</div>
            <div class="pipeline-row"><span class="pipeline-num">06</span>5-level risk meter — Authentic to Likely Deepfake</div>
            <div class="pipeline-row"><span class="pipeline-num">07</span>PDF forensic report — downloadable evidence</div>
        </div>
        <div class="about-card">
            <div class="about-card-title">Model Performance</div>
            <div class="metric-row"><span class="metric-label">Celeb-DF v2 AUC</span><span class="metric-value">0.9637</span></div>
            <div class="metric-row"><span class="metric-label">Celeb-DF v2 F1</span><span class="metric-value">0.9580</span></div>
            <div class="metric-row"><span class="metric-label">Celeb-DF v2 Accuracy</span><span class="metric-value">93.53%</span></div>
            <div class="metric-row"><span class="metric-label">Fake Detection</span><span class="metric-value">97.35%</span></div>
            <div class="metric-row"><span class="metric-label">Real Detection</span><span class="metric-value">81.48%</span></div>
            <div class="metric-row"><span class="metric-label">DFDC Val AUC</span><span class="metric-value">0.9507</span></div>
            <div class="metric-row"><span class="metric-label">Norm Threshold</span><span class="metric-value">{NORM_THRESHOLD}</span></div>
            <div class="metric-row"><span class="metric-label">Frame Guard</span><span class="metric-value">{FAKE_FRAME_RATIO}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-title">Score Normalisation</div>
            <p>
                NoCap uses a two-stage training approach — pretrained on DFDC (93,853 face crops),
                then fine-tuned on Celeb-DF v2 for domain adaptation. The fine-tuned model
                scores real videos around 0.54 and fake videos around 0.93 on average.<br><br>
                Score normalisation rescales these using empirical anchors: real anchor 0.54,
                fake anchor 0.93. After normalisation, real videos score near 0 and fakes near 1.
                A secondary frame guard requires 60%+ of frames to confirm the verdict,
                preventing isolated spike frames from causing false positives.
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
            <span class="tag">Celeb-DF v2</span>
            <span class="tag">Streamlit</span>
            <span class="tag">ReportLab</span>
        </div>
        """, unsafe_allow_html=True)
