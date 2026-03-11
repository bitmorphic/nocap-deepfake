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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, gdown, tempfile, io, base64
from datetime import datetime
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
html, body { background-color: #080810 !important; }
.stApp { background-color: #080810 !important; color: #e8e8f0 !important; font-family: 'DM Sans', sans-serif !important; }
p, span, div, label { color: #e8e8f0; font-family: 'DM Sans', sans-serif; }
.stMarkdown { color: #e8e8f0; }
/* noise overlay removed for compatibility */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; position: relative; z-index: 1; }

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
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,45,85,0.2), transparent); margin: 28px 0; }

/* Verdict */
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,45,85,0.08), rgba(255,45,85,0.03));
    border: 1px solid rgba(255,45,85,0.3); border-radius: 20px; padding: 28px 24px;
    text-align: center; position: relative; overflow: hidden;
}
.verdict-fake::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background: linear-gradient(90deg, transparent, #ff2d55, transparent); }
.verdict-real {
    background: linear-gradient(135deg, rgba(0,210,120,0.08), rgba(0,210,120,0.03));
    border: 1px solid rgba(0,210,120,0.3); border-radius: 20px; padding: 28px 24px;
    text-align: center; position: relative; overflow: hidden;
}
.verdict-real::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background: linear-gradient(90deg, transparent, #00d278, transparent); }
.verdict-label { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:4px; text-transform:uppercase; color:#555; margin-bottom:10px; }
.verdict-text { font-family:'Bebas Neue',sans-serif; font-size:3.6rem; letter-spacing:8px; line-height:1; }
.verdict-fake .verdict-text { color:#ff2d55; }
.verdict-real .verdict-text { color:#00d278; }
.verdict-conf { font-family:'DM Mono',monospace; font-size:0.8rem; color:#666; margin-top:10px; }

/* Risk meter */
.risk-wrap { margin-top: 16px; }
.risk-label { font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:3px; text-transform:uppercase; color:#444; margin-bottom:10px; }
.risk-bar-bg { background: rgba(255,255,255,0.04); border-radius: 99px; height: 10px; width:100%; overflow:hidden; position:relative; }
.risk-bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
.risk-levels { display:flex; justify-content:space-between; margin-top:6px; }
.risk-level-item { font-family:'DM Mono',monospace; font-size:0.55rem; color:#333; letter-spacing:1px; text-align:center; }

/* Score grid */
.score-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:14px; }
.score-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:14px; text-align:center; }
.score-card-label { font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:2px; text-transform:uppercase; color:#444; margin-bottom:8px; }
.score-card-value { font-family:'Bebas Neue',sans-serif; font-size:1.8rem; letter-spacing:2px; color:#ff6b35; }
.score-card-sub { font-family:'DM Mono',monospace; font-size:0.56rem; color:#333; margin-top:4px; letter-spacing:1px; }

/* Timestamps */
.ts-wrap { margin-top:8px; }
.ts-row {
    display:flex; align-items:center; gap:12px; padding:10px 14px;
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; margin-bottom:6px;
}
.ts-row.ts-high { border-color:rgba(255,45,85,0.3); background:rgba(255,45,85,0.04); }
.ts-row.ts-med  { border-color:rgba(255,107,53,0.25); background:rgba(255,107,53,0.03); }
.ts-time { font-family:'DM Mono',monospace; font-size:0.72rem; color:#888; min-width:48px; }
.ts-bar-bg { flex:1; background:rgba(255,255,255,0.06); border-radius:99px; height:6px; overflow:hidden; }
.ts-bar-fill { height:100%; border-radius:99px; }
.ts-score { font-family:'DM Mono',monospace; font-size:0.7rem; min-width:38px; text-align:right; }
.ts-tag { font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:2px; padding:2px 8px; border-radius:4px; }

/* Section title */
.section-title { font-family:'Bebas Neue',sans-serif; font-size:1.3rem; letter-spacing:4px; color:#555; margin:24px 0 12px 0; }

/* Gradcam */
.gradcam-caption { font-family:'DM Mono',monospace; font-size:0.6rem; color:#333; letter-spacing:2px; text-transform:uppercase; text-align:center; margin-top:6px; }

/* About */
.about-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:22px; margin-bottom:14px; }
.about-card-title { font-family:'DM Mono',monospace; font-size:0.66rem; letter-spacing:3px; text-transform:uppercase; color:#ff2d55; margin-bottom:12px; }
.about-card p { font-size:0.86rem; color:#888; line-height:1.8; }
.pipeline-row { display:flex; align-items:center; gap:12px; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.84rem; color:#aaa; }
.pipeline-row:last-child { border-bottom:none; }
.pipeline-num { font-family:'DM Mono',monospace; font-size:0.62rem; color:#ff2d55; min-width:22px; }
.metric-row { display:flex; justify-content:space-between; align-items:center; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.84rem; }
.metric-row:last-child { border-bottom:none; }
.metric-label { color:#666; font-family:'DM Mono',monospace; font-size:0.70rem; }
.metric-value { color:#00d278; font-weight:600; font-family:'DM Mono',monospace; }
.tag { display:inline-block; background:rgba(255,45,85,0.08); border:1px solid rgba(255,45,85,0.15); border-radius:6px; padding:3px 10px; font-family:'DM Mono',monospace; font-size:0.62rem; color:#ff6b35; margin:3px; letter-spacing:1px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid rgba(255,255,255,0.06) !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] { background:transparent!important; color:#888!important; font-family:'DM Mono',monospace!important; font-size:0.70rem!important; letter-spacing:2px!important; text-transform:uppercase!important; padding:12px 22px!important; border:none!important; border-radius:0!important; }
.stTabs [aria-selected="true"] { color:#ff2d55!important; border-bottom:2px solid #ff2d55!important; background:transparent!important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:28px; background:transparent!important; }
button[data-baseweb="tab"] { background:transparent!important; }
.stProgress > div > div { background:#ff2d55!important; }
[data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,0.02)!important; border:1.5px dashed rgba(255,45,85,0.25)!important; border-radius:16px!important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #ff2d55, #ff6b35)!important;
    color: white!important; border: none!important; border-radius: 10px!important;
    font-family: 'DM Mono', monospace!important; font-size: 0.72rem!important;
    letter-spacing: 2px!important; text-transform: uppercase!important;
    padding: 10px 24px!important; width: 100%!important;
    box-shadow: 0 0 20px rgba(255,45,85,0.3)!important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DRIVE = '/content/drive/MyDrive/NoCap-Deepfake'
MODEL_ID   = "13goF5n1TXIOtaimlNZPQ2s4mAr3uTJCp"

# Score normalisation anchors (empirically measured)
REAL_ANCHOR      = 0.88
FAKE_ANCHOR      = 0.97
NORM_THRESHOLD   = 0.50
FRAME_THRESHOLD  = 0.92
FAKE_FRAME_RATIO = 0.60

# Risk levels on normalised 0-1 scale
RISK_LEVELS = [
    (0.00, 0.20, "AUTHENTIC",      "#00d278", "#0a2e1e"),
    (0.20, 0.40, "LOW RISK",       "#4ecb71", "#0d2b16"),
    (0.40, 0.60, "SUSPICIOUS",     "#f5a623", "#2e2008"),
    (0.60, 0.80, "HIGH RISK",      "#ff6b35", "#2e1508"),
    (0.80, 1.00, "LIKELY DEEPFAKE","#ff2d55", "#2e0812"),
]

# ── Model ────────────────────────────────────────────────────
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
        transforms.Resize((224, 224)), transforms.ToTensor(),
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
        self.model=model; self.grads=None; self.acts=None
        model.features[-1].register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        model.features[-1].register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))
    def generate(self, inp):
        self.model.eval()
        out = self.model(inp); self.model.zero_grad(); out[0,0].backward()
        w = self.grads[0].mean(dim=(1,2))
        cam = F.relu((w[:,None,None]*self.acts[0]).sum(0))
        cam = cam - cam.min(); cam = cam/(cam.max()+1e-8)
        return cam.cpu().numpy()

def apply_heatmap(cam, img_np):
    arr = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((224,224)))/255.0
    heatmap = (cm.inferno(arr)[:,:,:3]*255).astype(np.uint8)
    overlay  = (0.55*img_np + 0.45*heatmap).astype(np.uint8)
    return heatmap, overlay

# ── Core helpers ─────────────────────────────────────────────
def normalise(raw):
    return float(np.clip((raw - REAL_ANCHOR)/(FAKE_ANCHOR - REAL_ANCHOR), 0.0, 1.0))

def get_risk_level(norm_score):
    for lo, hi, label, color, bg in RISK_LEVELS:
        if norm_score <= hi:
            return label, color, bg
    return RISK_LEVELS[-1][2], RISK_LEVELS[-1][3], RISK_LEVELS[-1][4]

def crop_face(pil_img, mtcnn):
    if mtcnn is None: return pil_img.resize((224,224))
    try:
        face = mtcnn(pil_img)
        if face is not None:
            fn = face.permute(1,2,0).numpy()
            fn = ((fn-fn.min())/(fn.max()-fn.min()+1e-8)*255).astype(np.uint8)
            return Image.fromarray(fn)
    except: pass
    return pil_img.resize((224,224))

def score_face(face_img, model, transform):
    inp = transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(inp)).item()

def extract_frames(video_path, max_frames=20):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    if total == 0: cap.release(); return [], fps
    idxs  = np.linspace(0, total-1, min(max_frames, total), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret: frames.append((Image.fromarray(frame[:,:,::-1].copy()), int(i)))
    cap.release()
    return frames, fps

def run_gradcam(model, transform, face):
    try:
        gc  = GradCAM(model)
        inp = transform(face).unsqueeze(0).to(DEVICE); inp.requires_grad = True
        cam = gc.generate(inp)
        img_np = np.array(face.resize((224,224)))
        return apply_heatmap(cam, img_np), img_np
    except: return None, None


# ── PDF Report Generator ─────────────────────────────────────
def generate_pdf_report(filename, verdict, risk_label, risk_color,
                         norm_score, avg_raw, fake_ratio,
                         scores, gradcam_overlay_np):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image as RLImage, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             leftMargin=0.75*inch, rightMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
    W = letter[0] - 1.5*inch
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold',
                                  fontSize=28, textColor=colors.HexColor('#C0392B'),
                                  alignment=TA_CENTER, spaceAfter=4)
    sub_style   = ParagraphStyle('Sub', fontName='Helvetica',
                                  fontSize=10, textColor=colors.HexColor('#888888'),
                                  alignment=TA_CENTER, spaceAfter=2)
    h2_style    = ParagraphStyle('H2', fontName='Helvetica-Bold',
                                  fontSize=13, textColor=colors.HexColor('#1A1A2E'),
                                  spaceBefore=16, spaceAfter=6)
    body_style  = ParagraphStyle('Body', fontName='Helvetica',
                                  fontSize=10, textColor=colors.HexColor('#333333'),
                                  leading=15, spaceAfter=6)
    mono_style  = ParagraphStyle('Mono', fontName='Courier',
                                  fontSize=9, textColor=colors.HexColor('#555555'),
                                  spaceAfter=4)

    # ── Header ──
    story.append(Paragraph("NOCAP", title_style))
    story.append(Paragraph("Deepfake Detection Forensic Report", sub_style))
    story.append(Paragraph("Team Quad Logic  ·  NIT Delhi  ·  2026", sub_style))
    story.append(HRFlowable(width=W, thickness=2, color=colors.HexColor('#C0392B'), spaceAfter=12))

    # ── Meta info table ──
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    meta_data = [
        ["File Analysed", filename],
        ["Analysis Date", now],
        ["Model", "EfficientNet-B4 (DFDC fine-tuned)"],
        ["AUC Score", "0.9507"],
    ]
    meta_table = Table(meta_data, colWidths=[1.8*inch, W-1.8*inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',    (0,0),(-1,-1), 'Helvetica'),
        ('FONTNAME',    (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0),(-1,-1), 9),
        ('TEXTCOLOR',   (0,0),(0,-1),  colors.HexColor('#888888')),
        ('TEXTCOLOR',   (1,0),(1,-1),  colors.HexColor('#222222')),
        ('ROWBACKGROUNDS', (0,0),(-1,-1), [colors.HexColor('#F8F8F8'), colors.white]),
        ('GRID',        (0,0),(-1,-1), 0.5, colors.HexColor('#EEEEEE')),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
        ('TOPPADDING',  (0,0),(-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ── Verdict box ──
    v_color = colors.HexColor('#C0392B') if verdict=="FAKE" else colors.HexColor('#1E8449')
    v_bg    = colors.HexColor('#FEF0EE') if verdict=="FAKE" else colors.HexColor('#EAFAF1')
    verdict_data = [[
        Paragraph(f'<font size="22" color="{v_color.hexval()}">'
                  f'<b>{verdict}</b></font>', ParagraphStyle('V', alignment=TA_CENTER)),
        Paragraph(f'<font size="11" color="#555555">Risk Level</font><br/>'
                  f'<font size="14" color="{risk_color}"><b>{risk_label}</b></font>',
                  ParagraphStyle('RL', alignment=TA_CENTER)),
        Paragraph(f'<font size="11" color="#555555">Fake Probability</font><br/>'
                  f'<font size="14" color="#C0392B"><b>{round(norm_score*100,1)}%</b></font>',
                  ParagraphStyle('FP', alignment=TA_CENTER)),
        Paragraph(f'<font size="11" color="#555555">Fake Frames</font><br/>'
                  f'<font size="14" color="#E67E22"><b>{round(fake_ratio*100)}%</b></font>',
                  ParagraphStyle('FF', alignment=TA_CENTER)),
    ]]
    vt = Table(verdict_data, colWidths=[W/4]*4)
    vt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), v_bg),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#DDDDDD')),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0),(-1,-1), 12),
        ('BOTTOMPADDING',(0,0),(-1,-1),12),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
        ('RIGHTPADDING',(0,0),(-1,-1), 8),
    ]))
    story.append(vt)
    story.append(Spacer(1, 16))

    # ── Score chart ──
    story.append(Paragraph("Frame-by-Frame Analysis", h2_style))
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))

    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#F8F8F8')
    xs = list(range(1, len(scores)+1))
    norm_scores = [normalise(s) for s in scores]
    ax.plot(xs, norm_scores, color='#C0392B', linewidth=2, marker='o', markersize=4, label='Fake Probability')
    ax.axhline(y=NORM_THRESHOLD, color='#888888', linestyle='--', linewidth=1, label=f'Threshold ({NORM_THRESHOLD})')
    ax.fill_between(xs, norm_scores, NORM_THRESHOLD,
                    where=[s > NORM_THRESHOLD for s in norm_scores],
                    alpha=0.15, color='#C0392B')
    ax.set_xlabel('Frame', fontsize=8, color='#555')
    ax.set_ylabel('Fake Probability', fontsize=8, color='#555')
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(scores))
    ax.tick_params(labelsize=7, colors='#777')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#CCCCCC')
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout(pad=0.5)

    chart_buf = io.BytesIO()
    plt.savefig(chart_buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    chart_buf.seek(0)
    story.append(RLImage(chart_buf, width=W, height=2.2*inch))
    story.append(Spacer(1, 14))

    # ── Grad-CAM image ──
    if gradcam_overlay_np is not None:
        story.append(Paragraph("Grad-CAM Activation Map", h2_style))
        story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))
        story.append(Paragraph(
            "The heatmap below shows which facial regions most influenced the model's decision. "
            "Bright regions (yellow/white) indicate high activation — areas the model identified "
            "as suspicious. Common deepfake artifacts appear at face boundaries, eye regions, "
            "and skin texture boundaries.", body_style))
        story.append(Spacer(1, 6))
        cam_pil = Image.fromarray(gradcam_overlay_np)
        cam_buf = io.BytesIO()
        cam_pil.save(cam_buf, format='PNG')
        cam_buf.seek(0)
        story.append(RLImage(cam_buf, width=2.2*inch, height=2.2*inch))
        story.append(Spacer(1, 14))


    # ── Interpretation ──
    story.append(Paragraph("Interpretation", h2_style))
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))
    interp = (
        f"The NoCap AI system analysed this video and returned a verdict of "
        f"<b>{verdict}</b> with a fake probability of {round(norm_score*100,1)}% "
        f"(normalised score: {round(norm_score,3)}). "
        f"The raw model average score was {round(avg_raw,4)}, "
        f"and {round(fake_ratio*100)}% of analysed frames exceeded the individual "
        f"frame threshold of {FRAME_THRESHOLD}.<br/><br/>"
        f"This report was generated automatically by the NoCap deepfake detection "
        f"system. Results should be treated as indicative and not as definitive "
        f"legal or forensic evidence. The model (EfficientNet-B4, AUC 0.9507) was "
        f"trained on the DFDC dataset and may not generalise to all manipulation types."
    )
    story.append(Paragraph(interp, body_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#C0392B'), spaceAfter=6))
    story.append(Paragraph("Team Quad Logic  ·  NIT Delhi  ·  NoCap Deepfake Detector  ·  2026",
                             ParagraphStyle('Footer', fontName='Helvetica', fontSize=8,
                                            textColor=colors.HexColor('#AAAAAA'), alignment=TA_CENTER)))
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ── UI Renderers ─────────────────────────────────────────────
def render_verdict_and_risk(verdict, norm_score, confidence, avg_raw, fake_ratio):
    risk_label, risk_color, risk_bg = get_risk_level(norm_score)
    v_class = "verdict-fake" if verdict=="FAKE" else "verdict-real"
    icon_svg = ("""<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 4L17 10L24 11L19 16L20 23L14 20L8 23L9 16L4 11L11 10L14 4Z" fill="#ff2d55" opacity="0.9"/>
        </svg>""" if verdict=="FAKE" else
        """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M6 14L12 20L22 8" stroke="#00d278" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""")

    # Verdict card
    st.markdown(f"""
    <div class="{v_class}">
        <div class="verdict-label">Detection Result</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:12px">
            {icon_svg}<div class="verdict-text">{verdict}</div>
        </div>
        <div class="verdict-conf">{confidence}% confidence &nbsp;·&nbsp; norm {round(norm_score,3)}</div>
    </div>""", unsafe_allow_html=True)

    # Risk level meter
    bar_pct   = round(norm_score * 100, 1)
    bar_color = risk_color
    st.markdown(f"""
    <div class="risk-wrap">
        <div class="risk-label">Risk Level &nbsp;—&nbsp;
            <span style="color:{risk_color};font-weight:600;">{risk_label}</span>
        </div>
        <div class="risk-bar-bg">
            <div class="risk-bar-fill" style="width:{bar_pct}%;
                background:linear-gradient(90deg,#00d278,#f5a623,#ff2d55);
                clip-path:inset(0 {100-bar_pct}% 0 0 round 99px);"></div>
        </div>
        <div class="risk-levels">
            <span class="risk-level-item" style="color:#00d278">AUTHENTIC</span>
            <span class="risk-level-item" style="color:#4ecb71">LOW</span>
            <span class="risk-level-item" style="color:#f5a623">SUSPICIOUS</span>
            <span class="risk-level-item" style="color:#ff6b35">HIGH</span>
            <span class="risk-level-item" style="color:#ff2d55">DEEPFAKE</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Score cards
    st.markdown(f"""
    <div class="score-grid">
        <div class="score-card">
            <div class="score-card-label">Fake Prob</div>
            <div class="score-card-value">{round(norm_score*100,0):.0f}%</div>
            <div class="score-card-sub">normalised</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Raw Score</div>
            <div class="score-card-value">{avg_raw:.3f}</div>
            <div class="score-card-sub">model output</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Fake Frames</div>
            <div class="score-card-value">{round(fake_ratio*100)}%</div>
            <div class="score-card-sub">above {FRAME_THRESHOLD}</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Risk Level</div>
            <div class="score-card-value" style="font-size:1.1rem;color:{risk_color};">{risk_label}</div>
            <div class="score-card-sub">5-level scale</div>
        </div>
    </div>""", unsafe_allow_html=True)

    return risk_label, risk_color


        # ── Grad-CAMt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, gdown, tempfile, io, base64
from datetime import datetime
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
html, body { background-color: #080810 !important; }
.stApp { background-color: #080810 !important; color: #e8e8f0 !important; font-family: 'DM Sans', sans-serif !important; }
p, span, div, label { color: #e8e8f0; font-family: 'DM Sans', sans-serif; }
.stMarkdown { color: #e8e8f0; }
/* noise overlay removed for compatibility */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1200px; position: relative; z-index: 1; }

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
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,45,85,0.2), transparent); margin: 28px 0; }

/* Verdict */
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,45,85,0.08), rgba(255,45,85,0.03));
    border: 1px solid rgba(255,45,85,0.3); border-radius: 20px; padding: 28px 24px;
    text-align: center; position: relative; overflow: hidden;
}
.verdict-fake::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background: linear-gradient(90deg, transparent, #ff2d55, transparent); }
.verdict-real {
    background: linear-gradient(135deg, rgba(0,210,120,0.08), rgba(0,210,120,0.03));
    border: 1px solid rgba(0,210,120,0.3); border-radius: 20px; padding: 28px 24px;
    text-align: center; position: relative; overflow: hidden;
}
.verdict-real::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background: linear-gradient(90deg, transparent, #00d278, transparent); }
.verdict-label { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:4px; text-transform:uppercase; color:#555; margin-bottom:10px; }
.verdict-text { font-family:'Bebas Neue',sans-serif; font-size:3.6rem; letter-spacing:8px; line-height:1; }
.verdict-fake .verdict-text { color:#ff2d55; }
.verdict-real .verdict-text { color:#00d278; }
.verdict-conf { font-family:'DM Mono',monospace; font-size:0.8rem; color:#666; margin-top:10px; }

/* Risk meter */
.risk-wrap { margin-top: 16px; }
.risk-label { font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:3px; text-transform:uppercase; color:#444; margin-bottom:10px; }
.risk-bar-bg { background: rgba(255,255,255,0.04); border-radius: 99px; height: 10px; width:100%; overflow:hidden; position:relative; }
.risk-bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
.risk-levels { display:flex; justify-content:space-between; margin-top:6px; }
.risk-level-item { font-family:'DM Mono',monospace; font-size:0.55rem; color:#333; letter-spacing:1px; text-align:center; }

/* Score grid */
.score-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:14px; }
.score-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:14px; text-align:center; }
.score-card-label { font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:2px; text-transform:uppercase; color:#444; margin-bottom:8px; }
.score-card-value { font-family:'Bebas Neue',sans-serif; font-size:1.8rem; letter-spacing:2px; color:#ff6b35; }
.score-card-sub { font-family:'DM Mono',monospace; font-size:0.56rem; color:#333; margin-top:4px; letter-spacing:1px; }

/* Timestamps */
.ts-wrap { margin-top:8px; }
.ts-row {
    display:flex; align-items:center; gap:12px; padding:10px 14px;
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);
    border-radius:10px; margin-bottom:6px;
}
.ts-row.ts-high { border-color:rgba(255,45,85,0.3); background:rgba(255,45,85,0.04); }
.ts-row.ts-med  { border-color:rgba(255,107,53,0.25); background:rgba(255,107,53,0.03); }
.ts-time { font-family:'DM Mono',monospace; font-size:0.72rem; color:#888; min-width:48px; }
.ts-bar-bg { flex:1; background:rgba(255,255,255,0.06); border-radius:99px; height:6px; overflow:hidden; }
.ts-bar-fill { height:100%; border-radius:99px; }
.ts-score { font-family:'DM Mono',monospace; font-size:0.7rem; min-width:38px; text-align:right; }
.ts-tag { font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:2px; padding:2px 8px; border-radius:4px; }

/* Section title */
.section-title { font-family:'Bebas Neue',sans-serif; font-size:1.3rem; letter-spacing:4px; color:#555; margin:24px 0 12px 0; }

/* Gradcam */
.gradcam-caption { font-family:'DM Mono',monospace; font-size:0.6rem; color:#333; letter-spacing:2px; text-transform:uppercase; text-align:center; margin-top:6px; }

/* About */
.about-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:22px; margin-bottom:14px; }
.about-card-title { font-family:'DM Mono',monospace; font-size:0.66rem; letter-spacing:3px; text-transform:uppercase; color:#ff2d55; margin-bottom:12px; }
.about-card p { font-size:0.86rem; color:#888; line-height:1.8; }
.pipeline-row { display:flex; align-items:center; gap:12px; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.84rem; color:#aaa; }
.pipeline-row:last-child { border-bottom:none; }
.pipeline-num { font-family:'DM Mono',monospace; font-size:0.62rem; color:#ff2d55; min-width:22px; }
.metric-row { display:flex; justify-content:space-between; align-items:center; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.84rem; }
.metric-row:last-child { border-bottom:none; }
.metric-label { color:#666; font-family:'DM Mono',monospace; font-size:0.70rem; }
.metric-value { color:#00d278; font-weight:600; font-family:'DM Mono',monospace; }
.tag { display:inline-block; background:rgba(255,45,85,0.08); border:1px solid rgba(255,45,85,0.15); border-radius:6px; padding:3px 10px; font-family:'DM Mono',monospace; font-size:0.62rem; color:#ff6b35; margin:3px; letter-spacing:1px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid rgba(255,255,255,0.06) !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] { background:transparent!important; color:#888!important; font-family:'DM Mono',monospace!important; font-size:0.70rem!important; letter-spacing:2px!important; text-transform:uppercase!important; padding:12px 22px!important; border:none!important; border-radius:0!important; }
.stTabs [aria-selected="true"] { color:#ff2d55!important; border-bottom:2px solid #ff2d55!important; background:transparent!important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:28px; background:transparent!important; }
button[data-baseweb="tab"] { background:transparent!important; }
.stProgress > div > div { background:#ff2d55!important; }
[data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,0.02)!important; border:1.5px dashed rgba(255,45,85,0.25)!important; border-radius:16px!important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #ff2d55, #ff6b35)!important;
    color: white!important; border: none!important; border-radius: 10px!important;
    font-family: 'DM Mono', monospace!important; font-size: 0.72rem!important;
    letter-spacing: 2px!important; text-transform: uppercase!important;
    padding: 10px 24px!important; width: 100%!important;
    box-shadow: 0 0 20px rgba(255,45,85,0.3)!important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DRIVE = '/content/drive/MyDrive/NoCap-Deepfake'
MODEL_ID   = "13goF5n1TXIOtaimlNZPQ2s4mAr3uTJCp"

# Score normalisation anchors (empirically measured)
REAL_ANCHOR      = 0.88
FAKE_ANCHOR      = 0.97
NORM_THRESHOLD   = 0.50
FRAME_THRESHOLD  = 0.92
FAKE_FRAME_RATIO = 0.60

# Risk levels on normalised 0-1 scale
RISK_LEVELS = [
    (0.00, 0.20, "AUTHENTIC",      "#00d278", "#0a2e1e"),
    (0.20, 0.40, "LOW RISK",       "#4ecb71", "#0d2b16"),
    (0.40, 0.60, "SUSPICIOUS",     "#f5a623", "#2e2008"),
    (0.60, 0.80, "HIGH RISK",      "#ff6b35", "#2e1508"),
    (0.80, 1.00, "LIKELY DEEPFAKE","#ff2d55", "#2e0812"),
]

# ── Model ────────────────────────────────────────────────────
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
        transforms.Resize((224, 224)), transforms.ToTensor(),
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
        self.model=model; self.grads=None; self.acts=None
        model.features[-1].register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        model.features[-1].register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))
    def generate(self, inp):
        self.model.eval()
        out = self.model(inp); self.model.zero_grad(); out[0,0].backward()
        w = self.grads[0].mean(dim=(1,2))
        cam = F.relu((w[:,None,None]*self.acts[0]).sum(0))
        cam = cam - cam.min(); cam = cam/(cam.max()+1e-8)
        return cam.cpu().numpy()

def apply_heatmap(cam, img_np):
    arr = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((224,224)))/255.0
    heatmap = (cm.inferno(arr)[:,:,:3]*255).astype(np.uint8)
    overlay  = (0.55*img_np + 0.45*heatmap).astype(np.uint8)
    return heatmap, overlay

# ── Core helpers ─────────────────────────────────────────────
def normalise(raw):
    return float(np.clip((raw - REAL_ANCHOR)/(FAKE_ANCHOR - REAL_ANCHOR), 0.0, 1.0))

def get_risk_level(norm_score):
    for lo, hi, label, color, bg in RISK_LEVELS:
        if norm_score <= hi:
            return label, color, bg
    return RISK_LEVELS[-1][2], RISK_LEVELS[-1][3], RISK_LEVELS[-1][4]

def crop_face(pil_img, mtcnn):
    if mtcnn is None: return pil_img.resize((224,224))
    try:
        face = mtcnn(pil_img)
        if face is not None:
            fn = face.permute(1,2,0).numpy()
            fn = ((fn-fn.min())/(fn.max()-fn.min()+1e-8)*255).astype(np.uint8)
            return Image.fromarray(fn)
    except: pass
    return pil_img.resize((224,224))

def score_face(face_img, model, transform):
    inp = transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(inp)).item()

def extract_frames(video_path, max_frames=20):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    if total == 0: cap.release(); return [], fps
    idxs  = np.linspace(0, total-1, min(max_frames, total), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret: frames.append((Image.fromarray(frame[:,:,::-1].copy()), int(i)))
    cap.release()
    return frames, fps

def run_gradcam(model, transform, face):
    try:
        gc  = GradCAM(model)
        inp = transform(face).unsqueeze(0).to(DEVICE); inp.requires_grad = True
        cam = gc.generate(inp)
        img_np = np.array(face.resize((224,224)))
        return apply_heatmap(cam, img_np), img_np
    except: return None, None


# ── PDF Report Generator ─────────────────────────────────────
def generate_pdf_report(filename, verdict, risk_label, risk_color,
                         norm_score, avg_raw, fake_ratio,
                         scores, gradcam_overlay_np):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image as RLImage, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             leftMargin=0.75*inch, rightMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
    W = letter[0] - 1.5*inch
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold',
                                  fontSize=28, textColor=colors.HexColor('#C0392B'),
                                  alignment=TA_CENTER, spaceAfter=4)
    sub_style   = ParagraphStyle('Sub', fontName='Helvetica',
                                  fontSize=10, textColor=colors.HexColor('#888888'),
                                  alignment=TA_CENTER, spaceAfter=2)
    h2_style    = ParagraphStyle('H2', fontName='Helvetica-Bold',
                                  fontSize=13, textColor=colors.HexColor('#1A1A2E'),
                                  spaceBefore=16, spaceAfter=6)
    body_style  = ParagraphStyle('Body', fontName='Helvetica',
                                  fontSize=10, textColor=colors.HexColor('#333333'),
                                  leading=15, spaceAfter=6)
    mono_style  = ParagraphStyle('Mono', fontName='Courier',
                                  fontSize=9, textColor=colors.HexColor('#555555'),
                                  spaceAfter=4)

    # ── Header ──
    story.append(Paragraph("NOCAP", title_style))
    story.append(Paragraph("Deepfake Detection Forensic Report", sub_style))
    story.append(Paragraph("Team Quad Logic  ·  NIT Delhi  ·  2026", sub_style))
    story.append(HRFlowable(width=W, thickness=2, color=colors.HexColor('#C0392B'), spaceAfter=12))

    # ── Meta info table ──
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    meta_data = [
        ["File Analysed", filename],
        ["Analysis Date", now],
        ["Model", "EfficientNet-B4 (DFDC fine-tuned)"],
        ["AUC Score", "0.9507"],
    ]
    meta_table = Table(meta_data, colWidths=[1.8*inch, W-1.8*inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',    (0,0),(-1,-1), 'Helvetica'),
        ('FONTNAME',    (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0),(-1,-1), 9),
        ('TEXTCOLOR',   (0,0),(0,-1),  colors.HexColor('#888888')),
        ('TEXTCOLOR',   (1,0),(1,-1),  colors.HexColor('#222222')),
        ('ROWBACKGROUNDS', (0,0),(-1,-1), [colors.HexColor('#F8F8F8'), colors.white]),
        ('GRID',        (0,0),(-1,-1), 0.5, colors.HexColor('#EEEEEE')),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
        ('TOPPADDING',  (0,0),(-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ── Verdict box ──
    v_color = colors.HexColor('#C0392B') if verdict=="FAKE" else colors.HexColor('#1E8449')
    v_bg    = colors.HexColor('#FEF0EE') if verdict=="FAKE" else colors.HexColor('#EAFAF1')
    verdict_data = [[
        Paragraph(f'<font size="22" color="{v_color.hexval()}">'
                  f'<b>{verdict}</b></font>', ParagraphStyle('V', alignment=TA_CENTER)),
        Paragraph(f'<font size="11" color="#555555">Risk Level</font><br/>'
                  f'<font size="14" color="{risk_color}"><b>{risk_label}</b></font>',
                  ParagraphStyle('RL', alignment=TA_CENTER)),
        Paragraph(f'<font size="11" color="#555555">Fake Probability</font><br/>'
                  f'<font size="14" color="#C0392B"><b>{round(norm_score*100,1)}%</b></font>',
                  ParagraphStyle('FP', alignment=TA_CENTER)),
        Paragraph(f'<font size="11" color="#555555">Fake Frames</font><br/>'
                  f'<font size="14" color="#E67E22"><b>{round(fake_ratio*100)}%</b></font>',
                  ParagraphStyle('FF', alignment=TA_CENTER)),
    ]]
    vt = Table(verdict_data, colWidths=[W/4]*4)
    vt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), v_bg),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#DDDDDD')),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0),(-1,-1), 12),
        ('BOTTOMPADDING',(0,0),(-1,-1),12),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
        ('RIGHTPADDING',(0,0),(-1,-1), 8),
    ]))
    story.append(vt)
    story.append(Spacer(1, 16))

    # ── Score chart ──
    story.append(Paragraph("Frame-by-Frame Analysis", h2_style))
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))

    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#F8F8F8')
    xs = list(range(1, len(scores)+1))
    norm_scores = [normalise(s) for s in scores]
    ax.plot(xs, norm_scores, color='#C0392B', linewidth=2, marker='o', markersize=4, label='Fake Probability')
    ax.axhline(y=NORM_THRESHOLD, color='#888888', linestyle='--', linewidth=1, label=f'Threshold ({NORM_THRESHOLD})')
    ax.fill_between(xs, norm_scores, NORM_THRESHOLD,
                    where=[s > NORM_THRESHOLD for s in norm_scores],
                    alpha=0.15, color='#C0392B')
    ax.set_xlabel('Frame', fontsize=8, color='#555')
    ax.set_ylabel('Fake Probability', fontsize=8, color='#555')
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(scores))
    ax.tick_params(labelsize=7, colors='#777')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#CCCCCC')
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout(pad=0.5)

    chart_buf = io.BytesIO()
    plt.savefig(chart_buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    chart_buf.seek(0)
    story.append(RLImage(chart_buf, width=W, height=2.2*inch))
    story.append(Spacer(1, 14))

    # ── Grad-CAM image ──
    if gradcam_overlay_np is not None:
        story.append(Paragraph("Grad-CAM Activation Map", h2_style))
        story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))
        story.append(Paragraph(
            "The heatmap below shows which facial regions most influenced the model's decision. "
            "Bright regions (yellow/white) indicate high activation — areas the model identified "
            "as suspicious. Common deepfake artifacts appear at face boundaries, eye regions, "
            "and skin texture boundaries.", body_style))
        story.append(Spacer(1, 6))
        cam_pil = Image.fromarray(gradcam_overlay_np)
        cam_buf = io.BytesIO()
        cam_pil.save(cam_buf, format='PNG')
        cam_buf.seek(0)
        story.append(RLImage(cam_buf, width=2.2*inch, height=2.2*inch))
        story.append(Spacer(1, 14))

    # ── Top suspicious timestamps ──
    story.append(Paragraph("Top Suspicious Moments", h2_style))
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))

    ts_header = [["Timestamp", "Risk Level", "Fake Prob", "Raw Score"]]
    ts_rows   = []
    for ts, norm, raw, label, color in timestamps[:8]:
        ts_rows.append([ts, label, f"{round(norm*100,1)}%", f"{round(raw,4)}"])
    ts_table = Table(ts_header + ts_rows, colWidths=[1.2*inch, 2.0*inch, 1.2*inch, 1.2*inch])
    ts_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0),(-1,0), colors.HexColor('#1A1A2E')),
        ('TEXTCOLOR',   (0,0),(-1,0), colors.white),
        ('FONTNAME',    (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0),(-1,-1),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F8F8F8'), colors.white]),
        ('GRID',        (0,0),(-1,-1), 0.5, colors.HexColor('#EEEEEE')),
        ('ALIGN',       (0,0),(-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0),(-1,-1), 8),
        ('TOPPADDING',  (0,0),(-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(ts_table)
    story.append(Spacer(1, 14))

    # ── Interpretation ──
    story.append(Paragraph("Interpretation", h2_style))
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#EEEEEE'), spaceAfter=8))
    interp = (
        f"The NoCap AI system analysed this video and returned a verdict of "
        f"<b>{verdict}</b> with a fake probability of {round(norm_score*100,1)}% "
        f"(normalised score: {round(norm_score,3)}). "
        f"The raw model average score was {round(avg_raw,4)}, "
        f"and {round(fake_ratio*100)}% of analysed frames exceeded the individual "
        f"frame threshold of {FRAME_THRESHOLD}.<br/><br/>"
        f"This report was generated automatically by the NoCap deepfake detection "
        f"system. Results should be treated as indicative and not as definitive "
        f"legal or forensic evidence. The model (EfficientNet-B4, AUC 0.9507) was "
        f"trained on the DFDC dataset and may not generalise to all manipulation types."
    )
    story.append(Paragraph(interp, body_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#C0392B'), spaceAfter=6))
    story.append(Paragraph("Team Quad Logic  ·  NIT Delhi  ·  NoCap Deepfake Detector  ·  2026",
                             ParagraphStyle('Footer', fontName='Helvetica', fontSize=8,
                                            textColor=colors.HexColor('#AAAAAA'), alignment=TA_CENTER)))
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ── UI Renderers ─────────────────────────────────────────────
def render_verdict_and_risk(verdict, norm_score, confidence, avg_raw, fake_ratio):
    risk_label, risk_color, risk_bg = get_risk_level(norm_score)
    v_class = "verdict-fake" if verdict=="FAKE" else "verdict-real"
    icon_svg = ("""<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 4L17 10L24 11L19 16L20 23L14 20L8 23L9 16L4 11L11 10L14 4Z" fill="#ff2d55" opacity="0.9"/>
        </svg>""" if verdict=="FAKE" else
        """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M6 14L12 20L22 8" stroke="#00d278" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>""")

    # Verdict card
    st.markdown(f"""
    <div class="{v_class}">
        <div class="verdict-label">Detection Result</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:12px">
            {icon_svg}<div class="verdict-text">{verdict}</div>
        </div>
        <div class="verdict-conf">{confidence}% confidence &nbsp;·&nbsp; norm {round(norm_score,3)}</div>
    </div>""", unsafe_allow_html=True)

    # Risk level meter
    bar_pct   = round(norm_score * 100, 1)
    bar_color = risk_color
    st.markdown(f"""
    <div class="risk-wrap">
        <div class="risk-label">Risk Level &nbsp;—&nbsp;
            <span style="color:{risk_color};font-weight:600;">{risk_label}</span>
        </div>
        <div class="risk-bar-bg">
            <div class="risk-bar-fill" style="width:{bar_pct}%;
                background:linear-gradient(90deg,#00d278,#f5a623,#ff2d55);
                clip-path:inset(0 {100-bar_pct}% 0 0 round 99px);"></div>
        </div>
        <div class="risk-levels">
            <span class="risk-level-item" style="color:#00d278">AUTHENTIC</span>
            <span class="risk-level-item" style="color:#4ecb71">LOW</span>
            <span class="risk-level-item" style="color:#f5a623">SUSPICIOUS</span>
            <span class="risk-level-item" style="color:#ff6b35">HIGH</span>
            <span class="risk-level-item" style="color:#ff2d55">DEEPFAKE</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Score cards
    st.markdown(f"""
    <div class="score-grid">
        <div class="score-card">
            <div class="score-card-label">Fake Prob</div>
            <div class="score-card-value">{round(norm_score*100,0):.0f}%</div>
            <div class="score-card-sub">normalised</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Raw Score</div>
            <div class="score-card-value">{avg_raw:.3f}</div>
            <div class="score-card-sub">model output</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Fake Frames</div>
            <div class="score-card-value">{round(fake_ratio*100)}%</div>
            <div class="score-card-sub">above {FRAME_THRESHOLD}</div>
        </div>
        <div class="score-card">
            <div class="score-card-label">Risk Level</div>
            <div class="score-card-value" style="font-size:1.1rem;color:{risk_color};">{risk_label}</div>
            <div class="score-card-sub">5-level scale</div>
        </div>
    </div>""", unsafe_allow_html=True)

    return risk_label, risk_color
