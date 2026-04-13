"""
app.py
──────
Interactive Comprehensive Dashboard for Course Project: 
Facial Beauty Prediction via Multi-Modal Gated Networks.

This application provides a professional research dashboard to:
1. EXPLORE: Deep scientific context on experimental methodology and results.
2. ANALYZE: Real-time aesthetic scoring with live interpretability visualizations.

Powered by Gradio, MediaPipe, and PyTorch.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Internal project imports
import config as C
from models import M1ImageCNN, M4AdaptiveFusion
from phase1_data_prep import extract_landmarks, anchor_normalise_2d, procrustes_align
from phase6_transformer_experiments import M7CrossAttentionFusion

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def load_models():
    """Load the primary architectures from checkpoints (supports wrapped state_dicts)."""
    models = {}
    
    def safe_load(model, path):
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        return True

    # M1: Texture branch
    m1 = M1ImageCNN(pretrained=False)
    if safe_load(m1, os.path.join(C.CHECKPOINT_DIR, "m1_cnn.pt")):
        models["M1: Texture (ResNet-18)"] = m1

    # M4: Gated Fusion branch
    m4 = M4AdaptiveFusion(pretrained=False)
    if safe_load(m4, os.path.join(C.CHECKPOINT_DIR, "m4_fusion.pt")):
        models["M4: Gated Fusion (Adaptive)"] = m4

    # M7: Attention Fusion branch
    m7 = M7CrossAttentionFusion(freeze_vit=True)
    if safe_load(m7, os.path.join(C.CHECKPOINT_DIR, "m7_transformer.pt")):
        models["M7: Cross-Attention (Transformer)"] = m7

    return models

# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_m1_saliency(model, image_tensor, original_img):
    """Computes gradient-based saliency highlighting texture-relevant regions."""
    image_tensor.requires_grad_()
    output = model(image_tensor)
    output.backward(torch.ones_like(output))
    
    saliency, _ = torch.max(image_tensor.grad.data.abs(), dim=1)
    saliency = saliency.reshape(224, 224).cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_img)
    h, w = original_img.shape[:2]
    saliency_resized = cv2.resize(saliency, (w, h))
    ax.imshow(saliency_resized, cmap='hot', alpha=0.5)
    ax.axis('off')
    ax.set_title("Texture Focus Analysis")
    plt.tight_layout()
    return fig

def create_gate_plot(alpha, beta):
    """Visualizes the modality trust values for Gated Fusion."""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(["Geometry (β)", "Texture (α)"], [beta, alpha], color=['#FF5733', '#33C1FF'])
    ax.set_xlim(0, 1)
    ax.set_title("Inference Modality Preference")
    plt.tight_layout()
    return fig

def create_attention_map(image_pil, landmarks, attn_weights):
    """Plots landmark importance based on Transformer cross-attention."""
    importance = attn_weights[0].mean(axis=1) 
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_pil)
    w, h = image_pil.size
    x, y = landmarks[:, 0] * w, landmarks[:, 1] * h
    sc = ax.scatter(x, y, s=12, c=importance, cmap='plasma', alpha=0.8)
    plt.colorbar(sc, label="Landmark Attention Weight")
    ax.axis('off')
    ax.set_title("Spatial Feature Importance")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = load_models()
TR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_analysis(input_img, model_name):
    if input_img is None: return None, None, "No Image provided"
    
    temp_p = "demo_live.jpg"
    Image.fromarray(input_img).save(temp_p)
    lm2d, lm3d = extract_landmarks(temp_p)
    os.remove(temp_p)
    
    if lm2d is None: return 0.0, None, "Face detection failed. Ensure the face is clearly visible."

    img_t = TR(Image.fromarray(input_img)).unsqueeze(0).to(DEVICE)
    lm2d_norm = anchor_normalise_2d(lm2d)
    ref = np.load(C.PROCRUSTES_REF) if os.path.exists(C.PROCRUSTES_REF) else lm3d
    lm3d_aligned, _ = procrustes_align(lm3d, ref)
    lm3d_t = torch.from_numpy(lm3d_aligned.flatten()).unsqueeze(0).to(DEVICE)

    model = MODELS[model_name]
    score, viz, logs = 0, None, f"**Active Engine**: {model_name}\n\n"

    if "M1" in model_name:
        score = model(img_t).item()
        viz = get_m1_saliency(model, img_t, input_img)
        logs += "Focus: Analyzing local pixel variations (skin texture, symmetry) via ResNet-18."
    elif "M4" in model_name:
        with torch.no_grad():
            y_f, a, b, _, _ = model(img_t, lm3d_t)
            score = y_f.item()
            viz = create_gate_plot(a.item(), b.item())
            logs += f"Gating State: Modality trust split — Texture ({a.item():.1%}), Geometry ({b.item():.1%})."
    elif "M7" in model_name:
        with torch.no_grad():
            y_f, attn = model(img_t, lm3d_t)
            score = y_f.item()
            viz = create_attention_map(Image.fromarray(input_img), lm2d, attn.cpu().numpy())
            logs += "Mechanism: Transformer cross-attention highlights geometrically critical facial patches."

    return round((score * 4) + 1, 2), viz, logs

def get_project_table():
    path = os.path.join(C.RESULTS_DIR, "final_metrics_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        "Model": ["M1", "M2", "M4", "M7"],
        "Architecture": ["ResNet-18", "Landmark MLP", "Gated Fusion", "Cross-Attention"],
        "Pearson ρ ↑": [0.8857, 0.7109, 0.8886, 0.8814]
    })

# ═══════════════════════════════════════════════════════════════════════════════
#  INTERFACE THEME & ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Facial Aesthetic Analytics", css="footer {visibility: hidden}") as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: white; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-size: 2.5em; letter-spacing: 1px;">🧠 Facial Beauty Analytics Dashboard</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;">Multi-Modal Gated Networks for Aesthetic Scoring & Interpretability</p>
        </div>
    """)

    with gr.Tabs():
        # --- TAB 1: RESEARCH INSIGHTS ---
        with gr.Tab("📊 Research Insights"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Quantitative Results")
                    results_table = gr.DataFrame(value=get_project_table(), label="Performance Benchmarks", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 🔍 Key Findings
                    - **Adaptive Fusion (M4)** is the state-of-the-art ($ρ=0.888$), effectively balancing biological proportions with surface aesthetics.
                    - **Disentanglement**: Our model successfully separates 'Texture' from 'Geometry', allowing for objective scoring even under varying lighting.
                    """)

            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🏆 Performance Benchmarking")
                    gr.Image(value=os.path.join(C.RESULTS_DIR, "final_performance_comparison.png") if os.path.exists(os.path.join(C.RESULTS_DIR, "final_performance_comparison.png")) else None, show_label=False)
                    gr.Markdown("""
                    Our **Architecture Comparison** demonstrates the superiority of multimodal fusion. By combining CNN features with Aligned landmarks, we achieve significant gains over unimodal baselines.
                    """)
                
                with gr.Column():
                    gr.Markdown("#### ⚖️ Modality Trust (Adaptive Gating)")
                    gr.Image(value=os.path.join(C.RESULTS_DIR, "m4_beta_distribution.png") if os.path.exists(os.path.join(C.RESULTS_DIR, "m4_beta_distribution.png")) else None, show_label=False)
                    gr.Markdown("""
                    The **Gating Network** calculates per-sample trust. The distribution shows that while texture is often dominant, the model shifts to geometry ($β$) for faces where structure is the primary aesthetic driver.
                    """)

                with gr.Column():
                    gr.Markdown("#### 👁️ Spatial Landmark Importance")
                    gr.Image(value=os.path.join(C.RESULTS_DIR, "m7_attention_spatial.png") if os.path.exists(os.path.join(C.RESULTS_DIR, "m7_attention_spatial.png")) else None, show_label=False)
                    gr.Markdown("""
                    Our **Transformer model (M7)** uses cross-attention to 'look' at specific landmarks. We visualize this as a spatial heatmap, confirming that the eyes, jawline, and mouth proportions are the strongest predictors.
                    """)

        # --- TAB 2: INTERACTIVE ANALYZER ---
        with gr.Tab("👩‍🔬 Face Beauty Lab (Try it Yourself)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Input")
                    user_img = gr.Image(label="Facial Portrait", type="numpy")
                    model_sel = gr.Radio(choices=list(MODELS.keys()), 
                                        label="Select Architecture", 
                                        value=list(MODELS.keys())[0] if MODELS else None)
                    analyze_btn = gr.Button("🚀 Run Real-time Analysis", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Output")
                    res_score = gr.Number(label="Predicted Aesthetic Score (1.0 - 5.0)")
                    res_plot = gr.Plot(label="Interpretability Visualization")
                    res_logs = gr.Markdown("#### 📝 Inference Logic")

            analyze_btn.click(fn=run_analysis, inputs=[user_img, model_sel], outputs=[res_score, res_plot, res_logs])

    gr.HTML("""
        <div style="margin-top: 30px; text-align: center; color: #64748b; font-size: 0.9em;">
            <p>EE655 Course Project — IIT Kanpur | <a href="https://github.com/Guna-Venkat/Multi-Modal-Gated-Networks-for-Aesthetic-Face-Scoring" style="color: #3b82f6;">Source Code</a></p>
        </div>
    """)

if __name__ == "__main__":
    # NOTE: server_name="0.0.0.0" is for Docker. Use "127.0.0.1" for local browser access.
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860,
        theme=gr.themes.Default()
    )
