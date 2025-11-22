import gradio as gr
import nest_asyncio

nest_asyncio.apply()

def ultimate_pipeline(img):
    if img is None: return None, "", None
    
    # 1. VISION (DenseNet)
    img_t = inference_transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_t)
        probs = F.softmax(output, dim=1)[0]
    
    score_pneumo = float(probs[1])
    
    # Décision
    if score_pneumo > 0.95:
        diag = "PNEUMONIE DÉTECTÉE"
        color = "red"
    else:
        diag = "POUMONS SAINS"
        color = "green"

    # 2. LANGAGE (Flan-T5 Generative AI)
    # On demande au LLM d'écrire le rapport basé sur le diagnostic
    print("Génération du texte médical en cours...")
    ai_report_text = generate_medical_report(diag, score_pneumo)
    
    # 3. EXPLICABILITÉ (Grad-CAM)
    heatmap_img = get_heatmap_image(model, img)
    
    # 4. PDF (Assemblage)
    pdf_bytes = create_smart_pdf(img, heatmap_img, diag, score_pneumo, ai_report_text)
    pdf_path = "Smart_Report.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Formatage HTML pour l'interface
    html_output = f"""
    <h2 style='color:{color}; border-bottom: 2px solid {color}'>{diag}</h2>
    <p><b>Niveau de Confiance :</b> {score_pneumo:.1%}</p>
    <div style='background-color: #f4f4f9; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6;'>
        <p><b>🤖 Compte-rendu généré par l'IA :</b></p>
        <p style='font-family: monospace;'>{ai_report_text}</p>
    </div>
    """
    
    return { "Pneumonie": score_pneumo, "Normal": float(probs[0]) }, html_output, pdf_path

# Interface Ultra-Pro
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="AI-Rad GenAI") as demo:
    gr.Markdown("""
    # 🧠 AI-Rad GenAI : Vision + Langage
    **Détection de Pneumonie (DenseNet) & Rédaction de Rapport Automatique (LLM)**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Radio Thoracique", height=400)
            btn = gr.Button("✨ ANALYSER ET GÉNÉRER LE RAPPORT", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔬 Résultats Cliniques")
            out_probs = gr.Label(label="Classification Probabiliste")
            out_report = gr.HTML(label="Rapport IA")
            out_pdf = gr.File(label="📥 Télécharger le PDF Officiel")

    btn.click(ultimate_pipeline, inputs=input_img, outputs=[out_probs, out_report, out_pdf])

    gr.Examples(
        examples=[
            ["/content/dataset/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"],
            ["/content/dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"]
        ],
        inputs=input_img
    )

print("🚀 Lancement du système complet (Vision + LLM)...")
demo.launch(share=True, debug=False)