import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Authentification
login(token=os.environ["HF_TOKEN"])

# Restructuration des modèles et de leurs informations
models_info = {
    "Meta-llama": {
        "Llama 2": {
            "7B": {"name": "meta-llama/Llama-2-7b-hf", "languages": ["en"]},
            "13B": {"name": "meta-llama/Llama-2-13b-hf", "languages": ["en"]},
            "70B": {"name": "meta-llama/Llama-2-70b-hf", "languages": ["en"]},
        },
        "Llama 3": {
            "8B": {"name": "meta-llama/Meta-Llama-3-8B", "languages": ["en"]},
            "3.2-3B": {"name": "meta-llama/Llama-3.2-3B", "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"]},
            "3.1-8B": {"name": "meta-llama/Llama-3.1-8B", "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"]},
        },
    },
    "Mistral AI": {
        "Mistral": {
            "7B-v0.1": {"name": "mistralai/Mistral-7B-v0.1", "languages": ["en"]},
            "7B-v0.3": {"name": "mistralai/Mistral-7B-v0.3", "languages": ["en"]},
        },
        "Mixtral": {
            "8x7B-v0.1": {"name": "mistralai/Mixtral-8x7B-v0.1", "languages": ["en", "fr", "it", "de", "es"]},
        },
    },
    "Google": {
        "Gemma": {
            "2B": {"name": "google/gemma-2-2b", "languages": ["en"]},
            "9B": {"name": "google/gemma-2-9b", "languages": ["en"]},
            "27B": {"name": "google/gemma-2-27b", "languages": ["en"]},
        },
    },
    "CroissantLLM": {
        "CroissantLLMBase": {
            "Base": {"name": "croissantllm/CroissantLLMBase", "languages": ["en", "fr"]},
        },
    },
}

# Paramètres recommandés pour chaque modèle
model_parameters = {
    "meta-llama/Llama-2-13b-hf": {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
    "meta-llama/Llama-2-7b-hf": {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
    "meta-llama/Llama-2-70b-hf": {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
    "meta-llama/Meta-Llama-3-8B": {"temperature": 0.75, "top_p": 0.9, "top_k": 50},
    "meta-llama/Llama-3.2-3B": {"temperature": 0.75, "top_p": 0.9, "top_k": 50},
    "meta-llama/Llama-3.1-8B": {"temperature": 0.75, "top_p": 0.9, "top_k": 50},
    "mistralai/Mistral-7B-v0.1": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
    "mistralai/Mixtral-8x7B-v0.1": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    "mistralai/Mistral-7B-v0.3": {"temperature": 0.7, "top_p": 0.9, "top_k": 50},
    "google/gemma-2-2b": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
    "google/gemma-2-9b": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
    "google/gemma-2-27b": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
    "croissantllm/CroissantLLMBase": {"temperature": 0.8, "top_p": 0.92, "top_k": 50}
}

# Variables globales
model = None
tokenizer = None
selected_language = None

def update_model_type(family):
    return gr.Dropdown(choices=list(models_info[family].keys()), value=None, interactive=True)

def update_model_variation(family, model_type):
    return gr.Dropdown(choices=list(models_info[family][model_type].keys()), value=None, interactive=True)

def update_selected_model(family, model_type, variation):
    if family and model_type and variation:
        model_name = models_info[family][model_type][variation]["name"]
        return model_name, gr.Dropdown(choices=models_info[family][model_type][variation]["languages"], value=models_info[family][model_type][variation]["languages"][0], visible=True, interactive=True)
    return "", gr.Dropdown(visible=False)

def load_model(model_name, progress=gr.Progress()):
    global model, tokenizer
    try:
        progress(0, desc="Chargement du tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        progress(0.5, desc="Chargement du modèle")
        
        # Configurations spécifiques par modèle
        if "mixtral" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        progress(1.0, desc="Modèle chargé")
        
        # Recherche des langues disponibles pour le modèle sélectionné
        available_languages = next(
            (info["languages"] for family in models_info.values()
             for model_type in family.values()
             for variation in model_type.values()
             if variation["name"] == model_name),
            ["en"]  # Défaut à l'anglais si non trouvé
        )
        
        # Mise à jour des sliders avec les valeurs recommandées
        params = model_parameters[model_name]
        return (
            f"Modèle {model_name} chargé avec succès. Langues disponibles : {', '.join(available_languages)}",
            gr.Dropdown(choices=available_languages, value=available_languages[0], visible=True, interactive=True),
            params["temperature"],
            params["top_p"],
            params["top_k"]
        )
    except Exception as e:
        return f"Erreur lors du chargement du modèle : {str(e)}", gr.Dropdown(visible=False), None, None, None

def set_language(lang):
    global selected_language
    selected_language = lang
    return f"Langue sélectionnée : {lang}"

def ensure_token_display(token):
    """Assure que le token est affiché correctement."""
    if token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
        return tokenizer.decode([int(token)])
    return token

def analyze_next_token(input_text, temperature, top_p, top_k):
    global model, tokenizer, selected_language
    
    if model is None or tokenizer is None:
        return "Veuillez d'abord charger un modèle.", None, None

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_token_logits = outputs.logits[0, -1, :]
        probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
        
        top_k = 10
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_words = [ensure_token_display(tokenizer.decode([idx.item()])) for idx in top_indices]
        prob_data = {word: prob.item() for word, prob in zip(top_words, top_probs)}
        
        prob_text = "Prochains tokens les plus probables :\n\n"
        for word, prob in prob_data.items():
            prob_text += f"{word}: {prob:.2%}\n"
        
        prob_plot = plot_probabilities(prob_data)
        attention_plot = plot_attention(inputs["input_ids"][0].cpu(), last_token_logits.cpu())
        
        return prob_text, attention_plot, prob_plot
    except Exception as e:
        return f"Erreur lors de l'analyse : {str(e)}", None, None

def generate_text(input_text, temperature, top_p, top_k):
    global model, tokenizer, selected_language
    
    if model is None or tokenizer is None:
        return "Veuillez d'abord charger un modèle."

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

def plot_probabilities(prob_data):
    words = list(prob_data.keys())
    probs = list(prob_data.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(words)), probs, color='lightgreen')
    ax.set_title("Probabilités des tokens suivants les plus probables")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Probabilité")
    
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    
    for i, (bar, word) in enumerate(zip(bars, words)):
        height = bar.get_height()
        ax.text(i, height, f'{height:.2%}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    return fig

def plot_attention(input_ids, last_token_logits):
    input_tokens = [ensure_token_display(tokenizer.decode([id])) for id in input_ids]
    attention_scores = torch.nn.functional.softmax(last_token_logits, dim=-1)
    top_k = min(len(input_tokens), 10)
    top_attention_scores, _ = torch.topk(attention_scores, top_k)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(top_attention_scores.unsqueeze(0).numpy(), annot=True, cmap="YlOrRd", cbar=True, ax=ax, fmt='.2%')
    ax.set_xticklabels(input_tokens[-top_k:], rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(["Attention"], rotation=0, fontsize=10)
    ax.set_title("Scores d'attention pour les derniers tokens", fontsize=16)
    
    cbar = ax.collections[0].colorbar
    cbar.set_label("Score d'attention", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    return fig

def reset():
    global model, tokenizer, selected_language
    model = None
    tokenizer = None
    selected_language = None
    return (
        "", 1.0, 1.0, 50, None, None, None, None,
        gr.Dropdown(choices=list(models_info.keys()), value=None, interactive=True),
        gr.Dropdown(choices=[], value=None, interactive=False),
        gr.Dropdown(choices=[], value=None, interactive=False),
        "", gr.Dropdown(visible=False), ""
    )

with gr.Blocks() as demo:
    gr.Markdown("# LLM&BIAS")
    
    with gr.Accordion("Sélection du modèle", open=True):
        with gr.Row():
            model_family = gr.Dropdown(choices=list(models_info.keys()), label="Famille de modèle", interactive=True)
            model_type = gr.Dropdown(choices=[], label="Type de modèle", interactive=False)
            model_variation = gr.Dropdown(choices=[], label="Variation du modèle", interactive=False)
        
        selected_model = gr.Textbox(label="Modèle sélectionné", interactive=False)
        load_button = gr.Button("Charger le modèle")
        load_output = gr.Textbox(label="Statut du chargement")
        language_dropdown = gr.Dropdown(label="Choisissez une langue", visible=False)
        language_output = gr.Textbox(label="Langue sélectionnée")
    
    with gr.Row():
        temperature = gr.Slider(0.1, 2.0, value=1.0, label="Température")
        top_p = gr.Slider(0.1, 1.0, value=1.0, label="Top-p")
        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k")
    
    input_text = gr.Textbox(label="Texte d'entrée", lines=3)
    analyze_button = gr.Button("Analyser le prochain token")
    
    next_token_probs = gr.Textbox(label="Probabilités du prochain token")
    
    with gr.Row():
        attention_plot = gr.Plot(label="Visualisation de l'attention")
        prob_plot = gr.Plot(label="Probabilités des tokens suivants")
    
    generate_button = gr.Button("Générer le prochain mot")
    generated_text = gr.Textbox(label="Texte généré")
    
    reset_button = gr.Button("Réinitialiser")
    
    # Événements pour la sélection du modèle
    model_family.change(
        update_model_type,
        inputs=[model_family],
        outputs=[model_type]
    )
    
    model_type.change(
        update_model_variation,
        inputs=[model_family, model_type],
        outputs=[model_variation]
    )
    
    model_variation.change(
        update_selected_model,
        inputs=[model_family, model_type, model_variation],
        outputs=[selected_model, language_dropdown]
    )
    
    load_button.click(
        load_model,
        inputs=[selected_model],
        outputs=[load_output, language_dropdown, temperature, top_p, top_k]
    )
    
    language_dropdown.change(
        set_language,
        inputs=[language_dropdown],
        outputs=[language_output]
    )
    
    analyze_button.click(
        analyze_next_token,
        inputs=[input_text, temperature, top_p, top_k],
        outputs=[next_token_probs, attention_plot, prob_plot]
    )
    
    generate_button.click(
        generate_text,
        inputs=[input_text, temperature, top_p, top_k],
        outputs=[generated_text]
    )
    
    reset_button.click(
        reset,
        outputs=[
            input_text, temperature, top_p, top_k,
            next_token_probs, attention_plot, prob_plot, generated_text,
            model_family, model_type, model_variation,
            selected_model, language_dropdown, language_output
        ]
    )

if __name__ == "__main__":
    demo.launch()
