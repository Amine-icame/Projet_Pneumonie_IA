# 🩻 AI-Rad Expert : Détection de Pneumonie par IA Multimodale

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Deep Learning](https://img.shields.io/badge/Model-DenseNet121-green)

## 📋 Description
Ce projet propose un système expert d'aide au diagnostic radiologique. Il utilise une architecture **DenseNet121** optimisée pour détecter la pneumonie sur des radiographies thoraciques (Chest X-Ray).
Le système intègre :
*   **Classification Haute Précision :** Rappel de 99% sur les cas pathologiques.
*   **Explicabilité (XAI) :** Visualisation des zones infectées via **Grad-CAM++**.
*   **IA Générative :** Rédaction automatique de rapports médicaux via un LLM (**BLOOMZ**).

## 🚀 Fonctionnalités Clés
*   **Prétraitement Avancé :** Zoom aléatoire (RandomResizedCrop) pour éviter le biais d'apprentissage.
*   **Entraînement Robuste :** Weighted Loss pour gérer le déséquilibre de classes.
*   **Calibration :** Seuil de décision optimisé à 0.95 pour minimiser les fausses alertes.
*   **Interface Web :** Démo interactive sous Gradio avec génération de PDF.

## 📊 Résultats
| Métrique | Score |
|----------|-------|
| **Recall (Pneumonie)** | **99%** |
| Accuracy Globale | 89% |
| F1-Score (Moyen) | 0.89 |

## 🛠️ Installation
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/VOTRE_NOM/Projet_Pneumonie_IA.git
Installez les dépendances :
code
Bash
pip install -r requirements.txt
Lancez l'application :
code
Bash
python app.py
👤 Auteur
Réalisé par Amine Içame/ Salma Benomar dans le cadre du module Deep Learning.
