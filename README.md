Offline Chat-Reply Recommendation System
========================================

## Overview
This project implements an offline chat-reply recommendation system using a fine-tuned DistilGPT-2 model. The system is designed to generate appropriate replies to chat messages based on historical conversation data.

## Workflow
1. **Data Loading**: Reads chat data from an Excel file (`conversationfile.xlsx`).
2. **Preprocessing**: Cleans and standardizes sender/message columns, detects user labels, and pairs messages for supervised training.
3. **Dataset Preparation**: Converts message pairs into tokenized format using GPT-2 tokenizer and creates a PyTorch dataset and dataloader.
4. **Model Training**: Fine-tunes DistilGPT-2 on the chat pairs using cross-entropy loss. Training is performed on CPU for demonstration purposes.
5. **Evaluation**: Calculates BLEU score and perplexity to assess model performance.
6. **Artifacts**: Saves the trained model weights (`ChatRec_Model.pt`), a joblib dump (`Model.joblib`), and a summary file (`ReadMe.txt`).

## Key Files
- `nchat.ipynb`: Main notebook containing all code for data processing, model training, and evaluation.
- `conversationfile.xlsx`: Source chat data for training.
- `ChatRec_Model.pt`: Saved PyTorch model weights.
- `Model.joblib`: Model weights in joblib format for portability.
- `ReadMe.txt`: Text summary of the project and artifacts.

## Main Libraries Used
- pandas
- torch
- transformers (HuggingFace)
- nltk (for BLEU score)
- joblib

## Usage
1. Place your chat data in `conversationfile.xlsx` with columns for sender, message, and optionally timestamp.
2. Run the notebook `nchat.ipynb` to preprocess data, train the model, and evaluate results.
3. The model can generate replies to input prompts using the `generate_reply` function.

## Metrics
- **BLEU Score**: Measures the quality of generated replies against actual responses.
- **Perplexity**: Evaluates model confidence and fluency.

## Notes
- Training is performed on CPU for demonstration; for larger datasets, GPU is recommended.
- The notebook is modular and can be adapted for other chat datasets with similar structure.

## Author
Pushpit Saluja

## Submission Artifacts
- `ChatRec_Model.pt`
- `Model.joblib`
- `ReadMe.txt`
