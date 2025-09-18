import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image

# Load the model and processor from Hugging Face
model_name = "domenicrosati/skin_disease_classification"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Define the class labels (example - adjust based on model's classes)
class_names = [
    "Acne",
    "Eczema",
    "Psoriasis",
    "Melanoma",
    "Healthy"
]

def predict_skin_disease(image):
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    
    return f"**Prediction:** {predicted_class}\n**Confidence:** {confidence_score:.2%}"

# Create the Gradio interface
title = "AI Dermatology Scanner"
description = """
⚠️ **Warning:** This tool is for educational purposes only and is **not a substitute for professional medical diagnosis.** 
Always consult a healthcare provider for medical concerns.
"""
examples = [["acne_example.jpg"], ["eczema_example.jpg"]]

demo = gr.Interface(
    fn=predict_skin_disease,
    inputs=gr.Image(type="pil", label="Upload Skin Image"),
    outputs=gr.Markdown(label="Analysis Result"),
    title=title,
    description=description,
    examples=examples
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
