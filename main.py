import torch
import clip
from PIL import Image
import glob
import os
import numpy as np

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function for embedding images
def get_image_embedding(image):
    processed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(processed_image)
    return image_embedding

# Function for embedding text
def get_text_embedding(text: str):
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
    return text_embedding

# Function for comparing image and text similarity
def compare(image, text: list):
    image = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return np.ravel(probs)

# Load all item images from the Images folder
image_paths = glob.glob('Images/*.png')  # Adjust path if necessary
item_names = [os.path.basename(path).replace(".png", "").replace("_", " ") for path in image_paths]

# Preload all reference images into embeddings
item_embeddings = []
for path in image_paths:
    image = Image.open(path)
    embedding = get_image_embedding(image)
    item_embeddings.append(embedding)

# Compare an inventory screenshot to the item list
inventory_image_path = 'inventory_screenshot.png'  # Replace with your inventory screenshot
inventory_image = Image.open(inventory_image_path).resize((224, 224))  # Adjust resizing as needed

# Calculate similarity with all items
inventory_embedding = get_image_embedding(inventory_image)
similarities = []

for idx, item_embedding in enumerate(item_embeddings):
    similarity = torch.cosine_similarity(inventory_embedding, item_embedding).item()
    similarities.append((item_names[idx], similarity))

# Sort and display the most similar item
similarities.sort(key=lambda x: x[1], reverse=True)
print("Top Matches:")
for item, score in similarities[:5]:  # Top 5 matches
    print(f"{item}: {score:.4f}")

