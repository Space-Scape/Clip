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

# Function to load and preprocess all reference item images
def load_reference_images(image_folder):
    image_paths = glob.glob(os.path.join(image_folder, "*.png"))
    item_names = [os.path.basename(path).replace(".png", "").replace("_", " ") for path in image_paths]
    item_embeddings = []

    print("Loading reference images...")
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            embedding = get_image_embedding(image)
            item_embeddings.append(embedding)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    print("Reference images loaded successfully.\n")
    return item_names, item_embeddings

# Function to compare inventory image to reference images
def compare_inventory_to_references(inventory_image_path, item_names, item_embeddings):
    print("Processing inventory image...")
    try:
        inventory_image = Image.open(inventory_image_path).resize((224, 224)).convert("RGB")
    except Exception as e:
        print(f"Error loading inventory image: {e}")
        return

    inventory_embedding = get_image_embedding(inventory_image)

    similarities = []
    for idx, item_embedding in enumerate(item_embeddings):
        similarity = torch.cosine_similarity(inventory_embedding, item_embedding).item()
        similarities.append((item_names[idx], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    print("Top Matches:")
    for item, score in similarities[:5]:  # Top 5 matches
        print(f"- {item}: {score:.4f}")

# Main execution
if __name__ == "__main__":
    # Define paths
    images_folder = "Images"
    inventory_image_path = "Test_Inventory/inventory.png"

    # Load reference item images
    item_names, item_embeddings = load_reference_images(images_folder)

    # Compare inventory image to reference items
    compare_inventory_to_references(inventory_image_path, item_names, item_embeddings)
