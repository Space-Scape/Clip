import discord
import torch
import clip
import cv2
import os
import glob
import numpy as np
from PIL import Image
import easyocr
import re

# ---- Discord Bot Configuration ---- #
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Get token from Railway environment variable
CHANNEL_ID = 1273094409432469605        # Replace with your target channel ID

# ---- CLIP Model Setup ---- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder paths
REFERENCE_FOLDER = "Images"          # Reference OSRS item images
REFERENCE_INVENTORY = "References/reference_inventory.png"  # Path to inventory panel reference
TEMP_FOLDER = "temp_images"          # Folder to temporarily save uploaded images

# Ensure temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# ---- Load Reference Item Images ---- #
def load_reference_images():
    image_paths = glob.glob(os.path.join(REFERENCE_FOLDER, "*.png"))
    item_names = [os.path.basename(path).replace(".png", "").replace("_", " ") for path in image_paths]
    item_embeddings = []

    print("Loading reference images...")
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            processed_image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(processed_image)
            item_embeddings.append(embedding)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    print("Reference images loaded successfully.")
    return item_names, item_embeddings

# ---- Crop Inventory Panel ---- #
def crop_inventory_panel(full_image_path, reference_path, save_path):
    """Crop the inventory panel using template matching."""
    full_image = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(reference_path, cv2.IMREAD_COLOR)

    if full_image is None or template is None:
        raise FileNotFoundError("Full image or reference image not found!")

    result = cv2.matchTemplate(full_image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Check match confidence
    if max_val < 0.7:  # Confidence threshold
        raise ValueError("Inventory panel not found! Match confidence too low.")

    # Extract the matched area
    top_left_x, top_left_y = max_loc
    bottom_right_x = top_left_x + template.shape[1]
    bottom_right_y = top_left_y + template.shape[0]

    cropped_inventory = full_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    cv2.imwrite(save_path, cropped_inventory)

    return save_path

# ---- Check Items Per Slot ---- #
def check_slots(cropped_image_path, item_names, item_embeddings, empty_embedding_index):
    cropped_image = Image.open(cropped_image_path).convert("RGB")
    detected_items = []

    # Hardcoded slot coordinates
    slot_coordinates = [
        # Row 1
        (45, 30, 117, 94), (131, 30, 203, 94), (216, 30, 288, 94), (301, 30, 373, 94),
        # Row 2
        (45, 103, 117, 168), (131, 103, 203, 168), (216, 103, 288, 168), (301, 103, 373, 168),
        # Row 3
        (45, 176, 117, 241), (131, 176, 203, 241), (216, 176, 288, 241), (301, 176, 373, 241),
        # Row 4
        (45, 249, 117, 314), (131, 249, 203, 314), (216, 249, 288, 314), (301, 249, 373, 314),
        # Row 5
        (45, 322, 117, 387), (131, 322, 203, 387), (216, 322, 288, 387), (301, 322, 373, 387),
        # Row 6
        (45, 395, 117, 460), (131, 395, 203, 460), (216, 395, 288, 460), (301, 395, 373, 460),
        # Row 7
        (45, 468, 117, 533), (131, 468, 203, 533), (216, 468, 288, 533), (301, 468, 373, 533)
    ]

    # Iterate over slots
    for idx, (left, upper, right, lower) in enumerate(slot_coordinates):
        slot_image = cropped_image.crop((left, upper, right, lower))

        # Compare with reference images
        slot_image_resized = slot_image.resize((224, 224))  # Resize for CLIP model
        processed_image = preprocess(slot_image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            slot_embedding = model.encode_image(processed_image)

        similarities = []
        for item_name, ref_embedding in zip(item_names, item_embeddings):
            similarity = torch.cosine_similarity(slot_embedding, ref_embedding).item()
            similarities.append((item_name, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_match = similarities[0]

        # Check if slot matches the empty slot
        if top_match[1] > 0.7 and top_match[0] != item_names[empty_embedding_index]:
            detected_items.append(f"Slot ({(idx // 4) + 1}, {(idx % 4) + 1}): {top_match[0]}")

    return detected_items

# ---- Extract Text After Colons and Before Parentheses ---- #
def extract_text_from_chatbox(image_path):
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_path)

        extracted_text = set()  # Use a set to ensure unique matches
        for _, text, _ in results:
            matches = re.findall(r':\s*([^:()\n]+)\s*\(', text)
            extracted_text.update(matches)

        return list(extracted_text)
    except Exception as e:
        print(f"Error during chatbox text extraction: {e}")
        return []

# ---- Discord Bot ---- #
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

# Preload the reference images
item_names, item_embeddings = load_reference_images()
empty_embedding_index = item_names.index("empty")  # Find the index of the empty slot embedding

@client.event
async def on_ready():
    print(f"Bot is ready! Logged in as {client.user}")

@client.event
async def on_message(message):
    # Ignore messages outside the target channel or from bots
    if message.channel.id != CHANNEL_ID or message.author.bot:
        return

    if message.attachments:
        await message.channel.send("Processing image... Please wait.")
        for attachment in message.attachments:
            if attachment.filename.endswith((".png", ".jpg", ".jpeg")):
                temp_image_path = os.path.join(TEMP_FOLDER, attachment.filename)
                await attachment.save(temp_image_path)

                try:
                    # Crop inventory panel
                    cropped_inventory_path = crop_inventory_panel(temp_image_path, REFERENCE_INVENTORY, "cropped_inventory.png")

                    # Analyze slots
                    slot_matches = check_slots(cropped_inventory_path, item_names, item_embeddings, empty_embedding_index)

                    # Build and send detected items response
                    if slot_matches:
                        response = "**Detected Items in Inventory:**\n"
                        for item in slot_matches:
                            response += f"{item}\n"
                        await message.channel.send(response)
                    else:
                        await message.channel.send("No matching items detected in the inventory.")

                    # Extract text from chatbox
                    extracted_text = extract_text_from_chatbox(temp_image_path)
                    if extracted_text:
                        response = "**Extracted Text from Chatbox:**\n" + "\n".join(extracted_text)
                        await message.channel.send(response)
                    else:
                        await message.channel.send("No matching text found in chatbox.")

                except Exception as e:
                    await message.channel.send(f"Error: {e}")

                os.remove(temp_image_path)  # Clean up temp image

if TOKEN:
    client.run(TOKEN)
else:
    print("Error: DISCORD_BOT_TOKEN not found.")
