import torch
import clip
from PIL import Image
import os
import glob
import numpy as np
import cv2  # For dynamic cropping
import discord
from io import BytesIO

# ---- Configurations ---- #
REFERENCE_GRID_PATH = "References/inventory_grid_reference.png"  # Inventory grid reference
ITEMS_FOLDER = "Images"  # Reference OSRS item images
TEMP_FOLDER = "temp_images"  # Folder for temporary processing
GRID_SLOT_WIDTH = 73  # Grid slot width in pixels
GRID_SLOT_HEIGHT = 65  # Grid slot height in pixels
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
TARGET_CHANNEL_ID = 1273094409432469605  # Replace with your Discord channel ID

# Ensure temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# ---- CLIP Model Setup ---- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---- Load Reference Grid Dimensions ---- #
def get_grid_slots(image_path):
    """ Splits the inventory reference into 7x4 grid slots. """
    grid_image = Image.open(image_path).convert("RGB")
    slots = []
    for row in range(7):
        for col in range(4):
            x0 = col * GRID_SLOT_WIDTH
            y0 = row * GRID_SLOT_HEIGHT
            x1 = x0 + GRID_SLOT_WIDTH
            y1 = y0 + GRID_SLOT_HEIGHT
            slot = grid_image.crop((x0, y0, x1, y1))
            slots.append(slot)
    return slots

# ---- Detect and Crop Inventory Panel ---- #
def crop_inventory_panel(full_screenshot_path, grid_reference_path):
    """ Dynamically crops the inventory panel using template matching. """
    full_image = cv2.imread(full_screenshot_path)
    grid_reference = cv2.imread(grid_reference_path)
    
    result = cv2.matchTemplate(full_image, grid_reference, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    # Crop the detected inventory panel
    h, w = grid_reference.shape[:2]
    x, y = max_loc
    cropped_panel = full_image[y:y+h, x:x+w]
    
    # Resize to match reference size
    cropped_panel_resized = cv2.resize(cropped_panel, (414, 559))
    panel_image = Image.fromarray(cv2.cvtColor(cropped_panel_resized, cv2.COLOR_BGR2RGB))
    return panel_image

# ---- Compare Each Slot Against Reference Items ---- #
def compare_slots_to_references(slots, reference_items):
    """ Compare each slot to reference item images. """
    matches = {}
    for idx, slot in enumerate(slots):
        processed_slot = preprocess(slot).unsqueeze(0).to(device)
        with torch.no_grad():
            slot_embedding = model.encode_image(processed_slot)
        
        # Compare to all reference images
        similarities = []
        for item_name, item_embedding in reference_items:
            similarity = torch.cosine_similarity(slot_embedding, item_embedding).item()
            similarities.append((item_name, similarity))
        
        # Sort and store top match
        similarities.sort(key=lambda x: x[1], reverse=True)
        matches[f"Slot {idx+1}"] = similarities[:1]  # Top 1 match per slot
    return matches

# ---- Preload Reference Items ---- #
def load_reference_items():
    """ Load and process all reference item images into embeddings. """
    print("Loading reference items...")
    reference_items = []
    image_paths = glob.glob(os.path.join(ITEMS_FOLDER, "*.png"))
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        processed_image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(processed_image)
        item_name = os.path.basename(path).replace(".png", "").replace("_", " ")
        reference_items.append((item_name, embedding))
    print("Reference items loaded.")
    return reference_items

# ---- Discord Bot Integration ---- #
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)
reference_items = load_reference_items()

@client.event
async def on_ready():
    print(f"Bot is ready and logged in as {client.user}")

@client.event
async def on_message(message):
    if message.channel.id != TARGET_CHANNEL_ID or message.author.bot:
        return
    
    if message.attachments:
        await message.channel.send("Processing image... Please wait.")
        for attachment in message.attachments:
            if attachment.filename.endswith(('.png', '.jpg', '.jpeg')):
                temp_path = os.path.join(TEMP_FOLDER, attachment.filename)
                await attachment.save(temp_path)
                
                # Crop and process the inventory panel
                cropped_inventory = crop_inventory_panel(temp_path, REFERENCE_GRID_PATH)
                slots = []
                for row in range(7):
                    for col in range(4):
                        x0 = col * GRID_SLOT_WIDTH
                        y0 = row * GRID_SLOT_HEIGHT
                        x1 = x0 + GRID_SLOT_WIDTH
                        y1 = y0 + GRID_SLOT_HEIGHT
                        slot = cropped_inventory.crop((x0, y0, x1, y1))
                        slots.append(slot)
                
                # Compare slots and build results
                matches = compare_slots_to_references(slots, reference_items)
                result_message = "**Inventory Analysis Results:**\n"
                for slot, match in matches.items():
                    result_message += f"{slot}: {match[0][0]} (Similarity: {match[0][1]:.4f})\n"
                
                # Send the cropped inventory for verification
                cropped_image_bytes = BytesIO()
                cropped_inventory.save(cropped_image_bytes, format='PNG')
                cropped_image_bytes.seek(0)
                
                await message.channel.send(result_message)
                await message.channel.send(file=discord.File(fp=cropped_image_bytes, filename="cropped_inventory.png"))
                
                os.remove(temp_path)  # Clean up

# ---- Run the Bot ---- #
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN:
        client.run(DISCORD_BOT_TOKEN)
    else:
        print("Error: DISCORD_BOT_TOKEN not found.")

