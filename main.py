iimport discord
import torch
import clip
from PIL import Image
import os
import glob
import numpy as np

# ---- Discord Bot Configuration ---- #
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = 1273094409432469605

# ---- CLIP Model Setup ---- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
REFERENCE_FOLDER = "Images"
TEMP_FOLDER = "temp_images"
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# Inventory grid settings
SLOT_WIDTH, SLOT_HEIGHT = 73, 65  # Size of each grid square
GRID_ROWS, GRID_COLS = 7, 4       # Grid dimensions

# ---- Load Reference Images ---- #
def load_reference_images():
    image_paths = glob.glob(os.path.join(REFERENCE_FOLDER, "*.png"))
    item_names = [os.path.basename(path).replace(".png", "").replace("_", " ") for path in image_paths]
    item_embeddings = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        processed_image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(processed_image)
        item_embeddings.append(embedding)
    return item_names, item_embeddings

item_names, item_embeddings = load_reference_images()

# ---- Crop Inventory Panel ---- #
def crop_inventory_panel(full_image):
    """
    Automatically detects and crops the inventory panel based on its known size and location.
    """
    width, height = full_image.size
    inventory_x = width - 414
    inventory_y = height - 559
    cropped_inventory = full_image.crop((inventory_x, inventory_y, width, height))
    return cropped_inventory

# ---- Check Items in Each Slot ---- #
def check_inventory_slots(inventory_image):
    """
    Splits the inventory into grid slots, checks each slot for matching items.
    """
    matches = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Crop individual slot
            x1 = col * SLOT_WIDTH
            y1 = row * SLOT_HEIGHT
            x2 = x1 + SLOT_WIDTH
            y2 = y1 + SLOT_HEIGHT
            slot_image = inventory_image.crop((x1, y1, x2, y2))

            # Compare to reference images
            processed_slot = preprocess(slot_image).unsqueeze(0).to(device)
            with torch.no_grad():
                slot_embedding = model.encode_image(processed_slot)
            similarities = [
                (item_names[idx], torch.cosine_similarity(slot_embedding, ref_embed).item())
                for idx, ref_embed in enumerate(item_embeddings)
            ]

            # Get the best match
            best_match = max(similarities, key=lambda x: x[1])
            if best_match[1] > 0.8:  # Threshold for confidence
                matches.append((row, col, best_match[0], best_match[1]))
    return matches

# ---- Discord Bot ---- #
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Bot is ready! Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.channel.id != CHANNEL_ID or message.author.bot:
        return

    if message.attachments:
        await message.channel.send("Processing image... Please wait.")

        for attachment in message.attachments:
            if attachment.filename.endswith((".png", ".jpg", ".jpeg")):
                temp_image_path = os.path.join(TEMP_FOLDER, attachment.filename)
                await attachment.save(temp_image_path)

                # Crop inventory panel
                full_image = Image.open(temp_image_path).convert("RGB")
                cropped_inventory = crop_inventory_panel(full_image)
                cropped_inventory_path = os.path.join(TEMP_FOLDER, "cropped_inventory.png")
                cropped_inventory.save(cropped_inventory_path)

                # Check each slot
                matches = check_inventory_slots(cropped_inventory)

                # Prepare and send results
                if matches:
                    response = "**Inventory Matches:**\n"
                    for row, col, item, score in matches:
                        response += f"Slot ({row+1}, {col+1}): {item} ({score:.4f})\n"
                    await message.channel.send(file=discord.File(cropped_inventory_path))
                    await message.channel.send(response)
                else:
                    await message.channel.send("No matching items found.")

                os.remove(temp_image_path)
                os.remove(cropped_inventory_path)

# Run the bot
if TOKEN:
    client.run(TOKEN)
else:
    print("Error: DISCORD_BOT_TOKEN not found.")
