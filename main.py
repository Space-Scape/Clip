import discord
import torch
import clip
import cv2
from PIL import Image
import os
import glob
import numpy as np

# ---- Discord Bot Configuration ---- #
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Get token from Railway environment variable
CHANNEL_ID = 1273094409432469605        # Replace with your target channel ID

# ---- CLIP Model Setup ---- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder paths
REFERENCE_FOLDER = "Images"            # Reference OSRS item images
TEMP_FOLDER = "temp_images"            # Folder to temporarily save uploaded images
CROPPED_INVENTORY = "cropped_inventory.png"
RREFERENCE_INVENTORY = "References/reference_inventory.png"

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

# ---- Crop Inventory Panel Using Template Matching ---- #
def crop_inventory_panel(full_image_path, template_image_path, save_path=CROPPED_INVENTORY):
    """ Detect and crop the inventory panel from the full client screenshot using template matching. """
    full_image = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(template_image_path, cv2.IMREAD_COLOR)

    # Perform template matching
    result = cv2.matchTemplate(full_image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Define confidence threshold
    confidence_threshold = 0.8  # Adjust if needed
    if max_val >= confidence_threshold:
        # Inventory panel found; crop the region
        top_left_x, top_left_y = max_loc
        bottom_right_x = top_left_x + template.shape[1]
        bottom_right_y = top_left_y + template.shape[0]

        cropped_inventory = full_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cv2.imwrite(save_path, cropped_inventory)  # Save the cropped panel
        return save_path
    else:
        raise ValueError("Error: Inventory panel not found in the image!")

# ---- Check Items Per Slot ---- #
def check_slots(cropped_image_path, item_names, item_embeddings):
    cropped_image = Image.open(cropped_image_path).convert("RGB")

    slot_width, slot_height = 73, 65
    slots = []

    # Iterate over the 7x4 grid
    for row in range(7):
        for col in range(4):
            left = col * slot_width
            upper = row * slot_height
            right = left + slot_width
            lower = upper + slot_height

            slot_image = cropped_image.crop((left, upper, right, lower))
            slot_image = slot_image.resize((224, 224))  # Resize for CLIP model
            processed_image = preprocess(slot_image).unsqueeze(0).to(device)

            with torch.no_grad():
                slot_embedding = model.encode_image(processed_image)

            # Compare slot with reference images
            similarities = []
            for idx, ref_embedding in enumerate(item_embeddings):
                similarity = torch.cosine_similarity(slot_embedding, ref_embedding).item()
                similarities.append((item_names[idx], similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_match = similarities[0]

            if top_match[1] > 0.9:  # Threshold for confidence
                slots.append((row + 1, col + 1, top_match[0], top_match[1]))

    return slots

# ---- Discord Bot ---- #
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Required for reading message content and attachments

client = discord.Client(intents=intents)

# Preload the reference images
item_names, item_embeddings = load_reference_images()

@client.event
async def on_ready():
    print(f"Bot is ready! Logged in as {client.user}")

@client.event
async def on_message(message):
    # Ignore messages outside the target channel or from bots
    if message.channel.id != CHANNEL_ID or message.author.bot:
        return

    # Automatically process any image attachments
    if message.attachments:
        await message.channel.send("Processing image... Please wait.")

        for attachment in message.attachments:
            if attachment.filename.endswith((".png", ".jpg", ".jpeg")):
                temp_image_path = os.path.join(TEMP_FOLDER, attachment.filename)
                await attachment.save(temp_image_path)  # Save the image temporarily

                try:
                    # Crop inventory dynamically
                    cropped_inventory_path = crop_inventory_panel(temp_image_path, REFERENCE_INVENTORY)
                    slot_matches = check_slots(cropped_inventory_path, item_names, item_embeddings)

                    # Send cropped inventory for verification
                    await message.channel.send(file=discord.File(cropped_inventory_path))

                    # Build response for detected items
                    if slot_matches:
                        response = "**Detected Items in Inventory:**\n"
                        for row, col, item, score in slot_matches:
                            response += f"Slot ({row}, {col}): {item} ({score:.2f})\n"
                        await message.channel.send(response)
                    else:
                        await message.channel.send("No matching items detected in the inventory.")
                except ValueError as e:
                    await message.channel.send(str(e))
                finally:
                    os.remove(temp_image_path)  # Clean up temp image

# Run the bot
if TOKEN:
    client.run(TOKEN)
else:
    print("Error: DISCORD_BOT_TOKEN not found. Set the environment variable.")

