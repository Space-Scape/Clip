import discord
import torch
import clip
import cv2
import os
import glob
import numpy as np
from PIL import Image

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
CROPPED_INVENTORY = "cropped_inventory.png"

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
    """Crop the inventory panel using template matching with debugging."""
    full_image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale full client image
    template = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)    # Grayscale reference inventory

    if full_image is None or template is None:
        raise FileNotFoundError("Full image or reference image not found!")

    # Perform template matching
    result = cv2.matchTemplate(full_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Debugging: Save the result map for inspection
    result_debug = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("template_match_result.png", result_debug)

    # Check match confidence
    print(f"Match Confidence: {max_val:.2f}")
    if max_val < 0.7:  # Confidence threshold
        raise ValueError("Inventory panel not found! Match confidence too low.")

    # Extract the matched area
    top_left_x, top_left_y = max_loc
    bottom_right_x = top_left_x + template.shape[1]
    bottom_right_y = top_left_y + template.shape[0]

    cropped_inventory = full_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    cv2.imwrite(save_path, cropped_inventory)

    print("Inventory cropped successfully.")
    return save_path

# ---- Remove Grid Lines ---- #
def remove_grid_lines(image_path, save_path):
    """Remove grid lines (white squares) from the inventory image."""
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define white color range in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Create a mask for white lines
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Replace white lines with the background color (median filtering)
    image[mask > 0] = [40, 40, 40]  # Replace with approximate inventory background color

    cv2.imwrite(save_path, image)
    return save_path

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
intents.message_content = True

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
                    # Crop inventory and clean it up
                    cropped_inventory_path = crop_inventory_panel(temp_image_path, REFERENCE_INVENTORY, "cropped_inventory.png")
                    cleaned_inventory_path = remove_grid_lines(cropped_inventory_path, "cleaned_inventory.png")

                    # Check slots for items
                    slot_matches = check_slots(cleaned_inventory_path, item_names, item_embeddings)

                    # Send cropped inventory for verification
                    await message.channel.send(file=discord.File(cleaned_inventory_path))

                    # Build response for detected items
                    if slot_matches:
                        response = "**Detected Items in Inventory:**\n"
                        for row, col, item, score in slot_matches:
                            response += f"Slot ({row}, {col}): {item} ({score:.2f})\n"
                        await message.channel.send(response)
                    else:
                        await message.channel.send("No matching items detected in the inventory.")

                except Exception as e:
                    await message.channel.send(f"Error: {str(e)}")

                os.remove(temp_image_path)  # Clean up temp image

# Run the bot
if TOKEN:
    client.run(TOKEN)
else:
    print("Error: DISCORD_BOT_TOKEN not found. Set the environment variable.")
