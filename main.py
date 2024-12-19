import discord
import torch
import clip
import cv2
import os
import glob
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

# Ensure temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# ---- Load Reference Item Images ---- #
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

# ---- Crop Inventory Panel ---- #
def crop_inventory_panel(full_image_path, reference_path, save_path):
    full_image = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
    template = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    result = cv2.matchTemplate(full_image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.7:  # Confidence threshold
        raise ValueError("Inventory panel not found!")

    top_left = max_loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cropped_inventory = full_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.imwrite(save_path, cropped_inventory)

    return save_path

# ---- Check Items Per Slot ---- #
def check_slots(cropped_image_path, item_names, item_embeddings, empty_embedding_index):
    cropped_image = Image.open(cropped_image_path).convert("RGB")
    detected_items = []
    slot_coordinates = [
        (45, 30, 117, 94), (131, 30, 203, 94), (216, 30, 288, 94), (301, 30, 373, 94),
        (45, 103, 117, 168), (131, 103, 203, 168), (216, 103, 288, 168), (301, 103, 373, 168),
        # Add remaining slots as needed
    ]

    for idx, (left, upper, right, lower) in enumerate(slot_coordinates):
        slot_image = cropped_image.crop((left, upper, right, lower))
        slot_image_resized = slot_image.resize((224, 224))
        processed_image = preprocess(slot_image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            slot_embedding = model.encode_image(processed_image)

        similarities = []
        for item_name, ref_embedding in zip(item_names, item_embeddings):
            similarity = torch.cosine_similarity(slot_embedding, ref_embedding).item()
            similarities.append((item_name, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_match = similarities[0]

        if top_match[1] > 0.7 and top_match[0] != item_names[empty_embedding_index]:
            detected_items.append(f"Slot ({(idx // 4) + 1}, {(idx % 4) + 1}): {top_match[0]}")

    return detected_items

# ---- Discord Bot ---- #
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

item_names, item_embeddings = load_reference_images()
empty_embedding_index = item_names.index("empty")

@client.event
async def on_ready():
    print(f"Bot is ready! Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.channel.id != CHANNEL_ID or message.author.bot:
        return

    if message.attachments:
        await message.channel.send("Processing inventory image... Please wait.")
        for attachment in message.attachments:
            if attachment.filename.endswith((".png", ".jpg", ".jpeg")):
                temp_image_path = os.path.join(TEMP_FOLDER, attachment.filename)
                await attachment.save(temp_image_path)

                try:
                    cropped_inventory_path = crop_inventory_panel(temp_image_path, REFERENCE_INVENTORY, "cropped_inventory.png")
                    slot_matches = check_slots(cropped_inventory_path, item_names, item_embeddings, empty_embedding_index)
                    await message.channel.send(file=discord.File(cropped_inventory_path))
                    response = "**Detected Items in Inventory:**\n" + "\n".join(slot_matches) if slot_matches else "No matching items detected."
                    await message.channel.send(response)

                except Exception as e:
                    await message.channel.send(f"Error: {e}")

                os.remove(temp_image_path)

if TOKEN:
    client.run(TOKEN)
else:
    print("Error: DISCORD_BOT_TOKEN not found.")
