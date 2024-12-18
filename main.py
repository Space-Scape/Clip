import discord
import torch
import clip
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
REFERENCE_FOLDER = "Images"          # Reference OSRS item images
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

# ---- Compare Uploaded Image ---- #
def compare_image_to_references(image_path, item_names, item_embeddings):
    try:
        uploaded_image = Image.open(image_path).resize((224, 224)).convert("RGB")
        processed_image = preprocess(uploaded_image).unsqueeze(0).to(device)
        with torch.no_grad():
            uploaded_embedding = model.encode_image(processed_image)
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        return None

    similarities = []
    for idx, ref_embedding in enumerate(item_embeddings):
        similarity = torch.cosine_similarity(uploaded_embedding, ref_embedding).item()
        similarities.append((item_names[idx], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:5]  # Return top 5 matches

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

                # Compare the uploaded image
                top_matches = compare_image_to_references(temp_image_path, item_names, item_embeddings)

                # Send results back to Discord
                if top_matches:
                    response = "**Top Matches:**\n"
                    for item, score in top_matches:
                        response += f"- {item}: {score:.4f}\n"
                    await message.channel.send(response)
                else:
                    await message.channel.send("Failed to process the image.")

                os.remove(temp_image_path)  # Clean up temp image

# Run the bot
if TOKEN:
    client.run(TOKEN)
else:
    print("Error: DISCORD_BOT_TOKEN not found. Set the environment variable.")
