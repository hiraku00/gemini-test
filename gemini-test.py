import os
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 4096,
}

# ===========================================
# model_name="gemini-pro"
# ===========================================
model_name = "gemini-pro"
model = genai.GenerativeModel(model_name=model_name,
                              generation_config=config)

# ----------------------------
# Execute (text)
# ----------------------------
prompt = [
    "こんにちはGemini！"
]
response = model.generate_content(prompt)
print(response.text)
print(f"===============================================")

# ----------------------------
# Execute Chat (text)
# ----------------------------
history = []
history.append({'role':'user',
                 'parts':["私は鈴木と言います"]})
history.append({'role':'model',
                 'parts':["こんにちは鈴木！"]})

chat = model.start_chat(history=history)
response = chat.send_message("こんにちは！私の名前は知っていますか？")
print(response.text)

# ===========================================
# model_name="gemini-pro-vision"
# ===========================================
model_name = "gemini-pro-vision"
model = genai.GenerativeModel(model_name=model_name,
                              generation_config=config)

# ----------------------------
# Execute (image)
# ----------------------------
img = PIL.Image.open('IMG_9768.jpg')
response = model.generate_content(img)
print(response.text)
print(f"===============================================")

# ----------------------------
# Execute (text & image)
# ----------------------------
prompt = [
    "この画像で作文を書いてみて"
]
response = model.generate_content([
    prompt,
    img
], stream=True)
response.resolve()
print(response.text)
print(f"===============================================")
