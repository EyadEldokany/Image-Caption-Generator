import streamlit as st
import torch
import random
from PIL import Image
from googletrans import Translator
import torchvision.transforms as transforms
# === Load Models ===
from ImageCaptionModel import ImageCaptionModel
from PositionalEncoding import PositionalEncoding
from ImageCaptionModelViT import ImageCaptionModelViT
from ImageCaptionModelDenseNet import ImageCaptionModelDenseNet
from ImageCaptionModelConvNeXT import ImageCaptionModelConvNeXT
# === Load Vocabulary ===
word_to_index = torch.load("word_mappings\word_to_index.pth")
index_to_word = torch.load("word_mappings\index_to_word.pth")
print(list(word_to_index.keys())[:10])  # Show first 10 keys

start_token = word_to_index["<start>"]
end_token = word_to_index["<end>"]
pad_token = word_to_index["<pad>"]
max_seq_len = 33

translator = Translator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model(model_path):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


models = {
    "ResNet": {
        "model": load_model("ResNet\BestModel_ResNet.pth"),
        "encoder_path": "ResNet/EncodedImageValidResNet.pkl"
    },
    "ViT": {
        "model": load_model("ViT/BestModelViT"),
        "encoder_path": "ViT/EncodedImageValidViT.pkl"
    },
    "DenseNet": {
        "model": load_model("DenseNet/BestModelDenseNet"),
        "encoder_path": "DenseNet/EncodedImageValidDenseNet.pkl"
    },
    "ConvNeXT": {
        "model": load_model("ConvNext/BestModelConvNeXT"),
        "encoder_path": "ConvNext/EncodedImageValidConvNeXT.pkl"
    }
}

# === Helper to generate caption ===
def generate_caption(model, image_embed, K=1):
    input_seq = [pad_token] * max_seq_len
    input_seq[0] = start_token
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)

    predicted_sentence = []
    with torch.no_grad():
        for i in range(1, max_seq_len):
            output, padding_mask = model(image_embed, input_seq)
            output = output[i - 1, 0, :]  # get the last word
            topk = torch.topk(output, K)
            next_word_index = random.choices(topk.indices.tolist(), topk.values.tolist(), k=1)[0]
            input_seq[:, i] = next_word_index
            next_word = index_to_word[next_word_index]
            if next_word == "<end>":
                break
            predicted_sentence.append(next_word)

    return " ".join(predicted_sentence)

# === Streamlit UI ===
st.set_page_config(page_title="Image Caption Generator", layout="centered", initial_sidebar_state="auto")

st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        .css-1d391kg { background-color: #0e1117 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image and select a model to generate a caption (with Arabic translation).")

# === Sidebar Model Selector ===
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))

# === Upload Image ===
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (depends on encoder used)
    # Preprocess the image (depends on encoder used)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    # This line is causing the issue for ViT and potentially other models
    image_embed = image_tensor.view(image_tensor.size(0), -1, image_tensor.size(1))  # simple flatten if needed
    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            model_info = models[model_name]
            caption = generate_caption(model_info["model"], image_embed)
            st.subheader("Predicted Caption (English)")
            st.write(caption)

            translation = translator.translate(caption, dest="ar").text
            st.subheader("Predicted Caption (Arabic)")
            st.write(translation)
