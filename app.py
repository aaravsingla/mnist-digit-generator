import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import io

# --- Model Architecture (must match the training script exactly) ---
# Hyperparameters (must match training script)
latent_dim = 100
num_classes = 10
embedding_dim = 100
img_shape = (1, 28, 28) # MNIST image shape (channels, height, width)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape, embedding_dim):
        super().__init__()
        self.img_shape = img_shape
        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        # Helper function to create a linear block
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # BatchNorm1d helps stabilize training
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # LeakyReLU for non-linearity, a common choice in GANs
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Generator's sequential model
        # It takes (noise + label embedding) as input and outputs a flattened image
        self.model = nn.Sequential(
            *block(latent_dim + embedding_dim, 128, normalize=False), # First layer without normalization
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # Output layer: linear to the total number of pixels, then Tanh to scale to [-1, 1]
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh() # Output activation to scale pixel values to [-1, 1]
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise vector
        # The label embedding provides conditioning for the generator
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        # Pass through the generator model
        img = self.model(gen_input)
        # Reshape the flattened output into image format (channels, height, width)
        img = img.view(img.size(0), *self.img_shape)
        return img

# --- Load the trained Generator model ---
# @st.cache_resource decorator caches the model loading, so it runs only once
# when the app starts, improving performance.
@st.cache_resource
def load_generator_model():
    """
    Loads the trained Generator model from 'final_generator.pth'.
    It maps the model to CPU as Streamlit Cloud environments usually don't have GPUs.
    Includes error handling for file not found or other loading issues.
    """
    generator = Generator(latent_dim, num_classes, img_shape, embedding_dim)
    model_path = 'final_generator.pth'
    try:
        # Load the model state dictionary, mapping to CPU
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        generator.eval() # Set the model to evaluation mode (important for BatchNorm/Dropout)
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory as app.py.")
        st.stop() # Stop the app execution if model is not found
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() # Stop the app execution on other loading errors
    return generator

# Load the generator model once
generator = load_generator_model()

# --- Streamlit UI Components ---
st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stSlider > div > div > div {
        background-color: #007bff;
    }
    .stButton > button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background-color: #218838;
        transform: translateY(-2px);
    }
    .stImage img {
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #333;
        text-align: center;
    }
    p {
        text-align: center;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔢 Handwritten Digit Generator")
st.write("Welcome! This app generates 5 unique handwritten images for any digit from 0 to 9 using a trained Generative Adversarial Network (GAN).")

# Slider for user to select the digit
digit_to_generate = st.slider("Choose the digit you want to generate:", 0, 9, 0)

# Button to trigger image generation
if st.button("✨ Generate Images"):
    st.info(f"Generating 5 images for digit: **{digit_to_generate}**...")

    generated_images_list = []
    with torch.no_grad(): # Disable gradient calculation for inference
        for i in range(5): # Generate 5 images
            # Generate random noise vector for input to the generator
            # Each image needs a different noise vector for diversity
            noise = torch.randn(1, latent_dim) 
            # Create a tensor for the desired label (digit)
            labels = torch.tensor([digit_to_generate], dtype=torch.long)
            
            # Generate the image using the model
            # Squeeze(0) removes the batch dimension (1, 1, 28, 28) -> (1, 28, 28)
            # .cpu() ensures the tensor is on CPU before converting to NumPy
            img_tensor = generator(noise, labels).squeeze(0).cpu() 
            
            # Denormalize from [-1, 1] to [0, 1] for display
            # The Tanh activation outputs values between -1 and 1.
            # To display as an image, pixel values typically need to be in [0, 1] or [0, 255].
            img_np = (img_tensor.numpy() + 1) / 2.0  
            # Convert to 0-255 range and then to unsigned 8-bit integer format
            img_np = (img_np * 255).astype(np.uint8) 

            # Convert NumPy array to PIL Image
            # For grayscale, PIL's Image.fromarray expects (H, W) or (H, W, 1)
            # Since img_np is (1, 28, 28) after squeeze(0), img_np[0] gives (28, 28)
            generated_images_list.append(Image.fromarray(img_np[0])) 

    st.subheader(f"✅ Generated Images for Digit {digit_to_generate}:")
    
    # Display images in a grid using Streamlit columns
    cols = st.columns(5) # Create 5 columns for layout
    for i, img_pil in enumerate(generated_images_list):
        with cols[i]: # Place each image in its own column
            st.image(img_pil, caption=f"Image {i+1}", width=100) # Adjust width as needed

st.markdown("---")
st.markdown("🌐 Powered by PyTorch and Streamlit.")
st.markdown("Model trained on the **MNIST dataset**.")

