import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
num_classes = 10
embedding_dim = 100 # For class embedding
epochs = 50 # You can adjust this based on desired quality and training time
batch_size = 128
lr = 0.0002
beta1 = 0.5

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# --- Generator Model ---
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape, embedding_dim):
        super().__init__()
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + embedding_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# --- Discriminator Model ---
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape, embedding_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))) + embedding_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        # Concatenate label embedding and image
        d_input = torch.cat((img_flat, self.label_embedding(labels)), -1)
        validity = self.model(d_input)
        return validity

# Initialize models
img_shape = (1, 28, 28)
generator = Generator(latent_dim, num_classes, img_shape, embedding_dim).to(device)
discriminator = Discriminator(num_classes, img_shape, embedding_dim).to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
os.makedirs("generated_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # Configure input
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()

        # Real images
        real_pred = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Fake images
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (imgs.size(0),)).to(device) # Generate random labels for fake images
        fake_imgs = generator(z, gen_labels)
        fake_pred = discriminator(fake_imgs.detach(), gen_labels) # Detach to prevent gradient flow to generator
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        optimizer_G.zero_grad()

        # Generate images with the original labels (or desired labels for generation)
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        # Use the real labels for generator training to encourage it to produce specific digits
        generated_imgs = generator(z, labels)
        g_pred = discriminator(generated_imgs, labels)
        g_loss = adversarial_loss(g_pred, valid)

        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if i % 100 == 0:
            tqdm.write(
                f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

    # Save generated images for visual inspection
    if epoch % 5 == 0:
        with torch.no_grad():
            # Generate 5 images for each digit (0-9)
            for digit in range(num_classes):
                # Ensure diversity by using different noise vectors for each image
                sample_noise = torch.randn(5, latent_dim).to(device)
                sample_labels = torch.full((5,), digit, dtype=torch.long).to(device)
                generated_samples = generator(sample_noise, sample_labels).cpu()

                # Create a grid of 5 images for the current digit
                filename = f"generated_images/epoch_{epoch:03d}_digit_{digit}.png"
                # CORRECTED LINE: Removed the 'range' argument
                save_image(generated_samples.data, filename, nrow=5, normalize=True)

    # Save model checkpoints
    torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch}.pth")

print("Training complete!")

# Save the final generator model
torch.save(generator.state_dict(), "saved_models/final_generator.pth")
