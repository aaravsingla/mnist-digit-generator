# 🔢 Handwritten Digit Generator

A web application that generates realistic handwritten digits (0-9) using a trained Generative Adversarial Network (GAN). Built with PyTorch and Streamlit, this project demonstrates the power of deep learning in creating synthetic handwritten digit images.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## 🌟 Features

- **Interactive Web Interface**: Easy-to-use Streamlit web app
- **Real-time Generation**: Generate 5 unique handwritten digit images instantly
- **Digit Selection**: Choose any digit from 0 to 9 using an intuitive slider
- **GAN-Powered**: Uses a trained conditional GAN for high-quality image generation
- **Responsive Design**: Clean, modern UI with hover effects and styling
- **Development Ready**: Includes dev container configuration for easy setup

## 🎯 Demo

The application allows users to:
1. Select a digit (0-9) using a slider
2. Click "Generate Images" to create 5 unique variations
3. View the generated handwritten digit images in a grid layout

## 🏗️ Architecture

### Model Components

- **Generator Network**: 
  - Conditional GAN generator with label embedding
  - Takes random noise + digit label as input
  - Outputs 28x28 grayscale images
  - Uses BatchNorm and LeakyReLU for stable training

- **Architecture Details**:
  - Latent dimension: 100
  - Embedding dimension: 100
  - Output image shape: 1×28×28 (MNIST format)
  - 10 classes (digits 0-9)

### Network Structure
```
Input: [Noise Vector (100) + Label Embedding (100)]
├── Linear(200 → 128) + BatchNorm + LeakyReLU
├── Linear(128 → 256) + BatchNorm + LeakyReLU  
├── Linear(256 → 512) + BatchNorm + LeakyReLU
├── Linear(512 → 1024) + BatchNorm + LeakyReLU
└── Linear(1024 → 784) + Tanh
Output: 28×28 grayscale image
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd handwritten-digit-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the trained model**
   - Place your `final_generator.pth` file in the project root directory
   - This file contains the trained GAN generator weights

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start generating handwritten digits!

## 🐳 Development with Dev Containers

This project includes a complete dev container setup for consistent development environments.

### Using VS Code Dev Containers

1. **Prerequisites**
   - Install [Docker](https://www.docker.com/get-started)
   - Install [VS Code](https://code.visualstudio.com/)
   - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open in Dev Container**
   - Open the project folder in VS Code
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Dev Containers: Reopen in Container"
   - Select the command and wait for the container to build

3. **Automatic Setup**
   - The container automatically installs all dependencies
   - Streamlit server starts automatically on port 8501
   - Ready to develop with Python extensions pre-configured

### Using GitHub Codespaces

1. Click the "Code" button on GitHub
2. Select "Create codespace on main"
3. Wait for the environment to set up
4. The app will automatically start and be available via port forwarding

## 📁 Project Structure

```
handwritten-digit-generator/
├── .devcontainer/
│   └── devcontainer.json          # Dev container configuration
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── final_generator.pth           # Trained GAN model (not in repo)
└── README.md                     # Project documentation
```

## 🔧 Configuration

### Model Parameters

The application uses these hyperparameters (must match training):

```python
latent_dim = 100        # Noise vector dimension
num_classes = 10        # Number of digit classes (0-9)
embedding_dim = 100     # Label embedding dimension
img_shape = (1, 28, 28) # MNIST image dimensions
```

### Streamlit Configuration

The app runs with:
- CORS disabled for dev containers
- XSRF protection disabled for dev environments
- Auto port forwarding on 8501

## 🎨 UI Features

- **Modern Design**: Clean, responsive interface with custom CSS
- **Interactive Elements**: Hover effects and smooth transitions
- **Grid Layout**: Images displayed in organized columns
- **Status Indicators**: Loading states and success messages
- **Error Handling**: Graceful error messages for missing models

## 🧠 Model Training

While the training code isn't included in this repository, the model was trained on:

- **Dataset**: MNIST handwritten digits
- **Architecture**: Conditional GAN
- **Training**: Adversarial training with Generator and Discriminator
- **Output**: `final_generator.pth` containing trained generator weights

## 🚨 Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   Error: Model file 'final_generator.pth' not found
   ```
   **Solution**: Ensure the trained model file is in the project root directory

2. **Memory issues**
   ```
   RuntimeError: out of memory
   ```
   **Solution**: The app runs on CPU by default. Ensure sufficient RAM is available.

3. **Port already in use**
   ```
   Port 8501 is already in use
   ```
   **Solution**: Stop other Streamlit apps or use a different port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## 🔮 Future Enhancements

- [ ] Support for custom image sizes
- [ ] Batch generation with download option
- [ ] Interactive training interface
- [ ] Additional datasets (Fashion-MNIST, CIFAR-10)
- [ ] Model comparison tools
- [ ] Advanced styling options for generated images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Classic handwritten digit dataset
- **PyTorch**: Deep learning framework
- **Streamlit**: Web app framework
- **Dev Containers**: Consistent development environments

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out via email.

---

⭐ **Star this repository if you found it helpful!**
