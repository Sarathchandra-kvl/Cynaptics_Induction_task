# 🎨 DCGAN - Deep Convolutional Generative Adversarial Network

This project is a PyTorch-based implementation of a Deep Convolutional GAN (DCGAN) for generating realistic-looking images from random noise. It demonstrates the foundational architecture of GANs through a generator-discriminator adversarial training framework.

## 🗂️ Project Structure

```
DCGAN/
├── main.py                # Full implementation of the DCGAN (the code you provided)
├── generated-samples/     # Folder to save generated sample images
└── README.md              # This file
```

## 📦 Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- OpenCV (cv2)
- tqdm
- numpy
- pandas

Install dependencies with:

```bash
pip install torch torchvision matplotlib opencv-python tqdm numpy pandas
```

## 📁 Dataset

Ensure your dataset is structured as an image folder (like `ImageFolder` expects) and update the path in:

```python
direc = 'Enter your dataset directory here'
```

## 🧠 Model Architecture

- **Generator:** Transforms latent noise vectors into realistic images via a stack of ConvTranspose2D layers.
- **Discriminator:** Binary classifier that distinguishes between real and fake images.

## 🔁 Training the GAN

The training loop consists of two key parts:

- **Discriminator training:** Maximizes ability to distinguish real from fake images.
- **Generator training:** Learns to fool the discriminator with more realistic fakes.

```python
run_GAN(epochs=50, lr=0.0002)
```

Sample images are saved to the specified directory after each epoch.

## 💾 Sample Output

During training, generated samples are saved using:

```python
sample_dir = 'Enter directory to save the samples produced'
```

Update this path to save results properly.

## 📈 Loss Tracking

- `loss_generator`: Tracks how well the generator is fooling the discriminator.
- `loss_discriminator`: Measures how accurately the discriminator can distinguish between real and fake images.

Progress and loss values are printed after each epoch.

## 🖼️ Image Normalization

Input images are normalized to `[-1, 1]` and the generator uses `Tanh()` for compatible output. De-normalization is applied before saving.

---



