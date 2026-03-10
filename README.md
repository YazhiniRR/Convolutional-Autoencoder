# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

The objective of this experiment is to develop an Autoencoder model using PyTorch to remove noise from images.
The MNIST dataset is used, which contains handwritten digit images (0–9) of size 28 × 28 pixels.
Noise is added to the images and the autoencoder is trained to reconstruct the original clean image from the noisy input.
<img width="563" height="417" alt="image" src="https://github.com/user-attachments/assets/f900d3b2-48d9-4446-a6a3-575dacbe996b" />


## DESIGN STEPS
## STEP 1:
Import the required Python libraries such as PyTorch, Torchvision, NumPy, and Matplotlib and load the MNIST dataset using DataLoader.
## STEP 2:
Add random noise to the input images and design the Autoencoder architecture with encoder and decoder layers.
## STEP 3:
Train the model using a suitable loss function (MSE Loss) and optimizer (Adam) to minimize the reconstruction error.
## STEP 4:
Test the trained model on noisy images and visualize the original, noisy, and denoised images.

## PROGRAM
### Name: YAZHINI R R
### Register Number: 212224100063
1️⃣ Define Autoencoder
```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```
2️⃣ Initialize Model
```
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
3️⃣ Training Function
```
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()

    for epoch in range(epochs):
        running_loss = 0

        for images, _ in loader:
            images = images.to(device)

            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
```

## OUTPUT


### Model Summary
<img width="841" height="608" alt="image" src="https://github.com/user-attachments/assets/e95e69a1-1240-49c9-8503-c90da5b8276d" />


### Original vs Noisy Vs Reconstructed Image
<img width="1071" height="629" alt="image" src="https://github.com/user-attachments/assets/620b0373-edf2-4108-8876-98dbde719aaf" />


## RESULT
The Denoising Autoencoder model was successfully implemented using PyTorch and trained on the MNIST dataset.
The model was able to reconstruct clean images from noisy input images.
The output visualization showed the original image, noisy image, and denoised image, demonstrating that the autoencoder effectively removed noise and restored the digit images.



