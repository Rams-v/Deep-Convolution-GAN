# DCGAN: Deep Convolutional Generative Adversarial Networks

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch. The model is trained to generate realistic images using the CelebA dataset.

---
## Dataset Preprocessing

### **1. Downloading the Dataset**
- Ensure that you have the CelebA dataset or any image dataset stored locally.
- The dataset should be organized in a folder, where each image is stored as an individual file.
- Update the dataset path in the script:
  ```python
  dataroot = "/kaggle/input/celeba-dataset"
  ```

### **2. Data Transformation**
- Images are preprocessed before being fed into the DCGAN model. The following transformations are applied:
  - Resize all images to **64×64** pixels to match the input size of the model.
  - Center crop to ensure images have the correct aspect ratio.
  - Convert images to tensors.
  - Normalize pixel values to the range **[-1, 1]** for stable training.

  ```python
  dataset = dset.ImageFolder(root=dataroot,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.CenterCrop(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
  ```

### **3. Loading Data**
- The dataset is loaded using PyTorch’s `DataLoader`, which efficiently batches and shuffles images for training:
  ```python
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
  ```

---
## Training the Model

### **1. Install Dependencies**
Ensure you have the required libraries installed:
```bash
pip install torch torchvision matplotlib numpy
```

### **2. Run the Training Script**
To train the DCGAN model, execute the script:
```bash
python train_dcgan.py
```

### **3. Training Parameters**
Key hyperparameters for training include:
- **Batch Size**: 128
- **Image Size**: 64×64
- **Latent Vector Size (nz)**: 100
- **Generator Feature Maps (ngf)**: 64
- **Discriminator Feature Maps (ndf)**: 64
- **Learning Rate (lr)**: 0.0002
- **Beta1 (Adam Optimizer Momentum Term)**: 0.5
- **Number of Epochs**: 5 (adjustable for better results)

---
## Testing the Model

### **1. Generate Images**
- Once the model is trained, it can generate new images using random noise input.
- A batch of fake images can be generated using:
  ```python
  with torch.no_grad():
      fake_images = netG(fixed_noise).detach().cpu()
  plt.imshow(np.transpose(vutils.make_grid(fake_images[:64], padding=2, normalize=True), (1,2,0)))
  ```

### **2. Evaluate Training Progress**
- During training, real and fake images are plotted to track improvements.
- Loss curves for both generator and discriminator are logged.
- At the end of training, a set of generated images is saved for evaluation.

---
## Expected Outputs

### **1. Training Process**
- The generator gradually learns to create more realistic images over time.
- The discriminator's loss stabilizes as it reaches equilibrium with the generator.

### **2. Generated Images**
- Initially, generated images may appear as noise.
- After a few epochs, the model starts producing recognizable facial structures.
- With enough training, generated images resemble real human faces.

### **3. Animation of Generated Images**
- The training script saves intermediate results and creates an animation of generated images improving over epochs.
  ```python
  fig = plt.figure(figsize=(8,8))
  ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
  anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
  ```

---
## References
- **DCGAN Paper**: [Radford et al.](https://arxiv.org/abs/1511.06434)
- **PyTorch DCGAN Tutorial**: [Official PyTorch Guide](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

For improvements, consider training with more epochs, experimenting with different datasets, or fine-tuning network architectures.

---
## License
This project is open-source and available for educational purposes.

