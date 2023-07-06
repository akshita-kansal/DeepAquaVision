import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.convx = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)


        # Decoder
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(768, 128, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(384, 64, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(192, 3, kernel_size=4, stride=2, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bnx = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.bny = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x)))
        x2 = F.relu(self.bn3(self.conv3(x1)))
        x3 = F.relu(self.bn4(self.conv4(x2)))
        x4 = F.relu(self.bn5(self.conv5(x3)))
        x5 = F.relu(self.bn6(self.conv6(x4)))
        xx = F.relu(self.bnx(self.convx(x5)))

        # Decoder
        x6 = F.relu(self.bn7(self.deconv1(xx)))
        x6 = torch.cat([x6, x5], dim=1)
        x7 = F.relu(self.bny(self.deconv2(x6)))
        x7 = torch.cat([x7, x4], dim=1)
        x8 = F.relu(self.bn8(self.deconv3(x7)))
        x8 = torch.cat([x8, x3], dim=1)
        x9 = F.relu(self.bn9(self.deconv4(x8)))
        x9 = torch.cat([x9, x2], dim=1)
        x10 = F.relu(self.bn10(self.deconv5(x9)))
        x10 = torch.cat([x10, x1], dim=1)
        x11 = self.deconv6(x10)
        x11 = torch.sigmoid(x11)
        return x11





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)

        self.conv8 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.flat=nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.relu4(self.bn4(self.conv4(out)))
        out = self.relu5(self.bn5(self.conv5(out)))
        out = self.relu7(self.bn7(self.conv7(out)))
        out = self.sigmoid(self.flat(self.conv8(out)))
        return out


class underwaterDataset(Dataset):
    def __init__(self, low_folder, high_folder):
        self.low_images = os.listdir(low_folder)
        self.high_images = os.listdir(high_folder)
        self.low_folder = low_folder
        self.high_folder = high_folder

    def __getitem__(self, index):
        low_path = os.path.join(self.low_folder, self.low_images[index])
        high_path = os.path.join(self.high_folder, self.high_images[index])

        low_image = Image.open(low_path)
        high_image = Image.open(high_path)

        transform = transforms.ToTensor()

        low_image = transform(low_image)
        high_image = transform(high_image)

        return low_image, high_image

    def __len__(self):
        return len(self.low_images)



Discriminator=Discriminator()
generator =generator()


criterion = nn.MSELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_Dis = optim.Adam(Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

dataset = underwaterDataset('low', 'high')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
num_epochs = 10


def psnr(target, prediction):
   
    target = torch.clamp(target, 0.0, 1.0)
    prediction = torch.clamp(prediction, 0.0, 1.0)

    mse = F.mse_loss(target, prediction)

 
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr_value


for epoch in range(num_epochs):
    for i, data in enumerate(data_loader):
        

        inputs, ground_truth = data
        batch_size = inputs.size(0)
        
        optimizer_Dis.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        real_predictions = Discriminator(ground_truth)
        d_loss_real = criterion(real_predictions, real_labels)

        fake_images = generator(inputs)
        fake_labels = torch.zeros(batch_size, 1)
        fake_predictions = Discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_predictions, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_Dis.step()
         
        optimizer_G.zero_grad()
        targets = torch.ones(batch_size, 1)
        fake_predictions = Discriminator(fake_images)
        g_loss = criterion(fake_predictions, targets)
        g_loss.backward()
        optimizer_G.step()
        psnr_value = psnr(ground_truth, fake_images)

    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(data_loader)}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f} PSNR: {psnr_value.item():.2f}")

