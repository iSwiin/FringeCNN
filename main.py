import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class FringeSharpeningNet(nn.Module):
    def __init__(self):
        super(FringeSharpeningNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class FringeDataset(Dataset):
    def __init__(self, input_image_path, target_image_path, transform=None):
        self.input_image_path = input_image_path
        self.target_image_path = target_image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        input_image = Image.open(self.input_image_path).convert('L')  # Convert to grayscale
        target_image = Image.open(self.target_image_path).convert('L')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image



input_image_path = "blurred.png"
target_image_path = "final.png"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


dataset = FringeDataset(input_image_path, target_image_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


model = FringeSharpeningNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    for input_image, target_image in dataloader:
        output = model(input_image)
        loss = criterion(output, target_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


torch.save(model.state_dict(), 'fringe_sharpening_net.pth')


model = FringeSharpeningNet()
model.load_state_dict(torch.load('fringe_sharpening_net.pth'))
model.eval()


def process_image(input_image_path, output_image_path):

    input_image = Image.open(input_image_path).convert('L')
    input_image = transform(input_image).unsqueeze(0)


    with torch.no_grad():
        output_image = model(input_image)


    output_image = output_image.squeeze(0).cpu().numpy()
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_image[0], mode='L')


    output_image.save(output_image_path)


new_input_image_path = 'test.png'
new_output_image_path = 'output_image.png'


process_image(new_input_image_path, new_output_image_path)