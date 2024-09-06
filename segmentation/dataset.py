import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random

from segmentation.puzzle_generator import PuzzleGenerator
from stn.resnet import ResNet, ResidualBlock
from stn.resnet import ResNet18, ResNet34

class MultiTaskDataset(Dataset):
    def __init__(self, image_paths, transform=None, difficulty='medium'):
        self.image_paths = image_paths
        self.transform = transform
        self.puzzle_generator = PuzzleGenerator(difficulty)

    def __getitem__(self, idx):
        image = "C:\\Users\\pasca\\Data Science\\Math Notes Model\\OCR\\test_data\\math_example.png"  

        # Contrastive task
        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
        
        # Puzzle task
        puzzle_image, puzzle_solution = self.puzzle_generator.generate_puzzle(image)
        if self.transform:
            puzzle_image = self.transform(puzzle_image)
        
        return image1, image2, puzzle_image, puzzle_solution
    
    def create_jigsaw_puzzle(self, image):
        # Split the image into tiles
        tiles = []
        tile_size = image.size[0] // self.jigsaw_tiles
        for i in range(self.jigsaw_tiles):
            for j in range(self.jigsaw_tiles):
                tile = image.crop((j*tile_size, i*tile_size, 
                                   (j+1)*tile_size, (i+1)*tile_size))
                tiles.append(tile)
        
        # Shuffle tiles
        order = list(range(self.jigsaw_tiles**2))
        random.shuffle(order)
        
        # Create shuffled image
        shuffled_image = Image.new('RGB', image.size)
        for i, idx in enumerate(order):
            row = i // self.jigsaw_tiles
            col = i % self.jigsaw_tiles
            shuffled_image.paste(tiles[idx], (col*tile_size, row*tile_size))
        
        if self.transform:
            shuffled_image = self.transform(shuffled_image)
        
        return shuffled_image, torch.tensor(order)

    def __len__(self):
        return len(self.image_paths)

class MultiTaskEncoder(nn.Module):
    def __init__(self, base_model='resnet18', num_classes=1000, feature_dim=2048):
        super(MultiTaskEncoder, self).__init__()
        self.feature_dim = feature_dim

        # Use the custom ResNet architecture
        if base_model == 'resnet18':
            self.base_model = ResNet18(num_classes=num_classes)
            self.base_feature_dim = 512
        elif base_model == 'resnet34':
            self.base_model = ResNet34(num_classes=num_classes)
            self.base_feature_dim = 512
        else:
            raise ValueError(f"Unsupported base model: {base_model}")

        # Remove the final fully connected layer from the base model
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Add a custom fully connected layer
        self.fc = nn.Linear(self.base_feature_dim, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Usage example
if __name__ == "__main__":
    # Create an instance of the encoder
    encoder = MultiTaskEncoder(base_model='resnet18', num_classes=1000, feature_dim=2048)
    
    # Create a sample input tensor
    sample_input = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)
    
    # Get the output
    output = encoder(sample_input)
    
    print(f"Output shape: {output.shape}")  # Should print: Output shape: torch.Size([1, 2048])

class MultiTaskModel(nn.Module):
    def __init__(self, base_model='resnet18', num_classes=1000, feature_dim=2048, num_jigsaw_tiles=3):
        super().__init__()
        self.encoder = MultiTaskEncoder(base_model=base_model, num_classes=num_classes, feature_dim=feature_dim)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.jigsaw_classifier = nn.Linear(2048, num_jigsaw_tiles**2)

    def forward(self, x, task='both'):
        features = self.encoder(x)
        if task == 'contrastive' or task == 'both':
            contrastive_output = self.contrastive_projection(features)
        if task == 'jigsaw' or task == 'both':
            jigsaw_output = self.jigsaw_classifier(features)
        
        if task == 'both':
            return contrastive_output, jigsaw_output
        elif task == 'contrastive':
            return contrastive_output
        elif task == 'jigsaw':
            return jigsaw_output

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.shape[0]
    s = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.exp(torch.mm(s, s.t().contiguous()) / temperature)
    mask = torch.eye(N, dtype=torch.bool).repeat(2, 2)
    sim_matrix = sim_matrix.masked_fill_(mask, 0)
    loss = -torch.log(sim_matrix.sum(dim=1) / (sim_matrix.sum(dim=1) - torch.diag(sim_matrix)))
    loss = loss.mean()
    return loss

def train_multi_task(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (x1, x2, jigsaw_x, jigsaw_target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Contrastive task
            z1, z2 = model(x1, task='contrastive'), model(x2, task='contrastive')
            contrastive_loss_val = contrastive_loss(z1, z2)
            
            # Jigsaw task
            jigsaw_output = model(jigsaw_x, task='jigsaw')
            jigsaw_loss = nn.CrossEntropyLoss()(jigsaw_output, jigsaw_target)
            
            # Combined loss
            loss = contrastive_loss_val + jigsaw_loss
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Usage example
if __name__ == "__main__":
    # Create an instance of the MultiTaskModel
    model = MultiTaskModel(base_model='resnet18', num_classes=1000, feature_dim=2048, num_jigsaw_tiles=3)
    
    # Create sample input tensors
    sample_input = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)
    
    # Get the outputs
    contrastive_output, jigsaw_output = model(sample_input, task='both')
    
    print(f"Contrastive output shape: {contrastive_output.shape}")  # Should be [1, 128]
    print(f"Jigsaw output shape: {jigsaw_output.shape}")  # Should be [1, 9] for 3x3 jigsaw puzzle




# Usage
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MultiTaskDataset(image_paths, transform=transform, jigsaw_tiles=3)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

model = MultiTaskModel(num_jigsaw_tiles=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_multi_task(model, dataloader, optimizer, epochs=100)