import torch
import torchvision.transforms as transforms
from PIL import Image
import random

class ContrastiveGenerator:
    def __init__(self, transform=None, num_views=2):
        self.transform = transform or self.default_transform()
        self.num_views = num_views

    def default_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate_contrastive_pair(self, image):
        """Generate a pair of augmented views for a single image."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        views = [self.transform(image) for _ in range(self.num_views)]
        return tuple(views)

    def generate_batch(self, images, batch_size):
        """Generate a batch of contrastive pairs."""
        batch = []
        for _ in range(batch_size):
            if isinstance(images, list):  # If images is a list of file paths
                image = random.choice(images)
            else:  # If images is a Dataset
                image = images[random.randint(0, len(images) - 1)]
            
            views = self.generate_contrastive_pair(image)
            batch.append(views)
        
        # Transpose the batch to have shape (num_views, batch_size, channels, height, width)
        return [torch.stack(view) for view in zip(*batch)]

# Usage example
if __name__ == "__main__":
    # Assuming you have a list of image paths or a Dataset object
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]  # or your Dataset object

    generator = ContrastiveGenerator()
    batch = generator.generate_batch(image_paths, batch_size=32)
    
    print(f"Generated batch shape: {[b.shape for b in batch]}")