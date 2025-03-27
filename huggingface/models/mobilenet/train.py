import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class MobileNetTrainer:
    def __init__(self):
        """
        Initialize the MobileNet trainer.
        """
        # Initialize MobileNetV2
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Modify the classifier for our needs
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 10)  # 10 classes example
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def prepare_dataset(self):
        """
        Prepare and preprocess the dataset.
        """
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load a sample dataset (replace with your actual dataset)
        dataset = datasets.ImageFolder(
            root='path/to/your/dataset',
            transform=transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        
        return train_loader
        
    def train(self, num_epochs=10):
        """
        Train the MobileNet model.
        """
        # Prepare the dataset
        train_loader = self.prepare_dataset()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
        
        # Save the model
        torch.save(self.model.state_dict(), './mobilenet.pth')

if __name__ == "__main__":
    trainer = MobileNetTrainer()
    trainer.train()
