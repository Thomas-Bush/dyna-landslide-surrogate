import torch

class Trainer:
    """Handles the training process for the CNN-LSTM model."""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        """Initializes the Trainer with model, data loaders, loss function, and optimizer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()  # Set the model to training mode
        for batch in self.train_loader:
            inputs, targets = batch
            self.optimizer.zero_grad()  # Zero the parameter gradients
            outputs = self.model(inputs)  # Forward pass
            loss = self.criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backward pass
            self.optimizer.step()  # Optimize the model

    def validate(self):
        """Validates the model on the validation dataset."""
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for batch in self.val_loader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def train(self, num_epochs):
        """Trains the model for a specified number of epochs."""
        for epoch in range(num_epochs):
            self.train_epoch()  # Train the model for one epoch
            val_loss = self.validate()  # Validate the model
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')