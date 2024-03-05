import torch
from torch.utils.data import DataLoader

class Trainer:
    """A trainer class for the CNN-LSTM model."""

    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, device):
        """
        Initializes the Trainer.

        Args:
            model: The CNN-LSTM model to be trained.
            train_dataloader: DataLoader for the training data.
            val_dataloader: DataLoader for the validation data.
            criterion: Loss function.
            optimizer: Optimizer for the model parameters.
            device: The device to run the training on ('cuda' or 'cpu').
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self):
        """Train the model for one epoch."""
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        return epoch_loss

    def validate(self):
        """Validate the model."""
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.val_dataloader.dataset)
        return epoch_loss

    def train(self, num_epochs):
        """Run the training process.

        Args:
            num_epochs: Number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            # Print epoch summary
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')

# Example usage:
# Assume you have created instances of the model, criterion, optimizer, and dataloaders.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trainer = Trainer(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
#                   criterion=criterion, optimizer=optimizer, device=device)
# trainer.train(num_epochs=25)