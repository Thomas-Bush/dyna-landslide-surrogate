import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, criterion, optimizer, device, augmentation=None):
        """Initialize the Trainer with model, criterion, optimizer, device, and augmentation."""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.augmentation = augmentation

    def train(self, train_loader, val_loader, epochs):
        """Train the model using the given data loaders and number of epochs."""
        self.model.to(self.device)
        for epoch in range(epochs):
            # Reset losses for each epoch
            train_loss = 0.0
            val_loss = 0.0
            num_train_samples = 0
            num_val_samples = 0

            # Training phase
            self.model.train()
            for inputs, targets in train_loader:
                # Apply augmentation if it is provided
                if self.augmentation:
                    inputs, targets = self.augmentation(inputs, targets)
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                num_train_samples += inputs.size(0)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    num_val_samples += inputs.size(0)

            # Calculate average losses
            avg_train_loss = train_loss / num_train_samples
            avg_val_loss = val_loss / num_val_samples

            # Print training and validation losses
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
          
  
    def test(self, test_loader, return_indices=None):
        """
        Test the model using the given data loader.
        
        Args:
            test_loader: DataLoader for the test data.
            return_indices (list or tuple, optional): Specific indices of the predictions and targets to return.
            
        Returns:
            tuple: A tuple containing the average test loss, predicted values, and target values.
        """
        
        test_loss = 0.0
        self.model.eval()
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        test_loss /= len(test_loader.dataset)
        
        # Concatenate all outputs and all targets
        all_outputs = torch.cat(all_outputs).cpu().numpy()
        all_targets = torch.cat(all_targets).cpu().numpy()
        
        # If return_indices is None, return the last sample by default
        if return_indices is None:
            predicted = all_outputs[-1]
            target = all_targets[-1]
        else:
            # Else, return the specified indices
            predicted = all_outputs[return_indices]
            target = all_targets[return_indices]
        
        return test_loss, predicted, target