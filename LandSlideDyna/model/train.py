import torch
from torch.utils.data import DataLoader

# class Trainer:
#     def __init__(self, model, criterion, optimizer, device, augmentation=None):
#         """
#         Initialize the Trainer with model, criterion, optimizer, device, and augmentation.
        
#         Args:
#             model: The neural network model to train.
#             criterion: The loss function used for training.
#             optimizer: The optimization algorithm used for training.
#             device: The device to run the training on ('cuda' or 'cpu').
#             augmentation: The augmentation to apply to the training data (default is None).
#         """
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.device = device
#         self.augmentation = augmentation

#     def train(self, train_loader, val_loader, epochs):
#         """
#         Train the model using the given data loaders and number of epochs.
        
#         Args:
#             train_loader: DataLoader for the training data.
#             val_loader: DataLoader for the validation data.
#             epochs: Number of epochs to train the model for.
#         """
#         self.model.to(self.device)
#         for epoch in range(epochs):
#             # Training phase
#             self.model.train()
#             train_loss = 0.0
#             for inputs, targets in train_loader:
#                 # Apply augmentation if it is provided
#                 if self.augmentation:
#                     inputs, targets = self.augmentation(inputs, targets)
                
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, targets)
#                 loss.backward()
#                 self.optimizer.step()
#                 train_loss += loss.item() * inputs.size(0)

#             # Validation phase
#             val_loss = 0.0
#             self.model.eval()
#             with torch.no_grad():
#                 for inputs, targets in val_loader:
#                     inputs, targets = inputs.to(self.device), targets.to(self.device)
#                     outputs = self.model(inputs)
#                     loss = self.criterion(outputs, targets)
#                     val_loss += loss.item() * inputs.size(0)

#             # Calculate average losses
#             train_loss /= len(train_loader.dataset)
#             val_loss /= len(val_loader.dataset)

#             # Print training and validation losses
#             print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


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

    # def test(self, test_loader):
    #     """
    #     Test the model using the given data loader.
        
    #     Args:
    #         test_loader: DataLoader for the test data.
        
    #     Returns:
    #         The average test loss.
    #     """
    #     test_loss = 0.0
    #     self.model.eval()
    #     with torch.no_grad():
    #         for inputs, targets in test_loader:
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             outputs = self.model(inputs)
    #             loss = self.criterion(outputs, targets)
    #             test_loss += loss.item() * inputs.size(0)
        
    #     test_loss /= len(test_loader.dataset)
    #     return test_loss
            
    def test(self, test_loader):
        """
            Test the model using the given data loader.
            
            Args:
                test_loader: DataLoader for the test data.
            
            Returns:
                tuple: A tuple containing the average test loss, predicted values, and target values.
        """
        test_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        
        test_loss /= len(test_loader.dataset)
        
        # Get the predicted and target values for the last sample in the test set
        predicted = outputs[-1].cpu().numpy()
        target = targets[-1].cpu().numpy()
        
        return test_loss, predicted, target