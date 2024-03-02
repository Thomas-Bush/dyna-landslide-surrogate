from dataset import VariableLengthDataset, split_dataset, collate_fn
from model import CNNLSTMModel
from train import Trainer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def main():
    # Load and split the dataset
    dataset = VariableLengthDataset('path/to/data')
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)  # For later use

    # Instantiate the model
    model = CNNLSTMModel()

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Initialize the Trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)

    # Train the model
    trainer.train(num_epochs=10)

    # After training, you can evaluate the model on the test dataset
    # test_loss = trainer.validate(test_loader)  # This requires adding test_loader to the validate method

if __name__ == "__main__":
    main()