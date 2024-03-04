import torch
from model import CNNLSTM  # Replace with your actual model class name if different
from dataset import DebrisFlowDataset  # Replace with your actual dataset class name if different
from train import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_dataset = DebrisFlowDataset('path/to/train/data')
    val_dataset = DebrisFlowDataset('path/to/val/data')

    # Define model
    model = CNNLSTM()  # Replace CNNLSTM with your actual model class name

    # Check if multiple GPUs are available and wrap the model using nn.DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # This will wrap the model for use with multiple GPUs
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()  # Choose the appropriate loss function for your problem
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Initialize Trainer
    batch_size = 32  # Define your batch size
    trainer = Trainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                      batch_size=batch_size, criterion=criterion, optimizer=optimizer, 
                      learning_rate=1e-3)

    # Train the model
    num_epochs = 10  # Set the number of epochs you wish to train for
    trainer.train(num_epochs)

if __name__ == '__main__':
    main()