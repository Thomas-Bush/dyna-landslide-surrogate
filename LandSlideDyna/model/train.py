import os
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import math

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau



# class TrainerPairs:
#     def __init__(self, model, optimizer, criterion, device, model_name="", checkpoint_dir="model_checkpoints"):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.device = device
#         self.model_name = model_name.strip()
#         self.checkpoint_dir = checkpoint_dir.strip()
#         self.training_losses = []
#         self.validation_losses = []
#         self.custom_loss = CustomDebrisLoss()

#         # Ensure the checkpoint directory exists
#         self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name) if self.model_name else self.checkpoint_dir
#         os.makedirs(self.checkpoint_dir, exist_ok=True)

#     def train(self, train_loader, val_loader, epochs, checkpoint_interval=5):
#         self.model.train()
#         for epoch in range(epochs):
#             total_loss = 0.0
#             for current, next_state in train_loader:
#                 current = current.to(self.device)
#                 next_state = next_state.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 predictions = self.model(current)
#                 loss = self.criterion(predictions, next_state)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
            
#             avg_loss = total_loss / len(train_loader)
#             self.training_losses.append(avg_loss)
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

#             # Validation step
#             val_loss = self.validate(val_loader)
#             self.validation_losses.append(val_loss)

#             # Save the model at the specified checkpoint interval
#             if (epoch + 1) % checkpoint_interval == 0:
#                 self.save_checkpoint(epoch + 1)
#                 self.save_losses(epoch + 1)

#         # Save checkpoint after the final epoch
#         self.save_checkpoint(epochs)

#         # Save losses after the final epoch
#         self.save_losses(epochs)

#         # After training, plot the training and validation losses
#         self.plot_losses()

        # self.scaling_factors = train_loader.dataset.dataset.scaling_factors

class TrainerPairs:
    def __init__(self, model, optimizer, criterion, device, model_name="", checkpoint_dir="model_checkpoints", patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name.strip()
        self.checkpoint_dir = os.path.join(checkpoint_dir.strip(), model_name) if model_name else checkpoint_dir.strip()
        self.training_losses = []
        self.validation_losses = []
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.custom_loss = CustomDebrisLoss(debris_weight=0.4)

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, train_loader, val_loader, epochs):
        self.model.train()
        checkpoint_interval = 5  # Save every 5 epochs
        for epoch in range(epochs):
            total_loss = 0.0
            for current, next_state in train_loader:
                current = current.to(self.device)
                next_state = next_state.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(current)
                loss = self.criterion(predictions, next_state)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            self.training_losses.append(avg_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

            # Validation step
            val_loss = self.validate(val_loader)
            self.validation_losses.append(val_loss)
            self.scheduler.step(val_loss)

            # Optionally, print the current learning rate
            current_lr = self.scheduler.get_last_lr()
            print(f"Current Learning Rate: {current_lr}")

            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("Early stopping")
                    self.early_stop = True
                    break

            # Save the model at the specified checkpoint interval
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)

        # Save checkpoint after the final epoch if it does not align with the interval
        if (epochs % checkpoint_interval != 0) and not self.early_stop:
            self.save_checkpoint(epochs)

        # Save losses after the final epoch
        self.save_losses(epochs)

        # After training, plot the training and validation losses
        self.plot_losses()

        # Retrieve scaling factors from the dataset for use in post-processing or inference
        self.scaling_factors = train_loader.dataset.dataset.scaling_factors

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for current, next_state in val_loader:
                current = current.to(self.device)
                next_state = next_state.to(self.device)
                predictions = self.model(current)
                loss = self.criterion(predictions, next_state)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

    def save_checkpoint(self, epoch):
        checkpoint_file = f'model_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path, train_loader):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        self.scaling_factors = train_loader.dataset.dataset.scaling_factors
        
        print(f'Model loaded from {checkpoint_path}')

    def save_losses(self, epoch):
        losses_file = f'losses_epoch_{epoch}.json'
        losses_path = os.path.join(self.checkpoint_dir, losses_file)

        losses = {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        
        with open(losses_path, 'w') as f:
            json.dump(losses, f, indent=4)
        
        print(f'Losses saved to {losses_path}')

    def load_losses(self):
        losses_file = 'losses.json'
        losses_path = os.path.join(self.checkpoint_dir, losses_file)

        with open(losses_path, 'r') as f:
            losses = json.load(f)
        self.training_losses = losses['training_losses']
        self.validation_losses = losses['validation_losses']
        print(f'Losses loaded from {losses_path}')


    def test(self, test_loader):
        self.model.eval()
        mse_criterion = nn.MSELoss()
        total_custom_loss = 0
        total_l1_loss = 0
        total_mse = 0
        total_psnr = 0
        
        with torch.no_grad():
            for current, next_state in test_loader:
                current = current.to(self.device)
                next_state = next_state.to(self.device)
                
                predictions = self.model(current)
                custom_loss = self.custom_loss(predictions, next_state)  # Calculate custom loss
                l1_loss = self.criterion(predictions, next_state)
                mse_loss = mse_criterion(predictions, next_state)
                
                total_custom_loss += custom_loss.item()
                total_l1_loss += l1_loss.item()
                total_mse += mse_loss.item()
                
                max_velocity = 30
                psnr = 20 * math.log10(max_velocity) - 10 * math.log10(mse_loss.item())
                total_psnr += psnr
        
        avg_custom_loss = total_custom_loss / len(test_loader)
        avg_l1_loss = total_l1_loss / len(test_loader)
        avg_mse = total_mse / len(test_loader)
        avg_rmse = math.sqrt(avg_mse)
        avg_psnr = total_psnr / len(test_loader)

        print(f'Test Custom Loss: {avg_custom_loss:.4f}')
        print(f'Test L1 Loss: {avg_l1_loss:.4f}')
        print(f'Test MSE: {avg_mse:.4f}')
        print(f'Test RMSE: {avg_rmse:.4f}')
        print(f'Test PSNR: {avg_psnr:.4f}')


    def create_inference_input(self, root_dir, model_number, state_number, array_size):
        # Construct the file paths based on the provided parameters
        model_dir = os.path.join(root_dir, str(model_number))
        elevation_file = os.path.join(model_dir, f'04_FinalProcessedData_{array_size}', 'elevation', f'{model_number}_elevation.npy')
        velocity_file = os.path.join(model_dir, f'04_FinalProcessedData_{array_size}', 'velocity', f'{model_number}_velocity_{state_number}.npy')
        thickness_file = os.path.join(model_dir, f'04_FinalProcessedData_{array_size}', 'thickness', f'{model_number}_thickness_{state_number}.npy')

        # Load the data arrays
        elevation = np.load(elevation_file)
        velocity = np.load(velocity_file)
        thickness = np.load(thickness_file)

        # Scale the data arrays
        min_elevation, max_elevation, min_velocity, max_velocity, min_thickness, max_thickness = self.scaling_factors
        elevation_scaled = (elevation - min_elevation) / (max_elevation - min_elevation)
        velocity_scaled = (velocity - min_velocity) / (max_velocity - min_velocity)
        thickness_scaled = (thickness - min_thickness) / (max_thickness - min_thickness)

        # Stack the scaled data arrays to create the input tensor
        input_data = np.stack((elevation_scaled, thickness_scaled, velocity_scaled), axis=0)

        # Convert to PyTorch tensor and move to the specified device
        input_tensor = torch.from_numpy(input_data).float().to(self.device)

        return input_tensor

    def infer(self, initial_input, num_timesteps):
        self.model.eval()
        device = self.device

        # Ensure initial_input is a PyTorch tensor and move to the device
        if not isinstance(initial_input, torch.Tensor):
            initial_input = torch.tensor(initial_input, dtype=torch.float32).to(device)
        else:
            initial_input = initial_input.to(device)

        # Extract the elevation channel from the initial input
        elevation = initial_input[0].unsqueeze(0)  # This is possibly incorrect if initial_input already includes a batch dimension

        print("Elevation shape initial:", elevation.shape)

        # Adjust elevation to have a channels dimension (assuming elevation data is a single channel)
        elevation = elevation.unsqueeze(1)  # Add a channel dimension making it [1, 1, height, width]

        print("Elevation shape adjusted:", elevation.shape)

        # Initialize the input tensor for the first timestep
        input_tensor = initial_input.unsqueeze(0)  # Check if a batch dimension is really needed here

        # Initialize a dictionary to store the inferred states
        inferred_states = {}

        with torch.no_grad():
            for t in range(num_timesteps):
                # Perform inference
                print("Input tensor shape:", input_tensor.shape)
                output = self.model(input_tensor)
                print("Model output shape:", output.shape)

                # Store the inferred state
                inferred_states[t + 1] = output.squeeze(0).cpu().numpy()

                # Concatenate elevation with output
                output_with_elevation = torch.cat((elevation, output), dim=1)

                print("Output with elevation shape:", output_with_elevation.shape)

                # Update the input tensor for the next iteration
                input_tensor = output_with_elevation

        return inferred_states

    def get_predictions(self, loader):
        self.model.eval()
        with torch.no_grad():
            # Get the first batch of data
            current, next_velocity_thickness = next(iter(loader))
            current = current.to(self.device)
            next_velocity_thickness = next_velocity_thickness.to(self.device)
            
            # Generate predictions
            predictions = self.model(current)
            
            # Move tensors back to the CPU for further processing/returning
            current = current.cpu().numpy()
            next_velocity_thickness = next_velocity_thickness.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Calculate the differences for velocity and thickness
            differences_velocity = np.abs(next_velocity_thickness[:, 0, :, :] - predictions[:, 0, :, :])
            differences_thickness = np.abs(next_velocity_thickness[:, 1, :, :] - predictions[:, 1, :, :])

            return {
                'current': current,
                'next_velocity_thickness': next_velocity_thickness,
                'predictions': predictions,
                'differences_velocity': differences_velocity,
                'differences_thickness': differences_thickness
            }

    def plot_predictions(self, loader, num_predictions=5, batch_index=0):
        self.model.eval()
        with torch.no_grad():
            # Iterate to the specified batch index
            for _ in range(batch_index + 1):
                current, next_velocity_thickness = next(iter(loader))
            
            current = current.to(self.device)
            next_velocity_thickness = next_velocity_thickness.to(self.device)
            
            predictions = self.model(current)

            # Move the tensors back to the CPU and convert to numpy for plotting
            current = current.cpu().numpy()
            next_velocity_thickness = next_velocity_thickness.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Calculate the differences for velocity and thickness
            differences_velocity = np.abs(next_velocity_thickness[:, 0, :, :] - predictions[:, 0, :, :])
            differences_thickness = np.abs(next_velocity_thickness[:, 1, :, :] - predictions[:, 1, :, :])
            
            # Determine the common color scales for current and next velocities and thicknesses
            common_scale = {
                'velocity': {
                    'min': min(np.min(current[:, 1, :, :]), np.min(next_velocity_thickness[:, 0, :, :])),
                    'max': max(np.max(current[:, 1, :, :]), np.max(next_velocity_thickness[:, 0, :, :]))
                },
                'thickness': {
                    'min': min(np.min(current[:, 2, :, :]), np.min(next_velocity_thickness[:, 1, :, :])),
                    'max': max(np.max(current[:, 2, :, :]), np.max(next_velocity_thickness[:, 1, :, :]))
                }
            }
            
            batch_size = current.shape[0]
            # Select random indices from the batch
            indices_to_plot = np.random.choice(batch_size, num_predictions, replace=False)
            
            for idx in indices_to_plot:
                plt.figure(figsize=(20, 8))



                # Row 1: Topography, Current Velocity, True Next Velocity, Predicted Next Velocity, Velocity Difference
                plt.subplot(2, 5, 1)
                plt.imshow(current[idx][0], cmap='gray')
                plt.title('Topography')
                plt.axis('off')

                plt.subplot(2, 5, 2)
                plt.imshow(current[idx][1], cmap='jet', vmin=common_scale['velocity']['min'], vmax=common_scale['velocity']['max'])
                plt.title('Current Velocity')
                plt.axis('off')
                plt.colorbar()

                plt.subplot(2, 5, 3)
                plt.imshow(next_velocity_thickness[idx][0], cmap='jet', vmin=common_scale['velocity']['min'], vmax=common_scale['velocity']['max'])
                plt.title('True Next Velocity')
                plt.axis('off')
                plt.colorbar()

                plt.subplot(2, 5, 4)
                plt.imshow(predictions[idx][0], cmap='jet', vmin=common_scale['velocity']['min'], vmax=common_scale['velocity']['max'])
                plt.title('Predicted Next Velocity')
                plt.axis('off')
                plt.colorbar()

                plt.subplot(2, 5, 5)
                plt.imshow(differences_velocity[idx], cmap='jet')
                plt.title('Velocity Difference')
                plt.axis('off')
                plt.colorbar()

                # Row 2: Topography, Current Thickness, True Next Thickness, Predicted Next Thickness, Thickness Difference
                plt.subplot(2, 5, 6)
                plt.imshow(current[idx][0], cmap='gray')  # Topography is the same as in the first row
                plt.axis('off')

                plt.subplot(2, 5, 7)
                plt.imshow(current[idx][2], cmap='jet', vmin=common_scale['thickness']['min'], vmax=common_scale['thickness']['max'])
                plt.title('Current Thickness')
                plt.axis('off')
                plt.colorbar()

                plt.subplot(2, 5, 8)
                plt.imshow(next_velocity_thickness[idx][1], cmap='jet', vmin=common_scale['thickness']['min'], vmax=common_scale['thickness']['max'])
                plt.title('True Next Thickness')
                plt.axis('off')
                plt.colorbar()

                plt.subplot(2, 5, 9)
                plt.imshow(predictions[idx][1], cmap='jet', vmin=common_scale['thickness']['min'], vmax=common_scale['thickness']['max'])
                plt.title('Predicted Next Thickness')
                plt.axis('off')
                plt.colorbar()

                plt.subplot(2, 5, 10)
                plt.imshow(differences_thickness[idx], cmap='jet')
                plt.title('Thickness Difference')
                plt.axis('off')
                plt.colorbar()

                plt.tight_layout()
                plt.show()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as an image file
        plot_file = 'loss_plot.png'
        plot_path = os.path.join(self.checkpoint_dir, plot_file)
        plt.savefig(plot_path)
        print(f'Loss plot saved to {plot_path}')

        # Show the plot
        plt.show()



class TrainerSeries:
    def __init__(self, model, optimizer, criterion, device, model_name="", checkpoint_dir="model_checkpoints", patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name.strip()
        self.checkpoint_dir = os.path.join(checkpoint_dir.strip(), model_name) if model_name else checkpoint_dir.strip()
        self.training_losses = []
        self.validation_losses = []
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, train_loader, val_loader, epochs):
        self.model.train()
        checkpoint_interval = 5  # Save every 5 epochs
        for epoch in range(epochs):
            total_loss = 0.0
            for sequence, next_state in train_loader:
                sequence = sequence.to(self.device)
                next_state = next_state.to(self.device)
                
                self.optimizer.zero_grad()
                predictions, _ = self.model(sequence)
                loss = self.criterion(predictions, next_state)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            self.training_losses.append(avg_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

            # Validation step
            val_loss = self.validate(val_loader)
            self.validation_losses.append(val_loss)
            self.scheduler.step(val_loss)

            # Optionally, print the current learning rate
            current_lr = self.scheduler.get_last_lr()
            print(f"Current Learning Rate: {current_lr}")

            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("Early stopping")
                    self.early_stop = True
                    break

            # Save the model at the specified checkpoint interval
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)

        # Save checkpoint after the final epoch if it does not align with the interval
        if (epochs % checkpoint_interval != 0) and not self.early_stop:
            self.save_checkpoint(epochs)

        # Save losses after the final epoch
        self.save_losses(epochs)

        # After training, plot the training and validation losses
        self.plot_losses()
        
        self.sequence_length = train_loader.dataset.dataset.sequence_length
        self.scaling_factors = train_loader.dataset.dataset.scaling_factors


# class TrainerSeries:
#     def __init__(self, model, optimizer, criterion, device, model_name="", checkpoint_dir="model_checkpoints"):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.device = device


#         self.model_name = model_name.strip()
#         self.checkpoint_dir = checkpoint_dir.strip()
#         self.training_losses = []
#         self.validation_losses = []


#         # Ensure the checkpoint directory exists
#         self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name) if self.model_name else self.checkpoint_dir
#         os.makedirs(self.checkpoint_dir, exist_ok=True)

#     def train(self, train_loader, val_loader, epochs, checkpoint_interval=5):
#         self.model.train()
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0.0
#             for sequence, next_state in train_loader:
#                 sequence = sequence.to(self.device)
#                 next_state = next_state.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 predictions, _ = self.model(sequence)

#                 # print("predictions shape: ", predictions.shape)  # Expected to match the shape of `next_state`
#                 # print("next state shape: ", next_state.shape)


#                 loss = self.criterion(predictions, next_state)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 total_loss += loss.item()
            
#             avg_loss = total_loss / len(train_loader)
#             self.training_losses.append(avg_loss)
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

#             # Validation step
#             val_loss = self.validate(val_loader)
#             self.validation_losses.append(val_loss)

#             # Save the model at the specified checkpoint interval
#             if (epoch + 1) % checkpoint_interval == 0:
#                 self.save_checkpoint(epoch + 1)
#                 self.save_losses(epoch + 1)

#         # Save checkpoint after the final epoch
#         self.save_checkpoint(epochs)

#         # Save losses after the final epoch
#         self.save_losses(epochs)

#         # After training, plot the training and validation losses
#         self.plot_losses()
        
#         self.sequence_length = train_loader.dataset.dataset.sequence_length
#         self.scaling_factors = train_loader.dataset.dataset.scaling_factors
    
    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for sequence, next_state in val_loader:
                sequence = sequence.to(self.device)
                next_state = next_state.to(self.device)
                predictions, _ = self.model(sequence)
                loss = self.criterion(predictions, next_state)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

    def save_checkpoint(self, epoch):
        checkpoint_file = f'model_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path, train_loader):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        print(f'Model loaded from {checkpoint_path}')
        self.scaling_factors = train_loader.dataset.dataset.scaling_factors
        self.sequence_length = train_loader.dataset.dataset.sequence_length

    def save_losses(self, epoch):
        losses_file = f'losses_epoch_{epoch}.json'
        losses_path = os.path.join(self.checkpoint_dir, losses_file)

        losses = {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        
        with open(losses_path, 'w') as f:
            json.dump(losses, f, indent=4)
        
        print(f'Losses saved to {losses_path}')

    def load_losses(self):
        losses_file = 'losses.json'
        losses_path = os.path.join(self.checkpoint_dir, losses_file)

        with open(losses_path, 'r') as f:
            losses = json.load(f)
        self.training_losses = losses['training_losses']
        self.validation_losses = losses['validation_losses']
        print(f'Losses loaded from {losses_path}')

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for current, next_velocity in test_loader:
                current = current.to(self.device)
                next_velocity = next_velocity.to(self.device)
                
                predictions, _ = self.model(current)  # Unpack the tuple if your model returns more than one output
                loss = self.criterion(predictions, next_velocity)
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f'Test Loss: {avg_loss:.4f}')

    def create_inference_input(self, root_dir, model_number, state_number, array_size):
        # Construct the file paths based on the provided parameters
        model_dir = os.path.join(root_dir, str(model_number))
        elevation_file = os.path.join(model_dir, f'04_FinalProcessedData_{array_size}', 'elevation', f'{model_number}_elevation.npy')
        velocity_file = os.path.join(model_dir, f'04_FinalProcessedData_{array_size}', 'velocity', f'{model_number}_velocity_{state_number}.npy')
        thickness_file = os.path.join(model_dir, f'04_FinalProcessedData_{array_size}', 'thickness', f'{model_number}_thickness_{state_number}.npy')

        # Load the data arrays
        elevation = np.load(elevation_file)
        velocity = np.load(velocity_file)
        thickness = np.load(thickness_file)

        # Scale the data arrays
        min_elevation, max_elevation, min_velocity, max_velocity, min_thickness, max_thickness = self.scaling_factors
        elevation_scaled = (elevation - min_elevation) / (max_elevation - min_elevation)
        velocity_scaled = (velocity - min_velocity) / (max_velocity - min_velocity)
        thickness_scaled = (thickness - min_thickness) / (max_thickness - min_thickness)

        # Stack the scaled data arrays to create the input tensor
        input_data = np.stack((elevation_scaled, thickness_scaled, velocity_scaled), axis=0)

        # Convert to PyTorch tensor and move to the specified device
        input_tensor = torch.from_numpy(input_data).float().to(self.device)

        return input_tensor

    def infer(self, initial_sequence, num_timesteps):
        self.model.eval()
        device = self.device
        sequence_length = self.sequence_length
        scaling_factors = self.scaling_factors
        
        # Ensure initial_sequence is a PyTorch tensor
        if not isinstance(initial_sequence, torch.Tensor):
            initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)

        # Move the initial sequence to the same device as the model
        initial_sequence = initial_sequence.to(device)
        print(f"Initial sequence shape: {initial_sequence.shape}")
        print(f"Initial sequence device: {initial_sequence.device}")

        # Extract the elevation channel from the initial sequence
        elevation = initial_sequence[0].unsqueeze(0)
        print(f"Elevation shape: {elevation.shape}")

        # Initialize the input tensor with zeros and the initial sequence
        input_tensor = [torch.zeros_like(initial_sequence) for _ in range(sequence_length - 1)]
        input_tensor.append(initial_sequence)
        print(f"Input tensor length: {len(input_tensor)}")


        # Initialize a dictionary to store the inferred states
        inferred_states = {}

        with torch.no_grad():
            for t in range(num_timesteps):
                print(f"\nTimestep: {t}")
                
                # Construct the input sequence
                input_sequence = torch.stack(input_tensor, dim=0)  # Shape: [seq_length, channels, height, width]
                input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension: [1, seq_length, channels, height, width]
                print(f"Input sequence shape: {input_sequence.shape}")

                # Perform inference
                next_state, _ = self.model(input_sequence)
                print(f"Next state shape: {next_state.shape}")

                # Scale and store the output
                min_elevation, max_elevation, min_velocity, max_velocity, min_thickness, max_thickness = scaling_factors
                next_state_unscaled = next_state.clone()
                next_state_unscaled[:, 0, ...] = next_state[:, 0, ...] * (max_thickness - min_thickness) + min_thickness
                next_state_unscaled[:, 1, ...] = next_state[:, 1, ...] * (max_velocity - min_velocity) + min_velocity
                inferred_states[t + 1] = next_state_unscaled.squeeze(0).cpu().numpy()
                print(f"Inferred state shape at timestep {t + 1}: {inferred_states[t + 1].shape}")

                # Stack the elevation channel with the inferred next state
                next_state_with_elevation = torch.cat((elevation, next_state.squeeze(0)), dim=0)
                print(f"Next state with elevation shape: {next_state_with_elevation.shape}")

                # Remove the oldest state and append the inferred next state with elevation to the input tensor
                input_tensor.pop(0)
                input_tensor.append(next_state_with_elevation)
                print(f"Input tensor length after updating: {len(input_tensor)}")

        print(f"\nFinal inferred states: {list(inferred_states.keys())}")

            # Compare inferred states with each other
        print("\nComparison of inferred states:")
        for i in range(1, num_timesteps + 1):
            for j in range(i + 1, num_timesteps + 1):
                state_i = inferred_states[i]
                state_j = inferred_states[j]

                # Calculate mean squared error (MSE) between the states
                mse = np.mean((state_i - state_j) ** 2)

                # Calculate mean absolute error (MAE) between the states
                mae = np.mean(np.abs(state_i - state_j))

                print(f"Timestep {i} vs Timestep {j}:")
                print(f"  Mean Squared Error (MSE): {mse:.4f}")
                print(f"  Mean Absolute Error (MAE): {mae:.4f}")

        return inferred_states


    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as an image file
        plot_file = 'loss_plot.png'
        plot_path = os.path.join(self.checkpoint_dir, plot_file)
        plt.savefig(plot_path)
        print(f'Loss plot saved to {plot_path}')

        # Show the plot
        plt.show()

    def plot_predictions(self, test_loader, num_predictions=5):
        self.model.eval()
        with torch.no_grad():
            for i, (sequence, next_state) in enumerate(test_loader):
                if i >= num_predictions:
                    break

                sequence = sequence.to(self.device)
                next_state = next_state.to(self.device)

                predictions, _ = self.model(sequence)

                # Move the data back to the CPU for plotting
                next_state = next_state.cpu().numpy()
                predictions = predictions.cpu().numpy()

                # Plot the expected and predicted results side by side
                fig, axes = plt.subplots(3, 2, figsize=(12, 18))

                # Process and plot each type of data (thickness, velocity)
                for index, title in enumerate(['Thickness', 'Velocity']):
                    # Determine color limits based on the expected values
                    vmin = np.min(next_state[0, index])
                    vmax = np.max(next_state[0, index])

                    # Expected
                    im = axes[index, 0].imshow(next_state[0, index], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[index, 0].set_title(f"Expected {title}")
                    axes[index, 0].axis('off')
                    fig.colorbar(im, ax=axes[index, 0], orientation='vertical')

                    # Predicted
                    im = axes[index, 1].imshow(predictions[0, index], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[index, 1].set_title(f"Predicted {title}")
                    axes[index, 1].axis('off')
                    fig.colorbar(im, ax=axes[index, 1], orientation='vertical')

                # Compute and plot the differences for thickness and velocity
                for index, title in enumerate(['Thickness', 'Velocity']):
                    diff = np.abs(next_state[0, index] - predictions[0, index])
                    vmax = np.max(diff)  # Dynamic range based on the actual differences observed

                    im = axes[2, index].imshow(diff, cmap='hot', vmin=0, vmax=vmax)
                    axes[2, index].set_title(f"Difference in {title}")
                    axes[2, index].axis('off')
                    fig.colorbar(im, ax=axes[2, index], orientation='vertical')

                plt.tight_layout()
                plt.show()

class CustomDebrisLoss(nn.Module):
    def __init__(self, loss_fn_zero=nn.MSELoss(), loss_fn_debris=nn.L1Loss(), debris_weight=0.75):
        super(CustomDebrisLoss, self).__init__()
        self.loss_fn_zero = loss_fn_zero
        self.loss_fn_debris = loss_fn_debris
        self.debris_weight = debris_weight

    def forward(self, outputs, targets):
        # Create a mask for debris areas (non-zero targets)
        debris_mask = targets != 0
        # Calculate loss for debris areas
        debris_loss = self.loss_fn_debris(outputs[debris_mask], targets[debris_mask])
        
        # Calculate loss for zero areas
        zero_mask = ~debris_mask
        zero_loss = self.loss_fn_zero(outputs[zero_mask], targets[zero_mask])
        
        # Weighted sum of the two losses
        total_loss = (self.debris_weight * debris_loss + 
                      (1 - self.debris_weight) * zero_loss)
        return total_loss
    
class SparseLoss(nn.Module):
    def __init__(self, zero_weight=0.1, debris_weight=0.9, epsilon=1e-8):
        super(SparseLoss, self).__init__()
        self.zero_weight = zero_weight
        self.debris_weight = debris_weight
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        # Create a mask for debris areas (non-zero targets)
        debris_mask = targets != 0
        zero_mask = ~debris_mask

        # Calculate the absolute error for debris and zero regions separately
        debris_error = torch.abs(outputs[debris_mask] - targets[debris_mask])
        zero_error = torch.abs(outputs[zero_mask] - targets[zero_mask])

        # Calculate the mean absolute error for debris and zero regions
        debris_mae = debris_error.mean()
        zero_mae = zero_error.mean()

        # Calculate the debris loss using mean squared error
        debris_loss = torch.mean(debris_error ** 2)

        # Calculate the zero loss using binary cross-entropy
        zero_pred = torch.clamp(outputs[zero_mask], self.epsilon, 1 - self.epsilon)
        zero_target = torch.zeros_like(zero_pred)
        zero_loss = nn.functional.binary_cross_entropy(zero_pred, zero_target)

        # Combine the losses using the specified weights
        total_loss = self.debris_weight * debris_loss + self.zero_weight * zero_loss

        return total_loss, debris_mae, zero_mae

class AdaptiveSparseLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(AdaptiveSparseLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        # Create a mask for debris areas (non-zero targets)
        debris_mask = targets != 0
        zero_mask = ~debris_mask

        # Calculate the absolute error for debris and zero regions separately
        debris_error = torch.abs(outputs[debris_mask] - targets[debris_mask])
        zero_error = torch.abs(outputs[zero_mask] - targets[zero_mask])

        # # Calculate the mean absolute error for debris and zero regions
        # debris_mae = debris_error.mean()
        # print(f"debris mae: {debris_mae}")
        # zero_mae = zero_error.mean()
        # print(f"zero mae: {zero_mae}")

        # Calculate the debris loss using mean squared error
        debris_loss = torch.mean(debris_error ** 2)

        # Calculate the zero loss using binary cross-entropy
        zero_pred = torch.clamp(outputs[zero_mask], self.epsilon, 1 - self.epsilon)
        zero_target = torch.zeros_like(zero_pred)
        zero_loss = nn.functional.binary_cross_entropy(zero_pred, zero_target)

        # Calculate the adaptive weights based on the sparsity of the data
        debris_weight = 1 - (debris_mask.sum() / targets.numel())
        zero_weight = 1 - debris_weight

        # Combine the losses using the adaptive weights
        total_loss = debris_weight * debris_loss + zero_weight * zero_loss

        return total_loss