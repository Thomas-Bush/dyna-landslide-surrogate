import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader



class TrainerPairs:
    def __init__(self, model, optimizer, criterion, device, model_name="", checkpoint_dir="model_checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name.strip()
        self.checkpoint_dir = checkpoint_dir.strip()
        self.training_losses = []
        self.validation_losses = []

        # Ensure the checkpoint directory exists
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name) if self.model_name else self.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, train_loader, val_loader, epochs, checkpoint_interval=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for current, next_velocity in train_loader:
                current = current.to(self.device)
                next_velocity = next_velocity.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(current)
                loss = self.criterion(predictions, next_velocity)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            self.training_losses.append(avg_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

            # Validation step
            val_loss = self.validate(val_loader)
            self.validation_losses.append(val_loss)

            # Save the model at the specified checkpoint interval
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)
                self.save_losses(epoch + 1)

        # Save checkpoint after the final epoch
        self.save_checkpoint(epochs)

        # Save losses after the final epoch
        self.save_losses(epochs)

        # After training, plot the training and validation losses
        self.plot_losses()

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for current, next_velocity in val_loader:
                current = current.to(self.device)
                next_velocity = next_velocity.to(self.device)
                predictions = self.model(current)
                loss = self.criterion(predictions, next_velocity)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

    def save_checkpoint(self, epoch):
        checkpoint_file = f'model_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_file)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path}')

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
                
                predictions = self.model(current)
                loss = self.criterion(predictions, next_velocity)
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f'Test Loss: {avg_loss:.4f}')

    def plot_predictions(self, loader, num_predictions=5):
        self.model.eval()
        with torch.no_grad():
            # Get the first batch of data
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
    def __init__(self, model, optimizer, criterion, device, model_name="", checkpoint_dir="model_checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name.strip()
        self.checkpoint_dir = checkpoint_dir.strip()
        self.training_losses = []
        self.validation_losses = []

        # Ensure the checkpoint directory exists
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_name) if self.model_name else self.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, train_loader, val_loader, epochs, checkpoint_interval=5):
        self.model.train()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for sequence, next_state in train_loader:
                sequence = sequence.to(self.device)
                next_state = next_state.to(self.device)
                
                self.optimizer.zero_grad()
                predictions, _ = self.model(sequence)

                # print("predictions shape: ", predictions.shape)  # Expected to match the shape of `next_state`
                # print("next state shape: ", next_state.shape)


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

            # Save the model at the specified checkpoint interval
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)
                self.save_losses(epoch + 1)

        # Save checkpoint after the final epoch
        self.save_checkpoint(epochs)

        # Save losses after the final epoch
        self.save_losses(epochs)

        # After training, plot the training and validation losses
        self.plot_losses()

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