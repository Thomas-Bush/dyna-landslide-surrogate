import torch
from torch import nn
from torch.optim import Adam

class Pix2PixGAN:
    def __init__(self, generator, discriminator, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        # Define loss functions
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        # Define optimizers for both generator and discriminator
        self.opt_gen = Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_disc = Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, train_loader, val_loader, epochs, val_interval=5):
        for epoch in range(epochs):
            for i, batch_data in enumerate(train_loader):
                # Unpack the dictionary
                source_image = batch_data['input'].to(self.device)
                target_image = batch_data['target'].to(self.device)
                
                # Check if mask is available in the batch_data
                if 'mask' in batch_data:
                    mask = batch_data['mask'].to(self.device)
                else:
                    # Create a dummy mask filled with ones if mask is not available
                    mask = torch.ones_like(source_image[:, 0:1, :, :])

                # True labels are set as ones
                real_labels = torch.ones(source_image.size(0), 1, 30, 30, device=self.device)

                # Fake labels are set as zeros
                fake_labels = torch.zeros(source_image.size(0), 1, 30, 30, device=self.device)

                # Train generator
                for _ in range(3):
                    self.opt_gen.zero_grad()

                    # Generate fake target image from the source image
                    fake_image = self.generator(source_image, mask)

                    # Calculate the generator loss
                    gen_loss = self.l1_loss(fake_image, target_image)

                    # Calculate the GAN loss
                    pred_fake = self.discriminator(fake_image, source_image)
                    gan_loss = self.bce_loss(pred_fake, real_labels)

                    total_gen_loss = gen_loss + gan_loss
                    total_gen_loss.backward()
                    self.opt_gen.step()

                # Train discriminator
                self.opt_disc.zero_grad()

                # Calculate discriminator loss on real images
                pred_real = self.discriminator(target_image, source_image)
                real_loss = self.bce_loss(pred_real, real_labels)

                # Calculate discriminator loss on fake images
                pred_fake = self.discriminator(fake_image.detach(), source_image)
                fake_loss = self.bce_loss(pred_fake, fake_labels)

                # Total discriminator loss
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                self.opt_disc.step()

                if i % 100 == 0:
                    print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], "
                        f"Discriminator Loss: {disc_loss.item()}, Generator Loss: {total_gen_loss.item()}")

            # Evaluate on validation set
            if epoch % val_interval == 0:
                self.generator.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data in val_loader:
                        source_image = batch_data['input'].to(self.device)
                        target_image = batch_data['target'].to(self.device)
                        
                        # Check if mask is available in the batch_data
                        if 'mask' in batch_data:
                            mask = batch_data['mask'].to(self.device)
                        else:
                            # Create a dummy mask filled with ones if mask is not available
                            mask = torch.ones_like(source_image[:, 0:1, :, :])
                        
                        fake_image = self.generator(source_image, mask)
                        val_loss += self.l1_loss(fake_image, target_image).item()
                val_loss /= len(val_loader)
                print(f"Epoch [{epoch}/{epochs}], Validation L1 Loss: {val_loss:.4f}")
                self.generator.train()

    
    def predict(self, dataloader):
        self.generator.eval()
        predictions = []

        with torch.no_grad():
            for batch_data in dataloader:
                source_image = batch_data['input'].to(self.device)
                mask = batch_data['mask'].to(self.device)
                fake_image = self.generator(source_image, mask)

                # Print the shape of the generated fake_image
                print("Generated fake_image shape:", fake_image.shape)

                # Print a sample value from the generated fake_image
                print("Sample value from fake_image:", fake_image[0, 0, 0, 0].item())

                fake_velocity, fake_thickness = fake_image[:, :1], fake_image[:, 1:]
                predictions.append((source_image.cpu(), fake_thickness.cpu(), fake_velocity.cpu()))

        return predictions



class Pix2PixGenerator:
    def __init__(self, generator, device):
        self.generator = generator
        self.device = device

        # Define loss function
        self.l1_loss = nn.L1Loss()

        # Define optimizer for the generator
        self.opt_gen = Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, train_loader, val_loader, epochs, val_interval=5):
        for epoch in range(epochs):
            for i, batch_data in enumerate(train_loader):
                # Unpack the dictionary
                source_image = batch_data['input'].to(self.device)
                target_image = batch_data['target'].to(self.device)
                
                # Check if mask is available in the batch_data
                if 'mask' in batch_data:
                    mask = batch_data['mask'].to(self.device)
                else:
                    # Create a dummy mask filled with ones if mask is not available
                    mask = torch.ones_like(source_image[:, 0:1, :, :])

                # Train generator
                self.opt_gen.zero_grad()

                # Generate fake target image from the source image
                fake_image = self.generator(source_image, mask)

                # Calculate the generator loss
                gen_loss = self.l1_loss(fake_image, target_image)

                gen_loss.backward()
                self.opt_gen.step()

                if i % 100 == 0:
                    print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], "
                        f"Generator Loss: {gen_loss.item()}")

            # Evaluate on validation set
            if epoch % val_interval == 0:
                self.generator.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data in val_loader:
                        source_image = batch_data['input'].to(self.device)
                        target_image = batch_data['target'].to(self.device)
                        
                        # Check if mask is available in the batch_data
                        if 'mask' in batch_data:
                            mask = batch_data['mask'].to(self.device)
                        else:
                            # Create a dummy mask filled with ones if mask is not available
                            mask = torch.ones_like(source_image[:, 0:1, :, :])
                        
                        fake_image = self.generator(source_image, mask)
                        val_loss += self.l1_loss(fake_image, target_image).item()
                val_loss /= len(val_loader)
                print(f"Epoch [{epoch}/{epochs}], Validation L1 Loss: {val_loss:.4f}")
                self.generator.train()

    def predict(self, dataloader):
        self.generator.eval()
        predictions = []

        with torch.no_grad():
            for batch_data in dataloader:
                source_image = batch_data['input'].to(self.device)
                mask = batch_data['mask'].to(self.device)
                fake_image = self.generator(source_image, mask)

                # Rescale the generated outputs
                fake_image = fake_image.cpu().numpy()
                fake_velocity = fake_image[:, 0, :, :]
                fake_thickness = fake_image[:, 1, :, :]

                # Reshape velocity and thickness to match the scaler's expected input shape
                fake_velocity = fake_velocity.reshape(fake_velocity.shape[0], -1)
                fake_thickness = fake_thickness.reshape(fake_thickness.shape[0], -1)

                # Apply inverse scaling transformation to velocity and thickness using the appropriate scalers
                velocity_max_scaler = dataloader.dataset.scaler['velocity_max']
                thickness_max_scaler = dataloader.dataset.scaler['thickness_max']

                fake_velocity = velocity_max_scaler.inverse_transform(fake_velocity)
                fake_thickness = thickness_max_scaler.inverse_transform(fake_thickness)

                # Reshape the rescaled velocity and thickness back to the original shape
                fake_velocity = fake_velocity.reshape(fake_velocity.shape[0], 1, fake_image.shape[2], fake_image.shape[3])
                fake_thickness = fake_thickness.reshape(fake_thickness.shape[0], 1, fake_image.shape[2], fake_image.shape[3])

                predictions.append((source_image.cpu(), torch.from_numpy(fake_thickness), torch.from_numpy(fake_velocity)))

        return predictions

    # def predict(self, dataloader):
    #     self.generator.eval()
    #     predictions = []

    #     with torch.no_grad():
    #         for batch_data in dataloader:
    #             source_image = batch_data['input'].to(self.device)
    #             mask = batch_data['mask'].to(self.device)
    #             fake_image = self.generator(source_image, mask)

    #             # Print the shape of the generated fake_image
    #             print("Generated fake_image shape:", fake_image.shape)

    #             # Print a sample value from the generated fake_image
    #             print("Sample value from fake_image:", fake_image[0, 0, 0, 0].item())

    #             fake_velocity, fake_thickness = fake_image[:, :1], fake_image[:, 1:]
    #             predictions.append((source_image.cpu(), fake_thickness.cpu(), fake_velocity.cpu()))

    #     return predictions
                
