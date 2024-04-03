import os
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from sklearn.preprocessing import MinMaxScaler

class LandslideDataset(Dataset):
    def __init__(self, base_dir, model_ids, scaler=None, transform=None):
        self.base_dir = base_dir
        self.model_ids = model_ids
        self.transform = transform
        self.scaler = scaler

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        model_id = self.model_ids[idx]
        model_path = os.path.join(self.base_dir, model_id, '04_FinalProcessedData_64', 'gan')

        # Load input data
        elevation_path = os.path.join(model_path, 'input', f'{model_id}_elevation.npy')
        thickness_0_path = os.path.join(model_path, 'input', f'{model_id}_thickness_0.npy')
        mask_path = os.path.join(model_path, 'input', f'{model_id}_mask.npy')

        elevation = np.load(elevation_path).astype(np.float32)
        thickness_0 = np.load(thickness_0_path).astype(np.float32)

        input_data = np.stack((elevation, thickness_0), axis=0)

        # Load target data
        thickness_max_path = os.path.join(model_path, 'target', f'{model_id}_thickness_max.npy')
        velocity_max_path = os.path.join(model_path, 'target', f'{model_id}_velocity_max.npy')

        thickness_max = np.load(thickness_max_path).astype(np.float32)
        velocity_max = np.load(velocity_max_path).astype(np.float32)

        target_data = np.stack((velocity_max, thickness_max), axis=0)

        if self.scaler:
            # Scale each feature independently to [0, 1] range
            elevation_scaler = self.scaler['elevation']
            thickness_0_scaler = self.scaler['thickness_0']
            thickness_max_scaler = self.scaler['thickness_max']
            velocity_max_scaler = self.scaler['velocity_max']

            elevation = elevation_scaler.transform(elevation.reshape(-1, 1)).reshape(elevation.shape)
            thickness_0 = thickness_0_scaler.transform(thickness_0.reshape(-1, 1)).reshape(thickness_0.shape)
            thickness_max = thickness_max_scaler.transform(thickness_max.reshape(-1, 1)).reshape(thickness_max.shape)
            velocity_max = velocity_max_scaler.transform(velocity_max.reshape(-1, 1)).reshape(velocity_max.shape)

            input_data = np.stack((elevation, thickness_0), axis=0)
            target_data = np.stack((velocity_max, thickness_max), axis=0)

        sample = {'input': input_data, 'target': target_data}

        # Load mask data if available
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)
            sample['mask'] = mask

        if self.transform:
            sample = self.transform(sample)

        # Convert to tensor
        sample['input'] = torch.from_numpy(sample['input'].copy()).float()
        sample['target'] = torch.from_numpy(sample['target'].copy()).float()

        # Convert mask to tensor if available
        if 'mask' in sample:
            sample['mask'] = torch.from_numpy(sample['mask'].copy()).float()

        return sample

class RandomRotationFlipTransform:
    def __call__(self, sample):
        input_data = sample['input']
        target_data = sample['target']

        # Random rotation by 90 degrees
        k = random.choice([0, 1, 2, 3])  # Number of times to rotate by 90 degrees
        input_data = np.rot90(input_data, k, axes=(1, 2))  # Rotate input
        target_data = np.rot90(target_data, k, axes=(1, 2))  # Rotate target

        # Random horizontal flip
        if random.random() > 0.5:
            input_data = np.flip(input_data, axis=2)  # Flip input horizontally
            target_data = np.flip(target_data, axis=2)  # Flip target horizontally

        # Random vertical flip
        if random.random() > 0.5:
            input_data = np.flip(input_data, axis=1)  # Flip input vertically
            target_data = np.flip(target_data, axis=1)  # Flip target vertically

        return {'input': input_data, 'target': target_data}



# import os
# import numpy as np
# from torch.utils.data import Dataset
# import torch
# import random
# from sklearn.preprocessing import StandardScaler

# class LandslideDataset(Dataset):
#     def __init__(self, base_dir, model_ids, scaler=None, transform=None):
#         self.base_dir = base_dir
#         self.model_ids = model_ids
#         self.transform = transform
#         self.scaler = scaler

#     def __len__(self):
#         return len(self.model_ids)

#     def __getitem__(self, idx):
#         model_id = self.model_ids[idx]
#         model_path = os.path.join(self.base_dir, model_id, '04_FinalProcessedData_256', 'gan')

#         # Load input data
#         elevation_path = os.path.join(model_path, 'input', f'{model_id}_elevation.npy')
#         thickness_0_path = os.path.join(model_path, 'input', f'{model_id}_thickness_0.npy')
#         mask_path = os.path.join(model_path, 'input', f'{model_id}_mask.npy')

#         elevation = np.load(elevation_path).astype(np.float32)
#         thickness_0 = np.load(thickness_0_path).astype(np.float32)
        
#         input_data = np.stack((elevation, thickness_0), axis=0)

#         # Load target data
#         thickness_max_path = os.path.join(model_path, 'target', f'{model_id}_thickness_max.npy')
#         velocity_max_path = os.path.join(model_path, 'target', f'{model_id}_velocity_max.npy')

#         thickness_max = np.load(thickness_max_path).astype(np.float32)
#         velocity_max = np.load(velocity_max_path).astype(np.float32)
        
#         target_data = np.stack((velocity_max, thickness_max), axis=0)

#         if self.scaler:
#             # Reshape the input_data from (channels, height, width) to (height * width, channels)
#             # so that each pixel is treated as a sample.
#             original_shape = input_data.shape
#             reshaped_input_data = input_data.reshape(-1, original_shape[0])
            
#             # Scale the data
#             scaled_input_data = self.scaler.transform(reshaped_input_data)
            
#             # Reshape the scaled data back to (channels, height, width)
#             input_data = scaled_input_data.reshape(original_shape).astype(np.float32)  

#         sample = {'input': input_data, 'target': target_data}

#         # Load mask data if available
#         if os.path.exists(mask_path):
#             mask = np.load(mask_path).astype(np.float32)
#             sample['mask'] = mask

#         if self.transform:
#             sample = self.transform(sample)

#         # Convert to tensor
#         sample['input'] = torch.from_numpy(sample['input'].copy()).float()
#         sample['target'] = torch.from_numpy(sample['target'].copy()).float()
        
#         # Convert mask to tensor if available
#         if 'mask' in sample:
#             sample['mask'] = torch.from_numpy(sample['mask'].copy()).float()

#         return sample


# class RandomRotationFlipTransform:
#     def __call__(self, sample):
#         input_data = sample['input']
#         target_data = sample['target']

#         # Random rotation by 90 degrees
#         k = random.choice([0, 1, 2, 3])  # Number of times to rotate by 90 degrees
#         input_data = np.rot90(input_data, k, axes=(1, 2))  # Rotate input
#         target_data = np.rot90(target_data, k, axes=(1, 2))  # Rotate target

#         # Random horizontal flip
#         if random.random() > 0.5:
#             input_data = np.flip(input_data, axis=2)  # Flip input horizontally
#             target_data = np.flip(target_data, axis=2)  # Flip target horizontally

#         # Random vertical flip
#         if random.random() > 0.5:
#             input_data = np.flip(input_data, axis=1)  # Flip input vertically
#             target_data = np.flip(target_data, axis=1)  # Flip target vertically

#         return {'input': input_data, 'target': target_data}


    

