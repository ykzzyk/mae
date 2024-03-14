import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .augmentation import get_albumentation_transform, get_custom_transform
import albumentations as A
import matplotlib.pyplot as plt
import torch

class SemiSupervisedDataloader:
  
    def __init__(self, unlabeld_loader, labeled_loader) -> None:
        self.dataset = unlabeld_loader.dataset
        self.unlabeld_loader = unlabeld_loader
        self.labeled_loader = labeled_loader
        self.unlabeld_iter = None
        self.labeld_iter = None

    def __iter__(self):
        self.unlabeld_iter = iter(self.unlabeld_loader)
        self.labeld_iter = iter(self.labeled_loader)
        return self
    
    def __len__(self):
        return min(len(self.unlabeld_loader), len(self.labeled_loader))

    def __next__(self):
        unlabeld = next(self.unlabeld_iter)
        labeled = next(self.labeld_iter)

        res = {}
        for k in unlabeld:
            res[k] = torch.cat([unlabeld[k], labeled[k]], dim=0)    
        return res

class BRPDataset(Dataset):

    def __init__(
            self, 
            data_root, 
            dataset_meta, 
            split,
            slice_per_volume,
            gt_scores,
            d_range=(3, 60),
            custom_transform=False,
            albumentation_transform=False,
            shadow_samples=0
        ):
        self.data_root = data_root
        self.dataset_meta = dataset_meta
        self.split = split
        self.cases = dataset_meta[split]
        self.slice_per_volume=slice_per_volume
        self.gt_scores = gt_scores
        self.d_range=d_range

        # define augmentations
        if custom_transform:
            self.custom_transform = get_custom_transform()
        else:
            self.custom_transform = lambda x: x

        if albumentation_transform:
            self.albumentation_transform = get_albumentation_transform()

        else:
            # equivalent to no transformations
            self.albumentation_transform = A.Compose([A.Transpose(p=0)])
        
        self.shadow_samples=shadow_samples

    
    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index, slice_indices=None):

        # index = 1

        case = self.cases[index]
        img = nib.load(f'{self.data_root}/{case}.nii')

        z_spacing = abs(img.affine[2, 2])
        z_shape = img.shape[2]

        max_z_len = z_spacing * z_shape

        if self.slice_per_volume == 'all':
            slice_indices = np.array(list(range(z_shape))[::-1])
            z = z_spacing
        elif slice_indices is not None:
            slice_indices = np.array(slice_indices)
            z = z_spacing * slice_indices[1] - slice_indices[0]
        else:
            d = np.random.uniform(low=self.d_range[0], high=min(self.d_range[1], max_z_len // self.slice_per_volume))
            k = d // z_spacing

            k = max(1, k)

            z = k * z_spacing

            slice_range = k * (self.slice_per_volume - 1) + 1

            # the slice closer to head and that has larger index value
            start_slice = np.random.randint(low=slice_range, high=z_shape - 1)

            slice_indices = np.array([start_slice - i * k for i in range(self.slice_per_volume)]).astype(np.int32)

        # try:
        assert slice_indices.min() >=0 and slice_indices.max() < img.shape[2]
        # except:
        #     print(1)

        raw_slices = np.stack([img.dataobj[:, :, i] for i in slice_indices], axis=-1)

        slices_aug = raw_slices.copy()
        
        for i in range(slices_aug.shape[2]):
            slices_aug[:, :, i] = self.custom_transform(slices_aug[:, :, i])
            slices_aug[:, :, i] = self.albumentation_transform(image=slices_aug[:, :, i])["image"]

        if case in self.gt_scores:
            gt_scores = np.array(self.gt_scores[case]['score'], dtype=np.float32)[slice_indices]
        else:
            gt_scores = np.array([np.nan] * self.slice_per_volume, dtype=np.float32)
        
        res = {'slices_raw': raw_slices.transpose(2, 0, 1), 
                'slices_aug': slices_aug.transpose(2, 0, 1), 
                'gt_scores': gt_scores, 
                'z': np.array([z] * (len(slice_indices) - 1), dtype=np.float32),
                'slice_indices': slice_indices,
                'sample_index': index
                }

        # generate shadow samples for semi-supervised learning
        for x in range(self.shadow_samples):
            slices_shadow = slices_shadow = raw_slices.copy()
            for i in range(slices_shadow.shape[2]):
                slices_shadow[:, :, i] = self.custom_transform(slices_shadow[:, :, i])
                slices_shadow[:, :, i] = self.albumentation_transform(image=slices_shadow[:, :, i])["image"]
            res['slices_shadow%s' % x] = slices_shadow.transpose(2, 0, 1)
        
        return res

    def plot(self, index):
        
        data = self.__getitem__(index)

        fig, axs = plt.subplots(2 + self.shadow_samples, self.slice_per_volume, figsize=(4 * self.slice_per_volume, 6 + 3 * self.shadow_samples))

        for i, item in enumerate([data['slices_raw'], data['slices_aug']] + [data['slices_shadow%s' % i] for i in range(self.shadow_samples)]):
            for j, slice in enumerate(item):
                if i == 0:
                    axs[i][j].set_title(int(data['slice_indices'][j]))
                axs[i][j].imshow(slice, cmap='gray')
        
        plt.show()


class LisaBPRDataset(Dataset):

    def __init__(
            self, 
            data_root, 
            dataset_meta, 
            split,
            slice_per_volume,
            gt_scores,
            d_range=(3, 60),
            custom_transform=False,
            albumentation_transform=False,
            shadow_samples=0,
            post_augmentation=False,
            p_lisa=0.2
        ):
        self.data_root = data_root
        self.dataset_meta = dataset_meta
        self.split = split
        self.cases = dataset_meta[split]
        self.slice_per_volume=slice_per_volume
        self.gt_scores = gt_scores
        self.d_range = d_range
        self.p_lisa = p_lisa

        self.mix_loader = BRPDataset(data_root, 
            dataset_meta, 
            split,
            slice_per_volume,
            gt_scores,
            d_range,
            custom_transform=False,
            albumentation_transform=False,
            shadow_samples=shadow_samples)

        self.normal_loader = BRPDataset(data_root, 
            dataset_meta, 
            split,
            slice_per_volume,
            gt_scores,
            d_range,
            custom_transform=custom_transform,
            albumentation_transform=albumentation_transform,
            shadow_samples=shadow_samples)

        # define augmentations
        if custom_transform:
            self.custom_transform = get_custom_transform()
        else:
            self.custom_transform = lambda x: x

        if albumentation_transform:
            self.albumentation_transform = get_albumentation_transform()

        else:
            # equivalent to no transformations
            self.albumentation_transform = A.Compose([A.Transpose(p=0)])
        
        self.shadow_samples=shadow_samples
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, index):
        
        if np.random.uniform(low=0, high=1) > self.p_lisa:
            return self.normal_loader[index]
        
        sample1 = self.mix_loader[index]

        index2 = [i for i in range(len(self.cases)) if i != index][np.random.choice(len(self.cases) - 1)]
        gt_scores2 = self.gt_scores[self.cases[index2]]['score']
        if 'slice_index' in self.gt_scores[self.cases[index2]]:
            gt_index2 = self.gt_scores[self.cases[index2]]['slice_index']
        else:
            gt_index2 = self.gt_scores[self.cases[index2]]['index']

        scores2_min = min(gt_scores2)
        scores2_max = max(gt_scores2)

        use_sample2 = []
        sample2_slice_indices = []
        min_diff = []
        for score in sample1['gt_scores']:
            if scores2_min < score < scores2_max:
                diff = np.abs(np.array(gt_scores2) - score)
                corr_index = gt_index2[diff.argmin()]
                min_diff.append(diff.min())
                if min_diff[-1] < 0.005:
                    use_sample2.append(1)
                    sample2_slice_indices.append(corr_index)
                else:
                    use_sample2.append(0)
                    sample2_slice_indices.append(0)
            else:
                use_sample2.append(0)
                sample2_slice_indices.append(0)
                min_diff.append(None)

        sample2 = self.mix_loader.__getitem__(index2, sample2_slice_indices)

        alpha = np.random.uniform(low=0.05, high=0.95, size=self.slice_per_volume)
        sample1_slices = sample1['slices_aug']
        sample2_slices = sample2['slices_aug']
        for i in range(self.slice_per_volume):
            if use_sample2[i] == 1:
                mixed_slice = sample1_slices[i] * alpha[i] + sample2_slices[i] * (1 - alpha[i])
                sample1_slices[i] = mixed_slice
            else:
                continue
        #sample1['slices_shadow0'] = sample2['slices_aug']
        #sample1['alpha'] = alpha
        #sample1['mixed'] = np.array(use_sample2)
        return sample1

    def plot(self, index):
        
        data = self.__getitem__(index)

        fig, axs = plt.subplots(2 + self.shadow_samples, self.slice_per_volume, figsize=(4 * self.slice_per_volume, 6 + 3 * self.shadow_samples))

        for i, item in enumerate([data['slices_raw'], data['slices_aug']] + [data['slices_shadow%s' % i] for i in range(self.shadow_samples)]):
            for j, slice in enumerate(item):
                if i == 0:
                    axs[i][j].set_title(int(data['slice_indices'][j]))
                if i == 1:
                    axs[i][j].set_title('%.3f, %d' % (data['alpha'][j], data['mixed'][j]))
                    
                axs[i][j].imshow(slice, cmap='gray')
        
        plt.show()