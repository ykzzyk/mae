import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import pathlib
import PIL.Image as Image
import matplotlib.pyplot as plt
import shutil
import models_mae
import os
import pandas as pd
import tqdm

data_dir = pathlib.Path("data")

def image_normalization(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img

def save_data(ct, pretained):
    output_dir = data_dir / "2Dimages_pretrained" if pretained else data_dir / "2Dimages"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(ct.shape[-1]):
        img = ct[..., i]
        img = image_normalization(img)
        if pretained: img = bilinear_resize(img, 224, 224)
        np.save(output_dir / f"ct_{i}.npy", img)
    return output_dir
        
def create_dirs(data_dir, dirs):
    for dir in dirs:
        os.makedirs(os.path.join(data_dir, dir), exist_ok=True)

def move_files(file_list, target_dir):
    for file_path in file_list:
        shutil.move(file_path, target_dir)
        
def split_data(image_dir, pretained, train_size=0.7, validate_size=0.15, test_size=0.15):
    # Ensure the sizes sum up to 1
    if (train_size + validate_size + test_size) != 1.0:
        raise ValueError("Train, validate, and test sizes must sum up to 1")

    files = os.listdir(image_dir)
    files = [os.path.join(image_dir, f) for f in files if f.endswith('.npy')]
    
    # Shuffle files
    np.random.shuffle(files)

    # Splitting the dataset
    train_files, validate_files, test_files = np.split(files, [int(.7*len(files)), int(.85*len(files))])
    
    output_dir = data_dir / "pretrained" if pretained else data_dir / "not_pretrained"
    os.makedirs(output_dir, exist_ok=True)
    
    # Creating directories for the splits
    create_dirs(output_dir, ['train', 'validate', 'test'])

    # Moving the files
    move_files(train_files, os.path.join(output_dir, 'train'))
    move_files(validate_files, os.path.join(output_dir, 'validate'))
    move_files(test_files, os.path.join(output_dir, 'test'))
    
    # Delete the folder
    shutil.rmtree(image_dir)

# visualization
def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(image[..., 0], cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def visualize(epoch, output_dir, model_mae, x, mask, y):
    y = model_mae.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model_mae.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model_mae.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.savefig(os.path.join(output_dir, f"epoch_{epoch}.png"))
    
def bilinear_resize(image, height, width):
    src_height, src_width = image.shape[:2]
    resized_image = np.zeros((height, width, *image.shape[2:]), dtype=image.dtype)

    x_ratio = src_width / width
    y_ratio = src_height / height

    for i in range(height):
        for j in range(width):
            x_l, y_l = np.floor([x_ratio * j, y_ratio * i]).astype(int)
            x_h, y_h = np.ceil([x_ratio * j, y_ratio * i]).astype(int)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h] if x_h < src_width else a
            c = image[y_h, x_l] if y_h < src_height else a
            d = image[y_h, x_h] if (x_h < src_width and y_h < src_height) else a

            pixel = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight
            resized_image[i, j] = pixel

    return resized_image
    
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def generate_dataset(pretained):
    # Generate 2D images from 3D CT
    # ct1 = nib.load(data_dir / "1.2.840.113654.2.70.1.100215820090357097111777802286335180398.nii.gz").get_fdata()
    # ct2 = nib.load(data_dir / "1.2.840.113654.2.70.1.101671239926961178188281461454089436958.nii.gz").get_fdata()
    # ct = np.concatenate([ct1, ct2], axis=-1)
    # output_dir = save_data(ct, pretained)
    # output_dir = split_data(output_dir, pretained)
    
    # dataset_train = [np.array(Image.open(os.path.join(output_dir / "train", f))) for f in os.listdir(output_dir / "train") if f.endswith('.npy')]
    # dataset_valid = [np.array(Image.open(os.path.join(output_dir / "validate", f))) for f in os.listdir(output_dir / "validate") if f.endswith('.npy')]
    # dataset_test = [np.array(Image.open(os.path.join(output_dir / "test", f))) for f in os.listdir(output_dir / "test") if f.endswith('.npy')]
    
    if pretained:
        dataset_train = [np.load(os.path.join(data_dir / "pretained" / "train", f)) for f in os.listdir(data_dir / "pretained" / "train") if f.endswith('.npy')]
        dataset_valid = [np.load(os.path.join(data_dir / "pretained" / "validate", f)) for f in os.listdir(data_dir / "pretained" / "validate") if f.endswith('.npy')]
        dataset_test = [np.load(os.path.join(data_dir / "pretained" / "test", f)) for f in os.listdir(data_dir / "pretained" / "test") if f.endswith('.npy')]
    else:
        dataset_train = [np.load(os.path.join(data_dir / "not_pretrained" / "train", f)) for f in os.listdir(data_dir / "not_pretrained" / "train") if f.endswith('.npy')]
        dataset_valid = [np.load(os.path.join(data_dir / "not_pretrained" / "validate", f)) for f in os.listdir(data_dir / "not_pretrained" / "validate") if f.endswith('.npy')]
        dataset_test = [np.load(os.path.join(data_dir / "not_pretrained" / "test", f)) for f in os.listdir(data_dir / "not_pretrained" / "test") if f.endswith('.npy')]

    dataset_valid.append(dataset_test)
        
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=10,
        pin_memory=True,
        drop_last=True,
    )
    
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=10,
        pin_memory=True,
        drop_last=True,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=10,
        pin_memory=True,
        drop_last=True,
    )
    
    return pretained, dataset_train, data_loader_train, data_loader_valid

# Run only one slices
def sanity_check(pretained, epoch_num, image, model_mae, optimizer):
    output_dir = pathlib.Path("output/sanity_check/pretrained" if pretained else "output/sanity_check/not_pretained")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "run", exist_ok=True)
    df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Valid Loss'])
    for epoch in tqdm.tqdm(range(epoch_num), total=epoch_num):
        optimizer.zero_grad()
        batch = torch.from_numpy(image).unsqueeze(0)
        x = batch.unsqueeze(dim=-1).repeat(1, 1, 1, 3)
        x = torch.einsum('nhwc->nchw', x)
        loss, y, mask = model_mae(x.float(), mask_ratio=0.75)
        loss.backward()
        optimizer.step()
        visualize(epoch, output_dir / "run", model_mae, x, mask, y)
        print(f"epoch: {epoch},loss: {loss.item()}")
        new_row = pd.DataFrame({'Epoch': [epoch], 'Train Loss': [loss.item()], 'Valid Loss': [0]})
        df = pd.concat([df, new_row], ignore_index=True)
        
    df.to_excel(output_dir / 'losses_curve.xlsx', index=False)
    
# Run all the slices
def run_all_samples(pretained, epoch_num, model_mae, optimizer, data_loader_train, data_loader_valid):
    total_train_loss = 0
    total_valid_loss = 0
    best_loss = 1e+8
    df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Valid Loss'])
    output_dir = pathlib.Path("output/run_all_samples/pretained" if pretained else "output/run_all_samples/not_pretained")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "run", exist_ok=True)
    for epoch in tqdm.tqdm(range(epoch_num), total=epoch_num):
        for _, batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            # make it a batch-like
            x = batch.unsqueeze(dim=-1).repeat(1, 1, 1, 3)
            x = torch.einsum('nhwc->nchw', x)
            # run MAE
            loss, y, mask = model_mae(x.float(), mask_ratio=0.75)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        with torch.no_grad():
            for _, batch in enumerate(data_loader_valid):
                # make it a batch-like
                x = batch.unsqueeze(dim=-1).repeat(1, 1, 1, 3)
                x = torch.einsum('nhwc->nchw', x)
                # run MAE
                _, y, mask = model_mae(x.float(), mask_ratio=0.75)
                total_valid_loss += loss.item()
                if total_valid_loss < best_loss:
                    best_loss = total_valid_loss
                    visualize(epoch, output_dir / "run", model_mae, x, mask, y)
                    torch.save(model_mae.state_dict(), output_dir / 'best_model.pth')
        
        train_loss_mean = total_train_loss / len(data_loader_train)
        valid_loss_mean = total_valid_loss / len(data_loader_valid)
        print(f"train_loss_mean: {train_loss_mean}, valid_loss_mean: {valid_loss_mean}")
        
        # Create a new row DataFrame to append
        new_row = pd.DataFrame({'Epoch': [epoch], 'Train Loss': [train_loss_mean], 'Valid Loss': [valid_loss_mean]})
        
        # Append the new row
        df = pd.concat([df, new_row], ignore_index=True)
        
    # Save the DataFrame to a CSV file
    df.to_excel(output_dir / 'losses_curve.xlsx', index=False)
    
if __name__ == "__main__":
    
    pretained, dataset_train, data_loader_train, data_loader_valid = generate_dataset(pretained=True)
     
    # Use the pretrained model
    if pretained:
        model_mae = prepare_model(data_dir.parent / 'mae_visualize_vit_large.pth', 'mae_vit_large_patch16')
    # train from scratch
    else:
        model_mae = getattr(models_mae, 'mae_vit_large_patch16')()
    model_mae.train(True)
    
    epochs = 200
    # match with the original training setting
    lr = 1e-3 * 10 / 256
    optimizer = torch.optim.AdamW(model_mae.parameters(), lr=lr)
    
    sanity_check(pretained, epochs, dataset_train[0], model_mae, optimizer)
    # run_all_samples(pretained, epochs, model_mae, optimizer, data_loader_train, data_loader_valid)