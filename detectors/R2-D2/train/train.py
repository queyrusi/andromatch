"""
This script trains an Inception v3 model to classify images as either 'goodware' or
'malware'.

It first checks if MPS or GPU is available and sets the target device accordingly. 
It then defines the transformations to apply to the images, which include resizing 
the images to 299x299 (as required by Inception v3) and converting them to tensors.

The script is expected to continue with loading the datasets, creating a DataLoader,
loading the pre-trained Inception v3 model, replacing the last layer to match the
number of classes, moving the model to the target device, defining the loss 
function and optimizer, and then training the model.

Usage:
    python train.py

Requirements:
    - PyTorch
    - torchvision
    - platform

"""
import os 
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import platform
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import warnings
import json
import argparse
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# Mute the warning
warnings.filterwarnings("ignore", category=UserWarning)

def get_device():

    # Check if MPS or GPU is available
    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch,'has_mps',False)

    device = "mps" if getattr(torch,'has_mps',False) \
        else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Python Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
    print("MPS is", "available" if has_mps else "NOT AVAILABLE")
    print(f"Target device is {device}")
    return device

def create_datasets(mw_folder, gw_folder, seed=42, log_dir='./log/splits'):
    # Define the transformation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((299, 299)),  # Resize to match Inception v3 input size
        torchvision.transforms.ToTensor(),  # Convert image to tensor
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load the datasets

    # Get the list of image files in each folder
    gw_images = [os.path.join(gw_folder, img) for img in os.listdir(gw_folder)]
    mw_images = [os.path.join(mw_folder, img) for img in os.listdir(mw_folder)]

    # Extract the dataset name from mw_folder path
    dataset_name = mw_folder.split('/mw/')[0].split('/')[-1]
    
    # Create a list of tuples containing image paths and labels
    data_list = [(img_path, 0) for img_path in gw_images if img_path.endswith('.jpg') or img_path.endswith('.png')] + \
        [(img_path, 1) for img_path in mw_images if img_path.endswith('.jpg') or img_path.endswith('.png')]
    
    dataset = CustomDataset(data_list, transform=transform)

    # Create train/test split
    train_size = int(0.5 * len(dataset))  # 5x2-CV exp. calls for 1/2 splits
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    # Record train sample paths in JSON
    train_sample_paths = []
    for img_path, _ in train_dataset.dataset.data_list:
        if '/gw/' in img_path:
            train_sample_paths.append(os.path.join(dataset_name, 'gw', os.path.basename(img_path).split('.')[0] + '.apk'))
        else:
            train_sample_paths.append(os.path.join(dataset_name, 'mw', os.path.basename(img_path).split('.')[0] + '.apk'))
    train_json = {'train_apks_paths': train_sample_paths}

    # Record test sample paths in JSON
    test_apks_paths = []
    for img_path, _ in test_dataset.dataset.data_list:
        if '/gw/' in img_path:
            test_apks_paths.append(os.path.join(dataset_name, 'gw', os.path.basename(img_path).split('.')[0] + '.apk'))
        else:
            test_apks_paths.append(os.path.join(dataset_name, 'mw', os.path.basename(img_path).split('.')[0] + '.apk'))
    test_json = {'test_apks_paths': test_apks_paths}

    # Save train JSON to file
    with open(os.path.join(log_dir, 'train.json'), 'w') as f:
        json.dump(train_json, f)

    # Save test JSON to file
    with open(os.path.join(log_dir, 'test.json'), 'w') as f:
        json.dump(test_json, f)

    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size=4):
    # Create DataLoaders for training and test
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader

def load_model():
    # Load the pre-trained Inception v3 model
    model = torchvision.models.inception_v3(pretrained=True)

    # Replace the last layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 1),
        torch.nn.Sigmoid()
    )

    model.aux_logits = False  # Ensure that auxiliary logits are not used for loss calculation

    # Move the model to the target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

def evaluate(model, test_dataloader, epoch):
    # Evaluate the model on the test set
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(test_dataloader, desc="running through test")):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs).cpu().T  # Convert probabilities to binary predictions
            labels = labels.cpu()

            # Calculate TP, FP, TN, FN
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

        if (2 * TP + FP + FN) != 0:
            f1 = 2 * TP / (2 * TP + FP + FN)
        else:
            f1 = 0
        
        if (TP + FP) != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        
        if (TP + FN) != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

    wandb.log({
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }, step=epoch)

    return TP, FP, TN, FN

def train(model, train_dataloader, test_dataloader, criterion, optimizer):
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_dataloader, 0), desc="passing through train"):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())  # Squeeze the output to remove extra dimension

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()


    model_dir = './model'
    # Create the ./model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model
    torch.save(model.state_dict(), model_dir + '/inception_v3.pth')
    
    return model, optimizer


if __name__ == "__main__":
    # Get info about current experiment
    parser = argparse.ArgumentParser(description="Get info about current experiment")
    parser.add_argument('--experiment_id', type=str, required=True, help='The ID of the experiment')
    parser.add_argument('--mw_folder', type=str, required=True, help='The path to the malware folder')
    parser.add_argument('--gw_folder', type=str, required=True, help='The path to the goodware folder')
    parser.add_argument('--seed', type=int, required=True, help='The seed for random split generation')
    parser.add_argument('--log_dir', type=str, required=True)
    
    args = parser.parse_args()
    experiment_id = args.experiment_id
    mw_folder = args.mw_folder
    gw_folder = args.gw_folder
    seed = args.seed
    log_dir = args.log_dir

    # Load the JSON config file
    config_path = 'detectors/R2-D2/train/wandb_configs/native.json'
    with open(config_path) as f:
        config = json.load(f)

    # Update the config with the parsed arguments
    config["experiment_id"] = experiment_id
    config["mw_folder"] = mw_folder
    config["gw_folder"] = gw_folder
    config["seed"] = seed
    config["log_dir"] = log_dir

    # Initialize a new run
    run = wandb.init(project="andromatch", config=config, dir='experiments')

    device = get_device()
    train_set, test_set = create_datasets(mw_folder, gw_folder, seed=seed, log_dir=log_dir)
    train_dataloader, test_dataloader = create_dataloaders(train_set, test_set, batch_size=run.config.batch_size)
    model = load_model().to(device)   
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=run.config.lr)

    # Train
    try:
        for epoch in range(config["epochs"]):
            model, optimizer = train(model, train_dataloader, test_dataloader, criterion, optimizer)
            evaluate(model, test_dataloader, epoch)
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        torch.save(model.state_dict(), model_dir + '/inception_v3.pth')
