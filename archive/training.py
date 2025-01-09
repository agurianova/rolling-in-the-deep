# Standard library imports
import time

# Third-party library imports
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import yaml

# Project-specific imports
from src.utils import seed_everything
from src.dataset_combine_7_channels import Dataset
from src.model import create_model


def main(config_path):

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def training(config):




    # 1. ⚙️ Settings
    tb = config['training']['tensorboard']

    seed_everything(seed = 42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if tb: #📍
        writer = SummaryWriter(config['training']['tensorboard_dir'])




    # 2. 🗂 Dataset 
    transform = T.Compose([
        T.Pad(config['dataset_T']['padding']),
        T.Normalize(config['dataset_T']['normalization']['mean'], config['dataset_T']['normalization']['std'])
    ])

    dataset_t = Dataset(csv_file=config['data']['dir']['train'], transform=transform)
    dataset_v = Dataset(csv_file=config['data']['dir']['val'], transform=transform)




    # 3. 🧺 Loader
    loader_t = DataLoader(dataset_t, batch_size=config['dataloader']['batch_size'], shuffle=True, num_workers=config['dataloader']['num_workers'])
    loader_v = DataLoader(dataset_v, batch_size=config['dataloader']['batch_size'], shuffle=False, num_workers=config['dataloader']['num_workers'])  
    
    if tb:
        batch = next(iter(loader_t))
        images = batch[0]
        grid = make_grid(images, nrow=4, normalize=False)
        writer.add_image('train/example_batch_images', grid)




    # 4. 🪷 Model
    model = create_model(config)
    model.to(device)

    #info
    tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tr_params_size = tr_params * 4 / (1024**2)
    print(f'{tr_params} train parameters, model size is {tr_params_size:.1f} MB')

    if tb:
        writer.add_graph(model, next(iter(loader_t)).to(device))
        writer.add_text('model_info', f'{tr_params} train parameters, model size is {tr_params_size:.1f} MB')

    


    # 5. 🚩 Training
    # 5. 🚩    1. 🎚 Set train hyperparameters
    dataset_t_y_true = [y_true for _, y_true in dataset_t]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(dataset_t_y_true), y=dataset_t_y_true) # balanced mean - calculates weights inversely (N/k*n)

    criterion = nn.CrossEntropyLoss(weight=class_weights) if config['training']['hyper']['loss']['type'] == 'CrossEntropyLoss' else None
    optimizer = optim.Adam(model.parameters(), lr=config['training']['hyper']['optimizer']['lr']) if config['training']['hyper']['optimizer']['type'] == 'Adam' else None
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['training']['hyper']['scheduler']['step_size'], 
                                                gamma=config['training']['hyperp']['scheduler']['gamma']) if config['training']['hyper']['sheduler']['type'] == 'StepLR' else None




    # 5. 🚩    2. 🗿 Iterate through all epochs
    epochs = config['training']['num_epochs']
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        

    # 5. 🚩    2. 🗿     1. 📈..train run
        model.train() #🪷 
        loader = loader_t #🧺

        # ----- ✍ run record ----- # 
        run_y_true = [] #🔜
        run_y_pred = [] #🔜
        run_loss = 0.0 #🔙
        # ------------------------- #

        for _, (images, y_true) in enumerate(tqdm(loader)):

            #  >>>>>>>>> another batch from 🧺 <<<<<<<<< #
            images, y_true = images.to(device), y_true.to(device) #[32,7,221,100], [32]


            optimizer.zero_grad() 
            # 🔜 forward pass ............................
            y_logit = model(images) # logit >                 .
            y_prob = F.softmax(y_logit, dim=1) # prob >  .
            y_pred = torch.argmax(y_prob, dim=1) # pred  .
            #.............................................
            
            # 🔙 backward pass ...........................
            loss = criterion(y_logit, y_true) # loss >   .
            loss.backward() # grad >                     .
            optimizer.step() # parameters updated        .
            #.............................................

            # ----------- ✍ batch record ------------- #
            run_y_true.extend(y_true.cpu().numpy())     #
            run_y_pred.extend(y_pred.cpu().numpy())     #
            run_loss += loss.item()                     #
            # ----------------------------------------- #

        # ----- ✍ run record ----- #
        run_loss = run_loss / len(loader)
        accuracy = (run_y_pred == run_y_true).float().mean().item()
        auc_ovr = roc_auc_score(run_y_true, run_y_pred, multi_class='ovr') # macro-averaging by default - arithm mean, each class is treated equally 
        auc_ovo = roc_auc_score(run_y_true, run_y_pred, multi_class='ovo')
        # ------------------------- #

        print(f'Training loss: {loss:.3f}, accuracy: {accuracy:.3f}, auc_ovr: {auc_ovr:.3f}')

        # 📍 Tensorboard
        if tb:
            writer.add_scalar('train_metrics/loss', loss, epoch+1)
            writer.add_scalar('train_metrics/accuracy' , accuracy, epoch+1)
            writer.add_scalar('train_metrics/auc_ovr', auc_ovr, epoch+1)
            writer.add_scalar('train_metrics/auc_ovo', auc_ovo, epoch+1)
                

        # Validation loop----------------------------------------------------------------
        
        model.eval()#🪷 
        loader = loader_v #🧺

        # ----- ✍ run record ----- # 
        run_images = [] #🔜
        run_y_true = [] #🔜
        run_y_prob = [] #🔜
        run_y_pred = [] #🔜
        run_loss = 0.0 #🔙
        # ------------------------- #

        with torch.no_grad():
            for _, (images, y_true) in enumerate(tqdm(loader)):

                #  >>>>>>>>> another batch from 🧺 <<<<<<<<< #
                images, y_true = images.to(device), y_true.to(device)


                #optimizer.zero_grad() 
                # 🔜 forward pass ............................
                y_logit = model(images) # logit >     
                y_prob = F.softmax(y_logit, dim=1) # prob >  .
                y_pred = torch.argmax(y_prob, dim=1) # pred  .
                #.............................................

                # 🔙 backward pass ...........................
                loss = criterion(y_logit, y_true) # loss >   .
                #loss.backward() # grad >                     .
                #optimizer.step() # parameters updated        .
                #.............................................
                
                # ----------- ✍ batch record ------------- #
                run_images.extend(images.cpu().numpy())     #
                run_y_true.extend(y_true.cpu().numpy())     #
                run_y_prob.extend(y_prob.cpu().numpy())     #
                run_y_pred.extend(y_pred.cpu().numpy())     #
                run_loss += loss.item()                     #
                # ----------------------------------------- # 

        
        # ----- ✍ run record ----- #
        run_loss = run_loss / len(loader)
        accuracy = (run_y_pred == run_y_true).float().mean().item()
        auc_ovr = roc_auc_score(run_y_true, run_y_pred, multi_class='ovr') # macro-averaging by default - arithm mean, each class is treated equally 
        auc_ovo = roc_auc_score(run_y_true, run_y_pred, multi_class='ovo')
        # ------------------------- #
                
        # 📍 Tensorboard
        if tb:
            writer.add_scalar('val_metrics/loss', loss, epoch+1)
            writer.add_scalar('val_metrics/accuracy' , accuracy, epoch+1)
            writer.add_scalar('val_metrics/auc_ovr' , auc_ovr, epoch+1)
            writer.add_scalar('val_metrics/auc_ovo' , auc_ovo, epoch+1)
        
        print(f'Validation loss: {loss:.3f}, accuracy: {accuracy:.3f}')

        scheduler.step() # adjust the learning rate
        
    writer.close()
              


To run training.py using a configuration file, you can design the training.py script to load parameters from a configuration file (e.g., YAML, JSON, or INI). Here's a step-by-step guide:

Step 1: Create a Configuration File
A configuration file stores parameters like hyperparameters, dataset paths, model settings, etc.

Example: config.yaml
yaml
Копировать код
# Configuration for training

# General settings
seed: 42
device: cuda

# Dataset
dataset:
  train_path: ./data/train
  val_path: ./data/val
  batch_size: 32
  num_workers: 4

# Model
model:
  name: resnet50
  num_classes: 10

# Training
training:
  epochs: 20
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: Adam
Step 2: Modify training.py to Load Configuration
Use a library like yaml or json to read the configuration file in your script.

Example: training.py
python
Копировать код
import yaml
import torch
from torch.utils.data import DataLoader
from src.utils import seed_everything
from src.dataset_combine_7_channels import Dataset
from src.model import create_model
from tqdm import tqdm

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set the seed for reproducibility
    seed_everything(config['seed'])

    # Select device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    train_dataset = Dataset(config['dataset']['train_path'])
    val_dataset = Dataset(config['dataset']['val_path'])

    train_loader = DataLoader(train_dataset, 
                              batch_size=config['dataset']['batch_size'], 
                              num_workers=config['dataset']['num_workers'], 
                              shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=config['dataset']['batch_size'], 
                            num_workers=config['dataset']['num_workers'], 
                            shuffle=False)

    # Create model
    model = create_model(config['model']['name'], config['model']['num_classes'])
    model.to(device)

    # Define optimizer
    if config['training']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config['training']['learning_rate'], 
                                     weight_decay=config['training']['weight_decay'])
    elif config['training']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config['training']['learning_rate'], 
                                    weight_decay=config['training']['weight_decay'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation step can be added here

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    main(args.config)