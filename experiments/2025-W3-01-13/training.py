import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import time
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.utils import make_grid
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import yaml
import pickle
from src.utils import seed_everything, dataset_mean_std
from src.dataset_combine_7_channels import Dataset
from src.model import create_model
from src.figs import fig_confusion_matrix, fig_image_details





def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def training(config):



    # 1. ⚙️ Settings
    #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
    tb = config['training']['tensorboard'] # bool: tensorboard - to store or not to store                              .
    # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

    seed_everything(seed = 42) # random.seed etc 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if tb: #📍 (futher all tb writings are highlighed with 📍)
        writer = SummaryWriter(config['training']['tensorboard_dir'])


    # 2. 🗂 Dataset
    #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐                                                        .
    csv = config['data']['dir'] #                                                                                      .
    padding = config['dataset_T']['padding'] # [1, 62, 2, 62]                                                          .
    # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

    # calculate mean and std on a given dataset
    mean, std = dataset_mean_std(csv, padding) # padding will fit image from 100x221 to 224x224
    transform = T.Compose([
        T.Pad(padding),
        T.Normalize(mean, std)
    ])

    dataset = Dataset(csv_file=csv, transform=transform)
    dataset_y_true = [dataset[i][1] for i in range(len(dataset))]

    # 3. 🍰 K-fold cross validation
    #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
    k_folds = config['cross_val_folds'] #                                                                              .
    # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # ⭐️ as alternative for tensorboard
    results = {
        'folds': []
        }

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset_y_true)), dataset_y_true)):
        print(f"Fold {fold + 1}/{k_folds}")

        print(train_idx)

        # ⭐️
        fold_results = {
        'fold': fold,
        'epochs': []
        }

        print(np.unique([y_true for image, y_true in dataset]))

        # 1. 🗂/🗂 Train-Val split 
        dataset_t = torch.utils.data.Subset(dataset, train_idx)
        
        class_counts = {}
        for _, y_true in dataset_t:
            label = int(y_true)
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        print("Training distribution:", class_counts)
        
        dataset_v = torch.utils.data.Subset(dataset, val_idx)

        class_counts = {}
        for _, y_true in dataset_v:
            label = int(y_true)
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        print("Validation distribution:", class_counts)


        # 2. 🧺 Loader
        #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
        batch_size = config['dataloader']['batch_size'] #                                                                   .
        num_workers = config['dataloader']['num_workers'] #                                                                 .
        # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

        loader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loader_v = DataLoader(dataset_v, batch_size=batch_size, shuffle=False, num_workers=num_workers)  


        # 3. 🪷 Model
        #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
        model_name = config['model']['name'] #                                                                              .
        model_classes = config['model']['classes'] #                                                                        .
        model_channels = config['model']['channels'] #                                                                      .
        # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

        model = create_model(model_name, model_classes, model_channels)
        model.to(device)

        # model info
        num_tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_tr_params = num_tr_params * 4 / (1024**2)
        print(f'{num_tr_params} train parameters, model size is {size_tr_params:.1f} MB')

        if tb: #📍
            #writer.add_graph(model, example_batch.to(device)) # dont work with 7 channel, need to fix
            writer.add_text('model_info', f'{num_tr_params} train parameters, model size is {size_tr_params:.1f} MB')


        # 4. 🎚 Set train hyperparameters
        #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
        loss = config['training']['hyper']['loss']['type'] #                                                                .
        optimizer_type = config['training']['hyper']['optimizer']['type'] #                                                 .
        lr = config['training']['hyper']['optimizer']['lr'] #                                                               .
        scheduler_type = config['training']['hyper']['scheduler']['type'] #                                                 .
        step = config['training']['hyper']['scheduler']['step_size'] #                                                      .
        gamma = config['training']['hyper']['scheduler']['gamma'] #                                                         .
        # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

        # 🔹 loss
        # to calculate weighted loss we will need a number of images in each class. to find it lets return a list of all labels (on trainig part):
        dataset_t_y_true = [y_true for image, y_true in dataset_t]
        # and number if classes (there are 3 classes: 0, 1, 2)
        dataset_t_classes = np.unique(dataset_t_y_true)
        class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=dataset_t_classes, y=dataset_t_y_true), dtype=torch.float32) # balanced mean - calculates weights inversely (N/k*n)
        criterion = nn.CrossEntropyLoss(weight=class_weights) if loss == 'CrossEntropyLoss' else None
        # 🔹 param optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer_type == 'Adam' else None
        # 🔹 optimizer scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma) if scheduler_type == 'StepLR' else None


        # 5. 🗿 Iterate through epochs
        #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
        epochs = config['training']['num_epochs'] #                                                                        .   
        # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')   

            # ⭐️
            epoch_results = {
                'epoch': epoch,
                'training': {
                    'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'auc_ovr': [],
                    'auc_ovo': []
                },
                'validation': {
                    'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'auc_ovr': [],
                    'auc_ovo': []
                }
            }

            # 1. 📈 Training 
            print('Training')
            model.train() #🪷 
            loader = loader_t #🧺

            # --------------------------------- ✍ run record ---------------------------------- #
            #run_images = [] #🔜 # I've just listed all possible items for tracking during run (epoch), I'm not intrested in images during training
            run_y_true = [] #🔜
            run_y_prob = [] #🔜
            run_y_pred = [] #🔜
            run_loss = 0.0 #🔙
            # ---------------------------------------------------------------------------------- #

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
                #run_images.append(images.cpu().numpy())     #
                run_y_true.append(y_true.cpu().numpy())     #
                run_y_prob.append(y_prob.cpu().detach().numpy())     #
                run_y_pred.append(y_pred.cpu().numpy())     #
                run_loss += loss.item()                     #
                # ----------------------------------------- #


            # --------------------------------- ✍ run record - metrics ---------------------------------- #
            #run_images = np.concatenate(run_images)
            run_y_true = np.concatenate(run_y_true)
            run_y_prob = np.concatenate(run_y_prob)
            run_y_pred = np.concatenate(run_y_pred)


            # 🔸 loss
            run_loss = run_loss / len(loader)
            # 🔸 accuracy
            correct_predictions = sum(1 for true, pred in zip(run_y_true, run_y_pred) if true == pred)
            accuracy = correct_predictions / len(run_y_true)
            # 🔸 precision
            precision = precision_score(run_y_true, run_y_pred, average='macro', zero_division=0)
            # 🔸 recall
            recall = recall_score(run_y_true, run_y_pred, average='macro', zero_division=0)
            # 🔸 f1
            f1 = f1_score(run_y_true, run_y_pred, average='macro', zero_division=0)
            # 🔸 auc
            auc_ovr = roc_auc_score(run_y_true, run_y_prob, multi_class='ovr') # macro-averaging by default - arithm mean, each class is treated equally 
            auc_ovo = roc_auc_score(run_y_true, run_y_prob, multi_class='ovo')
            # ------------------------------------------------------------------------------------- #

            # print
            print(f'Training loss: {loss:.3f}, accuracy: {accuracy:.3f}, auc_ovr: {auc_ovr:.3f}') # lets show just three metrics

            # tensorboard
            if tb: # 📍
                writer.add_scalar('train_metrics/loss', loss, epoch+1)
                writer.add_scalar('train_metrics/accuracy' , accuracy, epoch+1)
                writer.add_scalar('train_metrics/precision', precision, epoch+1)
                writer.add_scalar('train_metrics/recall', recall, epoch+1)
                writer.add_scalar('train_metrics/f1', f1, epoch+1)
                writer.add_scalar('train_metrics/auc_ovr' , auc_ovr, epoch+1)
                writer.add_scalar('train_metrics/auc_ovo' , auc_ovo, epoch+1)

            # ⭐️
            epoch_results['training']['loss'] = loss   
            epoch_results['training']['accuracy'] = accuracy
            epoch_results['training']['precision'] = precision
            epoch_results['training']['recall'] = recall
            epoch_results['training']['f1'] = f1
            epoch_results['training']['auc_ovr'] = auc_ovr
            epoch_results['training']['auc_ovo'] = auc_ovo
            

            # 2. ❓ Validation
            print('Validation')
            model.eval()#🪷 
            loader = loader_v #🧺

            # --------------------------------- ✍ run record ---------------------------------- #
            run_images = [] #🔜
            run_y_true = [] #🔜
            run_y_prob = [] #🔜
            run_y_pred = [] #🔜
            run_loss = 0.0 #🔙
            # ---------------------------------------------------------------------------------- #

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
                    #loss.backward() # grad >                     . # cos validation
                    #optimizer.step() # parameters updated        . # cos validation
                    #.............................................
                    
                    # ----------- ✍ batch record ------------- #
                    run_images.append(images.cpu().numpy())     #
                    run_y_true.append(y_true.cpu().numpy())     #
                    run_y_prob.append(y_prob.cpu().detach().numpy())     #
                    run_y_pred.append(y_pred.cpu().numpy())     #
                    run_loss += loss.item()                     #
                    # ----------------------------------------- # 

            # --------------------------------- ✍ run metrics ---------------------------------- #

            run_images = np.concatenate(run_images)
            run_y_true = np.concatenate(run_y_true)
            run_y_prob = np.concatenate(run_y_prob)
            run_y_pred = np.concatenate(run_y_pred)


            # 🔸 loss
            run_loss = run_loss / len(loader)
            # 🔸 accuracy
            correct_predictions = sum(1 for true, pred in zip(run_y_true, run_y_pred) if true == pred)
            accuracy = correct_predictions / len(run_y_true)
            # 🔸 precision
            precision = precision_score(run_y_true, run_y_pred, average='macro', zero_division=0)
            # 🔸 recall
            recall = recall_score(run_y_true, run_y_pred, average='macro', zero_division=0)
            # 🔸 f1
            f1 = f1_score(run_y_true, run_y_pred, average='macro', zero_division=0)
            # 🔸 auc
            auc_ovr = roc_auc_score(run_y_true, run_y_prob, multi_class='ovr') # macro-averaging by default - arithm mean, each class is treated equally 
            auc_ovo = roc_auc_score(run_y_true, run_y_prob, multi_class='ovo')

            if tb: # 📍
                writer.add_scalar('val_metrics/loss', loss, epoch+1)
                writer.add_scalar('val_metrics/accuracy' , accuracy, epoch+1)
                writer.add_scalar('val_metrics/precision', precision, epoch+1)
                writer.add_scalar('val_metrics/recall', recall, epoch+1)
                writer.add_scalar('val_metrics/f1', f1, epoch+1)
                writer.add_scalar('val_metrics/auc_ovr' , auc_ovr, epoch+1)
                writer.add_scalar('val_metrics/auc_ovo' , auc_ovo, epoch+1)

            # ⭐️
            epoch_results['validation']['loss'] = loss   
            epoch_results['validation']['accuracy'] = accuracy
            epoch_results['validation']['precision'] = precision
            epoch_results['validation']['recall'] = recall
            epoch_results['validation']['f1'] = f1
            epoch_results['validation']['auc_ovr'] = auc_ovr
            epoch_results['validation']['auc_ovo'] = auc_ovo

            # 🔸 confusion matrix for the last epoch
            if epoch+1 == epochs:
                fig = fig_confusion_matrix(y_true = run_y_true, y_pred = run_y_pred)
                if tb: # 📍
                    writer.add_figure("val_confusion_matrix", fig)
            
            # 🔸 per class stat - here for each combination (i - true class, j - predicted class) lets track percent and images
            # just want to see how many images of class i were predicted as class j (0/1/2) and show them on the last epoch as example
            for i in range(len(dataset_t_classes)):
                for j in range(len(dataset_t_classes)):
                    mask = (run_y_true == i) & (run_y_pred == j)

                    # - percent
                    percent = mask.sum() / (run_y_true == i).sum() * 100 if (run_y_true == i).sum() > 0 else 0
                    
                    if tb: # 📍
                        writer.add_scalar(f"val_metrics_class_{i}/predicted_as_{j}", percent, epoch+1)

                    # - images on the last epoch
                    if epoch+1 == epochs:
                        if mask.sum() > 0:
                            count = 0
                            fig_list = []
                            for image, y_true, y_prob in zip(run_images[mask], run_y_true[mask], run_y_prob[mask]):

                                while count < 10:
                                    fig = fig_image_details(image, y_true, y_prob) # fig is a tensor
                                    fig_list.append(fig) # list of tensors
                                    grid = make_grid(fig_list, nrow=1, normalize=True)
                                    count += 1
                                
                            if tb: # 📍
                                writer.add_image(f"val_metrics_fig_class_{i}/predicted_as_{j}", grid, epoch+1)
                                
            # --------------------------------------------------------------------------------------- #
                    
            print(f'Validation loss: {loss:.3f}, accuracy: {accuracy:.3f}, auc_ovr: {auc_ovr:.3f}')

            scheduler.step() # adjust the learning rate
        
            # ⭐️
            fold_results['epochs'].append(epoch_results)
        
        # ⭐️
        results['folds'].append(fold_results)

    writer.close()
        
    print(results)

    pickle_path = config['pickle_path']
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str)
    args = parser.parse_args()

    config = load_config(args.config) #'experiments/2025-W3-01-13/configs/EfficientNet_b0.yaml'
    print('Config is loaded..')

    training(config)

    #python experiments/2025-W3-01-13/training.py --config experiments/2025-W3-01-13/configs/EfficientNet_b0.yaml