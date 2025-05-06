import sys
import os
import argparse
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import time
import multiprocessing
import numpy as np
import torch
import gc
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP # 🪄
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # 🪄
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp # 🪄
from torch.distributed import init_process_group, destroy_process_group, get_rank # 🪄
import torch.distributed as dist
from torchvision import transforms as T
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
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



def training(rank, config, world_size):



    # 1. ⚙️ Settings
    #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
    tb = config['training']['tensorboard'] # bool: tensorboard - to store or not to store                              .
    # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

    seed_everything(seed = 42) # random.seed etc 

    torch.cuda.set_device(rank) # 🪄
    init_process_group(backend="nccl", rank=rank, world_size=world_size) # 🪄
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else 'cpu') # 🪄
    print(device)
    
    if tb: #📍 (futher all tb writings are highlighed with 📍)
        writer = SummaryWriter(config['training']['tensorboard_dir'])


    # 2. 🗂 Dataset
    #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐                                                        .
    csv = config['data']['dir'] #                                                                                      .
    padding = config['dataset_T']['padding'] # [1, 62, 2, 62]                                                          .
    # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

    # calculate mean and std on a given dataset
    print("Find mean, std..")
    mean, std = dataset_mean_std(csv, padding) # padding will fit image from 100x221 to 224x224
    transform = T.Compose([
        T.Pad(padding),
        T.Normalize(mean, std)
    ])

    print("Dataset..")
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
        print(f"Before fold {fold + 1}: {torch.cuda.memory_allocated() / 1e9} GB")


        # ⭐️
        fold_results = {
        'fold': fold,
        'epochs': []
        }


        # 1. 🗂/🗂 Train-Val split 
        dataset_t = torch.utils.data.Subset(dataset, train_idx)
        

        ## (Training/Validation distribution)
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

        loader_t = DataLoader(dataset_t, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=DistributedSampler(dataset_t, num_replicas=world_size, rank=rank, shuffle=True)) # 🪄
        loader_v = DataLoader(dataset_v, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=DistributedSampler(dataset_v, num_replicas=world_size, rank=rank, shuffle=False)) # 🪄  


        # 3. 🪷 Model
        #  config parameters for this part ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐
        model_name = config['model']['name'] #                                                                              .
        model_classes = config['model']['classes'] #                                                                        .
        model_channels = config['model']['channels'] #                                                                      .
        # ࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐࿐

        model = create_model(model_name, model_classes, model_channels)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank]) # 🪄

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
        class_weights = class_weights.to(device)
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

            print(f'Epoch {epoch+1}/{epochs} {torch.cuda.memory_allocated() / 1e9} GB')   
            # ⭐️

            loader_t.sampler.set_epoch(epoch)

            epoch_results = {
                'epoch': epoch,
                'training': {
                    'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'auc_ovr': [],
                    'auc_ovo': [],
                    'mcc': [],
                    'inference_time': []
                },
                'validation': {
                    'loss': [],
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'auc_ovr': [],
                    'auc_ovo': [],
                    'mcc': [],
                    'inference_time': []
                }
            }

            # 1. 📈 Training 
            print('Training')
            start_time = time.time()
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
                run_y_true.append(y_true.detach())     #
                run_y_prob.append(y_prob.detach())     #
                run_y_pred.append(y_pred.detach())     #
                run_loss += loss.item()                     #
                # ----------------------------------------- #


            # --------------------------------- ✍ run record - metrics ---------------------------------- #
            #run_images = np.concatenate(run_images)
            run_y_true = torch.cat(run_y_true, dim=0)
            run_y_prob = torch.cat(run_y_prob, dim=0)
            run_y_pred = torch.cat(run_y_pred, dim=0)
            run_loss_tensor = torch.tensor([run_loss / len(loader)], device=device)
            elapsed_time = torch.tensor([(time.time() - start_time) / 60], device=device)

            # -------- Distributed Gathering -------- #
            local_size = torch.tensor([run_y_true.size(0)], device=device)
            sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(sizes_list, local_size)
            max_size = max([s.item() for s in sizes_list])

            def pad_tensor(x, max_len):
                if x.dim() == 1:
                    return torch.cat([x, torch.zeros(max_len - x.size(0), device=x.device)])
                else:
                    return torch.cat([x, torch.zeros((max_len - x.size(0), x.size(1)), device=x.device)])

            run_y_true_pad = pad_tensor(run_y_true, max_size)
            run_y_prob_pad = pad_tensor(run_y_prob, max_size)
            run_y_pred_pad = pad_tensor(run_y_pred, max_size)

            true_gather = [torch.zeros_like(run_y_true_pad) for _ in range(world_size)]
            prob_gather = [torch.zeros_like(run_y_prob_pad) for _ in range(world_size)]
            pred_gather = [torch.zeros_like(run_y_pred_pad) for _ in range(world_size)]
            loss_gather = [torch.zeros_like(run_loss_tensor) for _ in range(world_size)]
            time_gather = [torch.zeros_like(elapsed_time) for _ in range(world_size)]

            torch.distributed.barrier()

            # all_gather from all GPU
            dist.all_gather(true_gather, run_y_true_pad)
            dist.all_gather(prob_gather, run_y_prob_pad)
            dist.all_gather(pred_gather, run_y_pred_pad)
            dist.all_gather(loss_gather, run_loss_tensor)
            dist.all_gather(time_gather, elapsed_time)

            if dist.get_rank() == 0:
                all_true, all_prob, all_pred = [], [], []
                for i in range(world_size):
                    n = sizes_list[i].item()
                    all_true.append(true_gather[i][:n])
                    all_prob.append(prob_gather[i][:n])
                    all_pred.append(pred_gather[i][:n])

                y_true = torch.cat(all_true).cpu().numpy()
                y_prob = torch.cat(all_prob).cpu().numpy()
                y_pred = torch.cat(all_pred).cpu().numpy()
                avg_loss = torch.stack(loss_gather).mean().item()
                avg_time = torch.stack(time_gather).mean().item()

                # Compute metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr')
                auc_ovo = roc_auc_score(y_true, y_prob, multi_class='ovo')
                mcc = matthews_corrcoef(y_true, y_pred)

                print(f"[Train] Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Time: {avg_time:.2f}min")

                # tensorboard
                if tb: # 📍 
                    writer.add_scalar('train_metrics/loss', avg_loss, epoch+1)
                    writer.add_scalar('train_metrics/accuracy' , accuracy, epoch+1)
                    writer.add_scalar('train_metrics/precision', precision, epoch+1)
                    writer.add_scalar('train_metrics/recall', recall, epoch+1)
                    writer.add_scalar('train_metrics/f1', f1, epoch+1)
                    writer.add_scalar('train_metrics/auc_ovr' , auc_ovr, epoch+1)
                    writer.add_scalar('train_metrics/auc_ovo' , auc_ovo, epoch+1)
                    writer.add_scalar('train_metrics/mcc' , mcc, epoch+1)
                    writer.add_scalar('train_metrics/inference_time', avg_time, epoch+1)

                # ⭐️
                epoch_results['training']['loss'] = avg_loss   
                epoch_results['training']['accuracy'] = accuracy
                epoch_results['training']['precision'] = precision
                epoch_results['training']['recall'] = recall
                epoch_results['training']['f1'] = f1
                epoch_results['training']['auc_ovr'] = auc_ovr
                epoch_results['training']['auc_ovo'] = auc_ovo
                epoch_results['training']['mcc'] = mcc
                epoch_results['training']['inference_time'] = avg_time
                

            # 2. ❓ Validation
            print('Validation')
            start_time = time.time()
            model.eval()#🪷 
            loader = loader_v #🧺

            # --------------------------------- ✍ run record ---------------------------------- #
            #run_images = [] #🔜
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
                    #run_images.append(images.detach().cpu().numpy())     #
                    run_y_true.append(y_true.detach())     #
                    run_y_prob.append(y_prob.detach())     #
                    run_y_pred.append(y_pred.detach())     #
                    run_loss += loss.item()                     #
                    # ----------------------------------------- # 

                # --------------------------------- ✍ run metrics ---------------------------------- #

                #run_images = np.concatenate(run_images)
                run_y_true = torch.cat(run_y_true, dim=0)
                run_y_prob = torch.cat(run_y_prob, dim=0)
                run_y_pred = torch.cat(run_y_pred, dim=0)
                run_loss_tensor = torch.tensor([run_loss / len(loader)], device=device)
                elapsed_time = torch.tensor([(time.time() - start_time) / 60], device=device)

                # -------- Distributed Gathering -------- #
                local_size = torch.tensor([run_y_true.size(0)], device=device)
                sizes_list = [torch.zeros_like(local_size) for _ in range(world_size)]
                dist.all_gather(sizes_list, local_size)
                max_size = max([s.item() for s in sizes_list])

                run_y_true_pad = pad_tensor(run_y_true, max_size)
                run_y_prob_pad = pad_tensor(run_y_prob, max_size)
                run_y_pred_pad = pad_tensor(run_y_pred, max_size)

                true_gather = [torch.zeros_like(run_y_true_pad) for _ in range(world_size)]
                prob_gather = [torch.zeros_like(run_y_prob_pad) for _ in range(world_size)]
                pred_gather = [torch.zeros_like(run_y_pred_pad) for _ in range(world_size)]
                loss_gather = [torch.zeros_like(run_loss_tensor) for _ in range(world_size)]
                time_gather = [torch.zeros_like(elapsed_time) for _ in range(world_size)]

                torch.distributed.barrier()
                dist.all_gather(true_gather, run_y_true_pad)
                dist.all_gather(prob_gather, run_y_prob_pad)
                dist.all_gather(pred_gather, run_y_pred_pad)
                dist.all_gather(loss_gather, run_loss_tensor)
                dist.all_gather(time_gather, elapsed_time)

                if dist.get_rank() == 0:
                    all_true, all_prob, all_pred = [], [], []
                    for i in range(world_size):
                        n = sizes_list[i].item()
                        all_true.append(true_gather[i][:n])
                        all_prob.append(prob_gather[i][:n])
                        all_pred.append(pred_gather[i][:n])

                    y_true = torch.cat(all_true).cpu().numpy()
                    y_prob = torch.cat(all_prob).cpu().numpy()
                    y_pred = torch.cat(all_pred).cpu().numpy()
                    avg_loss = torch.stack(loss_gather).mean().item()
                    avg_time = torch.stack(time_gather).mean().item()

                    # Compute metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    auc_ovo = roc_auc_score(y_true, y_prob, multi_class='ovo')
                    mcc = matthews_corrcoef(y_true, y_pred)

                    print(f"[Validation] Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Time: {avg_time:.2f}min")

                    if tb: # 📍
                        writer.add_scalar('val_metrics/loss', avg_loss, epoch+1)
                        writer.add_scalar('val_metrics/accuracy' , accuracy, epoch+1)
                        writer.add_scalar('val_metrics/precision', precision, epoch+1)
                        writer.add_scalar('val_metrics/recall', recall, epoch+1)
                        writer.add_scalar('val_metrics/f1', f1, epoch+1)
                        writer.add_scalar('val_metrics/auc_ovr' , auc_ovr, epoch+1)
                        writer.add_scalar('val_metrics/auc_ovo' , auc_ovo, epoch+1)
                        writer.add_scalar('val_metrics/mcc' , mcc, epoch+1)
                        writer.add_scalar('val_metrics/inference_time', avg_time, epoch+1)

                    # ⭐️
                    epoch_results['validation']['loss'] = avg_loss   
                    epoch_results['validation']['accuracy'] = accuracy
                    epoch_results['validation']['precision'] = precision
                    epoch_results['validation']['recall'] = recall
                    epoch_results['validation']['f1'] = f1
                    epoch_results['validation']['auc_ovr'] = auc_ovr
                    epoch_results['validation']['auc_ovo'] = auc_ovo
                    epoch_results['validation']['mcc'] = mcc
                    epoch_results['validation']['inference_time'] = avg_time

            # 🔸 confusion matrix for the last epoch
            #if epoch+1 == epochs:
            #    fig = fig_confusion_matrix(y_true = run_y_true, y_pred = run_y_pred)
            #    if tb: # 📍
            #        writer.add_figure("val_confusion_matrix", fig)
            
            # 🔸 per class stat - here for each combination (i - true class, j - predicted class) lets track percent and images
            # just want to see how many images of class i were predicted as class j (0/1/2) and show them on the last epoch as example
            #for i in range(len(dataset_t_classes)):
            #    for j in range(len(dataset_t_classes)):
            #        mask = (run_y_true == i) & (run_y_pred == j)
            #
            #        # - percent
            #        percent = mask.sum() / (run_y_true == i).sum() * 100 if (run_y_true == i).sum() > 0 else 0
            #        
            #        if tb: # 📍
            #            writer.add_scalar(f"val_metrics_class_{i}/predicted_as_{j}", percent, epoch+1)
            #
            #        # - images on the last epoch
            #        if epoch+1 == epochs:
            #            if mask.sum() > 0:
            #                count = 0
            #                fig_list = []
            #                for image, y_true, y_prob in zip(run_images[mask], run_y_true[mask], run_y_prob[mask]):
            #
            #                    while count < 10:
            #                        fig = fig_image_details(image, y_true, y_prob) # fig is a tensor
            #                        fig_list.append(fig) # list of tensors
            #                        grid = make_grid(fig_list, nrow=1, normalize=True)
            #                        count += 1
            #                    
            #                if tb: # 📍
            #                    writer.add_image(f"val_metrics_fig_class_{i}/predicted_as_{j}", grid, epoch+1)
                                
            # --------------------------------------------------------------------------------------- #
                    
            scheduler.step() # adjust the learning rate
        
            # ⭐️
            fold_results['epochs'].append(epoch_results)
        
        # ⭐️
        results['folds'].append(fold_results)

        print(f"Fold {fold + 1} befor del: {torch.cuda.memory_allocated() / 1e9} GB")
        del model, optimizer, loader, loader_t, loader_v
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Fold {fold + 1} after del: {torch.cuda.memory_allocated() / 1e9} GB")

    destroy_process_group() # 🪄

    if tb:
        writer.close()
        
    print(results)

    pickle_path = config['pickle_path']
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    print('Config is loaded..')

    world_size = torch.cuda.device_count()  # Number of GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(training, args=(config, world_size), nprocs=world_size, join=True)