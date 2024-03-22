import os
import random
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import copy
import argparse

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


from torcheeg.datasets import SEEDIVFeatureDataset, SEEDFeatureDataset, SEEDVFeatureDataset, DREAMERDataset

from torcheeg.datasets.constants.emotion_recognition import \
    SEED_GENERAL_REGION_LIST, SEED_FRONTAL_REGION_LIST, SEED_HEMISPHERE_REGION_LIST, SEED_NEIGHBOR_REGION_LIST, \
    SEED_POSTERIOR_REGION_LIST,\
    SEED_GENERAL_REGION_MASK_MATRIX, SEED_FRONTAL_REGION_MASK_MATRIX, SEED_HEMISPHERE_REGION_MASK_MATRIX, SEED_NEIGHBOR_REGION_MASK_MATRIX, \
    SEED_POSTERIOR_REGION_MASK_MATRIX

from torcheeg.datasets.constants.emotion_recognition import \
    DREAMER_GENERAL_REGION_LIST, DREAMER_FRONTAL_REGION_LIST, DREAMER_HEMISPHERE_REGION_LIST, DREAMER_NEIGHBOR_REGION_LIST, \
    DREAMER_POSTERIOR_REGION_LIST,\
    DREAMER_GENERAL_REGION_MASK_MATRIX, DREAMER_FRONTAL_REGION_MASK_MATRIX, DREAMER_HEMISPHERE_REGION_MASK_MATRIX, DREAMER_NEIGHBOR_REGION_MASK_MATRIX, \
    DREAMER_POSTERIOR_REGION_MASK_MATRIX

from torcheeg import transforms
from torcheeg.model_selection import \
    KFoldPerSubjectCrossTrial, Subcategory
from torcheeg.io import MetaInfoIO

from torcheeg.models.pyg import HDGNet

def arg_parse():
    parser = argparse.ArgumentParser(description='HDGNet')


    parser.add_argument('--dataset_name', dest='dataset_name',
                        type=str, help='Dataset name you use.')
    parser.add_argument('--exper_set', dest='exper_set',
                        type=str, help='Experiment setting.')
    parser.add_argument('--model_name', dest='model_name',
                        type=str, help='The network model name.')
    parser.add_argument('--lr', dest='lr',
                        type=float, help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs',
                        type=int, help='Number of epochs to train.')
    parser.add_argument('--num_classes', dest='num_classes',
                        type=int, help='The number of classes')
    parser.add_argument('--num_workers', dest='num_workers',
                        type=int, help='Number of workers to load data.')
    parser.add_argument('--threshold', dest='threshold',
                        type=float, help='The label threshold of binarization.')
    parser.add_argument('--emotion_key', dest='emotion_key',
                        type=str, help='Emotion task select')
    parser.add_argument('--random_seed', dest='random_seed',
                        type=int, help='Seed used for pseudorandom number generate.')
    parser.add_argument('--Split', dest='Split',
                        type=bool, help='Whether the data has been split. True is yes, False is no')
    parser.add_argument('--n_outer', dest='n_outer',
                        type=int, help='set the outer loop splits')
    parser.add_argument('--n_inner', dest='n_inner',
                        type=int, help='set the inner loop splits')

    # the model's parameter setting
    parser.add_argument('--graph_defi', dest='graph_defi',
                        type=str, help='Select local-global graph definition')
    parser.add_argument('--num_electrodes', dest='num_electrodes',
                        type=int, help='The number of electrodes.')
    parser.add_argument('--in_channels', dest='in_channels',
                        type=int, help='The feature dimension of each electrode.')
    parser.add_argument('--hid_channels', dest='hid_channels',
                        type=int, help='The number of hidden nodes in the local GNN layer.')
    parser.add_argument('--out_channels', dest='out_channels',
                        type=int, help='The number of hidden nodes in the global GNN layer.')

    parser.set_defaults(dataset_name='SEED', ###
                        exper_set='trial_nest',
                        model_name='examples_HDGNet',
                        lr=3e-4,
                        batch_size=200,
                        epochs=200,
                        num_classes=3,###
                        num_workers=0,
                        threshold=0.0, ### DRE
                        emotion_key='', ### DRE
                        random_seed=42,
                        Split=False,
                        n_outer=5, ###
                        n_inner=2, ###
                        graph_defi='POSTERIOR', ####
                        num_electrodes=62, ###
                        in_channels=5, ###
                        hid_channels=5, ###
                        out_channels=10 ###
        )
    return parser.parse_args()

# select the mask and region division
def select_region(dataset_name, graph_defi):

    if dataset_name=='SEED' or dataset_name=='SEED-IV' or dataset_name=='SEED-V':
       if graph_defi == 'GENERAL':
          return SEED_GENERAL_REGION_MASK_MATRIX, SEED_GENERAL_REGION_LIST
       elif graph_defi == 'FRONTAL':
           return SEED_FRONTAL_REGION_MASK_MATRIX, SEED_FRONTAL_REGION_LIST
       elif graph_defi == 'HEMISPHERE':
           return SEED_HEMISPHERE_REGION_MASK_MATRIX, SEED_HEMISPHERE_REGION_LIST
       elif graph_defi == 'NEIGHBOR':
           return SEED_NEIGHBOR_REGION_MASK_MATRIX, SEED_NEIGHBOR_REGION_LIST
       elif graph_defi == 'POSTERIOR':
           return SEED_POSTERIOR_REGION_MASK_MATRIX, SEED_POSTERIOR_REGION_LIST
       else:
           raise ValueError("Please use existing graph definitions or adding new code to deal new graph")
    elif dataset_name=='DREAMER':
        if graph_defi == 'GENERAL':
            return DREAMER_GENERAL_REGION_MASK_MATRIX, DREAMER_GENERAL_REGION_LIST
        elif graph_defi == 'FRONTAL':
            return DREAMER_FRONTAL_REGION_MASK_MATRIX, DREAMER_FRONTAL_REGION_LIST
        elif graph_defi == 'HEMISPHERE':
            return DREAMER_HEMISPHERE_REGION_MASK_MATRIX, DREAMER_HEMISPHERE_REGION_LIST
        elif graph_defi == 'NEIGHBOR':
            return DREAMER_NEIGHBOR_REGION_MASK_MATRIX, DREAMER_NEIGHBOR_REGION_LIST
        elif graph_defi == 'POSTERIOR':
            return DREAMER_POSTERIOR_REGION_MASK_MATRIX, DREAMER_POSTERIOR_REGION_LIST
        else:
            raise ValueError("Please use existing graph definitions or adding new code to deal new graph")
    else:
        raise ValueError("Please use existing dataset or adding new code to deal new dataset")

    return region_mask, region_list

###############################################################################
# Set the random number seed in all modules to guarantee the same result when running again.
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################################
# Set train and valid function
def train(dataloader, model, loss_fn, optimizer,):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            loss, current = loss.item(), batch_idx * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] \n")

def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    preds, ys = [], []
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            preds.append(pred.cpu().numpy())
            ys.append(y.unsqueeze(1).cpu().numpy())

    preds = np.vstack(preds)
    ys = np.vstack(ys)
    f1 = f1_score(ys, preds.argmax(1), average='macro')

    val_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f}, f1: {100*f1:>0.1f} \n")
    return round(100*correct, 2), round(val_loss,8), round(100*f1, 2)


###############################################################################
# Building Deep Learning Pipelines Using torcheeg
if __name__=="__main__":
    args = arg_parse()

    seed_everything(args.random_seed)

    ################################################################################
    # Get the folder where the dataset is stored
    # tmp_out: to store the output, include: log file, the processed data
    # tmp_in: to store the input, include: EEG dataset

    tmp_out = os.path.join(os.getcwd(), 'tmp_out')
    tmp_in = os.path.join(os.getcwd(), 'tmp_in')

    # Step 1: Initialize the Dataset
    if args.dataset_name == 'SEED':
        # Here we use the extracted EEG feature in public SEED- dataset
        dataset = SEEDFeatureDataset(io_path=os.path.join(tmp_out, args.model_name, f'{args.dataset_name}Feature'),
                                     root_path=os.path.join(tmp_in, args.dataset_name, 'ExtractedFeatures'),
                                     feature=['de_LDS'],
                                     online_transform=transforms.Compose([transforms.ToTensor()]),
                                     label_transform=transforms.Compose([
                                         transforms.Select('emotion'),
                                         transforms.Lambda(lambda x: int(x) + 1)]),
                                     num_worker=1,
                                     io_size=1 * 1024 * 1024 * 1024)

    elif args.dataset_name == 'SEED-IV':
        dataset = SEEDIVFeatureDataset(io_path=os.path.join(tmp_out, args.model_name, f'{args.dataset_name}Feature'),
                                       root_path=os.path.join(tmp_in, args.dataset_name, 'eeg_feature_smooth'),
                                       feature=['de_LDS'],
                                       online_transform=transforms.Compose([transforms.ToTensor()]),
                                       label_transform=transforms.Select('emotion'),
                                       num_worker=1,
                                       io_size=0.5 * 1024 * 1024 * 1024)

    elif args.dataset_name == 'SEED-V':
        dataset = SEEDVFeatureDataset(io_path=os.path.join(tmp_out, args.model_name, f'{args.dataset_name}Feature'),
                                      root_path=os.path.join(tmp_in, args.dataset_name, 'EEG_DE_features'),
                                      online_transform=transforms.Compose([transforms.ToTensor()]),
                                      label_transform=transforms.Select('emotion'),
                                      num_worker=1,
                                      io_size=0.5 * 1024 * 1024 * 1024)

    elif args.dataset_name == 'DREAMER':
        dataset = DREAMERDataset(io_path=os.path.join(tmp_out, args.model_name, f'{args.dataset_name}Feature'),
                                 mat_path=os.path.join(tmp_in, args.dataset_name, 'DREAMER.mat'),
                                 offline_transform=transforms.Compose([
                                     transforms.BandPowerSpectralDensity(band_dict={
                                         "theta": [4, 8],
                                         "alpha": [8, 13],
                                         "beta": [13, 20]
                                     }),
                                 ]),
                                 online_transform=transforms.ToTensor(),
                                 label_transform=transforms.Compose([
                                     transforms.Select(args.emotion_key),
                                     transforms.Binary(args.threshold),
                                 ]),
                                 num_worker=1,
                                 io_size=0.5 * 1024 * 1024 * 1024,
                                 io_mode='pickle')
    else:
        raise ValueError("Please use existing dataset or adding new code to deal new dataset")

    if not args.Split:
        if args.dataset_name == 'SEED':
            # Here we do not consider the impact of cross-session on the test results. Therefore, we first mark the session index on the sample according to the collection date. Next, we use :obj:`Subcategory` to divide the data set to obtain the sub-data set of the first session, the second session and the third session.
            # add the session information, SEED need this step when do the session split
            subject_info_list = []
            for subject_id in dataset.info['subject_id'].unique().tolist():
                subject_info = dataset.info[dataset.info['subject_id'] == subject_id]
                session_id_set = subject_info['date'].unique().tolist()
                session_id_map = {
                    session_id: i
                    for i, session_id in enumerate(session_id_set)
                }
                subject_info['session_id'] = subject_info['date'].apply(
                    lambda x: session_id_map[x])
                subject_info_list.append(subject_info)
            dataset.info = pd.concat(subject_info_list)

            # only use the first session for SEED dataset
            subset_sess = Subcategory(criteria='session_id',
                                      split_path=os.path.join(
                                          tmp_out, args.model_name,
                                          f'cross_{args.exper_set}', args.dataset_name,
                                          f'split_fix_{args.exper_set}_session'))
            for sub_sess, sub_dataset in enumerate(subset_sess.split(dataset)):
                if sub_sess == 0:
                    sess_data = sub_dataset
            dataset = sess_data

        elif args.dataset_name == 'DREAMER':
            # For DREAMER, only use the data of 60s after a trial
            new_info = pd.DataFrame(columns=dataset.info.columns)
            subject_id_index_list = dataset.info['subject_id'].unique().tolist()
            for subject_index, subject_name in enumerate(subject_id_index_list):
                one_subject_info = dataset.info[dataset.info['subject_id'] == subject_name]
                trial_id_index_list = one_subject_info['trial_id'].unique().tolist()
                for trail_index, trial_name in enumerate(trial_id_index_list):
                    one_trial_info = one_subject_info[one_subject_info['trial_id'] == trial_name]
                    trial_last_60s = one_trial_info.tail(60)
                    new_info = new_info.append(trial_last_60s, ignore_index=True)

            dataset.info = new_info

            # there are some info in the DREAMER data set that does not or does not conform to the rules, so we need to manually add adjustments
            # add session information
            dataset.info['session_id'] = 1

            # # add label information to emotion according selected class
            dataset.info['emotion'] = dataset.info[args.emotion_key].apply(lambda x: 1 if x >= args.threshold else 0)

        else:
            print("The dataset requires no special setup")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size
    result = {}
    result_path = os.path.join(tmp_out, args.model_name,
                               f'cross_{args.exper_set}', 'result')

    region_mask, region_list = select_region(args.dataset_name,
                                             args.graph_defi)

    Mymodel = HDGNet(region_mask=region_mask,
                     region_list=region_list,
                     num_electrodes=args.num_electrodes,
                     in_channels=args.in_channels,
                     hid_channels=args.hid_channels,
                     out_channels=args.out_channels,
                     num_classes=args.num_classes).to(device)

    # divide subject
    subset1 = Subcategory(criteria='subject_id',
                          split_path=os.path.join(
                              tmp_out, args.model_name,
                              f'cross_{args.exper_set}', args.dataset_name,
                              f'split_fix_{args.exper_set}_subject'))

    t_start = datetime.datetime.now()
    for sub_idex, sub_dataset in enumerate(subset1.split(dataset)):  # the number of subjects

        print(f'Start the subject{sub_idex+1}')

        subset2 = KFoldPerSubjectCrossTrial(n_splits=args.n_outer, shuffle=False,
                                            stratified=True, split_path=os.path.join(
                                                                    tmp_out,
                                                                    args.model_name,
                                                                    f'cross_{args.exper_set}', args.dataset_name,
                                                                    f'split_fix_{args.exper_set}_subject_{sub_idex + 1}'))

        for outer_idex, (outer_train_dataset, outer_test_dataset) in enumerate(subset2.split(sub_dataset)):  # outer loop k_fold
            print(f'Start the outer loop {outer_idex + 1}')

            subset3 = KFoldPerSubjectCrossTrial(n_splits=args.n_inner, shuffle=False,
                                                stratified=True, split_path=os.path.join(
                                                                    tmp_out,
                                                                    args.model_name,
                                                                    f'cross_{args.exper_set}', args.dataset_name,
                                                                    f'split_fix_{args.exper_set}_subject_{sub_idex + 1}_outer_{outer_idex}'))

            inner_best_val_acc = float('-inf')
            inner_best_model_params = None
            for inner_idex, (inner_train_dataset, inner_test_dataset) in enumerate(subset3.split(outer_train_dataset)):  # inner loop k_fold
                print(f'Start the outer/inner loop {outer_idex + 1}/{inner_idex + 1}')

                model = copy.deepcopy(Mymodel)
                loss_fn = nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

                train_loader = DataLoader(inner_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
                val_loader = DataLoader(inner_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

                epochs = args.epochs
                best_val_acc = float('-inf')
                best_model_params = None
                conti_count = 0
                correct_conti = 0
                for t in range(epochs):
                    print(f"Epoch {t + 1}\n-------------------------------")
                    train(train_loader, model, loss_fn, optimizer)
                    correct_tr, _, _ = valid(train_loader, model, loss_fn)
                    correct, loss, f1 = valid(val_loader, model, loss_fn)
                    if t % 1 == 0:
                        if correct > best_val_acc:  # get the best parameter from epochs based on validation
                            best_val_acc = correct
                            best_model_params = model.state_dict().copy()

                    if correct_tr > 99.99 and correct_conti != correct:
                        correct_conti = correct
                        conti_count = 0
                    if correct_tr > 99.99 and correct_conti == correct:
                        conti_count = conti_count + 1

                    if (correct_tr > 99.99 and conti_count > 10) or correct > 99.99:
                        break

                # Use the model parameters of the best epoch to predict the outer_train_dataset
                outer_train_loader = DataLoader(outer_train_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
                model = copy.deepcopy(Mymodel)
                model.load_state_dict(best_model_params)
                correct, loss, f1 = valid(outer_train_loader, model, loss_fn)
                if correct > inner_best_val_acc:
                    inner_best_val_acc = correct
                    inner_best_model_params = model.state_dict().copy()


            # Load the best set of model parameters in the inner loop to predict the outer_test_dataset
            test_loader = DataLoader(outer_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
            model = copy.deepcopy(Mymodel)
            model.load_state_dict(inner_best_model_params)
            correct, loss, f1 = valid(test_loader, model, loss_fn)
            outer_result = {
                'subject_id': sub_idex+1,
                'outer_id': outer_idex,
                'correct': correct,
                'f1': f1,
                'loss': loss
            }
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            meta_result_path = os.path.join(result_path, f'cross_fix_{args.exper_set}_result_{args.dataset_name}.csv')
            result_io = MetaInfoIO(meta_result_path)
            result_io.write_info(outer_result)

    t_end = datetime.datetime.now()
    print("model run time:", t_end - t_start)















