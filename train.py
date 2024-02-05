import time
import os
import random
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from ortho_classification_model import OCModel
from ortho_classification_dataset import OCDataset
import transforms


def write_logs(file_name,
               out_type,
               classes=None,
               epoch=None,
               total_epochs=None,
               step=None,
               total_steps=None,
               loss_train=None,
               loss_valid=None,
               accuracy=None,
               precision=None,
               recall=None,
               fscore=None,
               test_out=None):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name), 'a') as file:
        if out_type == 'classes info' and classes != None:
            classes_info = [f"{index}: {class_name}" for index,
                            class_name in enumerate(classes)]
            file.write('\n'.join(classes_info))
        elif out_type == 'train step':
            file.write("Epoch: {}/{}  Train Steps: {}/{}  Loss: {:.20f} \n".format(epoch,
                                                                                   total_epochs,
                                                                                   step,
                                                                                   total_steps,
                                                                                   loss_train))
        elif out_type == 'train stage end':
            file.write("Epoch: {}/{}  Accuracy: {:.20f}\n".format(epoch,
                                                                  total_epochs,
                                                                  accuracy))
        elif out_type == 'valid step':
            file.write("Epoch: {}/{}  Valid Steps: {}/{}  Loss: {:.20f} \n".format(epoch,
                                                                                   total_epochs,
                                                                                   step,
                                                                                   total_steps,
                                                                                   loss_valid))
        elif out_type == 'valid stage end':
            file.write("Epoch: {}/{}  Accuracy: {:.20f}  Precision: {:.20f}  Recall: {:.20f}  F1: {:.20f}\n".format(epoch,
                                                                                                                    total_epochs,
                                                                                                                    accuracy,
                                                                                                                    precision,
                                                                                                                    recall,
                                                                                                                    fscore))
        elif out_type == 'epoch end':
            file.write('Epoch: {}/{}  Train Loss: {:.20f}  Valid Loss: {:.20f}\n'.format(epoch,
                                                                                         total_epochs,
                                                                                         loss_train,
                                                                                         loss_valid))
        elif out_type == 'saving model':
            file.write("MODEL SAVED;  Epoch: {}/{}  Min Valid Loss:  {:.20f}\n".format(epoch,
                                                                                       total_epochs,
                                                                                       loss_valid))
        elif out_type == 'test set info' and test_out != None:
            file.write(test_out)


def split_dataset(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    total_len = len(dataset)
    indices = list(range(total_len))
    images_to_indices = [
        {
            "name": os.path.basename(dataset.image_paths[idx]),
            "path": dataset.image_paths[idx],
            "idx": idx,
        }
        for idx in indices]

    unique_image_names = list(set(os.path.basename(item["name"])
                                  for item in images_to_indices))
    random.shuffle(unique_image_names)

    unique_len = len(unique_image_names)
    train_size = int(train_ratio * unique_len)
    valid_size = int(valid_ratio * unique_len)
    test_size = unique_len - train_size - valid_size

    calculated_test_ratio = test_size / unique_len
    assert abs(test_ratio - calculated_test_ratio) < 0.01, "Incorrect test ratio"

    train_image_names = unique_image_names[:train_size]
    valid_image_names = unique_image_names[train_size:train_size + valid_size]
    test_image_names = unique_image_names[train_size + valid_size:]

    print(f"Train unique set size: {len(train_image_names)}")
    print(f"Valid unique set size: {len(valid_image_names)}")
    print(f"Test  unique set size: {len(test_image_names)}")

    train_indices = [item["idx"] for item in images_to_indices if os.path.basename(
        item["name"]) in train_image_names]
    valid_indices = [item["idx"] for item in images_to_indices if os.path.basename(
        item["name"]) in valid_image_names]
    test_indices = [item["idx"] for item in images_to_indices if os.path.basename(
        item["name"]) in test_image_names]

    # Check duplicates
    assert unique_len == \
        len(set([images_to_indices[idx]["name"] for idx in train_indices])) + \
        len(set([images_to_indices[idx]["name"] for idx in valid_indices])) + \
        len(set([images_to_indices[idx]["name"]
            for idx in test_indices])), 'Wrong train/valid/test split!'

    train_dataset = torch.utils.data.Subset(
        dataset, train_indices)  # type: ignore
    valid_dataset = torch.utils.data.Subset(
        dataset, valid_indices)  # type: ignore
    test_dataset = torch.utils.data.Subset(
        dataset, test_indices)  # type: ignore

    return train_dataset, valid_dataset, test_dataset


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True, drop_last=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=drop_last)


def calculate_weighted_sum(loss, accuracy, precision, recall, f1):
    loss_weight = 0.85
    accuracy_weight = 0.8
    precision_weight = 0.6
    recall_weight = 0.6
    f1_weight = 0.7

    weighted_sum = loss_weight * loss + accuracy_weight * \
        (1 - accuracy) + precision_weight * (1 - precision) + \
        recall_weight * (1 - recall) + f1_weight * (1 - f1)

    return weighted_sum


def train(device, num_epochs, early_stopping_limit, lr_scheduler_patience, model_name_for_saving, dataset_directory_path, model_weights_path=None):
    dataset = OCDataset([folder.path for folder in os.scandir(dataset_directory_path)],
                        transform=transforms.ImageTransforms())

    write_logs(file_name='classes_info.txt',
               out_type='classes info',
               classes=dataset.classes)

    train_set, valid_set, test_set = split_dataset(dataset,
                                                   train_ratio=0.8,
                                                   valid_ratio=0.1,
                                                   test_ratio=0.1)

    print(f"Train set size: {len(train_set)}")
    print(f"Valid set size: {len(valid_set)}")
    print(f"Test  set size: {len(test_set)}")

    train_loader = create_data_loader(train_set)
    valid_loader = create_data_loader(valid_set)
    test_loader = create_data_loader(test_set)

    torch.autograd.set_detect_anomaly(True)
    network = OCModel(num_classes=len(dataset.classes)).to(device)
    if model_weights_path:
        network.load_state_dict(torch.load(model_weights_path,
                                           map_location=torch.device(device)))

    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(dataset.targets),
                                         y=dataset.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(network.parameters(), lr=0.0003, weight_decay=1e-8)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=0.1,
                                               patience=lr_scheduler_patience,
                                               verbose=True)
    early_stopping_counter = 0
    best_weighted_sum = np.inf

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        loss_train_total = 0.0
        accuracy_train_total = 0.0
        n_samples_train_total = 0

        network.train()
        for step in range(1, len(train_loader)+1):
            images, targets = next(iter(train_loader))
            images = images.to(device)
            targets = targets.to(device)
            predicted_targets = network(images)
            loss_train_step = loss(predicted_targets, targets)

            predicted_probs = F.softmax(predicted_targets, dim=1)
            _, predicted_indices = torch.max(predicted_probs, dim=1)
            accuracy_train_step = accuracy_score(targets.cpu().numpy(),
                                                 predicted_indices.cpu().numpy())
            loss_train_step.backward()
            optimizer.step()
            optimizer.zero_grad()

            n_samples = images.shape[0]

            loss_train_total += loss_train_step.item() * n_samples
            accuracy_train_total += accuracy_train_step * n_samples
            n_samples_train_total += n_samples

            write_logs(file_name='train_info.txt',
                       out_type='train step',
                       epoch=epoch,
                       total_epochs=num_epochs,
                       step=step,
                       total_steps=len(train_loader),
                       loss_train=loss_train_step.item())

        loss_train_res = loss_train_total / n_samples_train_total
        accuracy_train_res = accuracy_train_total / n_samples_train_total

        write_logs(file_name='train_info.txt',
                   out_type='train stage end',
                   epoch=epoch,
                   total_epochs=num_epochs,
                   accuracy=accuracy_train_res)

        print(f'\rTrained epoch: {epoch}/{num_epochs}')

        network.eval()
        with torch.no_grad():
            all_targets = []
            all_predictions = []
            loss_valid_total = 0.0
            accuracy_valid_total = 0.0
            n_samples_valid_total = 0
            for step in range(1, len(valid_loader)+1):
                images, targets = next(iter(valid_loader))
                images = images.to(device)
                targets = targets.to(device)
                predicted_targets = network(images)
                loss_valid_step = loss(predicted_targets, targets)

                predicted_probs = F.softmax(predicted_targets, dim=1)
                _, predicted_indices = torch.max(predicted_probs, dim=1)
                accuracy_valid_step = accuracy_score(targets.cpu().numpy(),
                                                     predicted_indices.cpu().numpy())

                n_samples = images.shape[0]

                loss_valid_total += loss_valid_step.item() * n_samples
                accuracy_valid_total += accuracy_valid_step * n_samples
                n_samples_valid_total += n_samples

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted_indices.cpu().numpy())

                write_logs(file_name='train_info.txt',
                           out_type='valid step',
                           epoch=epoch,
                           total_epochs=num_epochs,
                           step=step,
                           total_steps=len(valid_loader),
                           loss_valid=loss_valid_step.item())

            loss_valid_res = loss_valid_total / n_samples_valid_total
            accuracy_valid_res = accuracy_valid_total / n_samples_valid_total

            precision, recall, fscore, _ = precision_recall_fscore_support(all_targets,
                                                                           all_predictions,
                                                                           average='samples')

            write_logs(file_name='accuracy_precision_recall_F1_info.txt',
                       out_type='valid stage end',
                       epoch=epoch,
                       total_epochs=num_epochs,
                       accuracy=accuracy_valid_res,
                       precision=precision,
                       recall=recall,
                       fscore=fscore)

        write_logs(file_name='train_info.txt',
                   out_type='epoch end',
                   epoch=epoch,
                   total_epochs=num_epochs,
                   loss_train=loss_train_res,
                   loss_valid=loss_valid_res)

        print(
            f'Epoch {epoch}/{num_epochs} ended || Train accuracy: {accuracy_train_res} || Valid accuracy: {accuracy_valid_res}')

        current_weighted_sum = calculate_weighted_sum(loss_valid_res,
                                                      accuracy_valid_res,
                                                      precision,
                                                      recall,
                                                      fscore)

        scheduler.step(loss_valid_res)

        if current_weighted_sum < best_weighted_sum:
            best_weighted_sum = current_weighted_sum

            torch.save(network.state_dict(),
                       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f'{network.model_name}_{model_name_for_saving}_weights_epoch_{epoch}.pth'))

            write_logs(file_name='train_info.txt',
                       out_type='saving model',
                       epoch=epoch,
                       total_epochs=num_epochs,
                       loss_valid=loss_valid_res)

            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_limit:
            print(f"Early stopping after {epoch} epochs without improvement")
            break

    network.eval()
    with torch.no_grad():
        all_targets = []
        all_predictions = []
        for step, (images, targets) in enumerate(test_loader, 1):

            images = images.to(device)
            targets = targets.to(device)
            predicted_targets = network(images)

            predicted_probs = F.softmax(predicted_targets, dim=1)
            _, predicted_indices = torch.max(predicted_probs, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted_indices.cpu().numpy())

        test_output = classification_report(
            all_targets, all_predictions, target_names=dataset.classes)

        print(test_output)

        write_logs(file_name='test_set_info.txt',
                   out_type='test set info',
                   test_out=test_output)

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time()-start_time))


if __name__ == '__main__':
    train(device=("cuda" if torch.cuda.is_available() else "cpu"),
          num_epochs=1000,
          early_stopping_limit=100,
          lr_scheduler_patience=25,
          model_name_for_saving='ortho_classification',
          dataset_directory_path=r'D:\repos\sorted_new_2023_10_24_dataset_balanced',
          model_weights_path=None)
