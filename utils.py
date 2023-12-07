import os
import sys
import json
import pickle
import random
import math
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import logging
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def read_split_data(root: str, val_rate: float = 0.2):
    # random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    pathology = os.path.join(root, 'pathology')
    endoscope = os.path.join(root, 'endoscope')

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(pathology) if os.path.isdir(os.path.join(pathology, cla))]   #分期
    # 排序，保证顺序一致
    flower_class.sort()    # 每期的图片
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)

    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_pimages_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    train_eimages_path = []  # 存储训练集的所有图片路径


    val_pimages_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    val_eimages_path = []  # 存储验证集的所有图片路径

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(pathology, cla)
        # 遍历获取supported支持的所有文件路径
        patients = [id for id in os.listdir(cla_path)] #just id

        # 获取该类别对应的索引
        pimage_class = class_indices[cla]
        # 记录该类别的样本数量 病人样本
        every_class_num.append(len(patients))
        # 按比例随机采样验证样本  病人数量比例8：2
        val_id = random.sample(patients, k=int(len(patients) * val_rate))

        for id in patients:
            id_path = os.path.join(pathology, cla, id)
            pimage = [os.path.join(pathology, cla, id, i) for i in os.listdir(id_path) #pimage_path
                      if os.path.splitext(i)[-1] in supported]
            only_pimage = random.sample(pimage, 1)  #每个ID里只随机取一张的路径 only_pimage_path

            id_path2 = os.path.join(endoscope, cla, id)
            eimage = [os.path.join(endoscope, cla, id, i) for i in os.listdir(id_path2)
                      if os.path.splitext(i)[-1] in supported]
            only_eimage = random.sample(eimage, 1)  # 每个ID里只随机取一张的路径 only_eimage_path

            if id in val_id:  # 如果该路径在采样的验证集样本中则存入验证集 ID为验证集
                val_pimages_path.append(only_pimage)
                val_eimages_path.append(only_eimage)
                val_images_label.append(pimage_class)

            else:  # 否则存入训练集
                train_pimages_path.append(only_pimage)
                train_eimages_path.append(only_eimage)
                train_images_label.append(pimage_class)


    print("{} patients were found in the dataset.".format(sum(every_class_num)))
    print("{}P and {}E images for training.".format(len(train_pimages_path), len(train_eimages_path)))
    print("{}P and {}E images for validation.".format(len(val_pimages_path), len(val_eimages_path)))

    assert len(train_pimages_path) > 0, "not find data for train."
    assert len(val_pimages_path) > 0, "not find data for eval"

    return train_pimages_path, train_eimages_path, train_images_label, \
           val_pimages_path, val_eimages_path, val_images_label


def read_multi_task_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Create dictionaries to store class indices for each task
    task_indices = {}
    class_indices = {}

    train_Ppatients_path = []
    train_Epatients_path = []
    train_patients_label = []

    pathology = os.path.join(root, 'pathology')
    endoscope = os.path.join(root, 'endoscope')

    for class_name in os.listdir(pathology):
        pclass_path = os.path.join(pathology, class_name)
        class_path = os.path.join(endoscope, class_name)

        if os.path.isdir(class_path):
            # Extract the class label (e.g., "1a") from the directory name
            class_label = class_name

            # Add the class label to the class_indices dictionary
            if class_label not in class_indices:
                class_indices[class_label] = len(class_indices)

            # List all subdirectories in the class directory (e.g., "103940348_G2", "234356464_G3")
            task_dirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            ptask_dirs = [d for d in os.listdir(pclass_path) if os.path.isdir(os.path.join(pclass_path, d))]

            # Process each task directory
            for task_dir in task_dirs:
                task_path = os.path.join(class_path, task_dir)

                # Check if the item in the task directory is a directory
                if os.path.isdir(task_path):
                    # Add the task label to the task_indices dictionary
                    task_label = task_dir.split('_')[-1]
                    if task_label not in task_indices:
                        task_indices[task_label] = len(task_indices)

                    # Get the class index and task index for the current patient
                    class_index = class_indices[class_label]
                    task_index = task_indices[task_label]

                    # Add the patient path, class index, and task index to the lists
                    train_Epatients_path.append(task_path)
                    train_patients_label.append((class_index, task_index))

            for ptask_dir in ptask_dirs:
                etask_path = os.path.join(pclass_path, ptask_dir)
                train_Ppatients_path.append(etask_path)

    # Print information about the classes, tasks, and the number of patients for training
    print("Classes:", list(class_indices.keys()))
    print("Tasks:", list(task_indices.keys()))
    print("Number of classes:", len(class_indices))
    print("Number of tasks:", len(task_indices))
    print("{} Ppatients for training.".format(len(train_Ppatients_path)))
    print("{} Epatients for training.".format(len(train_Epatients_path)))
    print("{} Labels for training.".format(len(train_patients_label)))

    return train_Epatients_path, train_Ppatients_path, train_patients_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, ncols=80)

    for step, data in enumerate(data_loader):
        pimgs, eimgs, labels = data  # new add
        sample_num += pimgs.shape[0]

        # Forward pass for both tasks
        stage_output, severity_output = model(pimgs.to(device), eimgs.to(device))

        # Calculate loss for each task
        loss_stage = loss_function(stage_output, labels[:, 0].to(device))
        loss_severity = loss_function(severity_output, labels[:, 1].to(device))

        # Combine the losses (you can use weighted sum if needed)
        total_loss = loss_stage + loss_severity

        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calculate accuracy for each task
        pred_stage = torch.max(stage_output, dim=1)[1]
        pred_severity = torch.max(severity_output, dim=1)[1]

        accu_stage = torch.eq(pred_stage, labels[:, 0].to(device)).sum()
        accu_severity = torch.eq(pred_severity, labels[:, 1].to(device)).sum()

        accu_loss += total_loss.item()
        accu_num += (accu_stage + accu_severity)

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / (sample_num * 2),  # Divide by 2 since we have two tasks
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(total_loss):
            logging.warning('WARNING: non-finite loss, ending training')
            break

        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / (sample_num * 2)  # Divide by 2 for two tasks


class MyDataSet(Dataset):
    def __init__(self, pimages_path: list, eimages_path: list, images_class: list, ptransform=None, etransform=None):
        self.pimages_path = pimages_path
        self.eimages_path = eimages_path
        self.images_class = images_class
        self.ptransform = ptransform
        self.etransform = etransform

    def __len__(self):
        return len(self.images_class)

    def __getitem__(self, item):
        pimg_folder = self.pimages_path[item]
        eimg_folder = self.eimages_path[item]

        pimg_files = os.listdir(pimg_folder)
        eimg_files = os.listdir(eimg_folder)

        # Randomly select an image from each folder
        pimg_filename = random.choice(pimg_files)
        eimg_filename = random.choice(eimg_files)

        pimg_path = os.path.join(pimg_folder, pimg_filename)
        eimg_path = os.path.join(eimg_folder, eimg_filename)

        pimg = Image.open(pimg_path)
        eimg = Image.open(eimg_path)

        if pimg.mode != 'RGB':
            pimg = pimg.convert("RGB")
        if eimg.mode != 'RGB':
            eimg = eimg.convert("RGB")

        label = self.images_class[item]

        if self.ptransform is not None:
            pimg = self.ptransform(pimg)
        if self.etransform is not None:
            eimg = self.etransform(eimg)

        return pimg, eimg, label

    @staticmethod
    def collate_fn(batch):
        pimgs, eimgs, labels = zip(*batch)

        pimgs = torch.stack(pimgs, dim=0)
        eimgs = torch.stack(eimgs, dim=0)
        labels = torch.tensor(labels, dtype=torch.int64)  # Assuming labels are integers

        return pimgs, eimgs, labels


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num_stage = torch.zeros(1).to(device)
    accu_loss_stage = torch.zeros(1).to(device)

    accu_num_severity = torch.zeros(1).to(device)
    accu_loss_severity = torch.zeros(1).to(device)

    sample_num = 0
    labels_stage = []
    labels_severity = []
    pred_classes_stage = []
    pred_classes_severity = []

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        pimgs, eimgs, labels = data
        sample_num += pimgs.shape[0]

        stage_output, severity_output = model(pimgs.to(device), eimgs.to(device))

        # Calculate loss for stage task
        loss_stage = loss_function(stage_output, labels[:, 0].to(device))
        accu_loss_stage += loss_stage

        pred_class_stage = torch.max(stage_output, dim=1)[1]
        accu_num_stage += torch.eq(pred_class_stage, labels[:, 0].to(device)).sum()

        # Calculate loss for severity task
        loss_severity = loss_function(severity_output, labels[:, 1].to(device))
        accu_loss_severity += loss_severity

        pred_class_severity = torch.max(severity_output, dim=1)[1]
        accu_num_severity += torch.eq(pred_class_severity, labels[:, 1].to(device)).sum()

        # Collect labels and predicted classes for both tasks
        labels_stage.extend(labels[:, 0].cpu().detach().numpy())
        labels_severity.extend(labels[:, 1].cpu().detach().numpy())
        pred_classes_stage.extend(pred_class_stage.cpu().detach().numpy())
        pred_classes_severity.extend(pred_class_severity.cpu().detach().numpy())

        data_loader.desc = "[valid epoch {}] stage loss: {:.4f}, stage acc: {:.4f}, severity loss: {:.4f}, severity acc: {:.4f}".format(
            epoch,
            accu_loss_stage.item() / (step + 1),
            accu_num_stage.item() / sample_num,
            accu_loss_severity.item() / (step + 1),
            accu_num_severity.item() / sample_num
        )

    labels_stage = np.array(labels_stage)
    labels_severity = np.array(labels_severity)
    pred_classes_stage = np.array(pred_classes_stage)
    pred_classes_severity = np.array(pred_classes_severity)

    # Calculate metrics separately for each task (stage and severity)
    precision_stage, recall_stage, f1_stage, _ = metrics.precision_recall_fscore_support(
        labels_stage, pred_classes_stage, average='macro', zero_division=False)

    precision_severity, recall_severity, f1_severity, _ = metrics.precision_recall_fscore_support(
        labels_severity, pred_classes_severity, average='macro', zero_division=False)

    # Calculate confusion matrices for both tasks (stage and severity)
    cm_stage = confusion_matrix(labels_stage, pred_classes_stage)
    cm_severity = confusion_matrix(labels_severity, pred_classes_severity)

    # Calculate accuracy for both tasks (stage and severity)
    acc_stage = metrics.accuracy_score(labels_stage, pred_classes_stage)
    acc_severity = metrics.accuracy_score(labels_severity, pred_classes_severity)

    # Calculate Cohen's Kappa for both tasks (stage and severity)
    kappa_stage = metrics.cohen_kappa_score(labels_stage, pred_classes_stage)
    kappa_severity = metrics.cohen_kappa_score(labels_severity, pred_classes_severity)

    # Calculate Matthews Correlation Coefficient (MCC) for both tasks (stage and severity)
    mcc_stage = metrics.matthews_corrcoef(labels_stage, pred_classes_stage)
    mcc_severity = metrics.matthews_corrcoef(labels_severity, pred_classes_severity)

    # Calculate precision, recall, and F1-score for both tasks (stage and severity)
    precision_stage, recall_stage, f1_stage, _ = metrics.precision_recall_fscore_support(
        labels_stage, pred_classes_stage, average='macro', zero_division=0)
    precision_severity, recall_severity, f1_severity, _ = metrics.precision_recall_fscore_support(
        labels_severity, pred_classes_severity, average='macro', zero_division=0)

    # Print the metrics
    print("Stage Metrics:")
    print("Accuracy: {:.4f}, Kappa: {:.4f}, MCC: {:.4f}".format(acc_stage, kappa_stage, mcc_stage))
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_stage, recall_stage, f1_stage))
    print("Confusion Matrix:")
    print(cm_stage)

    print("Severity Metrics:")
    print("Accuracy: {:.4f}, Kappa: {:.4f}, MCC: {:.4f}".format(acc_severity, kappa_severity, mcc_severity))
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision_severity, recall_severity, f1_severity))
    print("Confusion Matrix:")
    print(cm_severity)

    return (
        accu_loss_stage.item() / (step + 1),
        accu_num_stage.item() / sample_num,
        accu_loss_severity.item() / (step + 1),
        accu_num_severity.item() / sample_num
    )


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
