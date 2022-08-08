import glob
import pandas as pd
from tqdm import tqdm
import torch

def evaluate(model, valid_loader, device):
    model.to(device).eval()
    train_count, correct_preds = 0, 0
    for i, (images, labels) in enumerate(tqdm(valid_loader)):
        images, labels = images.to(device), labels.to(device)
        # _, labels = torch.max(labels.data, 1)
        outputs = model(images)
        _, preds = torch.max(outputs.data, 1)
        train_count += labels.shape[0]
        correct_preds += (preds == labels).sum().item()
    valid_acc = (correct_preds / train_count)
    return valid_acc

# def evaluate(model, valid_loader, device):
#     model.to(device).eval()
#     valid_count, correct_preds = 0, 0
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(tqdm(valid_loader)):
#             images, labels = images.to(device), labels.to(device)
#             _, labels = torch.max(labels.data, 1)
#             outputs = model(images)
#             _, preds = torch.max(outputs.data, 1)
#             valid_count += labels.shape[0]
#             correct_preds += (preds == labels).sum().item()
#         valid_acc = (correct_preds / valid_count)
#         return valid_acc

def train(model, train_loader, optimizer, loss_fn, device):
    model.to(device).train()
    train_count, correct_preds = 0, 0
    train_loss = 0.
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)
        # _, labels = torch.max(labels.data, 1)
        outputs = model(images)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs.data, 1)
        train_count += labels.shape[0]
        correct_preds += (preds == labels).sum().item()
        train_loss += loss.item() * labels.shape[0]
    train_acc = (correct_preds / train_count)
    train_loss = (train_loss / train_count)
    return train_acc, train_loss

def save_model(model, optimizer, epoch, loss, path):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path
    )
    print(f"\n-----------\nModel Saved At {path}\n-----------\n")


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return checkpoint, epoch, loss


def create_df(root_dir):
    all_imgs = glob.glob(root_dir + '*/*/*.*')
    paths, classes = [], []
    for img in all_imgs:
        img_path = img.replace('\\', '/')
        img_class = img_path.split('/')[-2]
        paths.append(img_path)
        classes.append(img_class)
    df = pd.DataFrame({
        'Path': paths,
        'Class': classes
    })
    df = pd.concat([df, pd.get_dummies(df.iloc[:, 1])], axis=1)
    return df


def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)