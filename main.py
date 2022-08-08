from dataloaders import RiceDataset
from utils import create_df, count_model_params, train, evaluate, save_model
from models import RiceLeafNet
import torchvision
import torch

root_dir = './Dataset/'
train_dir = root_dir + 'train'
valid_dir = root_dir + 'validation'
test_dir = root_dir + 'test'

df = create_df(root_dir)
transform = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root=train_dir, transform=transform),
    batch_size=32, shuffle=True, pin_memory=True, drop_last=True, num_workers=2
)

valid_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root=valid_dir, transform=transform),
    batch_size=32, shuffle=True, pin_memory=True, drop_last=False, num_workers=2
)

model = RiceLeafNet()

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = './checkpoints/model.pth'
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    min_loss = 10000000
    for i in range(100):
        print(f"Epoch: {i} / {100}")
        train_acc, train_loss = train(model, train_loader, optimizer, loss_fn, DEVICE)
        print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
        if train_loss < min_loss:
            valid_acc = evaluate(model, valid_loader, DEVICE)
            print(f"Valid Acc: {valid_acc*100:.2f}")
            save_model(model, optimizer, i, train_loss, PATH)
