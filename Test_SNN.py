import tonic
import matplotlib.pyplot as plt
import tonic.transforms as transforms

import torch
import torchvision

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from SResNet import spiking_resnet
from accuracy import AverageMeter, save_checkpoint, record_info, accuracy

# Load DVS Gesture dataset
dataset = tonic.datasets.DVSGesture(save_to='./data', train=True)
events, label = dataset[0]

# Dataset size
print("Dataset contains", len(dataset), "samples.")

# Number of events in the first sample
print("There are", len(events), "events in the first sample.")

# (x-pos, y-pos, polarity, timestamp)
print("A single event:", events[0])

size = tonic.datasets.DVSGesture.sensor_size

# Transformations
denoise_transform = transforms.Denoise(filter_time=10000)
frame_transform = transforms.ToFrame(sensor_size=size, n_time_bins=25)
all_transform = transforms.Compose([denoise_transform, frame_transform])

# View the transformed data
tf_frames = all_transform(events)
print("Shape of frames object:", tf_frames.shape)
plt.axis('off')
plt.imshow(tf_frames[0][0])

# Prepare datasets with transformations
train_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=True)
test_set = tonic.datasets.DVSGesture(save_to='./data', transform=all_transform, train=False)
print(len(train_set))
print(len(test_set))

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True, drop_last=True)

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define the SpikingResNet model
image_size = (128, 128)  # Adjusted for pooling layers
model = spiking_resnet(image_size=image_size, batch_size=64, nb_classes=11, channel=2).to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Training loop parameters
num_epochs = 50
best_acc = 0

# Training and validation loops
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training phase
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
    for i, (data, targets) in enumerate(progress):
        data = data.to(device).float()  # Ensure data is in float
        targets = targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        prec1, _ = accuracy(outputs, targets, topk=(1, 5))
        train_loss.update(loss.item(), data.size(0))
        train_acc.update(prec1.item(), data.size(0))

        # Update progress bar
        progress.set_postfix({
            "Loss": train_loss.avg,
            "Accuracy": train_acc.avg,
        })

    print(f"Train Loss: {train_loss.avg:.4f}, Train Accuracy: {train_acc.avg:.2f}%")

    # Validation phase
    model.eval()
    val_acc = AverageMeter()
    progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
    with torch.no_grad():
        for i, (data, targets) in enumerate(progress):
            data = data.to(device).float()  # Ensure data is in float
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)

            # Update metrics
            prec1, _ = accuracy(outputs, targets, topk=(1, 5))
            val_acc.update(prec1.item(), data.size(0))

            # Update progress bar
            progress.set_postfix({
                "Validation Accuracy": val_acc.avg,
            })

    print(f"Validation Accuracy: {val_acc.avg:.2f}%")

    # Checkpoint saving
    is_best = val_acc.avg > best_acc
    best_acc = max(val_acc.avg, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, 'checkpoint.pth.tar', 'model_best.pth.tar')

    scheduler.step()

print("Training complete.")