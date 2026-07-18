import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# ─────────────────────────────────────────────
# 1. RESIDUAL BLOCK
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super().__init__()
# Main path
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
#Projection shortcut to add the tensors of diff shapes 
        self.shortcut=nn.Sequential()
        if stride !=1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )                   

    def forward(self,x):
         identity = self.shortcut(x)
         out=self.relu(self.bn1(self.conv1(x)))
         out = self.bn2(self.conv2(out)) 
         out += identity 
         out = self.relu(out) 
         return out
# ─────────────────────────────────────────────
# 2. FULL RESNET
# ─────────────────────────────────────────────

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
       super().__init__()
       self.stem = nn.Sequential(
               nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True)
       )
       # Residual stages
       self.stage1 = self.make_stage(in_ch=64,  out_ch=64,  num_blocks=2, stride=1)
       self.stage2 = self.make_stage(in_ch=64,  out_ch=128, num_blocks=2, stride=2)
       self.stage3 = self.make_stage(in_ch=128, out_ch=256, num_blocks=2, stride=2)
       # Global Average Pooling + classifier
       self.gap = nn.AdaptiveAvgPool2d(1)
       self.fc  = nn.Linear(256, num_classes)
       # Weight initialization
       self.init_weights()

    def make_stage(self, in_ch, out_ch, num_blocks, stride):
         layers = [ResidualBlock(in_ch, out_ch, stride)]
         for x in range(num_blocks - 1):
             layers.append(ResidualBlock(out_ch, out_ch, stride=1))
         return nn.Sequential(*layers)
    
    def init_weights(self):
         for w in self.modules():
            if isinstance(w,nn.Conv2d):
                #  whats is fan in and fan out?
                 nn.init.kaiming_normal_(w.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(w,nn.BatchNorm2d):
                 nn.init.constant_(w.weight,1)
                 nn.init.constant_(w.bias,0)

    def forward(self,x):
         x=self.stem(x)
         x=self.stage1(x)
         x=self.stage2(x)
         x=self.stage3(x)
         x=self.gap(x)
         x=torch.flatten(x,1)
         return self.fc(x)
# sanity check 

def shape_check():
     model=CustomResNet()
     dummy=torch.rand(2,3,32,32)
     out=model(dummy)
    #  throws error is the shape is wrong 
     assert out.shape == (2,10) , f"expected string (2,10)  , got {out.shape}"
    #  if the shape is correct then 
     print(f"shape check passed :{out.shape}")
# printing the total params

     total_params=0
     for p in model.parameters():
        if p.requires_grad:
            total_params+=p.numel()
     print(f"total trainable params are {total_params}")

# ─────────────────────────────────────────────
# 4. DATA PIPELINE
# ─────────────────────────────────────────────

def get_loaders(batch_size=128):
     mean = (0.4914, 0.4822, 0.4465)
     std_deviation  = (0.2470, 0.2435, 0.2616)
    #   these are used for normalisation so that the values are centred around 0-1 normalised = (x - mean) / std to Make scales more consistent Help training converge faster(got a better learning in revision-2)
     train_transform = transforms.Compose([
         transforms.RandomCrop(32, padding=4),       # mild augmentation for 32x32
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std_deviation)
     ])
     test_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean, std_deviation)
     ])
 
     trainset = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_transform)
     testset  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
 
     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
     testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
 
     return trainloader, testloader

# ─────────────────────────────────────────────
# 5. TRAINING + VALIDATION LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model,loader,criterion,optimiser,device):
     model.train()
     total=0
     total_loss , correct , loss = 0,0,0
     for inputs , labels in loader:
          inputs = inputs.to(device) 
          labels= labels.to(device)
          optimiser.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs,labels)
          loss.backward()
          optimiser.step()
          total_loss+= loss.item()*inputs.size(0)
          correct    += outputs.argmax(1).eq(labels).sum().item()
          total      += inputs.size(0)
     return total_loss / total, correct / total
# evaluation/validation
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct    += outputs.argmax(1).eq(labels).sum().item()
            total      += inputs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# 6. FULL TRAINING RUN
# ─────────────────────────────────────────────

def train(num_epochs=10):
 # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
# Data
    trainloader, testloader = get_loaders()
# Model, loss, optimizer, scheduler
    model     = CustomResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)

        print(f"Epoch {epoch+1:>3}/{num_epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

        # Save best model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "best_resnet.pth")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model, trainloader, testloader, device, train_losses, val_losses, train_accs, val_accs


# ─────────────────────────────────────────────
# 7. EVALUATION & PLOTS
# ─────────────────────────────────────────────
     
CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def plot_curves(train_losses, val_losses, train_accs, val_accs):

    # Loss Curves
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png', dpi=150)
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=(6,4))
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curves.png', dpi=150)
    plt.show()

    print("Saved: loss_curves.png")
    print("Saved: accuracy_curves.png")
    



def full_evaluation(model, testloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:

            inputs = inputs.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\nClassification Report:")
    print(classification_report(all_labels,all_preds,target_names=CLASSES))
# confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # saving the confusion matrix 
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: sanity check shapes before anything else
    shape_check()

    # Step 2: full training run
    model, trainloader, testloader, device, \
    train_losses, val_losses, train_accs, val_accs = train(num_epochs=50)

    # Step 3: plots
    plot_curves(train_losses, val_losses, train_accs, val_accs)

    # Step 4: final evaluation
    full_evaluation(model, testloader, device)