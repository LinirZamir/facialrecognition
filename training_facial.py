import deeplake
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

# Load the datasets
train_ds = deeplake.load('hub://activeloop/fer2013-train')
public_test_ds = deeplake.load('hub://activeloop/fer2013-public-test')
private_test_ds = deeplake.load('hub://activeloop/fer2013-private-test')

# Convert to PyTorch dataloaders
train_dataloader = train_ds.pytorch(num_workers=0, batch_size=32, shuffle=True)
public_test_dataloader = public_test_ds.pytorch(num_workers=0, batch_size=32, shuffle=False)
private_test_dataloader = private_test_ds.pytorch(num_workers=0, batch_size=32, shuffle=False)


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*24*24, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 emotion classes in FER2013

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*24*24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(10):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['images'], data['labels']

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'emotion_model.pth')


