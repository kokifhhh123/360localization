def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)



def show_images (imgs, num_rows, num_cols, titles=None, scale=1.5);			
		
    figsize= (num_cols* scale, num_rows* scale)			
    _,axes = d21.plt.subplots (num_rows, num_cols, figsize-figsize)			
    axes = axes. flatten()			
    for i, (ax, img) in enumerate (zip (axes, imgs)):			
        if torch.is_tensor (img):			
#图像张量			
            ax. imshow(img.numpy())			
        else:			
# PIL			
            ax. imshow(img)			
        ax.axes.get_xaxis ().set_visible (False)			
        ax.axes.get_yaxis ().set_visible (False)			
        if titles:			
            ax.set_title(titles[i])			
    return axes		



import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # Flatten the tensor for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
num_classes = 10  # Number of classes in the classification task
model = SimpleCNN(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load and preprocess your dataset (e.g., using torchvision)

# Training loop
for epoch in range(10):  # Number of training epochs
    running_loss = 0.0
    for inputs, labels in train_loader:  # Iterate through batches of training data
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

print("Training finished!")

# Evaluate the model on test data and perform predictions



