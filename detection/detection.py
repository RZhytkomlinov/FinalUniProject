import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas
from bs4 import BeautifulSoup
import xmltodict


image_path = "dataset/images"
ann_path = "dataset/annotations"

for image_file in os.listdir(image_path):
    image_filepath = image_path + "/" + image_file
    ann_file = image_file.removesuffix(".png") + ".xml"
    ann_filepath = ann_path + "/" + ann_file
    with open(ann_filepath, "r") as f:
        data = f.read()
        data = xmltodict.parse(data)
    try:
        label = data["annotation"]["object"]["name"]
    except TypeError:
        label = data["annotation"]["object"][0]
    if label == "cat":
        os.makedirs("DogCatDataset/cat",exist_ok=True)
        os.rename(image_filepath,"DogCatDataset/cat" + "/" + image_file)
    elif label == "dog":
        os.makedirs("DogCatDataset/dog", exist_ok=True)
        os.rename(image_filepath,"DogCatDataset/dog" + "/" + image_file)

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0,1)
])

dataset = datasets.ImageFolder("DogCatDataset", transform=transform)
dataset_len = len(dataset)
train_len, test_len = dataset_len * 0.8, dataset_len * 0.2
train_set, test_set = torch.utils.data.random_split(dataset, [int(train_len), int(test_len)])

batch_size = 200
train_set = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
test_set = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=14, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=14, out_channels=16, kernel_size=4)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=4)

        self.fc1 = nn.Linear(in_features=20*4*4, out_features=250)
        self.fc2 = nn.Linear(in_features=250, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=50)
        self.fc4 = nn.Linear(in_features=50, out_features=10)
        self.fc5 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.reshape(-1, 20*4*4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x


net = Model().to(device)

crossentropy = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


def train_data(epochs):
    net.train()

    for epoch in range(epochs):
        total_correct = 0.0
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_set):
            inputs, labels = inputs.to(device), labels.to(device)
            output = net(inputs)
            output_idx = torch.argmax(output, dim=1)
            total_correct += (labels == output_idx).sum().item()
            optimizer.zero_grad()
            loss = crossentropy(output, labels)
            running_loss += loss.item() * inputs.size(0)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch} loss: {running_loss/train_len} Accurace:{(total_correct/train_len)*100}%")

    print("finished training")
    torch.save(net.state_dict(), 'catdogmodel.pt')


def test_data(checkpoint):
    with torch.no_grad():
        model = Model().to(device)
        model.load_state_dict(torch.load(checkpoint))
        net.eval()
        total_loss = 0.0
        total_correct = 0.0

        for inputs, labels in test_set:
            labels = labels.to(device)
            outputs = net(inputs.to(device))
            loss = crossentropy(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            output_idx = torch.argmax(outputs, dim=1)
            total_correct += sum(labels == output_idx)

        print(f"Loss: {total_loss/test_len} Accuracy: {(total_correct/test_len)*100}%")


def prediction(image):
    img = Image.open(image)
    img = transform(img).unsqueeze(dim=0).to(device)
    predict = net(img)
    print(torch.argmax(predict))


def load_model_from_checkpoint(checkpoint):
    with torch.no_grad():
        model = Model().to(device)
        model.load_state_dict(torch.load(checkpoint))
        model.eval()

        total_correct = 0.0

        for inputs, labels in test_set:
            labels = labels.to(device)
            outputs = model(inputs.to(device))
            output_idx = torch.argmax(outputs, dim=1)
            total_correct += sum(labels==output_idx)
        prediction("test.jpg")


#train_data(40)
# test_data("catdogmodel.pt")
load_model_from_checkpoint("catdogmodel.pt")
