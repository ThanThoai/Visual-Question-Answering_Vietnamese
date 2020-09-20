from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy


plt.ion()  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#prepare image for trainging and testing
def prepare_data(train_path, val_path, input_net = (3, 224, 224)):
    #Most of pretrained use input_shape 3*224*224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


    image_datasets = {"train": datasets.ImageFolder('images_new/train', transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Scale(input_net),
            transforms.ToTensor(),
            normalize
        ])), "test": datasets.ImageFolder('images_new/test', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Scale(input_net),
            transforms.ToTensor(),
            normalize
        ]))
    }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4, pin_memory=True)
              for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    train_class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, train_class_names

#training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#class model pretrained 
class Resnet18_extract(nn.Module):
    def __init__(self, output_shape = 2048):
        super(Resnet18_extract, self).__init()
        self.out_pretrained = nn.Sequential(*list(models.resnet18(pretrained=True)
          .features.children())[:-1])       
        # Do not retrain pretrained model
        for param in self.out_pretrained.parameters():
            param.requires_grad = False

        #output shape is 512*1*1
        self.flatten = nn.Flatten(self.out_pretrained)
        self.features = nn.Linear(self.flatten, output_shape)

    def forward(self, x):
        x = self.out_pretrained(x)
        x = self.flatten
        x = self.features
        return x

class Resnet34_extract(nn.Module):
    def __init__(self, output_shape = 2048):
        super(Resnet34_extract, self).__init()
        self.out_pretrained = nn.Sequential(*list(models.resnet34(pretrained=True)
          .features.children())[:-1])
        for param in self.out_pretrained.parameters():
            param.requires_grad = False

        #output shape is 512*1*1
        self.flatten = nn.Flatten(self.out_pretrained)
        self.features = nn.Linear(self.flatten, output_shape)

    def forward(self, x):
        x = self.out_pretrained(x)
        x = self.flatten
        x = self.features
        return x

class VGG16_extract(nn.Module):
    def __init__(self, output_shape = 2048):
        super(VGG16_extract, self).__init()
        self.out_pretrained = nn.Sequential(*list(models.vgg16(pretrained=True)
          .features.children())[:-1])
        for param in self.out_pretrained.parameters():
            param.requires_grad = False

        #output shape is 512*7*7
        self.flatten = nn.Flatten(self.out_pretrained)
        self.features = nn.Linear(self.flatten, output_shape)


    def forward(self, x):
        x = self.out_pretrained(x)
        x = self.flatten
        x = self.features
        return x

class VGG19_extract(nn.Module):
    def __init__(self, output_shape = 2048):
        super(VGG19_extract, self).__init()
        self.out_pretrained = nn.Sequential(*list(models.vgg19(pretrained=True)
          .features.children())[:-1])
        for param in self.out_pretrained.parameters():
            param.requires_grad = False
            
        #output shape is 512*7*7
        self.flatten = nn.Flatten(self.out_pretrained)
        self.features = nn.Linear(self.flatten, output_shape)

  
    def forward(self, x):
        x = self.out_pretrained(x)
        x = self.flatten
        x = self.features
        return x

# class extract feature image using 
class Extract():
    def __init__(self, model_name, output_shape = 2048):
        self.model_name = model_name
        self.output_shape = self.output_shape
        if (model_name == "vgg16"):
            self.model = VGG16_extract(output_shape)
        elif (model_name == "vgg19"):
            self.model = VGG16_extract(output_shape)
        elif (model_name == "resnet18"):
            self.model = Resnet18_extract(output_shape)
        elif (model_name == "resnet34"):
            self.model = Resnet18_extract(output_shape)
        else:
            print("No pretrained was found")

    def extract(self, image):
        image = self.model(image)

    def train(self):
        pass
