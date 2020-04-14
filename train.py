import torch, argparse
import torchvision.models as models
from torchvision import datasets, transforms
from collections import OrderedDict
from torch import nn
from torch import optim

def preprocess_data(data_dir):
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader

def make_model(arch):
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        layer_size = [25088, 500, 200 ,102]

    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        layer_size = [9216, 500, 200 ,102]
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(layer_size[0], layer_size[1])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(layer_size[1], layer_size[2])),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(layer_size[2], layer_size[3])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return model, layer_size

def train(model, optimizer, epochs, trainloader, validloader, gpu, criterion):

    for epoch in range(1, epochs + 1):
        train_loss = 0
        for images, labels in trainloader:

            if args.gpu == True:
                images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if epoch % 2 == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in validloader:

                    if args.gpu == True:
                        images, labels = images.cuda(), labels.cuda()
                    
                    logps = model.forward(images)
                    loss = criterion(logps, labels)

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    valid_loss += loss.item()
            print(f"Epoch {epoch}/{epochs}.. "
                  f"Train loss: {train_loss/len(trainloader):.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            model.train()

def test(model, testloader, gpu):
    
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        
        if gpu == True:
            images, labels = images.cuda(), labels.cuda()
        logps = model.forward(images)
        loss = criterion(logps, labels)

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        test_loss += loss.item()
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    
    
def save_model(model, layer_size, arch):
    
    checkpoint = {'number_of_layer': 3,
              'layer_size': layer_size,
              'pretrain_model': arch,
              'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description = 'Image classification')
    parser.add_argument('--data_dir', dest = 'data_dir', help = 'path of data directory', type = str)
    parser.add_argument('--save_dir', dest = 'save_dir',  help = 'path for saving model', default = '', type = str)
    parser.add_argument('--gpu', dest = 'gpu',  help = 'want to use gpu or not', default = True, type = bool)
    parser.add_argument('--arch', dest = 'arch',  help = 'vgg13 or alexnet', default ='alexnet', type = str)
    parser.add_argument('--learning_rate', dest = 'lr',  help = 'learning rate', default = 0.001, type = float)
    parser.add_argument('--epochs', dest = 'epochs',  help = 'number of epoch', default = 10, type = int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    trainloader, validloader, testloader = preprocess_data(args.data_dir)
    
    model, layer_size = make_model(args.arch)  

    if args.gpu == True:
        model = model.cuda()
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
    
    train(model, optimizer, args.epochs, trainloader, validloader, args.gpu, criterion)
    
    test(model, testloader, args.gpu)
        
    save_model(model, layer_size, args.arch)
  