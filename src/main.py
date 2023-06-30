from data import train_loader, valid_loader, test_loader
from model import PointNet, pointnet_loss
import torch
import torch.optim as optim
import os
import argparse

def train(args):   
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the model
    model = PointNet(args.num_classes)
    model.to(device)

    # Get the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Get the loss function
    criterion = pointnet_loss

    # Create the directory to save the trained models
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Train the model
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.permute(0, 2, 1)
            data, target = data.to(device), target.to(device)

            # Get the output
            output, m3, m64 = model(data)

            # Compute the loss
            loss = criterion(output, target, m3, m64, args.alpha)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            # Print and log the loss
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                
                with open(args.log_dir + '/log.txt', 'a') as f:
                    f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                
        # Save the model
        torch.save(model.state_dict(), args.model_dir + '/model_' + str(epoch) + '.pth')

        # Evaluate the model
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                data = data.permute(0, 2, 1)

                # Get the output
                output, m3, m64 = model(data)

                # Compute the loss
                test_loss += criterion(output, target, m3, m64, args.alpha).item()

                # Get the predicted class
                pred = output.argmax(dim=1, keepdim=True)

                # Update the number of correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute the average loss
        test_loss /= len(valid_loader.dataset)

        # Print the loss
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(valid_loader.dataset),
                100. * correct / len(valid_loader.dataset)))

        # Log the loss and accuracy
        with open(args.log_dir + '/log.txt', 'a') as f:
            f.write(str(epoch) + '\t' + str(test_loss) + '\t' + str(100. * correct / len(valid_loader.dataset)) + '\n')
  

def test_saved_model(args, model_path):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet(args.num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.permute(0, 2, 1)

            # Get the output
            output, m3, m64 = model(data)

            # Compute the loss
            test_loss += pointnet_loss(output, target, m3, m64, args.alpha).item()

            # Get the predicted class
            pred = output.argmax(dim=1, keepdim=True)

            # Update the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Compute the average loss
    test_loss /= len(test_loader.dataset)

    # Print the loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    # Log the loss and accuracy
    with open(args.log_dir + '/log.txt', 'a') as f:
        f.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_dir', type=str, default='../models')
    argparser.add_argument('--log_dir', type=str, default='../logs')
    argparser.add_argument('--num_classes', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--num_epochs', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--weight_decay', type=float, default=0.0001)
    argparser.add_argument('--alpha', type=float, default=0.0001)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--beta1', type=float, default=0.9)
    argparser.add_argument('--beta2', type=float, default=0.999)
    argparser.add_argument('--log_interval', type=int, default=10)
    argparser.add_argument('--model_path', type=str, default='../models/model_99.pth')
    argparser.add_argument('--mode', type=str, default='train')
    args = argparser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_saved_model(args, args.model_path)
    else:
        print('Invalid mode')
        raise NotImplementedError