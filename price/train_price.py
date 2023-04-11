import random
import pickle
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_price import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler




class Dataset(Dataset):

    def __init__(self, args, split):

        if split == "train":
            self.train = True
        else:
            self.train = False

        self.args = args

        self.root_data = os.path.join(args.dataset_root, f'{split}.csv')
        self.df = pd.read_csv(self.root_data)
        self.df = pd.DataFrame(self.df)
        self.df.columns = self.df.columns.str.lower()
        # self.df['negative sentiment'] = np.random.randint(2, size=len(self.df))
        # self.df['positive sentiment'] = np.random.randint(2, size=len(self.df))
        self.df.dropna(inplace=True)

        mms_x = MinMaxScaler(feature_range=(0, 1))
        mms_y = MinMaxScaler(feature_range=(0, 1))

        # xvar = ['return_lag_2', 'ma3', 'std3', 'log_vol', 'negative sentiment', 'positive sentiment', 'usdx']
        # xvar = ['return_lag_2', 'ma3', 'negative sentiment', 'positive sentiment']
        xvar = ['open', 'negative sentiment', 'positive sentiment', 'std3', 'log_vol', 'usdx']
        # xvar = ['open']
        yvar = ['open']
        # mms = StandardScaler()
        self.x_matrix = np.matrix(mms_x.fit_transform(self.df[xvar]))
        # x_matrix = np.matrix(self.df[xvar])
        self.y_matrix = np.matrix(mms_y.fit_transform(self.df[yvar]))
        # y_matrix = np.matrix(self.df[yvar])
        scalerfile_x = '/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/mms_x.sav'
        pickle.dump(mms_x, open(scalerfile_x, 'wb'))
        scalerfile_y = '/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/mms_y.sav'
        pickle.dump(mms_y, open(scalerfile_y, 'wb'))


    def __getitem__(self, idx): # idx will iterate in range(__len__) so if the length of data change in __getitem__ it will raise an error

        # x = torch.tensor(self.x_matrix)[idx + 1: idx + args.history + 1, :] # descending date
        # x = torch.flip(x, [0]) # ascending date
        # y = torch.tensor(self.y_matrix)[idx, :]
        # return x.float(), y.float() # x: [history, xvar] history: past nth days  same as [sen_len, wordvector]

        x = torch.tensor(self.x_matrix)[idx: idx + args.history, :] # ascending date
        y = torch.tensor(self.y_matrix)[idx + args.history, :]
        return x.float(), y.float()  # x: [history, xvar] history: past nth days  same as [sen_len, wordvector]

    def __len__(self):
        return (self.df.shape[0] - args.history) # use return at nth day, use factor from (n-11)th to (n-1)th day, in total len - 10 + 1 - 1



def train_model(model, args):

    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), weight_decay=args.weight_decay,
                                  lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** epoch)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=3)
    # criterion = nn.CrossEntropyLoss() #  nn.BCELoss(reduction='mean')
    criterion = nn.MSELoss() # nn.BCELoss(reduction='mean')

    train_dataset = Dataset(args, split='train')  # shuffle=False?????????
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False) # batch_size=train_dataset.__len__()

    val_dataset = Dataset(args, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_dict = dict(iter=[], loss=[])
    val_dict = dict(iter=[], loss=[], acc=[])

    best_val_loss = np.inf
    start_epoch = 0

    # load optimizer and epoch
    if args.load_pretrain:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'model_epoch_' + str(args.which_checkpoint) + '.pth'))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, args.num_epochs):

        model.train()

        total_loss = 0.

        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (x, y) in progress_bar:
            # y = y.unsqueeze(1)  # from (30) to (30, 1)
            # x_batch.to(device)
            # y_batch.to(device)
            # x = x.unsqueeze(1) # CNN: x [batch size, 1, 7]  1 is the kernel number
            y_pred = model(x)

            loss = criterion(y_pred.float(), y.float())  # float prevent tensor type difference
            total_loss += loss.data

            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            optimizer.step()

            avg_loss = total_loss / (i + 1)
            if (i + 1) % 5 == 0:  # every 5 batch
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch + 1, args.num_epochs,
                                                                                      i + 1, len(train_loader),
                                                                                      loss.data, avg_loss))
        if args.scheduler == True:
            scheduler.step()

        model.eval()

        validation_loss = 0.0
        accuracy = 0.0

        progress_bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (x, y) in progress_bar:

            # x_batch.to(device)
            # y_batch.to(device)
            # x = x.unsqueeze(1) # CNN: from (batch_size, history, xvar) to (batch_size, 1, history, xvar)
            y_pred = model(x)

            loss = criterion(y_pred.float(), y.float())  # float prevent tensor type difference
            validation_loss += loss.data

            y_pred_TP = (abs(y_pred - y) / y) > (y * 0.1) # use 10 percent as the threshold for accuracy
            y_pred_TP = y_pred_TP.detach().numpy()

            acc = accuracy_score(y_pred_TP, np.ones(y_pred_TP.shape))
            accuracy += acc

        validation_loss /= len(val_loader)
        print("validation loss:", validation_loss.detach().numpy())

        accuracy /= len(val_loader)
        print('accuracy:', accuracy)


        # save models
        if best_val_loss > validation_loss:
            best_val_loss = validation_loss

            checkpoint = {    # save the best parameters
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'model_best.pth'))

        checkpoint = {      # save the current parameters os that we can continue with current training
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'model_epoch_' + str(epoch+1) + '.pth'))

        torch.cuda.empty_cache()

        train_dict['iter'].append(epoch + 1)
        train_dict['loss'].append(np.array(avg_loss.cpu()))

        val_dict['iter'].append(epoch + 1)
        val_dict['loss'].append(np.array(validation_loss.cpu()))
        val_dict['acc'].append(accuracy)


    if not os.path.exists(args.loss_curve_dir):
        os.makedirs(args.loss_curve_dir)

    folder = os.path.join(args.loss_curve_dir, 'ep_{}_bat_{}_lr_{}_wd_{}_Sch_{}') \
        .format(args.num_epochs, args.batch_size, args.learning_rate, args.weight_decay, args.scheduler)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # np.save(os.path.join(args.loss_curve_dir, 'lr_{}_wd_{}_Sch_{}.npy') \
    #         .format(args.learning_rate, args.weight_decay, args.scheduler), train_dict)
    # np.save(os.path.join(args.loss_curve_dir, 'lr_{}_wd_{}_Sch_{}.npy') \
    #         .format(args.learning_rate, args.weight_decay, args.scheduler), val_dict)

    np.save(os.path.join(folder, 'train.npy'), train_dict)
    np.save(os.path.join(folder, 'val.npy'), val_dict)

    plt.plot(train_dict['iter'], train_dict['loss'], '.-', label='train')
    plt.plot(val_dict['iter'], val_dict['loss'], '.-', label='val')
    plt.legend()
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    # plt.savefig(os.path.join(args.loss_curve_dir, 'lr_{}_wd_{}_Sch_{}.png') \
    #             .format(args.learning_rate, args.weight_decay, args.scheduler))
    plt.savefig(os.path.join(folder, 'loss.png'))
    plt.show()

    plt.plot(val_dict['iter'], val_dict['acc'], '.-')
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    # plt.savefig(os.path.join(folder, 'lr_{}_wd_{}_Sch_{}.png') \
    #             .format(args.learning_rate, args.weight_decay, args.scheduler))
    plt.savefig(os.path.join(folder, 'accuracy.png'))
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser() # use argparse to store simple global variable, checkpoint for complex variable

    # model settings
    parser.add_argument('--EMBEDDING_DIM', default=6, type=int)  # len(xvar) number of factor
    parser.add_argument('--HIDDEN_DIM', default=64, type=int)  # 32
    parser.add_argument('--OUTPUT_DIM', default=1, type=int)
    parser.add_argument('--NUM_LAYERS', default=1, type=int)
    parser.add_argument('--history', default=30, type=int)
    # hyperparameters
    parser.add_argument('--load_pretrain', default=False, type=bool)
    parser.add_argument('--which_checkpoint', default=1, type=int, help='load the check point saved from the end of which epoch')
    parser.add_argument('--num_epochs', default=15, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate') #0.01
    parser.add_argument('--weight_decay', default=5e-8, type=float, help='weight decay for AdamW')
    parser.add_argument('--scheduler', default=False, type=bool, help='scheduler for learning rate')
    # files
    parser.add_argument('--dataset_root', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/dataset', type=str,
                        help='dataset root')
    parser.add_argument('--checkpoint_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/checkpoints', type=str,
                        help='output directory')
    parser.add_argument('--loss_curve_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/loss_curve', type=str)

    args = parser.parse_args()

    #################################################################################################
    # train = pd.read_excel('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/dataset/train.xlsx')
    # train = pd.DataFrame(train)
    #
    # train.columns = train.columns.str.lower()
    #
    # print(train.head(5))
    # print(train.shape)
    #
    # train['negative sentiment'] = np.random.randint(2, size=len(train))
    # train['positive sentiment'] = np.random.randint(2, size=len(train))
    #
    # print(set(train['negative sentiment']))
    # print(train.columns)
    # train.dropna(inplace=True)
    # # train = pd.DataFrame(train, columns=['return', 'return_lag_2', 'ma3', 'std3', 'log_vol', 'negative sentiment', 'positive sentiment', 'usdx'])
    # xvar = ['return_lag_2', 'ma3', 'std3', 'log_vol', 'negative sentiment', 'positive sentiment', 'usdx']
    # yvar = ['return']
    # x = np.matrix(train[xvar])
    # y = np.matrix(train[yvar])
    # print(x)
    # print(y)
    # model = CNN()
    model = RNN(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                num_layers=args.NUM_LAYERS, output_dim=args.OUTPUT_DIM, history=args.history)
    # model = model.to(device)


    ##################################################################################################

    # load model
    if args.load_pretrain:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'model_epoch_' + str(args.which_checkpoint) + '.pth'))
        model.load_state_dict(checkpoint['model'])


    train_model(model, args)

