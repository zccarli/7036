
import argparse
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import *
from model import *



# train_data, valid_data = train_test_split(train, test_size=0.2, random_state=42)


# def clean_sentiment(text):
#     if text == 'positive' or text == 'pos':
#         return 1
#     elif text == 'negative' or text == 'neg':
#         return -1
#     else:
#         return 0
#
# def soft(text):
#     if text == 1:
#         return [1, 0, 0]
#     elif text == -1:
#         return [0, 0, 1]
#     else:
#         return [0, 1, 0]
#
#
#
# def pre_process(text):
#     text = str(text)
#     # Remove links
#     text = re.sub('http://\S+|https://\S+', '', text)
#     text = re.sub('http[s]?://\S+', '', text)
#     text = re.sub(r"http\S+", "", text)
#
#     # Convert HTML references
#     text = re.sub('&amp', 'and', text)
#     text = re.sub('&lt', '<', text)
#     text = re.sub('&gt', '>', text)
#     text = re.sub('\xa0', ' ', text)
#
#     # Remove new line characters
#     text = re.sub('[\r\n]+', ' ', text)
#
#     # Remove mentions
#     text = re.sub(r'@\w+', '', text)
#
#     # Remove hashtags
#     text = re.sub(r'#\w+', '', text)
#
#     # Remove multiple space characters
#     text = re.sub('\s+', ' ', text)
#
#     pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
#     text = pattern.sub('', text)
#
#     text = text.lower()
#
#     return text
#
#
# def expand_contractions(text):
#     try:
#         return contractions.fix(text)
#     except:
#         return text





# wordsList = np.load('/Users/Rui/Desktop/MFFINTECH/NN/LSTM/training_data/wordsList.npy')
# print('Loaded the word list!')
# wordsList = wordsList.tolist()  # Originally loaded as numpy array
# wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
# wordsList = dict(zip(wordsList, range(0, len(wordsList))))
# wordVectors = np.load('/Users/Rui/Desktop/MFFINTECH/NN/LSTM/training_data/wordVectors.npy')
# print('Loaded the word vectors!')



# tokenizer = get_tokenizer('basic_english')  # 分词器 类似split
# def text_transform(sentence, maxSeqLength=maxSeqLength):
#     #     sentence_vector = np.zeros(maxSeqLength)
#     sentence_vector = []
#     for token in tokenizer(sentence):
#         try:
#             sentence_vector.append(wordsList[token])
#         except:  # exclude non english sentence and not recognized word （gibberish)
#             sentence_vector.append(0)
#     if len(sentence_vector) > maxSeqLength:
#         sentence_vector = sentence_vector[:maxSeqLength]
#     elif len(sentence_vector) < maxSeqLength:
#         sentence_vector.extend(np.zeros(maxSeqLength - len(sentence_vector), dtype='int64'))
#     return sentence_vector






class Dataset(Dataset):

    def __init__(self, args, split):

        if split == "train":
            self.train = True
        else:
            self.train = False

        self.args = args

        self.root_data = os.path.join(args.dataset_root, split, 'sentiment.csv')
        self.df = pd.read_csv(self.root_data)
        self.df = pd.DataFrame(self.df)
        self.df.columns = self.df.columns.str.lower()
        self.df['sentiment'] = self.df['sentiment'].apply(clean_sentiment)
        self.df['sentiment'] = self.df['sentiment'].apply(soft)
        self.df['sentence'] = self.df['sentence'].apply(pre_process)
        self.df['sentence'] = self.df['sentence'].apply(expand_contractions)
        self.df['sentence'] = self.df['sentence'].apply(lambda sentence: text_transform(sentence, maxSeqLength=args.maxSeqLength))
        self.df = self.df.dropna()

    def __getitem__(self, idx): # idx will iterate in range(__len__) so if the length of data changes in __getitem__ it will raise an error

        y = self.df['sentiment'].iloc[idx]
        x = self.df['sentence'].iloc[idx]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    def __len__(self):
        return self.df.shape[0]



# class RNN(nn.Module):
#     #     self train embedding
#     #     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
#     #         super().__init__()
#     #         self.embedding = nn.Embedding(input_dim, embedding_dim)
#     #         self.rnn = nn.RNN(embedding_dim, hidden_dim)
#     #         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     #   pretrained embedding
#     def __init__(self, wordVectors, embedding_dim, hidden_dim, output_dim, maxseqlength=maxSeqLength):
#         super().__init__()
#         self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(wordVectors).float())
#         self.rnn = nn.RNN(embedding_dim, hidden_dim)  # similar to one hidden layer
#         self.fc = nn.Linear(maxseqlength * hidden_dim, output_dim)
#         self.hidden_dim = hidden_dim
#         self.maxseqlength = maxseqlength
#
#
#     def forward(self, text): # text = [batch size, sent len]
#
#         embedded = self.embedding(text) # embedded = [batch size, sent len, emb dim]
#         output, hidden = self.rnn(embedded) # output = [batch size, sent len, hid dim] # hidden = [batch size, 1, hid dim]
#         output = output.view(output.size(0), self.maxseqlength * self.hidden_dim)
#         output = self.fc(output)
#         # output = torch.sigmoid(output)  # torch.nn.Sigmoid, torch.nn.functional.sigmoid()
#         # output = nn.functional.softmax(output, dim=1)
#         return output





def train_model(model, args):

    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), weight_decay=args.weight_decay,
                                  lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** epoch)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=3)
    criterion = nn.CrossEntropyLoss() #  nn.BCELoss(reduction='mean')

    train_dataset = Dataset(args, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = Dataset(args, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

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
        for i, (x, y) in progress_bar: # x is sentence, y is sentiment

            # y = y.unsqueeze(1)  # from (30) to (30, 1)
            # x_batch.to(device)
            # y_batch.to(device)

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

            # y = y.unsqueeze(1)  # from (30) to (30, 1)
            # x_batch.to(device)
            # y_batch.to(device)

            y_pred = model(x)

            loss = criterion(y_pred.float(), y.float())  # float prevent tensor type difference
            validation_loss += loss.data

            y_pred_max = torch.zeros(y_pred.size())
            y_pred_max[(torch.arange(y_pred.size(0)).unsqueeze(1), torch.topk(y_pred,1).indices)] = 1

            y_pred_max = y_pred_max.detach().numpy()
            y = y.detach().numpy()

            acc = accuracy_score(y_pred_max, y)
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
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'model_epoch_' + str(epoch+1) + '.pth')) # start at the 0th epoch, empirically start at the 1st

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

    # np.save(os.path.join(folder, 'lr_{}_wd_{}_Sch_{}.npy') \
    #         .format(args.learning_rate, args.weight_decay, args.scheduler), train_dict)
    # np.save(os.path.join(folder, 'lr_{}_wd_{}_Sch_{}.npy') \
    #         .format(args.learning_rate, args.weight_decay, args.scheduler), val_dict)

    np.save(os.path.join(folder, 'train.npy'), train_dict)
    np.save(os.path.join(folder, 'val.npy'), val_dict)

    plt.plot(train_dict['iter'], train_dict['loss'], '.-', label='train')
    plt.plot(val_dict['iter'], val_dict['loss'], '.-', label='val')
    plt.legend()
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    # plt.savefig(os.path.join(folder, 'lr_{}_wd_{}_Sch_{}.png') \
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
    parser.add_argument('--EMBEDDING_DIM', default=50, type=int) #  wordVectors.shape[1] = 50
    parser.add_argument('--HIDDEN_DIM', default=128, type=int)
    parser.add_argument('--OUTPUT_DIM', default=3, type=int) # -1, 0, 1
    parser.add_argument('--NUM_LAYERS', default=2, type=int)
    parser.add_argument('--maxSeqLength', type=int)
    # hyperparameters
    parser.add_argument('--load_pretrain', default=False, type=bool)
    parser.add_argument('--which_checkpoint', default=1, type=int, help='load the check point saved from the end of which epoch')
    parser.add_argument('--num_epochs', default=70, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate') # 5e-5
    parser.add_argument('--weight_decay', default=5e-8, type=float, help='weight decay for AdamW')
    parser.add_argument('--scheduler', default=False, type=bool, help='scheduler for learning rate')
    # files path
    parser.add_argument('--dataset_root', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/dataset', type=str,
                        help='dataset root')
    parser.add_argument('--checkpoint_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/checkpoints', type=str,
                        help='output directory')
    parser.add_argument('--lexicon_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project', type=str)
    parser.add_argument('--loss_curve_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/loss_curve', type=str)

    args = parser.parse_args()

    #################################################################################################
    train = pd.read_csv('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/dataset/train/sentiment.csv')
    train = pd.DataFrame(train)

    train.columns = train.columns.str.lower()
    train['sentiment'] = train['sentiment'].apply(clean_sentiment)
    train['sentiment'] = train['sentiment'].apply(soft)
    train['sentence'] = train['sentence'].apply(pre_process)
    train['sentence'] = train['sentence'].apply(expand_contractions)

    print(train.head(5))
    print(train.shape)

    numWords = []
    for sentence in train['sentence']:
        numWords.append(len(sentence))
    # check for sentence length distribution
    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.show()
    print('mean:', np.mean(numWords))
    print('median:', np.median(numWords))
    print('mode:', stats.mode(numWords))

    maxSeqLength = int(np.median(numWords))
    args.maxSeqLength = maxSeqLength

    train['sentence'] = train['sentence'].apply(lambda sentence: text_transform(sentence, maxSeqLength=args.maxSeqLength))
    train = train.dropna()

    print(train.head(5))
    print(train.shape)

    model = RNN(wordVectors=wordVectors, embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                num_layers=args.NUM_LAYERS, output_dim=args.OUTPUT_DIM, maxseqlength=args.maxSeqLength)
    # model = LSTM(wordVectors=wordVectors, embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
    #             num_layers=args.NUM_LAYERS, output_dim=args.OUTPUT_DIM, maxseqlength=args.maxSeqLength)
    # model = model.to(device)


    # # 查看data shape
    # train_dataset = Dataset(args, split='train')
    # dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # x, y = next(iter(dl))
    # print(x.size)
    # embedding = nn.Embedding.from_pretrained(torch.from_numpy(wordVectors).float())
    # print(x.shape)
    # print(embedding(x).shape)
    # y = y
    # print(y.shape)

    ##################################################################################################

    # load model
    if args.load_pretrain:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'model_epoch_' + str(args.which_checkpoint) + '.pth'))
        model.load_state_dict(checkpoint['model'])

    train_model(model, args)