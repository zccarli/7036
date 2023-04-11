import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from model_price import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#
scalerfile_x = '/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/mms_x.sav'
mms_x = pickle.load(open(scalerfile_x, 'rb'))
scalerfile_y = '/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/mms_y.sav'
mms_y = pickle.load(open(scalerfile_y, 'rb'))

def predict(args, model):  # may use different data
    test_path = args.test_path
    df = pd.read_csv(test_path)
    df = pd.DataFrame(df)
    df.columns = df.columns.str.lower()
    df.dropna(inplace=True) # drop first
    # xvar = ['return_lag_2', 'ma3', 'std3', 'log_vol', 'negative sentiment', 'positive sentiment', 'usdx']
    # xvar = ['open', 'positive sentiment', 'negative sentiment']
    xvar = ['open', 'negative sentiment', 'positive sentiment', 'std3', 'log_vol', 'usdx']
    # xvar = ['open']
    yvar = ['open']
    # x_matrix = np.matrix(df[xvar])
    # mms_x = MinMaxScaler(feature_range=(0, 1))
    # mms_x = StandardScaler()
    # mms_y = MinMaxScaler(feature_range=(0, 1))
    # mms_y = StandardScaler()
    # x_matrix = np.matrix(mms_x.fit_transform(df[xvar]))
    # mms_y.fit(df[yvar])

    # df = df.sort_values('date', ascending=True).reset_index(drop=True) # ascending date because it is predicting forward!!!
    # print(df.head(20))
    x_matrix = np.matrix(mms_x.transform(df[xvar]))

    print('PREDICTING...')

    pred_list = []
    for idx in range(args.history, df.shape[0] + 1): # total df.shape[0] - args.history + 1
        with torch.no_grad():
            x = torch.tensor(x_matrix)[idx - args.history: idx, :].float() # total args.history
            # x = torch.flip(x, [0]).float() # float() to prevent 'expected scalar type Double but found Float'
            x = torch.tensor(x).unsqueeze(0) # (history, xvar) to (1, history, xvar)
            # x = torch.tensor(x).unsqueeze(0) # for CNN: (1, history, xvar) to (1, 1, history, xvar)
            y_pred = model(x) # 2d tensor
            y_pred = mms_y.inverse_transform(y_pred) # inverse transform expect 2d tensor and transform to 2d list
            # y_pred = torch.flatten(y_pred) # 1d tensor
            pred_list.append(float(y_pred))

    print(pred_list)
    df['predict_price'] = [np.NaN] * (df.shape[0] - len(pred_list)) + pred_list
    df.to_csv(os.path.join(result_dir, 'price_result.csv'))

    # df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(8, 8))
    plt.plot(df['date'], df['open'], label='real')
    plt.plot(df['date'], df['predict_price'], label='pred')
    plt.legend()
    plt.xlabel('DATE')
    plt.ylabel('PRICE')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(args.result_dir, 'result.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--EMBEDDING_DIM', default=6, type=int)  # len(xvar) number of factor
    parser.add_argument('--HIDDEN_DIM', default=64, type=int)
    parser.add_argument('--OUTPUT_DIM', default=1, type=int)
    parser.add_argument('--NUM_LAYERS', default=1, type=int)
    parser.add_argument('--history', default=30, type=int)
    # files
    parser.add_argument('--test_path', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/dataset/val.csv',
                        type=str)
    parser.add_argument('--checkpoint_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/checkpoints', type=str,
                        help='output directory')
    parser.add_argument('--result_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project_price/result', type=str)

    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # model = CNN()
    model = RNN(embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                num_layers=args.NUM_LAYERS, output_dim=args.OUTPUT_DIM, history=args.history)

    print('LOADING MODEL...')
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'model_best.pth'))['model'])
    model.eval()
    predict(args, model)
    # test_path = args.test_path
    # df = pd.read_csv(test_path, encoding='latin')
    # df = pd.DataFrame(df, columns=['sentence'])
    # print(df.shape[0])
    # print(df.tail())

