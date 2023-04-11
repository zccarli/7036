import os
import pandas as pd
import argparse

from utils import *
from model import *



def predict(args, model):  # may use different data
    test_path = args.test_path
    df = pd.read_parquet(test_path)
    df = pd.DataFrame(df) # , columns=['sentence'
    df.rename(columns={'Sentences_c':'sentence'}, inplace=True)
    # df['sentence'] = df['sentence'].apply(pre_process)
    df['sentence'] = df['sentence'].apply(expand_contractions)
    raw_sentence = df['sentence']
    df['sentence'] = df['sentence'].apply(lambda sentence: text_transform(sentence, maxSeqLength=args.maxSeqLength))
    df = df.dropna()

    print('PREDICTING...')

    pred_list = []
    for idx in range(df.shape[0]):
        with torch.no_grad():
            x = df['sentence'].iloc[idx]
            x = torch.tensor(x).unsqueeze(0) # (senquence_length) to (1, senquence_length) same as batch input (batch_size, sequence_length)
            y_pred = model(x) # 2d tensor
            y_pred = nn.functional.softmax(y_pred, dim=1)  # because nn.CrossEntropyLoss in train_sentiment has include softmax as the activation funciton
            y_pred = torch.flatten(y_pred) # 1d tensor
            pred_list.append(y_pred.tolist()) # pred_list: 2d list

    pred_matrix = np.matrix(pred_list) # pred_matrix [samples, 3] 3: [posive, neutral, negative]
    df['positive sentiment'] = pred_matrix[:, 0]
    df['negative sentiment'] = pred_matrix[:, 2]
    df['sentence'] = raw_sentence
    df.to_csv(os.path.join(result_dir, 'sentiment_result.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument('--EMBEDDING_DIM', default=50, type=int)  # wordVectors.shape[1] = 50
    parser.add_argument('--HIDDEN_DIM', default=128, type=int)
    parser.add_argument('--OUTPUT_DIM', default=3, type=int)  # -1, 0, 1
    parser.add_argument('--NUM_LAYERS', default=2, type=int)
    parser.add_argument('--maxSeqLength', default=81, type=int) # from the train_sentiment.py: np.median(sequence_length)
    # files path
    parser.add_argument('--test_path', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/dataset/test/sentiment.parquet',
                        type=str)
    parser.add_argument('--checkpoint_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/checkpoints', type=str,
                        help='output directory')
    parser.add_argument('--lexicon_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project', type=str)
    parser.add_argument('--result_dir', default='/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/result', type=str)

    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model = RNN(wordVectors=wordVectors, embedding_dim=args.EMBEDDING_DIM, hidden_dim=args.HIDDEN_DIM,
                num_layers=args.NUM_LAYERS, output_dim=args.OUTPUT_DIM, maxseqlength=args.maxSeqLength)

    print('LOADING MODEL...')
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'model_best.pth'))['model'])
    model.eval()
    predict(args, model)
    # df = pd.read_parquet(args.test_path)
    # df = pd.DataFrame(df)
    # df.rename(columns={'Sentences_c': 'sentence'}, inplace=True)
    # print(df.head(20))
    # test_path = args.test_path
    # df = pd.read_csv(test_path, encoding='latin')
    # df = pd.DataFrame(df, columns=['sentence'])
    # print(df.shape[0])
    # print(df.tail())

