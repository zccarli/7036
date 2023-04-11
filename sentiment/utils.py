
##################################### clean and load lexicon ##################################
import re
import numpy as np
import contractions
from torchtext.data.utils import get_tokenizer
from gensim.parsing.preprocessing import STOPWORDS


def clean_sentiment(text):
    if text == 'positive' or text == 'pos':
        return 1
    elif text == 'negative' or text == 'neg':
        return -1
    else:
        return 0

def soft(text):
    if text == 1:
        return [1, 0, 0] # y_pred [0.8, 0.2, 0.5] -> [1, 0, 0]
    elif text == -1:
        return [0, 0, 1]
    else:
        return [0, 1, 0]



def pre_process(text):
    text = str(text)
    # Remove links
    text = re.sub('http://\S+|https://\S+', '', text)
    text = re.sub('http[s]?://\S+', '', text)
    text = re.sub(r"http\S+", "", text)

    # Convert HTML references
    text = re.sub('&amp', 'and', text)
    text = re.sub('&lt', '<', text)
    text = re.sub('&gt', '>', text)
    text = re.sub('\xa0', ' ', text)

    # Remove new line characters
    text = re.sub('[\r\n]+', ' ', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove multiple space characters
    text = re.sub('\s+', ' ', text)

    pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
    text = pattern.sub('', text)

    text = text.lower()

    return text


def expand_contractions(text):
    try:
        return contractions.fix(text)
    except:
        return text


tokenizer = get_tokenizer('basic_english')
def text_transform(sentence, maxSeqLength): # word2vector
    #     sentence_vector = np.zeros(maxSeqLength)
    sentence_vector = []
    for token in tokenizer(sentence):
        try:
            sentence_vector.append(wordsList[token])
        except:  # exclude non english sentence and not recognized word （gibberish)
            sentence_vector.append(0)
    if len(sentence_vector) > maxSeqLength:
        sentence_vector = sentence_vector[:maxSeqLength]
    elif len(sentence_vector) < maxSeqLength:
        sentence_vector.extend(np.zeros(maxSeqLength - len(sentence_vector), dtype='int64'))
    return sentence_vector


wordsList = np.load('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/lexicon/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()  # Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordsList = dict(zip(wordsList, range(0, len(wordsList))))
wordVectors = np.load('/Users/Rui/Desktop/MFFINTECH/MFIN7036/project/lexicon/wordVectors.npy')
print('Loaded the word vectors!')

# print(wordsList['pce'])
