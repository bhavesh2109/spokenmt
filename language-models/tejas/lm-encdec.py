import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from xml.dom import minidom
import nltk
import math
import pickle

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#torch.manual_seed(1)

START_TOKEN = 'SENT_START'
END_TOKEN = 'SENT_END'

# Load data
print 'Loading XML file...'
xmldoc = minidom.parse('ted_fr-20160408.xml')
print 'Getting French text...'
textList = xmldoc.getElementsByTagName('content')
print 'Number of transcripts:', len(textList)
# print textList[0].childNodes[0].nodeValue

sentenceList = []
NUM_TRANSCRIPTS = 1000

# Get tokenized sentences from French transcripts
for s in textList[:NUM_TRANSCRIPTS]:

    text = s.childNodes[0].nodeValue
    
    # Split text into sentences
    sentences = nltk.sent_tokenize(text.decode('utf-8').lower())
    '''if i==0:
        for sent in sentences:
            print sent
    '''
    
    # Split sentences into words
    tokenized_sents = [nltk.word_tokenize(s) for s in sentences]
    tokenized_sents = [[START_TOKEN] + t + [END_TOKEN] for t in tokenized_sents]
    '''if i == 0:
        for t in tokenized_sents[0]:
            print t
    '''
    # i = 1

    sentenceList += tokenized_sents

# Use only some of the sentences in vocabulary
NUM_SENTENCES = 20000
print 'Taking only', NUM_SENTENCES, 'sentences...'
sentenceList = sentenceList[:NUM_SENTENCES]


# Create word_to_index dict from transcripts
word_to_index = {}
index_to_word = []
for s in sentenceList:
    for word in s:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
            index_to_word.append(word)
print 'Size of vocabulary:', len(word_to_index)

# print len(index_to_word)

# Define some constants
EMBEDDING_DIM = 32
HIDDEN_DIM = 100
VOCAB_SIZE = len(word_to_index)
MAX_LENGTH

# Prepare sequences for LSTM input and output
X_data = []
Y_data = []

for s in sentenceList:
    X_data.append([word_to_index[w] for w in s[:-1]])
    Y_data.append([word_to_index[w] for w in s[1:]])

print 'Example input sequence:', X_data[100]
print 'Example output sequence:', Y_data[100]

# print len(X_data), len(Y_data)
training_data = [(x, y) for x, y in zip(X_data, Y_data)]
print 'Number of sentences in dataset:', len(training_data)

TRAINING_SIZE = int(0.8*len(training_data))
TESTING_SIZE = len(training_data) - TRAINING_SIZE
# TRAINING_SIZE = 500
# TESTING_SIZE = 500
print 'Training data size:', TRAINING_SIZE
print 'Testing data size:', TESTING_SIZE


# Write training_data to file
def writeDataToFile(writeData=False):

    if writeData is False:
        return
    # print len(X_data), len(Y_data)

    with open('seq_in_data.txt', 'w') as fpX:
        pickle.dump(X_data, fpX)

    with open('seq_out_data.txt', 'w') as fpY:
        pickle.dump(Y_data, fpY)



# Convert sequence to Torch Tensor and return it in Variable wrapper
def seq_tensor(seq):
    tensor = torch.LongTensor(seq).view(-1, 1)
    return autograd.Variable(tensor)


# LM encoder
class LMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers = 1):

        super(LMEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def initHidden(self):

        result = autograd.Variable(torch.zeros(1, 1, self.hidden_size))
        return result

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden


#LM decoder
class LMDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):

        super(LMDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def initHidden(self):

        result = autograd.Variable(torch.zeros(1, 1, self.hidden_size))
        return result

    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


def train(input_variable, target_variable, encoder, decoder, enc_optim, dec_optim, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    enc_optim.zero_grad()
    dec_optim.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = autograd.Variable(torch.zeros(max_length, encoder.hidden_size))

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = autograd.Variable(torch.LongTensor([[word_to_index[START_TOKEN]]]))

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, d
