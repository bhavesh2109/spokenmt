import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# Get French text
xmldoc = minidom.parse('ted_fr-20160408.xml')
print 'Getting French text...'

# Get English text
# xmldoc = minidom.parse('ted_en-20160408.xml')
# print 'Getting English text...'

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
NUM_SENTENCES = 30000
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
MAX_LENGTH = 50

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
def seq2Tensor(seq):
    tensor = torch.LongTensor(seq).view(-1, 1)
    return autograd.Variable(tensor).cuda()

# Convert training sequences to Torch Tensors
def list2Variables(training_list):

    training_pairs = []
    for list_in, list_out in training_list:
        var_in = seq2Tensor(list_in)
        var_out = seq2Tensor(list_out)
        training_pairs.append((var_in, var_out))

    return training_pairs
        

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
        return result.cuda()

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden


#LM decoder
class LMAttnDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1,  max_length=MAX_LENGTH):

        super(LMAttnDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size*2, max_length)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def initHidden(self):

        result = autograd.Variable(torch.zeros(1, 1, self.hidden_size))
        return result.cuda()

    def forward(self, input, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights


def train(input_variable, target_variable, encoder, decoder, enc_optim, dec_optim, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    enc_optim.zero_grad()
    dec_optim.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = autograd.Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda()

    # print 'New training instance'
    # print 'Input length:', input_length
    # print 'Target length:', target_length

    loss = 0

    for ei in range(input_length):
        if ei == max_length:
            break
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        # print 'Encoder output size:', encoder_output.size(), 'Encoder hidden size:', encoder_hidden.size()
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = autograd.Variable(torch.LongTensor([word_to_index[START_TOKEN]]))
    decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # print 'Decoder input size:', decoder_input.size(), 'Decoder output size:', decoder_output.size(), 'Decoder hidden size:', decoder_hidden.size()
        loss += criterion(decoder_output, target_variable[di])
        decoder_input = target_variable[di]

    # print '--------------------------------'
    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss.data[0]/target_length

def trainIters(encoder, decoder, num_iters, print_every=100, plot_every=100, learning_rate=0.01):

    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)

    plot_loss_total = 0
    plot_losses = []

    training_pairs = list2Variables(training_data[:TRAINING_SIZE])
    criterion = nn.NLLLoss()

    for iter in range(1, num_iters+1):

        training_pair = training_pairs[iter-1]
        input_var = training_pair[0]
        output_var = training_pair[1]

        loss = train(input_var, output_var, encoder, decoder, encoder_optim, decoder_optim, criterion)
        plot_loss_total += loss

        if iter % print_every == 0:
            print 'Iteration:', iter, 'Loss:', loss, 'total loss:', plot_loss_total

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    evaluate(encoder, decoder)

    # showPlot(plot_losses)

def evaluate(encoder, decoder, max_length=MAX_LENGTH):

    testing_pairs = list2Variables(training_data[TRAINING_SIZE:TRAINING_SIZE+TESTING_SIZE])

    log_prob_sum = 0
    total_seq_len = 0
    encoder_hidden = encoder.initHidden()

    print 'Evaluating model...'

    for i in range(TESTING_SIZE):

        if i % 100 == 0:
            print 'Sentence number', i

        word_prob_sum = 0

        testing_pair = testing_pairs[i]
        input_var = testing_pair[0]
        target_var = testing_pair[1]

        input_length = input_var.size()[0]
        target_length = target_var.size()[0]

        encoder_outputs = autograd.Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda()

        # print 'New training instance'
        # print 'Input length:', input_length
        # print 'Target length:', target_length

        loss = 0

        for ei in range(input_length):
            if ei == max_length:
                break
            encoder_output, encoder_hidden = encoder(input_var[ei], encoder_hidden)
            # print 'Encoder output size:', encoder_output.size(), 'Encoder hidden size:', encoder_hidden.size()
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = autograd.Variable(torch.LongTensor([word_to_index[START_TOKEN]]))
        decoder_input = decoder_input.cuda()

        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # print 'Decoder input size:', decoder_input.size(), 'Decoder output size:', decoder_output.size(), 'Decoder hidden size:', decoder_hidden.size()
            # print 'Target variable:', (target_var[di])
            # print 'Target variable val:', (target_var[di].data)[0]
            # print 'target log prob:', (decoder_output.data)[0][(target_var[di].data)[0]]
            word_prob_sum += (decoder_output.data)[0][(target_var[di].data)[0]]
            decoder_input = target_var[di]

        total_seq_len += target_length
        log_prob_sum += word_prob_sum

    log_prob_mean = -1.0*log_prob_sum/total_seq_len
    print log_prob_mean

    perplexity = math.exp(log_prob_mean)
    print 'Perplexity:', perplexity


def showPlot(points):

    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points) 

encoder = LMEncoder(VOCAB_SIZE, HIDDEN_DIM, n_layers=3)
decoder = LMAttnDecoder(HIDDEN_DIM, VOCAB_SIZE, n_layers=3)

encoder.cuda()
decoder.cuda()

trainIters(encoder, decoder, TRAINING_SIZE)
