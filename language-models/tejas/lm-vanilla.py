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
def writeDataToFile(writeData):

    if writeData is False:
        return
    # print len(X_data), len(Y_data)

    with open('seq_in_data.txt', 'w') as fpX:
        pickle.dump(X_data, fpX)

    with open('seq_out_data.txt', 'w') as fpY:
        pickle.dump(Y_data, fpY)



# Convert sequence to Torch Tensor and return it in Variable wrapper
def seq_tensor(seq):
    tensor = torch.LongTensor(seq)
    return autograd.Variable(tensor)

# Create the LSTM-based Language Model
class LSTMLangModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers):
        
        super(LSTMLangModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm_embed = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

        self.hidden2words = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.initHidden()

    def initHidden(self):

        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm_embed(embeds.view(len(sentence), 1, -1), self.hidden)
        # print lstm_out.size()
        for i in range(self.num_layers):
            # print lstm_out.size()
            lstm_out, self.hidden = self.lstm(lstm_out, self.hidden)
        words_space = self.hidden2words(lstm_out.view(len(sentence), -1))
        words_probs = F.log_softmax(words_space)
        return words_probs

# Train the model
model = LSTMLangModel(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, 0)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print 'Training model...'
for epoch in range(1):

    print 'epoch', epoch
    i = 0
    for seq_in, seq_out in training_data[:TRAINING_SIZE]:

        if i % 100 == 0:
            print 'Sentence number', i
        i += 1

        # Clear the pytorch gradients
        model.zero_grad()

        # clear out the model's hidden state
        model.hidden = model.initHidden()

        # Convert sequences to tensors and wrap in Variable
        in_var = seq_tensor(seq_in)
        out_var = seq_tensor(seq_out)

        # Forward pass
        words_scores = model(in_var)

        # Calculate loss and backprop gradients
        loss = loss_function(words_scores, out_var)
        loss.backward()
        optimizer.step()

log_prob_sum = 0
total_seq_len = 0
j = 0

# Evaluation
print 'Evaluating model...'
for seq_in, seq_out in training_data[TRAINING_SIZE:TRAINING_SIZE+TESTING_SIZE]:

    words_probs = model(seq_tensor(seq_in))
    temp_prob = 0
    if j % 100 == 0:
        print 'Sentence number', j

    for i, next_word in enumerate(seq_out):
        # print i, next_word
        # print (words_probs.data)[i][next_word]
        temp_prob += (words_probs.data)[i][next_word]

    log_prob_sum += temp_prob
    total_seq_len += len(seq_out)
    # print total_seq_len, len(seq_out)
    j += 1 

log_prob_mean = -1.0*log_prob_sum/total_seq_len
print log_prob_mean

perplexity = math.exp(log_prob_mean)
print 'Perplexity:', perplexity

# Generation
def generate_sentence():

    new_sentence = [word_to_index[START_TOKEN]]
    while new_sentence[-1] != word_to_index[END_TOKEN]:

        word_probs = model(seq_tensor(new_sentence))
        print word_probs.size()
        # print word_prob.sum()
        max_prob, index = torch.max(word_probs.data, 1)
        # print index
        new_sentence.append(index[-1])
        # print new_sentence

        gen_sentence = [index_to_word[s] for s in new_sentence[1:]]
        print " ".join(gen_sentence)

    gen_sentence = []
    print new_sentence
    for s in new_sentence[1:-1]:
        gen_sentence.append(index_to_word[s])

    print gen_sentence.join(' ')

if __name__=='__main__':
    print
    writeDataToFile(False)
    # generate_sentence()
