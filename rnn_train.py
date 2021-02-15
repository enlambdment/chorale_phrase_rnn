from bach_preprocess import *
import math
import time
import torch
import torch.nn as nn
from random import *


def noteOrRestToTensor(nr):
    '''
    Given a music21 Note or Rest, convert it into a one-hot
    Tensor.
    '''
    # initialize a 'blank' Tensor of only zeroes
    tensor = torch.zeros(1, n_tuples)
    # access & modify the Tensor at the 0th row of tensor
    tensor[0][noteOrRestToIndex(nr)] = 1
    return tensor


def motiveToTensor(motive):
    '''
    Given a motive (list of Notes / Rests), convert it into
    a Tensor made up of one-hot encodings for the Notes / Rests.
    '''
    tensor = torch.zeros(len(motive), 1, n_tuples)
    for nr_i, nr in enumerate(motive):
        tensor[nr_i][0][noteOrRestToIndex(nr)] = 1
    return tensor


'''
e.g.
>>> myRandPart = randomBachPart()
>>> myRandMotives = bachPartsToMotives([myRandPart])[0]
>>> len(myRandMotive)
12
>>> for x in myRandMotive: print(x)
(...)
>>> len(myRandMotive)
12
>>> myRandMotiveTensor = motiveToTensor(myRandMotive)
>>> myRandMotiveTensor.size()
torch.Size([12, 1, 338])
'''

'''
We now have to build and train a "musical language model" that will
a) train on tensors of size <len_motive * 1 * num_tuples>
b) learn based upon the discrepancy of pred vs. tgt,
   where pred is what the rnn produces in "forward" direction
   while tgt  is the expected tensor of size <1 * 1 * num_tuples> (?)
              to follow
'''

'''
CREATING THE NETWORK
(Adapted from PyTorch intermediate tutorial on using RNN to build
character-level language model for operating on names)

Following the i2o lyer, we'll also add a linear layer o2o to
give the network more muscle to work with. This will be followed
by a dropout layer, which randomly zeros parts of its input
with a given probability (here 0.1) and is usually used to fuzz
inputs, to prevent overfitting.
Here we're using it towards the end of the network to purposely
add some chaos and increase sampling variety.
'''


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # operating on vectors of size 'input_size + hidden_size'
        # is effectively the same as defining 2 separate Linear
        # operations on input vector & hidden vector
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden),  1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


'''
TRAINING
PREPARING FOR TRAINING
For each timestep (i.e. for each note/rest in a training motive)
the inputs of the network will be (current note/rest, hidden state) and the
    outputs               will be (next note/rest, next hidden state).
We will always be predicting the next n/r from the current n/r,
so the n/r part of these pairs consists of consecutive n/r's -

(in representation casting to tuples (duration, pitch))
(we can think of the rests, cast as (d, None), like <EOS> tokens
 in an ordinary char-level NLP task)
      (1.0, 60) (1.0, 58) (2.0, 60)     <- input
               /         /
      (1.0, 58) (2.0, 60) (1.0, None)   <- output
'''

# One-hot matrix of first to last notes (not incl. final rest) for input


def inputTensor(motive):
    '''
    Given a motive (list of music21 Notes / Rests), convert into
    a Tensor containing one-hot encoding for all but the last
    Note / Rest.
    This is the input for one round of multi-step RNN training
    on a sequence.
    '''
    motive_init = motive[:-1]
    init_tensor = motiveToTensor(motive_init)
    return init_tensor

# LongTensor of second note to end (EOS) for target.
# (The main modification vs. the tutorial is that we don't
# have to manually include an additional index value of
# 'n_tuples - 1' to append onto the indexes, because the
# rest which lands at the end of the motive is already our
# 'EOM' (end-of-motive) token.)


def targetTensor(motive):
    '''
    Given a motive (list of music21 Notes / Rests), convert into
    a Tensor containing one-hot encoding for all but the first
    Note / Rest.
    This is the target for one round of multi-step RNN training
    on a sequence.
    '''
    nr_indexes = [noteOrRestToIndex(motive[j])
                  for j in range(1, len(motive))]
    return torch.LongTensor(nr_indexes)


# Make input and target tensors from a random motive
# (how to get random list element?)


def getRandomMotive():
    '''
    Obtain a random motive, of length at least 2, from the Bach chorale
    subcorpus consisting only of works in 4/4 time.
    '''
    # motives must have length at least 2 - i.e. long enough to contain
    # 1 note followed by 1 rest (minimal example)
    randMotive = []
    while len(randMotive) < 2:
        randPartMotives = []
        # part-motives list must be non-empty to select from
        while randPartMotives == []:
            randPart = randomBachPart()
            randPartMotivesList = bachPartsToMotives([randPart])
            randPartMotives = list(it.chain.from_iterable(randPartMotivesList))
        randMotive = choice(randPartMotives)
    return randMotive


'''
Example of getting a random motive (lists of notes ended by a
rest), making it into a stream, showing it (MuseScore is configured
to pop up with the musically notated example), and making input
and target Tensors out of it:

>>> myRandMotive = getRandomMotive()
>>> myRandStream = stream.Part(myRandMotive)
>>> myRandStream.show()
>>> len(myRandMotive)
10
>>> myRandInput = inputTensor(myRandMotive)
>>> myRandInput.size()
torch.Size([9, 1, 338])
>>> myRandTarget = targetTensor(myRandMotive)
>>> myRandTarget.size()
torch.Size([9])
'''

# Make input and target tensors from a random motive


def randomTrainingExample():
    '''
    Obtain a pair of input and target Tensors (training data)
    from a randomly selected motive in the common-time Bach chorale
    subcorpus.
    '''
    motive = getRandomMotive()
    input_tensor = inputTensor(motive)
    target_tensor = targetTensor(motive)
    return input_tensor, target_tensor


'''
TRAINING THE NETWORK
In contrast to classification, where only the last output is used,
we are making a prediction at every step, so we are calculating
loss at every step.
The magic of autograd allows you to simply sum these losses at each step,
and call backward at the end (i.e. once the whole sequence has been
trained on & losses at all (l-1) positions, l the length of the
sequence, calculated and summed up.)
'''
criterion = nn.NLLLoss()

learning_rate = 0.0005

n_hidden = 800

rnn = RNN(n_tuples, n_hidden, n_tuples)


def train(input_motive_tensor, target_motive_tensor):
    '''
    Given a training input and training target, conduct one round
    of RNN training on the sequence which the input / target
    describe.
    '''
    # 'unsqueeze in place' - this creates a new tensor based
    # on the tensor this is applied to, with dim. 1 inserted
    # at the location in the .size() dim's list specified by
    # the arg. to unsqueeze_.
    # Therefore, we are creating a torch.Size([9, 1]) tensor
    target_motive_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    # zero the gradient buffers
    rnn.zero_grad()

    # This will be accumulated over and eventually back-prop'ed on, so
    # it needs to be a Tensor.
    loss = torch.tensor(0.0, requires_grad=True)

    output = None
    # 0th part of tensor size lists out how many one-hots in the tensor
    # i.e. length of current sequence being trained on
    for i in range(input_motive_tensor.size(0)):
        output, hidden = rnn(input_motive_tensor[i], hidden)
        l = criterion(output, target_motive_tensor[i])
        # loss += l  # <- RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        # The operation that we use in order to add up the cumulative loss must be non-in-place.
        loss = torch.add(loss, l)

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_motive_tensor.size(0)


'''
Training is business as usual - call 'train' a bunch of times and
wait a few minutes, printing the current time and loss every
'print_every'-many examples, and keeping store of an average
loss per 'plot_every'-many example in all_losses for plotting later.
'''


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def training_loop():
    '''
    n_iters =   100000
    print_every = 5000
    plot_every = 500
    '''

    # experiment first with (much) small n_iters and [x]_every batch sizes
    n_iters = 48000
    print_every = 2400
    plot_every = 240
    all_losses = []
    total_loss = 0  # reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        inp, tgt = randomTrainingExample()
        output, loss = train(inp, tgt)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter,
                                         iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0


# Modified Friday 12/11/2020 in order to train
# and save an alternate model with more training rounds
PATH = './note_level_rnn.pth'


def train_and_save_model():
    training_loop()
    torch.save(rnn.state_dict(), PATH)
