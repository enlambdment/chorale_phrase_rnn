import torch
import torch.nn
from rnn_train import *
from music21 import *

'''
SAMPLING THE NETWORK
To sample we give the network a note and ask what the next one is, feed that in
as the next note, and repeat until a Rest is reached.
For this to work, we need to
a) provide some starting note that actually matches (in terms of its tuple
   encoding, as pair of duration & pitch) with an element of tupleList
b) start collecting the successive model predictions
c) check the tupleList item that each prediction corresponds to, and
   get the tupleToNoteOrRest(t) of said tuple t
d) as soon as an item is reached that is a Rest (i.e. t[1] = None), stop

Index into tupleList by the index of a vector having the greatest value,
then convert that tuple using tupleToNoteOrRest(..) on it
'''

# to illustrate loading back a saved model
trained_model = RNN(n_tuples, n_hidden, n_tuples)
trained_model.load_state_dict(torch.load(PATH))

max_length = 20

# Sample from a category and starting letter.
# We'll take middle C quarter note to be like our start_note default
start_note_default = note.Note(pitch=pitch.Pitch(60),
                               duration=duration.Duration(1.0))
# some example start notes

n1 = tupleToNoteOrRest(tupleList[30])

n2 = tupleToNoteOrRest(tupleList[40])

n3 = tupleToNoteOrRest(tupleList[50])

n4 = tupleToNoteOrRest(tupleList[60])

n5 = tupleToNoteOrRest(tupleList[70])


def motive_sample(start_note=start_note_default):
    start_tup = noteOrRestToTuple(start_note)
    if start_tup in tupleList:
        with torch.no_grad():  # no need to track history for sampling
            input = motiveToTensor([start_note])
            hidden = trained_model.initHidden()

            output_nrs = [start_note]

            for i in range(max_length):
                output, hidden = trained_model(input[0], hidden)
                topv, topi = output.topk(1)
                topi = topi[0][0]

                tup = tupleList[topi]
                nr = tupleToNoteOrRest(tup)
                output_nrs.append(nr)

                if isinstance(nr, note.Rest):
                    break
                else:
                    input = motiveToTensor([nr])

            return output_nrs

    else:
        print("Sorry, provided input note doesn't exist in corpus sample.")


m1 = motive_sample(n1)
part1 = stream.Part(m1)

m2 = motive_sample(n2)
part2 = stream.Part(m2)

m3 = motive_sample(n3)
part3 = stream.Part(m3)

m4 = motive_sample(n4)
part4 = stream.Part(m4)

m5 = motive_sample(n5)
part5 = stream.Part(m5)
