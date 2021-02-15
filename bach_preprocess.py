from music21 import *
import pathlib as pl
from random import *
import itertools as it

'''
1. Locate
a) all chorales composed by J. S. Bach
b) that are in 4/4 time.

In general, TimeSignature objects are found within Measure
objects inside a Part object. Example:
'''

bwv66_6 = corpus.parse('bach/bwv66.6')
bach_parts = bwv66_6.parts
bach_soprano = bach_parts['Soprano']
soprano_timesig = bach_soprano.recurse().getElementsByClass(meter.TimeSignature)[0]


'''
>>> bach_paths = corpus.getComposer('bach')
>>> bach_path0 = bach_paths[0]
This give me the *full* paths dependent on installation site,
but I just want stuff like 'bach/bwv1.6' rather than '(...)/bach/bwv1.6.mxl'
>>> bach_path0.stem
'bwv1.6'
>>> bach_path0.parent
PosixPath('/opt/anaconda3/lib/python3.8/site-packages/music21/corpus/bach')
>>> bach_path0.parent.stem
'bach'
'''

bach_paths = corpus.getComposer('bach')
bach_score_names = [p.parent.stem + '/' + p.stem for p in bach_paths]

'''
>>> bach_score_names[0]
'bach/bwv1.6'
>>> bach_score_names[34]
'bach/bwv137.5'
'''

'''
So now I can pass each in turn to corpus.parse(..) & check to see which of these have
4/4 time sigs
'''

# Only take non-bass parts
bach_common_time_parts = [
    part for name in bach_score_names
    for part in corpus.parse(name).parts
    for timesigs in part.recurse().getElementsByClass(meter.TimeSignature)
    if part.partName != 'Bass'
    and timesigs.ratioString == '4/4']


def randomBachPart():
    '''
    Obtain a randomly selected Part from the Bach chorales in 4/4 time.
    '''
    rand_j = randint(0, len(bach_common_time_parts) - 1)
    return bach_common_time_parts[rand_j]


# test a list arising from use of '.expressions' attribute to see
# if it contains a Fermata object
def hasFermata(list):
    '''
    Predicate to check whether an object is marked with a fermata.
    '''
    return any(isinstance(x, expressions.Fermata) for x in list)


def isRest(nr):
    '''
    Predicate to check whether an object is a Rest. 
    '''
    return isinstance(nr, note.Rest)


'''
e.g.

>>> myRandPart = randomBachPart()
>>> myGSharp = list(myRandPart.flat)[6]
>>> myGSharp.expressions
[]
>>> myGSharpWithFermata = list(myRandPart.flat)[12]
>>> myGSharpWithFermata
<music21.note.Note G#>
>>> myGSharpWithFermata.expressions
[<music21.expressions.Fermata>]
>>> hasFermata(myGSharpWithFermata.expressions)
True
>>> hasFermata(myGSharp.expressions)
False
'''

'''
I eventually want to treat lists of non-rests like sub-melodic ('motive-like')
sequences for training an RNN on.
So, I want to split lists of non-rests arising from the use of .flat (plus
some additional filtering?) on:
  a) rests;      b) fermata (which are ultimately like rests as they indicate
                             a pause in introducing further tonal material)
How?

>>> myRandNotesAndRests = myRandPart.flat.notesAndRests
>>> type(myRandNotesAndRests)
<class 'music21.stream.iterator.StreamIterator'>
>>> for nr in myRandNotesAndRests: print(nr)

Again, I'm not splitting on *values*, I'm splitting on *satisfaction of
either predicate lambda x: hasFermata(x.expressions)
or     predicate lambda x: isRest(x)

The generator
  (i for i, v in enumerate(myNotesAndRests) if isRest(v) or hasFermata(v.expressions))
lazily computes all index-value pairs along myNotesAndRests for the list
elements that either are a rest or are marked with a fermata.

'''


def restAndFermataIndices(nrs):
    '''
    Given a list made up of notes or rests, obtain the indices locating all
    Rests and fermata-marked Notes in the list.
    '''
    return (i for i, v in enumerate(nrs) if isRest(v) or hasFermata(v.expressions))


# break up a list created by using .flat.notesAndRests method on some Part object,
# doing so at j+1 for every index j of an item that either has a fermata or is a rest.
# goal is to get back a list of melodic motives.
def getMotives(nrs):
    '''
    Given a list made up of notes or rests, split up the list into sublists.
    Each sublist either ends with a Rest, or has a fermata-marked Note as its
    last Note and is given a quarter-length Rest as its final element.
    This way, all sublists (motives) will consistently conclude with a Rest.
    '''
    indices_gen = restAndFermataIndices(nrs)
    j_start = 0
    subnrs = []
    for j_stop in indices_gen:
        # get list slice
        subnr = nrs[j_start: j_stop + 1]
        # check to see if the slice is empty; if not, work on it
        # before adding to subnrs
        if subnr != []:
            if not isRest(subnr[-1]):
                q = note.Rest('quarter')
                subnr.append(q)
            # add it to subnrs
            subnrs.append(subnr)
        # update j_start
        j_start = j_stop + 1
    return subnrs


def partToMotives(part):
    '''
    Given a Part extracted from a Score, obtain all sublists (motives)
    contained in the Part. A motive is a sequence of Notes concluding
    in the original Score with either a Rest or a fermata-marked Note.
    '''
    partNotesAndRests = part.flat.notesAndRests
    partMotives = getMotives(partNotesAndRests)
    return partMotives


# 1. Function to sample the list bach_common_time_parts without replacement


def randNBachParts(n):
    '''
    Obtain n random Parts from the Bach chorales in 4/4 time, sampled without
    replacement.
    '''
    return sample(bach_common_time_parts, n)

# 2. Function to extract all motives, delimited by rest or fermata, from a list of parts:


def bachPartsToMotives(parts):
    '''
    Given a list of Parts, split each Part into its note-sublists (motives.)
    '''
    return [partToMotives(part) for part in parts]


# List of all notes & rests that arise in the data set selected from Bach corpus.


def all_motive_items():
    '''
    Obtains the superset of all distinct music21 objects contained in the
    entire subset of 4/4 (common-time) Bach chorales available via
    library corpus.
    '''
    all_items = []
    # a motive ~ a list of notes & rests. So this is a list of (lists of lists-of-notes/rests)
    # because each part in general gives rise to multiple motives when we split it up
    all_motives = [motive for part in bach_common_time_parts
                   for motive in partToMotives(part)]
    for motive in all_motives:
        all_items.extend(motive)
    return all_items


def noteOrRestToTuple(nr):
    '''
    Cast a Note or Rest object into a Tuple of its duration (as quarter length) and
    pitch (as midi semitone count). Rests are assigned a pitch of None in this
    representation. This representation:
    a) strips incidental note/rest-level material from the music which is of no
       interest for our defined learning task
    b) results in a hashable data type that we can form a set over, thus
       identifying unique tuples.
    '''
    if isinstance(nr, note.Note):
        return((nr.duration.quarterLength, nr.pitch.midi))
    elif isinstance(nr, note.Rest):
        return((nr.duration.quarterLength, None))


# Tuples are hashable! So I can do this now:
tupleList = list(set(
    noteOrRestToTuple(x) for x in all_motive_items()))

# For recasting back into music21 representation


def tupleToNoteOrRest(tup):
    '''
    Given a tuple of float (representing a count of quarter-note lengths)
    and pitch (possibly None), return the music21 Note or (for pitch of
    None) Rest that it represents.
    '''
    if tup[1] is not None:
        mkDur = duration.Duration(tup[0])
        mkPitch = pitch.Pitch(tup[1])
        return note.Note(pitch=mkPitch, duration=mkDur)
    elif tup[1] is None:
        mkDur = duration.Duration(tup[0])
        return note.Rest(duration=mkDur)


'''
At this writing there are 338 distinct kinds of Notes / Rests, identifying
distinct items in this count by duration-pitch pairing:
>>> len(tupleList)
338
'''
n_tuples = len(tupleList)


def noteOrRestToIndex(nr):
    '''
    Given a music21 Note or Rest, find the index of its duration-pitch
    tuple in the sub-corpus universe of all such tuples (if exists
    in this universe.)
    '''
    return tupleList.index(noteOrRestToTuple(nr))
