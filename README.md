# chorale-phrase-rnn

Employing the principles and `torch` functionality described in the 
[*NLP From Scratch: Classifying Names with a Character-Level RNN*](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial)
 series, I set up and trained a "note-level RNN" to work on musical phrases obtained from a selection of Bach chorales, available via the 
 computer music library `music21`.

## bach_preprocess
In order to perform recurrent neural network (RNN) training on musical phrases of variable length, it was first necessary to obtain
the phrases from the chorale parts in a subset of J. S. Bach chorales and pre-process these into a form suitale for one-hot encoding 
(the tensor encoding chosen in the char-rnn series and in my implementation as well.) By analogy with *characters* as the constituents of
words (names), *notes / rests* (possessing pitch or no pitch, respectively, but both containing a duration) were viewed as the constituents of
musical phrases, and the phrases in turn were obtained by subdividing the individual voices of 4-part chorales at the locations of fermatas or rests.
Because typical constituents could be notes of arbitrary pitch and duration, or rests of arbitrary duration (but no pitch), pre-processing of 
such constituent sequences also involved encoding all observed distinct pairs of pitch-duration into indexes of an overall list of all possible
observed items from the training data, as a preparatory step for one-hot encoding.

## rnn_train
One-hot encoding was employed in order to specify a function for transforming a musical phrase (again, of arbitrary length) into a 2D Tensor
consisting of consecutive one-hot vectors (as many vectors as there are items in the phrase being encoded.) An `RNN` class was defined and trained
on the data, and the model weights making up the final trained RNN were stored to the current directory for later use in generating phrases from
arbitrary starting notes / rests of the user's choosing.

## rnn_sample 
This module loads the previously trained model, and provides a function for generating the musical phrase which the RNN "spells out" 
based upon its input parameter as the starting note, and feeding the output at each time step back into the RNN as the following input.