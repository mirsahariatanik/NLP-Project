# NLP-Project


Implemented a calendar management tool using Google speech processing system and a dialogue act classifier with a sequence labeler system. The speech processing block convert the speech to text file. The input to the NLP part is a sequence of words and the output is a frame containing the key information to schedule the event in the calender. 
 
First, I am implemented an RNN+CRF based encoder for labeling and slot filling. For each word, the encoder predict highest-scoring label for that word. Implemented a labeler that just guesses, for each word, the label with the highest probability for that word.

To replace all words seen only once in the BIO training data "train_bio.txt" with unk, use "fourcount.py".

To train the model, run "python python part3_hw04.py.py --train train.four.unk --dev data/dev_bio.txt --save modproject1.pth".

The best labeler model is saved as "modproject1.pth".

Then I have used a Transformer encoder based Discriminative Classifiers for the data act classification.

To replace all words seen only once in the frame training data "train_data_act.txt" with unk, use "Part1_count.py".

To train the model, run "python part1_dac.py --train train.five.unk --dev data/dev_data_act.txt --save modproject2.pth".

The best data act model is saved as "modproject2.pth".

Then, integrated the dialogue act classifier with the slot filler. The file is "part2.py'.

In this part, data act classifier and slot filler models are combined to a single object "Model" and with torch.load, the trained models can easily be loaded separately.

To run the model for dev dataset, use "python part2.py --load modproject2.pth data/dev_words.txt --o project1.dev".

To compute the exact matching accuracy score, run "python3 score_frames.py project1.dev data/dev_frame.txt".

The scores of dev dataset are:

exact match:         0.74625
frame type accuracy: 1.0
argument F1:         0.9677419354838711

