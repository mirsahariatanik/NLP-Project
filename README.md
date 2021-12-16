# NLP-Project


Implemented a calendar management tool using Google speech processing system and a dialogue act classifier with a sequence labeler system. The speech processing block convert the speech to text file. The input to the NLP part is a sequence of words and the output is a frame containing the key information to schedule the event in the calender.

The varieties of NLP data can be generated with files nlp.py to nlp7.py and tweaking the codes a little. 
 
First, I have implemented an RNN+CRF based encoder for slot filler. For each word, the encoder predict highest-scoring label for that word. Implemented a labeler that just guesses, for each word, the label with the highest probability for that word.

To replace all words seen only once in the BIO training data "train_bio.txt" with unk, use "biocount.py".

To train the model, run "python python part_bio.py --train train.four.unk --dev data/dev_bio.txt --save modproject1.pth".

The best slot filler model is saved as "modproject1.pth".

Then I have used a Transformer encoder based Discriminative Classifiers for the data act classification.

To replace all words seen only once in the data act training data "train_data_act.txt" with unk, use "dac_count.py".

To train the model, run "python part_dac.py --train train.five.unk --dev data/dev_data_act.txt --save modproject2.pth".

The best data act model is saved as "modproject2.pth".

Then, integrated the dialogue act classifier with the slot filler. The file is "part_frame.py'.

In this part, data act classifier and slot filler models are combined to a single object "Model" and with torch.load, the trained models can easily be loaded separately.

To run the model for dev dataset, use "python part_frame.py --load modproject2.pth data/dev_words.txt --o project1.dev". The output is the desired frame from the sequence of words.

To compute the exact matching accuracy score, run "python3 score_frames.py project1.dev data/dev_frame.txt".

The scores of dev dataset is:

exact match:         0.74625

frame type accuracy: 1.0

argument F1:         0.9677419354838711

For the test case, to convert the speech to text, use "asr.py" or "speech.py" based on the type of the audio file. There are some speech samples available from training_1.mp3 to training_30.mp3.

To compare with the baseline, use "baseline.py" and save the data to another file.

I have tried to calculate the exact match score from the output of the basline method, somehow it did not work.

