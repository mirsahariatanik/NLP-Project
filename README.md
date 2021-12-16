# NLP-Project


Implemented a calendar management tool using dialogue act classifier with a sequence labeler system. The input is a sequence of words and the output is a frame containing the key information to schedule the event in the calender. I have used a Transformer encoder based Discriminative Classifiers for the data act classification and a RNN+CRF based encoder for labeling and slot filling.

For each word, the encoder predict highest-scoring label for that word. Implemented a labeler that just guesses, for each word, the label with the highest probability for that word.

To replace all words seen only once in the BIO training data "train_bio.txt" with unk, used "fourcount.py".

To train the model, run "python python part3_hw04.py.py --train train.four.unk --dev data/dev_bio.txt --save modproject1.pth".

The best model is saved as "modproject1.pth".

To replace all words seen only once in the frame training data "train.txt" with unk, use "Part1_count.py".

To train the model, run "python part1_dac.py --train train.five.unk --dev data/dev.txt --save modproject2.pth".

The best model is saved as "modproject2.pth".


To load the model and run on the test datasets, use "python part1_dac.py --load model1.pth --o test.types.out data.part1/test.words

To determine the test accuracy, run "python score_types.py test.types.out data.part1/test.types".

The test accuracy is 0.8376068376068376.



Part 02:

Integrated the dialogue act classifier with the slot filler from HW4. The file is "part2.py'.

In part 2 and 3, data act classifier and slot filler models are combined to a single object "Model" and with torch.load, the trained models can easily be loaded separately.

To run the model for dev dataset, use "python part2.py --load model1.pth data.part2/dev.words --o part21.dev".

To compute the accuracy score, run "python3 score_frames.py part21.dev data.part2/dev.frames".

The scores of dev dataset are:

exact match:         0.7159090909090909

frame type accuracy: 0.8481404958677686

argument F1:         0.8294392523364484

To run the model for test dataset, use "python part2.py --load model1.pth data.part2/test.words --o part21.test".

To compute the accuracy score, run "python3 score_frames.py part21.test data.part2/test.frames".

The scores for test dataset are:

exact match:         0.7122507122507122

frame type accuracy: 0.8271604938271605

argument F1:         0.8498206048180421

Integrated the system with the backend to make an interactive system in "part2_combined".

To run the code, use "python part2_combined.py --load model1.pth data.part2/test.words --o part22.test" and enter the required string. 
