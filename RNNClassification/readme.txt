#RNNClassification

The RNN structure mainly used the https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/rnn_text_clf.py

1. text_classification_py3.py contains the main code, it includes the structure of RNN and building dictionary. it take about 30 minutes to train the data.
the output of the training procedure is:

Train 4774 samples
Epoch 1 | Step 037 | Train loss: 2.3953 | Train acc: 0.1562 | lr: 0.0050
Epoch 2 | Step 037 | Train loss: 1.5066 | Train acc: 0.5000 | lr: 0.0042
Epoch 3 | Step 037 | Train loss: 1.6113 | Train acc: 0.4609 | lr: 0.0036
Epoch 4 | Step 037 | Train loss: 1.2793 | Train acc: 0.6250 | lr: 0.0031
Epoch 5 | Step 037 | Train loss: 0.9938 | Train acc: 0.6562 | lr: 0.0026
Epoch 6 | Step 037 | Train loss: 0.8916 | Train acc: 0.7188 | lr: 0.0022
Epoch 7 | Step 037 | Train loss: 0.6299 | Train acc: 0.8359 | lr: 0.0019
Epoch 8 | Step 037 | Train loss: 0.4825 | Train acc: 0.8516 | lr: 0.0016
Epoch 9 | Step 037 | Train loss: 0.3443 | Train acc: 0.8750 | lr: 0.0013
Epoch 10 | Step 037 | Train loss: 0.4324 | Train acc: 0.8750 | lr: 0.0011
3749.2543437480927
1519016980.5994093
training time: 3749.254344, testing time: 1519016980.599409
[ 5  5  8 ...,  5  8 10]

2. model file is the serialized model of RNN
3. serialized_text is the serialized text, you can retrain your model or load this text
4. result is the prediction of the classification labels
5. data file contains train and test dataset