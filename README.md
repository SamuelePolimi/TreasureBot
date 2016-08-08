# TreasureBot
A bot for financial signal

Requirements: 
- Tensorflow
- Numpy

stock1.py is just an enviroment to simulate series of stock prices.

Giving the following code you can generate and save on file a serie of stock price:

from stock1 import Stock1
s = Stock1()
s.savePriceList(1000,'stock')

it will save a serie of length=1000 on a file named "stock1.npy"

then if you want to visualize it, you could use utils/plotDataset.py

if you want to save a dataset with more than one series you could do:

from stock1 import Stock1
s = Stock1()
s.saveDataset(1000, 1000,'train')
s.saveDataset(200, 1000, 'test')

where 1000 and 200 are the number of series contained in the files.

If you want to build a very simple Recurrent Neural Network that learn how to trade, just type:

python direct_rl.py dataset/train.npy dataset/test.npy 300 500

where 300 is the length of the window size (for the moment 300 is a good number) and 500 is the number of series that we would like to use for the train phase (in such a way we can just generate a huge dataset once and use a wondered smaller batch)

The output you'll see is the gain that you'll have in average both in the trainset and in the testset. The really important one is the one that you have on the test set.
That's all folks!
