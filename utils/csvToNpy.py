import sys
import csv
import numpy as np

try:
    filename = sys.argv[1]              #name of the file
    column = int(sys.argv[2])           #select the column of interest
    jump_row = int(sys.argv[3])         #how many rows do I jump?
except:
    print """Hi, an error was catched. This means that probably you didn't insert the correct number/type of parameters
    don't worry at all.
    You should type:
        python csvToNpy.py <filename> <column> <jump row>
    where:
        filename: is the name of the csv
        column: is the column to select (for example the price_close)
        jump_row: normally the first row is a title row: you could select hence jump_row=1"""

prices = []
with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    i = 0
    for row in reader:
        if i<=jump_row:
            i+=1
            continue        
        i+=1
        prices.append(float(row[column]))
        
prices.reverse()
print "Data read!"

d = 'N'
while d!='Y':
    series_length = int(raw_input("what is the length of the series?"))
    n_series = len(prices) / series_length
    n_out = len(prices) % series_length
    d = raw_input("There will be " + str(n_series) + " series and " + str(n_out) + " values out. Is it okay? [Y,N]") 

n_test = int(raw_input("How many series for the test?"))

data = []

for i in xrange(n_series):
    data.append(prices[i * series_length:(i + 1) * series_length])
    
npdata = np.array(data)

np.random.shuffle(npdata)

nptest = npdata[0:n_test,:]
nptrain = npdata[n_test:,:]

np.save('train',nptrain)
np.save('test',nptest)
print "You'll find train.npy and test.npy in this folder. (In future, you'll find train.npy, validation.npy, test.npy)."