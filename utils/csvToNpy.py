import sys
import csv
import numpy as np
import os

from datetime import datetime

print """Hi, 
This script will generate train_set.npy, validation_set.npy, test_set.npy, config.json with the following meaning:
train_set.npy: 3D npy array with shape (n_sample, serie_length, n_series*2 + n_features)
validation_set.npy: 3D npy array with shape (n_sample, serie_length, n_series*2 + n_features)
test_set.npy: 3D npy array with shape (n_sample, serie_length, n_series*2 + n_features)

they respect the rule:

train_set.shape = (a,b,c)
validation_set.shape = (d,b,c)
test_set.shape = (e,b,c)

for train and validation sets, the 3rd dimension has the following meaning:

set[s,t,0:n_series] : derivative of the stocks of sample s at time t
set[s,t,n_series:n_series*2] : cost of transactions of sample s at time t
set[s,t,n_series*2:] :  features of sample s at time t

for test_set 

set[s,t,0:n_series] : derivative of the stocks of sample s at time t
set[s,t,n_series:n_series*2] : value of stocks of sample s at time t (cost will be computed from here by the finalObject)
set[s,t,n_series*2:] :  features of sample s at time t

config.json: {'n_series':n_series, 'n_features':n_features, ..}

The input needed are just csv files :).

Let's start.. 

"""


"""
TODO1:

imagine to have the following dataset

date         open     close      max      min
12.09.2012    10.      12.        14.      9.
13.09.2012    13.      12.5       15.      8.
14.09.2012    14.      11.        16.      10.

and we wand to delete the second row (for a synchronization issue, suppose), we won't have:

date         open     close      max      min
12.09.2012    10.      12.        14.      9.
14.09.2012    14.      11.        16.      10.

but the following:

date         open     close      max      min
12.09.2012    10.      12.5       15.      8.
14.09.2012    14.      11.        16.      10.

"""

def printExampleFile(path):
    print("File example:\n")
    with open(path, "r") as ins:
        count = 0
        for line in ins:
            print("\t" + line[:-1])
            if count > 5:
                print("")
                return
            count+=1
            
def selectCol(path, column, type_):
    ret = []
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        i = 0
        for row in reader:
            if i<1:
                i+=1
                continue        
            i+=1
            ret.append(type_(row[column]))
    return ret
    
def syncronize(time1,time2):
    i=0 #index of time1
    j=0 #index of time2
    time1_index = []
    time2_index = []
    while True:
        if(time1[i] == time2[j]):
            time1_index.append(i)
            time2_index.append(j)
            i+=1
            j+=1
            if j >= len(time2) or i >=len(time1):
                break
        if(time1[i] > time2[j]):
            j+=1
            if j >= len(time2):
                break
        if(time1[i] < time2[j]):
            i+=1
            if i >=len(time1):
                break
        
    return (time1_index, time2_index)
    
def indexise(indexes, date_time, open_price, close_price, min_price, max_price, volume, features):    
    ret = []    
    for last_indx, next_indx in zip(indexes[:-1],indexes[1:]):
        ret_date_time = date_time[last_indx]
        ret_open_price = open_price[last_indx]
        ret_close_price = close_price[next_indx-1]
        ret_min_price = min(min_price[last_indx:next_indx])
        ret_max_price = max(max_price[last_indx:next_indx])
        ret_volume = sum(volume[last_indx:next_indx])
        ret_features = []
        for i in features.shape[1]:
            ret_features.append(features[last_indx,i])
        ret.append([ret_date_time, ret_open_price, ret_close_price, ret_min_price, ret_max_price, ret_volume] + ret_features)
    final_features = []
    for f in features:
        final_features.append(f[-1])
    ret.append([date_time[-1],open_price[-1],close_price[-1],min_price[-1],max_price[-1],volume[-1]] + final_features)
    assert len(indexes)==len(ret)
    return ret
        
        
def ask(message, err_message, function):
    while True:
        try:
            ret = raw_input(message)
            ret = function(ret)
            #ret = selectCol(filename,date_column,lambda x: datetime.strptime(x, format_))
            break
        except:
            print("select valid number please")
    return ret
    
class Dataset:
    
    def __init__(self, open_price, close_price, min_price, max_price, volume, commission_f):
        self.open_price = open_price
        self.close_price = close_price
        self.min_price = min_price
        self.max_price = max_price
        self.volume = volume
        self.commission_f = commission_f
        
    def indexise(self, indexes):
        open_price = []
        close_price = []
        min_price = []
        max_price = []
        volume = []
        
        for last_indx, next_indx in zip(indexes[:-1],indexes[1:]):
            open_price.append(self.open_price[last_indx])
            close_price.append(self.close_price[next_indx-1])
            min_price.append(min(self.min_price[last_indx:next_indx]))
            max_price.append(max(self.max_price[last_indx:next_indx]))
            volume.append(sum(self.volume[last_indx:next_indx]))
        
        open_price.append(self.open_price[-1])
        close_price.append(self.close_price[-1])
        min_price.append(self.min_price[-1])
        max_price.append(self.max_price[-1])
        volume.append(self.volume[-1])
        
        self.open_price = open_price
        self.close_price = close_price
        self.min_price = min_price
        self.max_price = max_price
        self.volume = volume
        
    def get_derivate_matrix(self):
        return (np.matrix([self.close_price]) - np.matrix([self.open_price])).T
        
    def get_commission_matrix(self):
        return np.matrix([map(self.commission_f, self.open_price)]).T

    def get_features_matrix(self, min_, max_, volume_):
        ret = []
        if min_:
            ret.append(self.min_price)
        if max_:
            ret.append(self.max_price)
        if volume_:
            ret.append(self.volume)
        return np.matrix(ret).T
        
        
        
        
if __name__ == "__main__":
    cont = True
    
    format_ = '%d.%m.%Y %H:%M:%S.000'
    
    first_cycle = True

    
    datasets = []
    """The shape of dataset in the code will be:
    datasets = [v1, v2, .. vn]
    where vi is the ith dataset and
    
    vi = [c1, c2, c3, c4, c5, c6, ... cn]
    
    where 
        c1 = open_price
        c2 = close_price
        c3 = min_price
        c4 = max_price
        c5 = volume
        c6.. cn = features
        """
    
    #select columns and files
    while cont:
        filename = raw_input("Select the csv file: ")
    
        #check path existence    
        if not os.path.isfile(filename):
            print "The file selected is not existing."
            continue
        
        printExampleFile(filename)
        
        if not raw_input("Is the following time format correct " + format_ + "? (Y | n)? ") in ['Y','y','yes',1,'1']:
            format_ = raw_input("Input the time format: ")
        
        dates = ask("Select the column of the timestamp (counting start by 0): ","select valid number please",lambda y: selectCol(filename,int(y),lambda x: datetime.strptime(x, format_)))        
        
        open_prices = ask("Select the column of the stock's open price (counting start by 0): ","select valid number please",lambda y: selectCol(filename, int(y), float))
        close_prices = ask("Select the column of the stock's close price (counting start by 0): ","select valid number please",lambda y: selectCol(filename, int(y), float))
        min_prices = ask("Select the column of the stock's min price (counting start by 0): ","select valid number please",lambda y: selectCol(filename, int(y), float))
        max_prices = ask("Select the column of the stock's max price (counting start by 0): ","select valid number please",lambda y: selectCol(filename, int(y), float))
        volumes = ask("Select the column of the stock's volume (counting start by 0): ","select valid number please",lambda y: selectCol(filename, int(y), float))
        
        min_commission = ask("Select the minimum commission: ","select valid commission please", float)
        max_commission = ask("Select the maximum commission: ","select valid commission please", float)
        per_commission = ask("Select the rate commission: ","select valid commission please", float)
        
        
        cont = raw_input("Would you like to insert another file? (Y | n)? ") in ['Y','y','yes',1,'1']
        
        #Syncronization block
        if first_cycle:
            last_dates = np.matrix(dates).T
            new_ds = Dataset(open_prices, close_prices, min_prices, max_prices, volumes, lambda x: max(min(x*per_commission, max_commission),min_commission) )
        else:
            index1, index2 = syncronize(last_dates,dates)
            for ds in datasets:
                ds.indexise(index1)
            new_ds = Dataset(open_prices, close_prices, min_prices, max_prices, volumes, lambda x: max(min(x*per_commission, max_commission),min_commission) )
            new_ds.indexise(index2)
            last_dates = last_dates[index1]
        datasets.append(new_ds)
    
    min_ = raw_input("Would you like to use min_price as feature? (Y | n)? ") in ['Y','y','yes',1,'1']
    max_ = raw_input("Would you like to use max_price as feature? (Y | n)? ") in ['Y','y','yes',1,'1']
    volume_ = raw_input("Would you like to use volume as feature? (Y | n)? ") in ['Y','y','yes',1,'1']
    timing_ = raw_input("Would you like to use the time duration? (Y | n)? ") in ['Y','y','yes',1,'1']

    n_rows = ask("How many rows would you like to skip (if you say 1, you would not skip any row, with 2 you'll skip 1 row out of 2, ..)?", "Please give a reasonable (numeric) answer",int)
    derivate_matrix = None
    commission_matrix = None
    
    indexes = range(0,last_dates.shape[0],n_rows)
    for ds in datasets:
        ds.indexise(indexes)
    last_dates = last_dates[indexes]
    
            
    init_features = False
    feature_matrix = None
    
    #if at least one of the features is enabled
    if min_ or max_ or volume_ or timing_:
        if timing_:
            diff_dates = last_dates[1:] - last_dates[:-1]
            for ds in datasets:
                ds.indexise(range(0,diff_dates.shape[0]))
            features_matrix = np.matrix(diff_dates)
            init_features = True
        if min_ or max_ or volume_:
            for ds in datasets:
                if not init_features:
                    features_matrix = ds.get_features_matrix(min_, max_, volume_)
                    init_features=True
                else:
                    print features_matrix.shape
                    print ds.get_features_matrix(min_,max_,volume_).shape
                    features_matrix = np.append(features_matrix, ds.get_features_matrix(min_, max_, volume_),axis=1)
    
        
    first_cycle=True
    for ds in datasets:
        if first_cycle:
            derivate_matrix = ds.get_derivate_matrix()
            commission_matrix = ds.get_commission_matrix()
            first_cycle=False
        else:
            derivate_matrix = np.append(derivate_matrix,ds.get_derivate_matrix(),axis=1)
            commission_matrix = np.append(commission_matrix, ds.get_commission_matrix(),axis=1)
            
    if min_ or max_ or volume_ or timing_:
        final_matrix = np.append(np.append(derivate_matrix,commission_matrix,axis=1),features_matrix,axis=1)
    else:        
        final_matrix = np.append(derivate_matrix,commission_matrix,axis=1)
        
    print "Your data has the shape: ", final_matrix.shape
    
    while True:
        try:
            time_series = raw_input("How long you would like to be your time series?: ")
            time_series = int(time_series)
            n_sample = final_matrix.shape[0]/time_series
            if n_sample>3:
                if not raw_input("You'll have " + str(n_sample)+" sample. \nIs it okay for you? (Y | n)? ") in ['Y','y','yes',1,'1']:
                    continue
                break
            else:
                print("Your series should be less than: " + str(final_matrix.shape[0]/3) + " .. so be nice and don't insert stupid values once again")
        except:
            print("select valid number dude")
            
    samples = np.array([final_matrix[0:time_series]])
    
    for i in range(1,n_sample):
        sample = np.array([final_matrix[time_series*i:time_series*(i+1)]])
        samples = np.append(samples, sample, axis=0)
    
    print "Now the shape of the data you have is: " ,samples.shape
    
    np.random.shuffle(samples)
    
    while True:
        try:
            n_train = raw_input("How many samples do you want in your train_set.npy?: ")
            n_train = int(n_train)
            if n_train < n_sample - 2:
                break
            else:
                print("Come on, you need at least 2 samples outside your trainset. At least one for the validation set and one for the test set. be nice.")
        except:
            print("select valid number")

    while True:
        try:
            n_val = raw_input("How many samples do you want in your validation_set.npy?: ")
            n_val = int(n_train)
            if n_val < n_sample - n_train - 1:
                break
            else:
                print("Come on, you need at least 1 sample outside your trainset and validation set.")
        except:
            print("select valid number")
        
    np.save("train_set",samples[:n_train])
    np.save("validation_set",samples[n_train:n_train+n_val])
    np.save("test_set",samples[n_train+n_val:])
    
    print "Finally datasets are done! you'll find train_set.npy and the others."
    print "Now help me to write also the configuration file. This file is made to describe what is the content of the dataset, so you can remember it.\nI'm asking you just a couple of minutes more, don't worry"