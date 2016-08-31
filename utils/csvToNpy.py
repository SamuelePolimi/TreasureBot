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
    
if __name__ == "__main__":
    cont = True
    
    format_ = '%d.%m.%Y %H:%M:%S.000'
    
    first_cycle = True
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
        
        while True:
            try:
                date_column = raw_input("Select the column of the timestamp (counting start by 0): ")
                date_column = int(date_column)
                dates = selectCol(filename,date_column,lambda x: datetime.strptime(x, format_))
                break
            except:
                print("select valid number please")
        
        while True:
            try:
                prices_column = raw_input("Select the column of the stock's open price (counting start by 0): ")
                prices_column = int(prices_column)
                open_prices = selectCol(filename,prices_column,float)
                break
            except:
                print("select valid number please")
        
        while True:
            try:
                prices_column = raw_input("Select the column of the stock's close price (counting start by 0): ")
                prices_column = int(prices_column)
                close_prices = selectCol(filename,prices_column,float)
                break
            except:
                print("select valid number please")
                
        features = []
        
        next_col = raw_input("Would you like to select a feature column? (Y | n)? ") in ['Y','y','yes',1,'1']
        while next_col:
            try:
                feature_column = raw_input("Select the column of the feature: ")
                feature_column = int(feature_column)
                features.append(selectCol(filename,feature_column,float))
                if not raw_input("Would you like to select a feature column? (Y | n)? ") in ['Y','y','yes',1,'1']:
                    break
            except:
                print("select valid number please")
        
        while True:
            try:
                min_commission = raw_input("Select the minimum commission: ")
                min_commission = float(min_commission)
                break
            except:
                print("select valid commission please")
                
        while True:
            try:
                max_commission = raw_input("Select the maximum commission: ")
                max_commission = float(max_commission)
                break
            except:
                print("select valid commission please")   
          
        while True:
            try:
                per_commission = raw_input("Select the rate of commission (1 is 100%): ")
                per_commission = float(max_commission)
                break
            except:
                print("select valid commission please") 
                
        
        cont = raw_input("Would you like to insert another file? (Y | n)? ") in ['Y','y','yes',1,'1']
        
        #Syncronization block
        if first_cycle:
            last_dates = np.matrix(dates).T
            last_derivates = (np.matrix([close_prices]) - np.matrix([open_prices])).T     
            last_cost = np.matrix([map(
                                       lambda x: max(min(x*per_commission, max_commission),min_commission), open_prices)
                                       ]).T
            last_features = np.matrix(features).T
            first_cycle = False
        else:
            index1, index2 = syncronize(last_dates,dates)
            
            if not raw_input("Merging the last files, you'll match " + str(len(index1)) +" data points. \nWould you like to merge this file too? (Y | n)? ") in ['Y','y','yes',1,'1']:
                print("Last file discarted!")                
                continue
            last_dates = last_dates[index1]
            last_derivates = np.append(last_derivates[index1], (np.matrix([close_prices]) - np.matrix([open_prices])).T[index2], axis=1)
            last_cost = np.append(last_cost,
                                  np.matrix(
                                  [map(
                                       lambda x: max(min(x*per_commission, max_commission),min_commission), open_prices)
                                       ]).T,axis=1)
            if(len(features) > 0):
                if last_features.shape[0]> 0:
                    last_features = np.append(last_features[index1], np.matrix(features)[index2].T, axis=1)
                else:
                    last_features = np.matrix(features).T
                    
    
    final_matrix = np.append(np.append(last_derivates, last_cost,  axis=1), last_features, axis=1)
    
    print "Your data has the shape: ", final_matrix.shape
    
    while True:
        try:
            jrow = raw_input("Select how many rows would you like to jump: ")
            jrow = int(jrow)
            break
        except:
            print("select valid commission please") 
    
    index = range(0,final_matrix.shape[0]-1,jrow)
    final_matrix = final_matrix[index]
    last_dates = last_dates[index]
    
    print "Now your data has the shape: ", final_matrix.shape
    
    while True:
        try:
            time_series = raw_input("How long you would like to be you time series?: ")
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
                continue
            else:
                print("Come on, you need at least 2 samples outside your trainset. At least one for the validation set and one for the test set. be nice.")
        except:
            print("select valid number")

    while True:
        try:
            n_val = raw_input("How many samples do you want in your train_set.npy?: ")
            n_val = int(n_train)
            if n_val < n_sample - n_train - 1:
                continue
            else:
                print("Come on, you need at least 1 sample outside your trainset and validation set.")
        except:
            print("select valid number")
        
    np.save("train_set",samples[:n_train])
    np.save("validation_set",samples[n_train:n_train+n_val])
    np.save("test_set",samples[n_train+n_val:])
    
    print "Finally datasets are done! you'll find train_set.npy and the others."
    print "Now help me to write also the configuration file. This file is made to describe what is the content of the dataset, so you can remember it.\nI'm asking you just a couple of minutes more, don't worry"