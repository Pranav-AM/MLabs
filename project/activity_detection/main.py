#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import glob
import re
import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
import ConfigParser

config = ConfigParser.ConfigParser()
config.read('../../config.ini')

FRAME_PATH = '../' + config.get("DEFAULT", "FRAME_PATH")
OBJECT_PATH = '../' + config.get("DEFAULT", "OBJECT_PATH")
POSE_PATH = '../' + config.get("DEFAULT", "POSE_PATH")
ACTIVITY_PATH = '../' + config.get("DEFAULT", "ACTIVITY_PATH")
INVENTORY_PATH = '../' + config.get("DEFAULT", "INVENTORY_PATH")
IMAGE_PATH = '../' + config.get("DEFAULT", "IMAGE_PATH")

SAVED_MODEL = '../' + config.get("ACTIVITY-DETECTION", "SAVED_MODEL")

TRAIN_MODEL = config.get("ACTIVITY-DETECTION", "TRAIN_MODEL").lower()

class Object:
    def __init__(self):
        self.type = 0
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.xres = 0
        self.yres = 0
        
class Image:
    def __init__(self):
        self.number = ''
        self.location = ''
        self.objectList = []
        self.personList = []
        self.labelPath = ''
        self.imgPath = ''
        self.labelList = []
    
class Sequence:
    def __init__(self):
        self.imageDataList = []
        self.label = []
        self.dirName = ''

class Person:
    def __init__(self):
        self.name = ""
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.jointLocations = []
        # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
        # i have taken only 18 locations according to the parse_pickle script.
        #access via 2) index location
        self.pickedUpItems = []
        self.Wallet = 0
        self.inBagItems = []

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def calculateDistance(obj1,obj2):
    if obj1[0] == -1 or obj1[1] == -1 or obj2[0] == -1 or obj2[1] == -1:
        return -1   #return a large distance.
    return math.sqrt( pow(obj1[0]-obj2[0],2) + pow(obj1[1]-obj2[1],2))
    
def calculateCentroid(obj):
    lis = []
    lis.append((float(obj.xtop) + float(obj.xbot))/2)
    lis.append((float(obj.ytop) + float(obj.ybot))/2)
    return lis

def get_bag_and_other_objects(objects):
    bags = []
    others = []
    for item in objects:
        obj = Object()
        if type(item) != type(obj):
            print("function get_bag_and_other_objects: erroneous item type")
            print(type(item))
            print(type(obj))
        if item.type == 8:
            bags.append(item)
        else:
            others.append(item)
    return bags,others
    
def get_closest_bag(objects,person):
    #adds only the closest bag to all the objects and proceeds.
    bags, others = get_bag_and_other_objects(objects)
    min_dist = 999999999
    closest_bag = ''
    for item in bags:
        ans = calculateDistance(calculateCentroid(item), calculateCentroid(person))
        if ans < min_dist:
            min_dist = ans
            closest_bag = item
    if closest_bag == '':
        return []
    else:
        others.append(closest_bag)
    return others


def fill_in_missing_objects(modobjects):
    '''
    First finds out the objects in the modobjects and creates a Object instance for
    all the object types not present in modobjects
    '''
    seenobjects= {}
    for obj in modobjects:
        if (obj.type - 1)  not in seenobjects.keys():
            seenobjects[obj.type - 1] = 1
    for i in range(8):
        if i  not in seenobjects.keys():
            nullobj = Object()
            nullobj.type= i + 1
            modobjects.append(nullobj)
            seenobjects[i] = 1
    return modobjects


def generate_person_object_locations_prevframes(sequence):
    '''
        Process a sequence of image frames and returns (objects, person) pair in each frame for each person in any frame
        
        Input: Sequence object
        Output: Dictionary with keys as person names and values a list of (objects, person) tuples for each frame
    '''
    person_set = {}
    # person_set is a dictionary mapping person name to the list of tuples (object, person) over the past 5 frames.
    # person_set["karthik"] = [(objects,person) , (objects,person) ,(objects,person) , (objects,person), (objects,person)]
    #there might be missing objects but the person is in all the frames with default values added in.
    idx = 0
    if len(sequence.imageDataList) != 5:
        print(" Maintain the length of sequence as 5")
        print(sequence.dirName)
            
    '''
        Populates the person_set with (idx, objects, person) triplets
        idx - Frame number
        objects - List of Object instances(all objects + bag corresponding to the person)
        person - Person instance

        * Frames in which a person is not present will have its 'objects' list as empty list 
          for that person 

    '''
    for item in sequence.imageDataList:
        idx += 1
	#print(idx, ' ', item)
 
        objects = item.objectList
        persons = item.personList
        for person in persons:
	    #print("Person ", person.name)
            objectsmod = get_closest_bag(objects,person)
            objectsmod = fill_in_missing_objects(objectsmod)
            if person.name in person_set.keys():
                lis = person_set[person.name]
                last = lis[-1]
                previdx = last[0]
                # happens if the person wasn't in the initial frames
                # add null values for those frames 
                while previdx != idx-1:
                    per = Person()
                    per.name = person.name
                    lis.append((previdx + 1,[],per))
                    previdx += 1
                lis.append((idx,objectsmod,person))
                person_set[person.name] = lis
            else:
                lis = []
                previdx = 0
                # happens if the person wasn't in the initial frames
                # add null values for those frames 
                while previdx != idx-1:
                    per = Person()
                    per.name = person.name
                    lis.append((previdx + 1,[],per))
                    previdx += 1
                lis.append((idx,objectsmod,person))
                person_set[person.name] = lis
    # accounts for missing persons in later frames
    for item in person_set.keys():
	#print('item', item)
        lis = person_set[item]
        last = lis[-1]
        previdx = last[0]
        while previdx <5:
            per = Person()
            per.name = item
            lis.append((previdx + 1,[],per))
            previdx += 1
    
    '''
        Repopulate the person_set dictionary with only (objects, person) pairs
    '''
    # get the (objects, person) pair
    for item in person_set.keys():
	#print('item1', item)
        lis = person_set[item]
        newlis = []
        for it in lis:
            newlis.append( ( it[1] , it[2] ) )
        #print(len(newlis))
        person_set[item] = newlis
        
    #also filters with only the bag closest to the person added to the objects corresponding to that person.
    # now for each person you have a the set of object locations and pose locations over the past 5 frames.
    return person_set


def generate_featuremap_lstm(sequence):
    '''
        Process a sequence of images and returns feature vectors corresponding to each person in any
        of the image
        
        Input:
            Sequence object
            
        Output:
            List of (person_name, features) tuple
            
    '''
    #output: generates each feature over a time frame individually.
             #feature1 [x x1 x2 x3 x4]
             #feature2 [y y1 y2 y3 y4]
             # feature here means the locations of objects and joint locations.
    person_set = generate_person_object_locations_prevframes(sequence)
    feature_list = []
    
    for item in sorted(person_set.keys()):
        name = item # person name
        
        item = person_set[item] # list of (objects, person)
        
        # usually 8 objects (1 + 7 objects ) and 18 joint locations == total
        features = [] # over frames
        
        # one each for a frame (we have 5 frames in a sequence)
        for i in range(5):
            features.append([])
            
        idx = 0
        for frameitem in item:  
            '''
                frameitem:
                    (objects, person)
                    
                iterating over frames and appending features
                features:
                    * object location([x, y] coordinate of centroid of bounding box) for objects (8nos-7+1)
                    * [x, y] cordinate for each joint location
                      
                if 'objects' list is empty, add first 8 features i.e, location of
                8 objects as [-1, -1]
                
                if 'person.jointLocations' list is empty, add 18 features i.e, location of
                18 joint locations as [-1, -1]
                
                features list is extended for each frameitem thus giving one large feature list
                for a person i.e, 26 * 2 * 5 = 260 dimensional feature vector
             
                >>> features[idx] = []
                >>> features[idx].extend([1, 2])
                [1, 2]
                >>> features[idx].extend([5, 6])
                [1, 2, 5, 6]
                >>> features[idx].append([7, 8])
                [1, 2, 5, 6, [7, 8]]
                
            ''' 
            if len(frameitem[0]) == 0:
                for i in range(8):
                    features[idx].extend([-1, -1])
            else:
                for obj in frameitem[0]:
                    features[idx].extend(calculateCentroid(obj))
            
            joint_locations = frameitem[1].jointLocations
            if len(joint_locations) == 0:
                for i in range(8,26):
                    features[idx].extend([ -1, -1 ])
            else:
                j = 0
                for i in range(8,26):
                    features[idx].extend([ joint_locations[j],joint_locations[j+1] ])    
                    j += 2
            idx += 1
        
        feature_list.append((name,features))
    return feature_list
         
def generate_data(all_sequences):
    all_features = []
    all_labels = []
    for seq in all_sequences:
        #print('seq', seq)
        features = generate_featuremap_lstm(seq)
        #print(len(features))
        labels = seq.label
        #print(labels)
        if(len(labels) == len(features)):
            for it in range(0,len(labels)):
                #print(len(features[it][1]))
                #if len(features[it][1]) == 150: #calculated from the constant value
                all_features.append(features[it][1])
                all_labels.append(labels[it])
    #print(all_features)
    return all_features,all_labels

def discretize(labels):
    newlabels=[]
    for item in labels:
        #print(item)
        if item == "picking":
            newlabels.append(0)
        elif item == "placing":
            newlabels.append(1)
        elif item == "cart":
            newlabels.append(2)
        elif item == "idle":
            newlabels.append(3)
        else:
            print("ERROR when discretizing.")
    return newlabels


def loadDir():
    all_sequences = []
    ##############################################################################################
    imagedir = ACTIVITY_PATH
    labeldir = OBJECT_PATH
    persondir = POSE_PATH
    ###############################################################################################
    dirList = sorted_alphanumeric(glob.glob(os.path.join(imagedir, '**')))
    if len(dirList) == 0:
        print("No directories found")
    
    # iterate over all the activity detection data folders(5 images + label file)
    for directory in dirList:
        
        cursequence = Sequence()
        cursequence.dirName = directory
        try:
            targetlabelfile = directory + "/label.txt"
            targetlabel = open(targetlabelfile, 'r')
        except IOError:
            print("Error: can\'t find file or read data from label file", directory)
        tar_labels=[]
        for eachline in targetlabel:
            eachline = eachline.rstrip('\n')
            tar_labels.append(eachline)
        cursequence.label = tar_labels
        #print('cursequence.label', cursequence.label)
        imageList = sorted_alphanumeric(glob.glob(os.path.join(directory, '*.jpg')))
        num = 0
        imgdatalist = []
        for image in imageList:
            curimage = Image()
            
            curimage.persons = len(tar_labels)  #REMOVE THIS ONCE YOU GET JOINT LOCATIONS
            curimage.path = image
            #print('curimage.path', curimage.path)
            curimage.number = num
            num = num + 1  
            segments = image.split('/')
            imgname = segments[len(segments) - 1]
            imgsegments = imgname.split('.')
            imgnumber = imgsegments[0]
            labelname = imgnumber + ".txt"
            
            
            try:
                location = labeldir +'/'+ labelname
                #print('location', location)
                curimage.labelPath = location
                labelFile = open(location , 'r')
                objects = []
                for line in labelFile:
                    curobject = Object()
                    input_numbers = line.split(' ')
                    if len(input_numbers) == 7:
                        curobject.type = int(input_numbers[0])
                        curobject.xtop = int(input_numbers[1])
                        curobject.ytop = int(input_numbers[2])
                        curobject.xbot = int(input_numbers[3])
                        curobject.ybot = int(input_numbers[4])
                        curobject.xres = int(input_numbers[5])
                        curobject.yres = int(input_numbers[6])
                    else:
                        print("Read ERROR len not 7 " , labelname)
                    objects.append(curobject)
                curimage.objectList = objects
            except IOError:
               print("Error: can\'t find file or read data")
            
            
            persons = []
            #write code to get the joint locations
            try:
                location = persondir +'/'+ labelname
                personFile = open(location , 'r')
                for line in personFile:
                    curperson= Person()
                    input_numbers = line.split(' ')
                    #print( input_numbers)
                    if len(input_numbers) == 42:   #as it inlcudes '\n' at the end. usually its 41
                        curperson.name = input_numbers[0]
                        curperson.xtop = int(input_numbers[1])
                        curperson.ytop = int(input_numbers[2])
                        curperson.xbot = int(input_numbers[3])
                        curperson.ybot = int(input_numbers[4])
                        for idx in range(5,41):
                            curperson.jointLocations.append(int(input_numbers[idx]))
                    else:
                        print("Read ERROR len not 41 " , labelname)
                        
                    #print('curperson.name', curperson.name)
                    
                    persons.append(curperson)
                curimage.personList = persons
            except IOError:
               print("Error: can\'t find file or read data " , labelname)
            
            imgdatalist.append(curimage)
        cursequence.imageDataList = imgdatalist  
        #print(len(cursequence.imageDataList[0].personList), cursequence.imageDataList[0].personList[0].name)
        all_sequences.append(cursequence)
    return all_sequences
        
#sequences = loadDir()

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

from keras.models import load_model

#from sklearn.metrics import mean_squared_error
#from sklearn.utils import shuffle

import time
from numpy import array

def convert_to_lstm_format(all_features):
        npfeatures = array(all_features) 	
        features = npfeatures.reshape(len(all_features), 5, 52)
        print(features.shape)
        return npfeatures      

def normalize_time_series(all_features):
    new_list = []

    #normalization
    for item in all_features:
        for item2 in item:
            new_list.append(item2)

    from sklearn.preprocessing import MinMaxScaler  
    scaler = MinMaxScaler(feature_range = (0, 1))
    all_features_scaled = scaler.fit_transform(new_list)  
    
    final_list = []
    lis= []
    #converting back to time series
    for idx in range(len(all_features_scaled)):
        lis.append(all_features_scaled[idx])
        if (idx + 1) % 5 ==0:
            final_list.append(lis)
            lis=[]
    print("Normalized")
    return final_list


def fit_model(train_X, train_Y, no_of_frames = 5):	
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    
    start = time.time()
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(no_of_frames,52)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=150, batch_size=5, verbose=False)#verbose=2)   
    
    end = time.time()
    print('Time: ', end - start)
    return(model)
    
def train_model(train_d,train_l):
    model = fit_model(train_d, train_l)
    ####################################################################################################
    model.save(SAVED_MODEL)
    ####################################################################################################
    return model

def load_trained_model():
    ######################################################################################
    model = load_model(SAVED_MODEL)
    ######################################################################################
    return model

def to_categorical_tensor( x, n_cls ) :
      y = keras.utils.to_categorical( x, num_classes = n_cls )
      return y
    
#load_trained_model()



import keras
def train_test_split(features, labels):
    data = features
    labels = np.asarray(discretize(labels))
    ######################################
    kf = KFold(n_splits=4, random_state=42) # Added random state
    '''
        for train_index, test_index in kf.split(data):
           print("TRAIN:", train_index, "TEST:", test_index)
    '''
    
    ######################################
    conf_mat=np.full((4,4),0)
    scores=[]
    for train_index, test_index in kf.split(data):
        train_d = []
        train_l = []
        test_d = []
        test_l = []
        for it in range(0,len(features)):
            if it in train_index:
                train_d.append(features[it])
                train_l.append(labels[it])
            if it in test_index:
                test_d.append(features[it])
                test_l.append(labels[it])
        print("done")

        train_d = convert_to_lstm_format(train_d)
        test_d = convert_to_lstm_format(test_d)

        ##one hot encoding must be performed before
        train_l = to_categorical_tensor(train_l,4)
        test_l = to_categorical_tensor(test_l,4)
        
   	#print(TRAIN_MODEL)
	if(TRAIN_MODEL == 'true'):
            ## training
	    print 'training model'
            model = train_model(train_d, train_l)
	else:
            ## load pretrained model
	    #print 'pretrained model'
            model = load_trained_model() 
	
    	# evaluate model
        test_pred = model.predict(test_d)
        #print(test_pred)
        #print('\n')
        '''
          when any of the labels aren't in the test data, the confusion_matrix will
          be 3x3, whereas we require all the confusion matrices to be 4x4
          * check which label is missing
        '''
        def get_missing_label_indices():
          label_indices = [i for i in range(4)]
          # get the indices present in test_l or test_pred
          test_indices = list(set(np.concatenate([test_l.argmax(axis=1), test_pred.argmax(axis=1)] )))
          for index in test_indices:
            label_indices.remove(index)
          
          return label_indices
        
        #print(np.vstack([test_l.argmax(axis=1), test_pred.argmax(axis=1)]))
        
        cur_matrix = confusion_matrix(test_l.argmax(axis=1), test_pred.argmax(axis=1))
        
        missing_label_indices = get_missing_label_indices()
        
        
        #print('original cur_matrix', cur_matrix)

        if(len(missing_label_indices) != 0):
          #print("Missing labels: ", missing_label_indices)
          for index in missing_label_indices:
            # add new zero row at index position(add zeros to columns at index position)
            cur_matrix = np.insert(cur_matrix, index, np.zeros(cur_matrix.shape[1]), 0)
            # add new zero column at index position
            cur_matrix = np.insert(cur_matrix, index, np.zeros(cur_matrix.shape[0]), 1)
          
        #print('constructed cur_matrix',  cur_matrix)
        print('\n')
        '''
        print(train_d.shape, train_l.shape)
        print(train_l, '\n', test_l, '\n')

        print(test_l.shape, type(test_l))
        '''
        conf_mat=np.add(conf_mat, cur_matrix)
        
        #print(cur_matrix)
        _,accuracytrain = model.evaluate(train_d, train_l, batch_size=20, verbose=False)#, verbose=2)
        _,accuracytest = model.evaluate(test_d, test_l, batch_size=20, verbose=False)#, verbose=2)
        print(accuracytrain, accuracytest, '\n')
        scores.append(accuracytest*100)
    scores=np.asarray(scores)    
    print(scores.mean())
    print(conf_mat)


if __name__ == "__main__":
    all_sequences = loadDir()   

    all_features, all_labels = generate_data(all_sequences)

    all_features = normalize_time_series(all_features)

    train_test_split(all_features, all_labels)

'''
l = [0, 3, 2, 0, 2, 0, 0, 3]
pred = [3, 0, 2, 0, 2, 3, 3, 3]

def get_missing_label_indices():
          label_indices = [i for i in range(4)]
          # get the indices present in test_l or test_pred
          test_indices = list(set(np.concatenate([l, pred] )))
          for index in test_indices:
            label_indices.remove(index)
          
          return label_indices

missing_label_indices = get_missing_label_indices() 

cur_matrix = confusion_matrix(l, pred)
print(cur_matrix)

if(len(missing_label_indices) != 0):
          for index in missing_label_indices:
            # add new zero row at index position(add zeros to columns at index position)
            cur_matrix = np.insert(cur_matrix, index, np.zeros(cur_matrix.shape[1]), 0)
            # add new zero column at index position
            cur_matrix = np.insert(cur_matrix, index, np.zeros(cur_matrix.shape[0]), 1)

print(cur_matrix)
'''





