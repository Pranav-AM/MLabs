from main import *


imagedir = '../../data/mod-data'#'/content/drive/My Drive/MLabs/Project/activity_rec/mod-data/'
labeldir = '../../data/labels'#'/content/drive/My Drive/MLabs/Project/activity_rec/labels'
persondir = '../../data/persondata'#'/content/drive/My Drive/MLabs/Project/activity_rec/persondata'

'''
Change 3 to another folder in imagedir for predicting
'''
# --------------------------------------------
directory = os.path.join(imagedir, '3')
# --------------------------------------------


cursequence = Sequence()
cursequence.dirName = directory

imageList = sorted_alphanumeric(glob.glob(os.path.join(directory, '*.jpg')))
num = 0
imgdatalist = []
for image in imageList:
    curimage = Image()
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

features = generate_featuremap_lstm(cursequence)

#print(features)
all_features = []
for it in range(0, len(features)):
	all_features.append(features[it][1])

features = normalize_time_series(all_features)

model = load_trained_model()


#print(to_categorical_tensor(labels, 4))

test_pred = model.predict(convert_to_lstm_format(features))

label = test_pred.argmax(axis=1)

label_dict = {3:'idle', 1:'placing', 0:'picking', 2:'cart'}

print '\n', 40*'*'
print 'Predicted activity is: ', label_dict[label[0]]
print 40*'*'
