import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def class_name(cls):
	out=None
	labels = 'labels/labels.txt'
        with open(labels) as f:
		for (i, line) in enumerate(f):
                    tmp = [(t.strip()) for t in line.split()]
		    if cls==tmp[0]:
		    	out=tmp[2]
			break
	return out
def txt_to_csv(path,t):
    txt_list = []
    for txt_file in glob.glob(path + '/*.txt'):
        with open(txt_file) as f:
		for (i, line) in enumerate(f):
                    tmp = [int(t.strip()) for t in line.split()]
		    if t==0:
	                    value=((txt_file[:-3]+"jpg"),
		            int(tmp[5]),
		            int(tmp[6]),
		            class_name(str(tmp[0])),
		            int(tmp[1]),
		            int(tmp[2]),
		            int(tmp[3]),
		            int(tmp[4]))
		            txt_list.append(value)
		    else:
	                    value=((txt_file[:-3]+"jpg"),
		            int(tmp[5]),
		            int(tmp[6]),
		            class_name(str(tmp[0])),
		            int(tmp[1]),
		            int(tmp[2]),

		            int(tmp[3]),
		            int(tmp[4]))
		    	    txt_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    txt_df = pd.DataFrame(txt_list, columns=column_name)
    return txt_df


def main():
    image_path = os.path.join(os.getcwd(), 'labels/train')
    txt_df = txt_to_csv(image_path,0)
    txt_df.to_csv('data/train_labels.csv', index=None)
    print('Successfully converted train txt to csv.')
    image_path = os.path.join(os.getcwd(), 'labels/test')
    txt_df = txt_to_csv(image_path,1)
    txt_df.to_csv('data/test_labels.csv', index=None)
    print('Successfully converted test txt to csv.')


main()

