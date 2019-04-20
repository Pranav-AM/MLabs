'''
    To add folders:
        $ 1..100 | % {New-Item -Name ".\labels/$_" -ItemType 'Directory'}
    To add label.txt to all the folders(powershell in this script's directory):
        $ 1..100 | % { New-Item -Path "./labels/$_" -Name “label.txt” -ItemType file}


    MAKE SURE THERE ARE NO OTHER FILES IN THE activity_labels FOLDER
'''

import os
import glob
import re
'''
    directory structure
        |
        |- Activity_detection_annotator.py
        |- activity_labels
            |
            |- 1
                - label.txt
            |- 2
                - label.txt
            |- 3
                - label.txt
            etc
'''
        
label_dir = r'activity_labels'

label_files = glob.glob(label_dir + '/*/label.txt')
label_files = sorted(label_files, key=lambda w: int(re.search('([0-9]+)', w).group(1)))

folders = [item for item in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, item)) ]
folders = sorted(folders, key=lambda w: int(re.match('([0-9]+)', w).string))

label_dict = {'i':'idle', 'pl':'placing', 'pi':'picking', 'c':'cart'}

# for index, file in enumerate(label_files):
index = 0
while(index < len(label_files)):
    cmd = input("Enter 'y' to label folder {} or 'g' to goto another folder: ".format(folders[index]))
    
    if (cmd.lower() == 'y'):
        with open(label_files[index], mode='w') as f:
            label1 = input("Enter the left label for the folder {}(i/pl/pi/c): ".format(folders[index]))        
            label2 = input("Enter the right label for the folder {}: ".format(folders[index]))
            labels = [label_dict[label1], label_dict[label2]]

            f.write('\n'.join(labels))
            index += 1
            print('\n')
    elif(cmd.lower() == 'g'):
        goto = input("Enter the folder to goto: ")
        index = int(goto) - 1
        
