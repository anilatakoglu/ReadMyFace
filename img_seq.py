
# Firstly, we applied that class to implement CK++ dataset


import glob as gb
from shutil import copyfile

emotions_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  #indexed according to dataset labels
emotions_folders = gb.glob(
    "C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\emotions\\*")  # Returns a list of all folders with participant numbers

for x in emotions_folders:
    participant = "%s" % x[-4:]  # store current participant number
    for sessions in gb.glob("%s\\*" % x):
        for files in gb.glob("%s\\*.txt" % sessions):
            if (files is not None):
                current_session = files[55:-30]
                file = open(files, 'r')

                emotion = int(float(file.readline()))
                # get last image in sequence.It contains emotion
                sourcefile_emotion = gb.glob("C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\images\\%s\\%s\\*png" % (
                participant, current_session))[-1]

                # get first image in sequence.It contains neutral image of corresponding emotion
                sourcefile_neutral = gb.glob("C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\images\\%s\\%s\\*png" % (
                participant, current_session))[0]
                # Generate path to put neutral image
                dest_neut = "C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\selected_set\\neutral\\%s" % sourcefile_neutral[56:]
                # Do same for emotion containing image
                dest_emot = "C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\selected_set\\%s\\%s" % (
                emotions_list[emotion], sourcefile_emotion[56:])

                copyfile(sourcefile_neutral, dest_neut)  # Copy file
                copyfile(sourcefile_emotion, dest_emot)  # Copy file
            else:
                print "It's empty !"
