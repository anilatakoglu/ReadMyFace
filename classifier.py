import cv2
import glob as gb
import random
import numpy as np


# Emotion list
emojis = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
#  0 = neutral, 1 =anger, 2= contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
# Initialize fisher face classifier
fisherface = cv2.createFisherFaceRecognizer() #approximately 40% low eliminate this
data = {}


# Function defination to get file list, randomly shuffle it and split 67/33
def getFiles(emotion):
    files = gb.glob("C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\final_dataset\\%s\\*jpg" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.67)]  # get first 67% of file list
    prediction = files[-int(len(files) * 0.33):]  # get last 33% of file list
    return training, prediction

# make changes here and specify the emotion
def makeSet():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emojis:
        training, prediction = getFiles(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert to grayscale
            training_data.append(gray)  # append image array to training data list
            training_labels.append(emojis.index(emotion))

        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emojis.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


def runClassifier():
    training_data, training_labels, prediction_data, prediction_labels = makeSet()

    print "training fisher face classifier using the training data"
    print "size of training set is:", len(training_labels), "images"
    fisherface.train(training_data, np.asarray(training_labels))
    print "classification prediction"
    counter = 0
    right = 0
    wrong = 0
    for image in prediction_data:
        pred, conf = fisherface.predict(image)
        if pred == prediction_labels[counter]:
            right += 1
            #counter += 1
        else:
            #cv2.imwrite("C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\difficult\\%s_%s_%s.jpg" % (emojis[prediction_labels[counter]], emojis[pred], counter),image)
            wrong += 1
            #counter += 1
    return ((100 * right) / (right + wrong))


# Now run the classifier
metascore = []
for i in range(0, 5):
    right = runClassifier()
    print "-->",i,"got", right, "percent right!"
    print "---------------------------------------------"
    metascore.append(right)

print "\n\nend score:", np.mean(metascore), "percent right!"
print "!!! Finish !!!"
