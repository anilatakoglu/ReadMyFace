import cv2, glob, random, math, numpy as np, dlib
from sklearn.svm import SVC



emotions = ["anger", "disgust", "happy", "neutral","surprise"]  # 8 emotion %60 - 5 emotion %90

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector() #haarscade filter
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #spape predictor
clf = SVC(kernel='linear', probability=True,tol=1e-3)

# face filters
face_detector1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_detector2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_detector3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_detector4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

def get_files(emotion):  # Get raining set
    files = glob.glob("C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\final_dataset\\%s\\*jpg" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 1.0)]
    return training


def get_landmarks(image): # creating landmarks on face
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)  # Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]  # get distance between each point and the central point in both axes
        ycentral = [(y - ymean) for y in ylist]

        if xlist[26] == xlist[29]:  # If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)

            anglerelative = (math.atan((z - ymean) / (w - xmean)) * 180 / math.pi) - anglenose

            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    if len(detections) < 1:
        landmarks_vectorised = "error"

    return landmarks_vectorised


def make_sets():
    training_data = []
    training_labels = []

    for emotion in emotions:
        training = get_files(emotion) #, prediction
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                training_data.append(landmarks_vectorised)  # append image array to training data list
                training_labels.append(emotions.index(emotion))



    return training_data, training_labels


if __name__== "__main__":
    accur_lin = []
    prediction_data = []
    for i in range(1, 2):
        print ("--> Cretaing set %s time" %i)  # Make sets by random sampling 80/20%
        training_data, training_labels = make_sets() #,prediction_data,prediction_labels

        npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
        npar_trainlabs = np.array(training_labels)
        print("--> Training SVM linear %s" %i )  # train SVM
        clf.fit(npar_train, npar_trainlabs)

        print "--> Got source image "

        files = glob.glob("testBorahanHoca.jpg")  # Get list of all images with emotion

        filename = "test"
        for f in files:
            frame = cv2.imread(f)  # Open image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

            # Detect face using 4 different HAAR filters
            face1 = face_detector1.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face2 = face_detector2.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face3 = face_detector3.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            face4 = face_detector4.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            # Go over detected faces, stop at first detected face, return empty if no face.
            if len(face1) == 1:
                facefeatures = face1
            elif len(face2) == 1:
                facefeatures == face2
            elif len(face3) == 1:
                facefeatures = face3
            elif len(face4) == 1:
                facefeatures = face4
            else:
                facefeatures = ""

            # Cut and save face
            for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
                #print "Number of found faces: %s" % f
                gray = gray[y:y + h, x:x + w]  # Cut the frame to size

                try:
                    out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                    cv2.imwrite("C:\\Users\\N550\\PycharmProjects\\ReadMyFace\\%s.jpg" % (filename),out)  # Write image

                except:
                    pass  # pass the file on error

        print "--> Analyzing source image to prediction"
        input_image = cv2.imread("test.jpg")
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        landmarks_vectorised = get_landmarks(clahe_image)
        if landmarks_vectorised == "error":
            pass
        else:
            prediction_data.append(landmarks_vectorised)

        print("--> Making prediction %s" %i)  # Use score() function to get accuracy
        npar_pred = np.array(prediction_data)

        # pred_lin = clf.score(npar_pred, prediction_labels)

        xx = clf.predict_proba(npar_pred)
        print "prediction_proba --> ", xx


        # print "linear: " ,pred_lin

        # accur_lin.append(pred_lin)  # Store accuracy in a list

        # 0 = anger, 1=disgust, 2=happy, 3=Neutral, 4=Surprise
        print "--------------------------------------------------------"
        print "Anger    :", xx.item(0)
        print "Disgust  :", xx.item(1)
        print "Happy    :", xx.item(2)
        print "Neutral  :", xx.item(3)
        print "Surprise :", xx.item(4)
        print "--------------------------------------------------------"


    # print("Mean value lin svm: %.3f" % np.mean(accur_lin))


