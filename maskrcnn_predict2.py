# USAGE
# python maskrcnn_predict.py --weights mask_rcnn_coco.h5 --labels coco_labels.txt --image images/30th_birthday.jpg

# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import random
# Import the required module for text  
# to speech conversion 
from gtts import gTTS 
  
# This module is imported so that we can  
# play the converted audio 
import os 
  

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
	help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=True,
	help="path to class labels file")
ap.add_argument("-i", "--image", required=True,
	help="path to input image to apply Mask R-CNN to")
args = vars(ap.parse_args())

# load the class label names from disk, one label per line
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")

# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"

	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = len(CLASS_NAMES)

# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)

# perform a forward pass of the network to obtain the results
print("[INFO] making predictions with Mask R-CNN...")
r = model.detect([image], verbose=1)[0]

# loop over of the detected object's bounding boxes and masks
for i in range(0, r["rois"].shape[0]):
	# extract the class ID and mask for the current detection, then
	# grab the color to visualize the mask (in BGR format)
	classID = r["class_ids"][i]
	#print(classID)
	mask = r["masks"][:, :, i]
	color = COLORS[classID][::-1]

	# visualize the pixel-wise mask of the object
	image = visualize.apply_mask(image, mask, color, alpha=0.5)

# convert the image back to BGR so we can use OpenCV's drawing
# functions
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
dic = {"BG":0, "person":0, "bicycle":0,"car":0,"motorcycle":0,"airplane":0,"bus":0,"train":0,"truck":0,"boat":0,"traffic light":0,"fire hydrant":0,"stop sign":0,"parking meter":0,"bench":0,"bird":0,"cat":0,"dog":0,"horse":0,"sheep":0,"cow":0,"elephant":0,"bear":0,"zebra":0,"giraffe":0,"backpack":0,"umbrella":0,"handbag":0,"tie":0,"suitcase":0,"frisbee":0,"skis":0,"snowboard":0,"sports ball":0,"kite":0,"baseball bat":0,"baseball glove":0,"skateboard":0,"surfboard":0,"tennis racket":0,"bottle":0,"wine glass":0,"cup":0,"fork":0,"knife":0,"spoon":0,"bowl":0,"banana":0,"apple":0,"sandwich":0,"orange":0,"broccoli":0,"carrot":0,"hot dog":0,"pizza":0,"donut":0,"cake":0,"chair":0,"couch":0,"potted plant":0,"bed":0,"dining table":0,"toilet":0,"tv":0,"laptop":0,"mouse":0,"remote":0,"keyboard":0,"cell phone":0,"microwave":0,"oven":0,"toaster":0,"sink":0,"refrigerator":0,"book":0,"clock":0,"vase":0,"scissors":0,"teddy bear":0,"hair drier":0,"toothbrush":0}
k = 0
mytext='In front of you'
# loop over the predicted scores and class labels
for i in range(0, len(r["scores"])):
	# extract the bounding box information, class ID, label, predicted
	# probability, and visualization color
	(startY, startX, endY, endX) = r["rois"][i]
	classID = r["class_ids"][i]
	label = CLASS_NAMES[classID]
	dic[label]=dic[label]+1
	k=k+1
	
	

	score = r["scores"][i]
	color = [int(c) for c in np.array(COLORS[classID]) * 255]

	# draw the bounding box, class label, and score of the object
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	text = "{}: {:.3f}".format(label, score)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, color, 2)

# show the output image
cv2.imshow("Output", image)
b=0
l=0
distance = {0:'',1:' not so far',2:' further',3:' also,',4:' it looks like'}
prepos = {0:'there are , ',1:' , ',2:' , ',3:' , ',4:'and ,'}
# The text that you want to convert to audio 
for cle in dic.keys():
    if dic[cle] == 1:
            mytext = mytext + distance[l%5] + prepos[l%5]+str(dic[cle])+ str(cle)+" ."
            l=l+1
            b=1
    if dic[cle] > 1:
            mytext = mytext+ distance[l%5] + prepos[l%5]+str(dic[cle])+ str(cle)+"s ."
            l=l+1
            b=1
    
#Prudence
if b == 1:
        mytext = mytext +" Wait, you should pass carefully"
if b == 0:
        mytext = mytext +" You can pass."
# Language in which you want to convert 
language = 'en'
print(mytext)
  
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file named 
# welcome  
myobj.save("welcome.mp3") 
  
# Playing the converted file 
os.system("welcome.mp3") 
cv2.waitKey()