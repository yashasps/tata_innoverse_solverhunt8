
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from config_detect_folder import configurations as config
from config_detect_folder.detection import detect_people
from scipy.spatial import distance as dist
import argparse
import imutils
import numpy as np
import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True,
	help="path to input image")
ap.add_argument("-m", "--model_fold", required=True,
	help="path to (models, weights, prototext) folder")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["model_fold"], "deploy.prototxt"])
weightsPath = 	os.path.sep.join([args["model_fold"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
model = load_model(os.path.sep.join([args["model_fold"], "mask_detector.model"]))

path = args["video"]

labelsPath = os.path.sep.join([args["model_fold"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([args["model_fold"], "yolov3.weights"])
configPath = os.path.sep.join([args["model_fold"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
ynet = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = ynet.getLayerNames()
ln = [ln[i[0] - 1] for i in ynet.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(path)

savepath = path.split(".")[0] + "_prediced.mp4"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(savepath, fourcc, 25,
						 (720, 480), True)
while True:
	count = 0
	(ret, frame) = vs.read()
	if ret == False:break
	image = imutils.resize(frame, width=700)
	orig = image.copy()
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()



	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			(mask, withoutMask) = model.predict(face)[0]


			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			if label == "No Mask":count += 1
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


			cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)



	results = detect_people(image, ynet, ln, personIdx=LABELS.index("person"))

	violate = set()

	if len(results) >= 2:

		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):

				if D[i, j] < config.MIN_DISTANCE:

					violate.add(i)
					violate.add(j)

	for (i, (prob, bbox, centroid)) in enumerate(results):

		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)


		if i in violate:
			color = (0, 0, 255)


		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
		cv2.circle(image, (cX, cY), 5, color, 1)


	text = "Social Distancing Violations: {}".format(len(violate))
	masktext = f"mask wearing violators: {count}"
	cv2.putText(image, text, (10, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
	cv2.putText(image, masktext, (10, image.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	cv2.imshow("frame", image)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break


	#writer.write(frame)