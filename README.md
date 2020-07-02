# TATA_Innoverse_Solverhunt8
This is a submission for the tata innoverse solverhunt8 competition.

## Face-Mask detection and Social-Distancing system

  the main objective of this system is to alert people 
    1. who are no wearing mask at all
    2. who are not wearing there mask properly
    3. who are not maintaing social distancing

## Technologies
  this is created using :                          
      * Python : 3.68                          
      * OpenCV : 4.2.0                       
      * Keras : 1.0.8(application) & 1.1.0(preprocessing)                             
      * Tensorflow : 2.1.0  ......
     
     
     
     
This Project is done in 4 steps:                                        
    1. developing dataset for mask detection                                     
    2. building and training mask detection model                               
    3. loading yolo_v3 model, mask-detector model and predictng people with mask and also without                                 
    4. calculating euclidean distance and mapping them
    
input image |predicted image:
----------------------------|----------------------------------------
<img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/example_images/example_05.jpg" width=350 height=250>              |            <img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/predicted_images/example_05_predicted.jpg" width=350 height=250 >

input image |predicted image:
----------------------------|----------------------------------------
<img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/example_images/example_02.png" width=250 height=250>              |            <img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/predicted_images/pred_example_02.png" width=250 height=250 >

#### it provides best results when the camera is placed above a person's height and 45* to the face of that person like security camera
#### it can also be used against videos or real time video for processing (requires good performance device)
       
### step 1:
it is  done by:                                         
  Taking normal images of faces and
  then creating a custom computer vision Python script to add face masks to them, thereby creating an artificial        dataset which is also applicable for real-world application.                                
  this is done by applying facial landmarking system where it locates faces in that frame .                          
  From there, we apply face detection to compute the bounding box location of the face in the image.                                
  Once we know where in the image the face is, we can extract the face Region of Interest (ROI) to locate each facial structures like nose, eyes ...                                                          
  Then using an image of a mask and also points of nose, chin and mouth, an image of a person is created wearing a mask. Repeating this process a datset of people wearing mask are created                             
  ![refer](https://github.com/prajnasb/observations) for more information
    
Example :

<img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/training%20mask%20detector/dataset/without_mask/158.jpg" width=200 height=250>              |            <img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/training%20mask%20detector/dataset/with_mask/158-with-mask.jpg" width=200 height=250 >

### step 2:
Taking a mobilenetv2 model is taken aand initaize the hidden layers to non-trainable layers and ading some more dense layers a model is created(transfer learning)
train_mask_detector.py: Accepts our input dataset and trains the mobilenet model to create our mask_detector.model. A training history plot.png containing accuracy/loss curves is also produced.

### step 3:
download a yolo_v3 model from ![darknet](https://pjreddie.com/darknet/)                        
load it using opencv's readnetfromdarknet method and load mask-detector model using keras-loadmodel .                         
Then predict the people with yolo and predict face of people using caffemodel and also predict the face having mask and not having it.

### step 4:
calculate the centroid of the predicted people and calculate the euclidean diatance between them.            
if the distanc e is less than threshold disatnce determined in configurations.py file then treat them as violators of social distancing rule.                               
put bonding boxes for social distance violators and non violators and also for the faces wearing mask and not wearing.                                             
calsulate social distance violators and mask wearing violators and put them on the frame.                                    
here are more examples:

<img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/predicted_images/example_01_predicted.png" width=300 height=250>| <img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/predicted_images/images_predicted.jpg" width=300 height=250> 
<img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/predicted_images/example_04_predicted.jpg" width=400 height=250> |
<img src="https://github.com/yashasps/tata_innoverse_solverhunt8/blob/master/predicted_images/pred_onemask.jpg" width=400 height=250>    
