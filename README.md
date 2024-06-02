# POSM analysis
The project of ITMO x Nexign about detecting priority, parity and disparity of POSM.
---
## Project info files
Task description available in the [PDF](/task-description.pdf).

The architecture of system available in the [folder](system_architecture/).

Project requirements stored in the [requirements.txt](/src/requirements.txt).

The final presentation of project available in the [PDF](/final_presentation.pdf) it's include:
 - Problem statement
 - Proposed solution
 - System interface
 - System architecture
 - Software stack
 - Metrics for ML models (instance segmentation model, classification model)
 - Overall accuracy of system prediction
 - Project roadmap
---
## Brief description of system operation
The task of the system is to analyze the competition of mobile operators by the number of advertising materials (POSM) at a certain point of sale. The comparison should be made for MegaFon and Yota operators in relation to BeeLine, Tele2 and MTS operators. As a final evaluation it is required to assign priority, parity, disparity for MegaFon and Yota to each outlet.

Initially, a dataset with objects marked for segmentation was proposed for the project (the markup was performed using the Supervisory service). The objects had 3 classes: *MegaFon*, *Yota*, *other operators*. 

The developed system consists of 3 main components.
1. Backend server
2. Segmentation model server
3. Classification model server

All components are separate Flask servers and realize communication between each other using HTTP requests. 
### Backend server
The backend server receives a path to an Excel file from the client. Each row of the file contains the following information about a point of sales (POS): pos ID, photos of each POSM placement zone, and other details.

Next, the backend reads this Excel file and processes all the points of sales. For each POS, the photos of each placement zone are segmented using the segment server microservice. Then, for the "other" category, the real mobile operator is determined using the classifier microservice. Finally, the area occupied by POSM is calculated for each operator accordingly. Based on this area, parity, disparity, or priority is assigned for each POS.

The client receives from the server a path to an Excel file that contains information about the competitiveness of each POS.
### Segmentation model server
A pre-trained YOLOv8n-seg was taken as the model for the instance segmentation task, and then it was pre-trained on the original dataset for 150 epochs. 

This module accepts POST requests, where the input is the URL to the image (for which segmentation is required), and outputs the segmentation results of the *ultralytics.engine.results.Results class*.

Achieved metrics:
 - mAP50-95: 0.68 (where for MegaFon: 0.8, for Yota: 0.76, for others: 0.47)
 - mAP50: 0.81
 - Recall: 0.86
 - Precision: 0.85
### Classification model server
A classification model is used to separate operators into separate classes. It takes as input cropped images (using the *contourArea()* method of the OpenCV library) containing advertising material of an operator from the *other operators* class. The output is the class of operator: Tele2, MTS or BeeLine. 

VGG16 model pre-trained on ImageNet dataset was used as a model. For our task we added additional layers: 
1. *Flatten()*
2. *Dense(512, activation='relu')*
3. *Dropout(0.5)*
4. *Dense(3, activation='softmax')*
The parameters for training were chosen as follows: *optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']*. The model was trained for 10 epochs.

Achieved metrics:
 - Accuracy: 0.83
 - Precision: 0.74
 - Recall: 0.80
 - F1-score: 0.77
