# Cervical Spine Fracture Detection and Localization

### Group : Naman Raghuvanshi (nvr5386@psu.edu), Siddharth Rayabharam (nqr5356@psu.edu) and Sumant Suryawanshi (szs7220@psu.edu)

## Abstract

### Purpose

To develop a deep learning model for the detection and localization of cervical spine fractures in the axial CT scans.

### Methods

We use the dataset consisting of cervical spine CT scans provided by the Radiological Society of North America (RSNA). The dataset consists of 3000 studies of individual patients for the training and testing combined. Out of these 3000 patient studies, 83 of these studies also contains segmentation data. Additionally, for x studies the dataset contains bounding box coordinates data. The dataset popu- lation is split into 90% for training and 10% for validation. We use EfficientNetV2[1] to learn the segmentation, YOLOv5 model for detect- ing the fractures and drawing bounding boxes around the fracture area.

### Results

The segmentation model predicts the vertebrae label with the accuracy of 95.13% whereas the fracture detection model predicts fractures with an accuracy of 94%. The frac- ture detection model also draws bounding boxes localizing the fractures and provides the objectiveness of the prediction.

## Introduction

Cervical spine injury is very common injury with more than 3 million cases per year that are being evaluated for cervical spine injury in North America[2]. In United States, more than 1 million patients with blunt force injury are sus- pected to suffer cervical spine injury[3]. Since cervical spine injury is associated with high morbidity and mortality, quick diagnosis of the injury is crucial. Any delay in diagnosis may result in devastating consequences for the patient. So, any additional aid to the radiologists can reduce the morbidity or mortality of the patient.

In recent years, a machine deep learning technique known as deep convo- lutional neural network (DCNN) has been applied to image recognition tasks. DCNN’s are well suited for images. So, they have been used extensively in the field of medicine to classify medical images.

In past few years, there have been many studies that have tried to use DCNN[4][5][6] on medical radiographs. In these studies, the reference standard for the training and testing images was based on the assessment of human readers determining which were visible, only within a radiograph. Many radi- ologist fail to detect ”occult fracture” because of the difficulty in detecting such fracture in a radiograph. These extraction methodologies could adversely influence the classification accuracy and occult fracture being assessed as a “non-fracture case”. A proficient algorithm may help identify and triage studies for the radiologist to review more urgently, helping to ensure faster diagnoses.

The purpose of our study is to develop an automated deep learning sys- tem for detecting cervical spine fractures using CT a gold standard annotated by radiologists, and to evaluate the diagnostic performance inclusive of the experienced readers in detecting cervical spine fractures on radiographs.

## Literature Review

Many studies have been conducted to utilize deep learning techniques in tele- medicine field. We have adequate amount of literature that is relevant to our project. However, The CNN model proposed by J.E. Small et al. to detect the cervical spine fracture detection with the accuracy of 92%[7]. According to their study, the radiologists are 95% accurate in the cervical spine fracture detection. The fractures missed by the proposed model and the radiologists were similar by level and location which is mostly included fractures which are obscured by CT beam attenuations. Yee Liang Thian et al.[8] have imple- mented Inception-ResNet Faster R-CNN architecture to detect and localize the fractures on wrist radiographs. The model detected and correctly localized 310 (91.2%) of 340 and 236 (96.3%) of 245 of all radius and ulna fractures on the frontal and lateral views, respectively.

Yoga Dwi Pranata et al. have implemented ResNet and VGG for auto- mated classification and detection of calcaneus fracture in CT images[9]. The bone fracture detection algorithm incorporated fracture area matching using the speeded-up robust features (SURF) method, Canny edge detection, and contour tracing. Their results showed that ResNet was comparable in accu- racy, of 98%, to the VFF network for bone fracture classification but achieved better performance. Tsubasa Mawatari et al. have studied about the effect of deep convolutional networks on radiologists performance in the detection of hip fractures on digital pelvic radiograph. According to the study, the average AUC of the 7 readers was 0.832. The AUC of DCNN alone was 0.905. The average AUC of the 7 readers with DCNN outputs was 0.876. The AUC of readers with DCNN output were higher than those without[10].

## Future Work

Future works can implement include fracture highlighting which will cre- ate a highlight on top of the predicted bone crack. We intend to use the study to detect cracks in concrete structures using deep learning and image manipulation techniques[16].
On the other hand, current flow of prediction can be improved by creating an ensemble of two models or by create a single model to perform segmentation, fracture detection and localization tasks. Further improvements can be made in segmentation model to impose monotonicity on the predictions i.e., to reduce predictions where the slice is predicted to be in two or more vertebrae.

## Conclusion

We have demonstrated that deep learning models are able to provide good fracture detection predictions and localization from the CT radiographs. The results have shown the feasibility of the above mentioned models to perform cervical spine fracture detection with 95% in segmentation and 94% in fracture detection and localization. This that the CNNs holds promise at both worklist prioritization and assisting radiologists in cervical spine fracture detection on CT radiographs.
