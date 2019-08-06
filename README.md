# SDC--TrafficSignClassification
Classification problem has an effective role in self-driving cars. However, classifying weather the showing object is a car, traffic sign, or building is needed, but "how it can be implemented?" This takes many phases. - Traffic sign recognition is a multi-class classification problem. Traffic signs can provide a wide range of variations between classes in terms of color, shape, and the presence of pictograms or text. However, there exist subsets of classes (e. g., speed limit signs) that are very similar to each other. - The classifier has to cope with large variations in visual appearances due to illumination changes, partial occlusions, rotations, weather conditions, etc. - Humans are capable of recognizing the large variety of existing road signs with close to 100% correctness. This does not only apply to real-world driving, which provides both context and multiple views of a single traffic sign, but also to the recognition from single images. - In this project we will focus on the mechanism of recognizing the traffic signs and differentiating between them. In order to achieve this, several steps had been taken.

# LeNet Architecture:

![modifiedLeNet](https://user-images.githubusercontent.com/53750465/62506420-e865b080-b800-11e9-8f21-e353a7117b06.jpeg)

# Importing train.p , valid.p and test.p dataset from the hard drive

![output_6_0](https://user-images.githubusercontent.com/53750465/62506421-e865b080-b800-11e9-8ae9-1a75d496adee.png)

# Data histogram:

![output_7_0](https://user-images.githubusercontent.com/53750465/62506422-e8fe4700-b800-11e9-8bf9-26d44a4d5791.png)

# Data agumentation 


![output_17_2](https://user-images.githubusercontent.com/53750465/62506450-092e0600-b801-11e9-9d4a-dc044b1514da.png)

![output_22_1](https://user-images.githubusercontent.com/53750465/62506451-092e0600-b801-11e9-8a9b-cabf4f83ed6d.png)

![output_23_1](https://user-images.githubusercontent.com/53750465/62506452-09c69c80-b801-11e9-9932-6df3368c6170.png)

![output_24_1](https://user-images.githubusercontent.com/53750465/62506453-09c69c80-b801-11e9-83ec-dc1098b13f84.png)

![output_25_1](https://user-images.githubusercontent.com/53750465/62506454-0a5f3300-b801-11e9-9b33-a221ffcbc380.png)

![output_30_0](https://user-images.githubusercontent.com/53750465/62506449-092e0600-b801-11e9-96b9-ec6efbf12bea.png)

![output_57_1](https://user-images.githubusercontent.com/53750465/62506512-38dd0e00-b801-11e9-9ad2-16428d29ca01.png)

# Module Results:

![Prob Test](https://user-images.githubusercontent.com/53750465/62506510-38dd0e00-b801-11e9-8045-89448a55a595.PNG)

![Prob Test2](https://user-images.githubusercontent.com/53750465/62506511-38dd0e00-b801-11e9-91e3-c60eb1efb549.PNG)

![output_63_0](https://user-images.githubusercontent.com/53750465/62506532-4abeb100-b801-11e9-9c67-c56f503c5c7c.png)

![test](https://user-images.githubusercontent.com/53750465/62506534-4b574780-b801-11e9-9e1d-b80a69e63184.PNG)
