# Crowd Sound Affect
### Model for potential real-time crowd sound emotional affect technology.

Emotional affect is a way to determine arousal of emotions based on what emotion it is and the level or degree of intensity of that emotion.

It’s common to identify emotions in individuals, but what about crowds?

Crowds tend to use ‘mirroring’ and synchronization, 
but can also have multiple different emotions

Could we use this in real-time to perceive crowds ‘likeness’ in music or events for marketing, or maybe aiding police with predicting if a crowd may turn into a mob?


[Process](#Process)

[The Data](#The Data)

[What are Spectrograms?](#What are Spectrograms?)

[Approaches to Models](#Approaches to Models)

[Results](#Results)

[Further Research](#Further Research)


To see more please visit the ![original paper](https://link.springer.com/article/10.1007/s11042-020-09428-x), or use the ![original dataset](https://ieee-dataport.org/open-access/emotional-crowd-sound)


# Process
- Collecting data and researching previous models 
- Starting with a base Sequential Convolutional Neural Network (CNN)
- Transfer Learning
- Final Model and app deployment with Streamlit

# The Data

- 01 Normalized between 20 - 20,000 Hertz (Hz) which is sound audible to humans
- 02 Silence Blocks Removed
- 03 Spectrograms made with spgrambw function in MATLAB
    - 400 samples with frame increment of 4.5 milliseconds
- 04 Splitting the data. It was heavily unbalanced with Neutral having over 5k images, Approval having around 3k images and Disapproval having just over 300. I decreased the amount of neutral to below 3k to be a closer match approval.

The paper originally classified a range of frequencies, Bark (0–3.5 kHz range), Erb (2–4 kHz range), Log (0.02-2 kHz range) and Mel (4–6 kHz range), then used the emotions Approval, Disapproval and Neutral.

I, on the other hand, only classifed Approval, Disapproval and Neutral for my labels. I split the data into testing, training and validation sets. 

# What are Spectrograms?
Spectrograms can be thought of as a 3-D visulaization of audio sounds. 
The X axis is Time
The Y axis is Frequency 
Color Intensity is Amplitude

Amplitude can be interpreted as ‘loudness’ of the frequency
### Spectrograms. 
The first graph are randomly chosen spectrograms from the testing data set. There is a lot of varity within each classification. 

Labels of the images in order: 

| Neutral |             Approval |          Approval |             Approval |                   Disapproval |          Disapproval |          Disapproval |          Disapproval |
![shuffled_8_test](https://user-images.githubusercontent.com/61055286/129239920-b92cad62-a77b-4a9e-b7f1-c6a85776ac5e.png)

The graph below shows spectrograms that were converted in BGR color scheme, this is a little easier for the human eyes to recognize some patterns. There are 3 images with a density of amplitude on the bottom, each a different class (Neutral, Disapproval and Approval). Normally Approval's amplitude is condensed at the top, but the last image is showing more amplitude at the bottom. This is where Independent component analysis could help distinguish if there is both approval or another emotion within the crowd. 

Labels of the images in order: 

| Approval    |     Neutral      |     Approval        |       Disapproval  |      Disapproval    |    Disapproval    |      Approval   |           Approval |
![Preprocessed_shuffle](https://user-images.githubusercontent.com/61055286/129240138-66e4c694-9b58-4b6b-b75b-364cb82140fe.png)

# Approaches to Models
The original paper's approach used an AlexNet with 4 epochs, L2 Regularization and were also getting a validation score at around 97% average over 4 networks

I used a basic Sequenital Convolutional Neural Network Model. I chose to use this and work with parameters to learn about how the model reacts and learns the spectrograms. 

I stretched the layers out (more nodes), compacted them with larger pooling sizes, and changes the strides the filters would take. I generally kept to either a batch of 16 or 32. My First model (with a small training set) produced the confusion matrix below:


![confusion](https://user-images.githubusercontent.com/61055286/129240288-5ef1ad3f-312d-4d5a-85e4-f2383f7e6420.png)

My final model, the one I called the 'Compacted Sequential Model' produced these results:

|   Classification Report | Precision   | Recall   | Support|
|-----------------|--------|---------|-------|
|    Approval      | 0.93     | 0.95     | 1432|
| Disapproval      | 0.84    |  0.88      | 312|
|     Neutral      | 0.98   |   0.96     | 2928|

Disapproval is significantly lower than Approval and Neutral classifications and I believe that may have two factors: one due to the Disapprovals likeness in some neutral and approvals spectrograms and also the amount of data points compared to approval and neutral. 


|   Avgerages |  | support|
|-----------|--------|--------|
|    accuracy |0.96 |   4672|


|   Avgerages | Precision   | Recall   | Support|
|--------------|------------|------------|--------|
|   Macro avg    |   0.92    |  0.93   |   4672|
| Weighted avg  |     0.96    |  0.96 |     4672|


![confustion_strides](https://user-images.githubusercontent.com/61055286/129240377-5a8fbee0-7add-4fc7-b1da-6bf625b8c317.png)

# Results
I specifically randomly chose photos that were classified incorrectly and correctly to continue my data analysis to potentially find what the data is saying, and what my model it trying to say. 

 The images below are tsix testing spectrograms I used to predict with my model, the relatedness between the Approval and Neutral images with patterns are very closely similar, the main difference is the denstiy in the amplitude (which we talked about before with the BGR photos). The model is just barley predicting approval correctly or incorrectly. This was part of the reason I took out more of the Neutral data points to even out the model. 

![Approval_Neutral_predict](https://user-images.githubusercontent.com/61055286/129246995-501b5f60-2e20-47e0-884b-6cbc0d7d0038.png)

I wanted to see more of the dissapproval (since the percentages were lower) and below there is a lot of variation to the spectrograms in each one and the output for each of them. I believe that the problem could be fixed with more data points here. 

![disapproval_predict](https://user-images.githubusercontent.com/61055286/129246999-5f7cef7a-5513-4fa4-ba85-c1e05f0a4653.png)


# Further Research

* Even though my process was to get to transfer learning, I did not however, my next steps would be to use Global Average Pooling Layers after transfer learning model

* If this research is going to be used in real time, I think it would be important to include and to train the model to know what silence is and not take them out.

* Also, as stated above, I would like to use Independent Component Analysis to aid in learning to distinguish and predict emotional affect within crowds where there are multiple different emotions. 


