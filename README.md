# Crowd Sound Affect
### Model for potential real-time crowd sound emotional affect technology.

Emotional affect is a way to determine arousal of emotions based on what emotion it is and the level or degree of intensity of that emotion.

It’s common to identify emotions in individuals, but what about crowds?

Crowds tend to use ‘mirroring’ and synchronization, 
but can also have multiple different emotions

Could we use this in real-time to perceive crowds ‘likeness’ in music or events for marketing, or maybe aiding police with predicting if a crowd may turn into a mob?
[Process](#Process)
[Data](#Data)

## Process
- Collecting data and researching previous models 
- Starting with a base Sequential Convolutional Neural Network (CNN)
- Transfer Learning
- Final Model and app deployment with Streamlit

# Data

- 01 Normalized between 20 - 20,000 Hertz (Hz) which is sound audible to humans
- 02 Silence Blocks Removed
- 03 Spectrograms made with spgrambw in MATLAB
    - 400 samples with frame increment of 4.5 milliseconds
- 04 Splitting the data. It was heavily unbalanced with Neutral having over 5k images, Approval having around 3k images and Disapproval having just over 300


