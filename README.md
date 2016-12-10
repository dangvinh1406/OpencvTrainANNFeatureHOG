# OVERVIEW
This application is about training artificial neural network using HOG feature to predict wherether an image contains the interested object or not.

## Requirement:
- "positive folder" contains cropped positive images which are the same size and has only 1 object at center of image.
- "negative folder" contains negative images which have arbitrary size but larger than "width" and "height" respectively.
- "width" and "height" should be multipication of 16.

## Noted from OpenCV training ANN example 
OpenCV neural network (MLP) implementation does not support categorical variables explicitly. So, instead of the output class label, we will use a binary vector of {0,0 ... 1,0,0} components (one element by class) for training and therefore, MLP will give us a vector of "probabilities" at the prediction stage - the highest probability can be accepted as the "winning" class label output by the network.

# Detailed instruction
## Combining aplication
.......

## Running application
./train_HOG_ANN [positive folder] [negative folder] [width] [height] [model storing directory]
For more convenience, we can put all positive images and negative images to my created folders. After that, we can use the following command:

