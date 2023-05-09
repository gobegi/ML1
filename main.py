"""
First machine learning program following tutorial https://www.youtube.com/watch?v=jztwpsIzEGc&t=1800s

Program that uses machine learning in order to differentiate 2 folders of images

In this case I am going to differentiate basketball players and lacrosse players

All that is needed is a folders change in the data folder and we can differentiate whatever we want

0 is basketball and a 1 is lacrosse

IMGHDR cannot be used as in the tutorial so we use cv2 instead as seen around line 50
"""


#imports
import tensorflow as tf
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy


#avoid out of memory and vram errors
#cannot find a single GPU, may be an issue later
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#variable to hold path to data directory
data_dir = 'data'

#makes our good image exts
image_exts = ['jpeg','bmp','png']


#takes out bad images
#MAY NEED IMGHDR HERE
#lacrosse and basketball are image classes
for image_class in os.listdir(data_dir):
    #for each image in those directories
    for image in os.listdir(os.path.join(data_dir, image_class)):
        #gets image path
        image_path = os.path.join(data_dir, image_class, image)
        try:
            #reads in the image and creats a form variable that will give us some sort of .jpg .png etc
            img = cv2.imread(image_path)
            form = ''
            #if we have image then give us the format
            if img is not None:
                img_format = image_path.split('.')[-1]
                form = img_format
            else:
                print("Failed to read image")
            #if the form is not a valid extension then remove the image
            if form not in image_exts:
                print('Image is not in the extensions list{}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('issue with image {}'.format(image_path))

#makes it so we do not need to build labels or classes and will resize images 
#go into jupyter notebook to look at batch sizes and such. Can always pass these in as arguements after 'data'
data = tf.keras.utils.image_dataset_from_directory('data')

#converts data into numpy iterator
data_iterator = data.as_numpy_iterator()



#PRE PROCESSING STEPS


#scales our data to be betwwen 0 and 1 instead of 0 and 255
# x is images and y is target variable
data = data.map(lambda x,y: (x/255, y))

#split the data between training and testing and validation

train_size = int(len(data)*.7)
val_size = 0
test_size = 0 

#makes it so we are spread evenly over val and test
val_add = 0
while train_size + val_size + test_size < len(data):
    if val_add == 0:
        val_size +=1
        val_add+=1
    else:
        test_size +=1
        val_add= 0
    

#skip skips previously used images, takes takes images
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


#DEEP MODEL
model = Sequential()
#uses filters to make a classification
#relu means only see the positive values, makes all negative values 0
model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (256, 256, 3)))
#takes max value
model.add(MaxPooling2D())

#same things down here
model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

#flattens the values
model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
#maps betwwen 0 and 1 
model.add(Dense(1, activation = 'sigmoid'))

#adam is optimizer, we want to track accuracy
model.compile('adam', loss= tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

#TRAIN THE NETWORK

#logs our model training
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

#epochs are how many runs we do over a training set. we pass over our training data to run an evaluation on our evaluation data and we want to log our information to tensorboard
hist = model.fit(train, epochs=20, validation_data = val, callbacks = [tensorboard_callback])

#now we are totally trained

#visual loss and data loss
fig = plt.figure()
plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'red', label = 'val_loss')
fig.suptitle('Loss', fontsize = 20)
plt.legend(loc = "upper left")
plt.show()


#visualize the accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'teal', label = 'accuracy')
plt.plot(hist.history['val_accuracy'], color = 'red', label = 'val_accuracy')
fig.suptitle('Accuracy', fontsize = 20)
plt.legend(loc = "upper left")
plt.show()


#EVALUATE OUR PERFORMANCE/TEST

#makes instances of classes we need
pre = Precision()
re = Recall()
acc = BinaryAccuracy()


#Look out for these, says model is not good when it actually is working great
for batch in test.as_numpy_iterator():
    x, y = batch
    #returns a value between 0 and 1 
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print("Tests on our test data stats: ")
print(str(pre.result().numpy()) + " = Precision result")
print(str(re.result().numpy()) + " = Recall result")
print(str(acc.result().numpy()) + " = Accuracy result")


"""
#Shows an image of an outside test
img = cv2.imread('notb.jpeg')
#resize
resize = tf.image.resize(img, (256,256))

#adds an extra dimanesion as our model expects a batch of images not just 1
#divided by 255 to scale it
np.expand_dims(resize, 0)
yhat = model.predict(np.expand_dims(resize/255, 0))

label = ''
print(yhat)
if yhat >= .5:
    label = 'This is a picture of lacrosse'
else:
    label = 'This is a picture of basketball'

plt.title(label)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


#Shows an image of an outside test
img = cv2.imread('lacrosseTest.jpeg')
#resize
resize = tf.image.resize(img, (256,256))

#adds an extra dimanesion as our model expects a batch of images not just 1
#divided by 255 to scale it
np.expand_dims(resize, 0)
yhat = model.predict(np.expand_dims(resize/255, 0))

label = ''
print(yhat)
if yhat >= .5:
    label = 'This is a picture of lacrosse'
else:
    label = 'This is a picture of basketball'

plt.title(label)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

"""

#GOES THROUGH FILES INSTEAD OF 1 IMAGE

lacrosse_and_basketall_dir = 'lacrosse_and_basketball'

for image in os.listdir(os.path.join(lacrosse_and_basketall_dir)):
    #gets image path
    image_path = os.path.join(lacrosse_and_basketall_dir, image)
    try:
        #reads in the image and creats a form variable that will give us some sort of .jpg .png etc
        img = cv2.imread(image_path)
        form = ''
        #if we have image then give us the format
        if img is not None:
            img_format = image_path.split('.')[-1]
            form = img_format
        else:
            print("Failed to read image")
        #if the form is not a valid extension then remove the image
        if form not in image_exts:
            print('Image is not in the extensions list{}'.format(image_path))
            os.remove(image_path)
    except Exception as e:
        print('issue with image {}'.format(image_path))

for image in os.listdir(os.path.join(lacrosse_and_basketall_dir)):
    #gets image path
    image_path = os.path.join(lacrosse_and_basketall_dir, image)
    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (256,256))

    #adds an extra dimanesion as our model expects a batch of images not just 1
    #divided by 255 to scale it
    np.expand_dims(resize, 0)
    yhat = model.predict(np.expand_dims(resize/255, 0))

    #makes our save location
    save_location = ''
   
    if yhat >= .5:
        save_location = 'sortedLacrosse'
    else:
        save_location = 'sortedBasketball'

    
    # create the full file path by combining the folder path and file name
    file_path = os.path.join(save_location, image)

    # saves img to specified location
    cv2.imwrite(file_path, img)

 
#SAVE THE MODEL

model.save(os.path.join('models', 'basketandlacrosse.h5'))
new_model = load_model(os.path.join('models', 'basketandlacrosse.h5'))