import argparse
import glob
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential, Model 
from keras.layers import Input, Dense, Dropout,GlobalAveragePooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf 
import os
os.environ ['KMP_DUPLICATE_LIB_OK'] = 'True'


#args
perser = argparse.ArgumentParser()
perser.add_argument('--train_dir', default='./data/train', help='set train_dir')
perser.add_argument('--val_dir', default='./data/val', help='set val_dir')
perser.add_argument('--model_dir', default='./models/train1', help='set model_dir')
perser.add_argument('--epochs', default='30', help='set num_epochs')
perser.add_argument('--batch_size', default='8', help='set batch_size')
args = perser.parse_args()
train_dir, val_dir, model_dir, epochs, batch_size = args.train_dir, args.val_dir, args.model_dir, args.epochs, args.batch_size

classes = ['bird','cat','dog']
nb_classes = len(classes)
epochs = int(epochs)
batch_size = int(batch_size)
img_size = 96

#hom many samples?
train_samples = glob.glob(train_dir + '/*/*.jpg')
val_samples = glob.glob(val_dir + '/*/*.jpg')
train_samples = len(train_samples)
val_samples = len(val_samples)
print(train_samples)
print(val_samples)

#set Augumentations
train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, vertical_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255)

#set generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size,img_size),
    color_mode = 'rgb',
    classes = classes,
    class_mode = 'categorical',
    batch_size = batch_size
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size,img_size),
    color_mode = 'rgb',
    classes = classes,
    class_mode = 'categorical',
    batch_size = batch_size
)


#callbacks
#Adjustment of learning rate
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor=0.1,
    patience=3,
    mode='auto',
    min_lr=0.000000001
)

#To save model
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    filepath = model_dir + '/model_{epoch:02d}.hdf5',
    save_best_only = True
)


### model ###
input_tensor = Input(shape=(img_size,img_size,3))
MobileNetV2 = MobileNetV2(alpha=1.0, include_top=False, weights='imagenet', input_tensor=input_tensor, pooling=None, classes=nb_classes)
#def top
top_model = Sequential()
top_model.add(GlobalAveragePooling2D())
top_model.add(Dropout(0.20))
top_model.add(Dense(nb_classes, activation='softmax'))

model = Model(input=MobileNetV2.input,output=top_model(MobileNetV2.output))


#fine tuning
MobileNetV2.trainable = True

layer_names = [l.name for l in MobileNetV2.layers]
idx = layer_names.index('block_13_expand')
print(idx)

for layer in MobileNetV2.layers[:idx]:
    layer.trainable = False

#model compile
#set hyper param
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=0.0001),
    metrics=['accuracy']
)
model.summary()

#train
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_samples // batch_size,
    epochs = epochs,
    validation_data = val_generator,
    validation_steps = val_samples // batch_size,
    callbacks = [reduce_lr,checkpoint]

)

print(history.history)

#plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(1,2,1)
plt.plot(range(1,len(acc)+1),acc,'b',label = 'traning accracy')
plt.plot(range(1,len(acc)+1),val_acc,'r',label='validation accracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,len(loss)+1), loss, 'bo', label='Training loss')
plt.plot(range(1,len(loss)+1), val_loss, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('result')
