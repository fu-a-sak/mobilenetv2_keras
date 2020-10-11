import argparse
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import glob
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import pandas as pd 


#args
perser = argparse.ArgumentParser()
perser.add_argument('--test_dir', default='./data/test', help='set test_dir')
perser.add_argument('--model_dir', default='./models/train', help='set model_dir')
perser.add_argument('--train_number', default='1', help='set train_number')
perser.add_argument('--weight_number', default='1', help='set weight_number')
args = perser.parse_args()
test_dir, model_dir, train_number, weight_number = args.test_dir, args.model_dir, args.train_number, args.weight_number

classes = ['bird','cat','dog']
nb_classes = len(classes)

model_dir = model_dir + train_number

test_samples = glob.glob(test_dir + '/*/*.jpg')

img_size = 96

np.set_printoptions(suppress=True)
label = []
result = []

model = load_model(model_dir + '/model_' + weight_number + '.hdf5')

for i in enumerate(test_samples):
#    print(i[1])
    a = i[1].split('/')[-2]
    label.append(str(a))
    x = load_img(i[1],color_mode='rgb',target_size=(img_size,img_size))
    x = img_to_array(x)/255
    x = x[None,...]
    predict = model.predict(x,batch_size=1,verbose=1)
    pred = np.argmax(predict)

    if pred == 0: 
        result.append('bird')
    elif pred == 1: 
        result.append('cat')
    elif pred == 2: 
        result.append('dog')
    else: 
        pass

print('labels-------')
print(label)
print('results-------')
print(result)

labels = sorted(list(set(label)))
cm = confusion_matrix(label,result,labels=labels)
df_cmx = pd.DataFrame(cm,index=labels,columns=labels)

plt.figure()
sns.heatmap(df_cmx,annot=True,xticklabels='auto',yticklabels='auto')
plt.ylabel('label')
plt.xlabel('pred')
plt.savefig('./mobilenetv2_train' + train_number + '.png')