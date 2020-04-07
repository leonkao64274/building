import cv2,csv
import numpy as np
import datetime
from keras import layers,models,optimizers
from keras.applications.xception import Xception
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split as ttsplit
np.random.seed(0)
time= datetime.datetime.now()

def Xception_model(imgsize_,channel=3,drop=0.3,class_=3):
    model = models.Sequential()  
    conv_base =Xception(include_top=False, weights='imagenet',input_tensor=None,
                        input_shape=(imgsize,imgsize,3),pooling=None, classes=class_)
    model.add(conv_base)   
    model.add(layers.Flatten())
    model.add(layers.Dropout(drop))
    model.add(layers.Dense(8192, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(class_, activation='sigmoid'))    
    model.summary()      
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    return model

def data_pross(data, n_data):   #定義影像降解析的函式
    imglist = np.zeros((n_data,imgsize,imgsize,3))
    for i in range(n_data):
        img=data[i] 
        imglist[i]=cv2.resize(img,(imgsize,imgsize),interpolation=cv2.INTER_CUBIC)
    return imglist

#######################################主程式(執行...)
data_Xtrain=np.load('X_train.npy')
data_Ytrain=np.load('y_train.npy')
data_Xtest=np.load('X_test.npy')   

'''可調參數'''
imgdata_nmber=len(data_Xtrain) 
imgsize=110       #影像降解析(邊長)
drop_=0.01        #Dropout
batch=100
epoch=9
test_n=0.3        #test_size
validation_n=0.3  #validation_split

X =data_pross(data_Xtrain[0:imgdata_nmber], imgdata_nmber)/255.0
Y =to_categorical(data_Ytrain[0:imgdata_nmber])
Xtrain, Xtest, ytrain, ytest = ttsplit(X,Y,random_state=20,test_size =test_n)

model_=Xception_model(imgsize,channel=3,drop=drop_,class_=3)
history = model_.fit(Xtrain,ytrain,batch_size=batch,epochs=epoch,validation_split=validation_n)
model_.save('task1_cnn.h5')
######################################輸出成果+繪圖
'''成果輸出成 output.csv檔 '''
X__test=data_pross(data_Xtest,len(data_Xtest))/255.0
yfit = model_.predict_classes(X__test)
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index','Pred']) 
    for i in range(len(data_Xtest)):            
        writer.writerow([i+1,yfit[i]])

'''打印 執行時間及設定參數值'''
time0= datetime.datetime.now()  
interval=(time0-time).seconds
print('____________________________________________________________')
print("CNN score: %.2f%%"%(model_.evaluate(Xtest,ytest)[1]*100))
print('input imgsize   : ',imgsize)
print('batch           : ',batch)
print('Dropout         : ',drop_)
print('test data size  : ',test_n)
print('validation_split: ',validation_n)
print('Time interval   : ',int(interval/60/60),':',int(interval/60%60),':',interval%60)

'''繪製 acc & loss 曲線'''
import matplotlib.pyplot as plt   
acc,val_acc = history.history['acc'],history.history['val_acc']
loss,val_loss = history.history['loss'],history.history['val_loss']

plt.plot(range(epoch+1)[1:-1], acc, 'ro', label='Training acc')
plt.plot(range(epoch+1)[1:-1], val_acc, 'b.-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(epoch+1)[1:-1], loss, 'ro', label='Training loss')
plt.plot(range(epoch+1)[1:-1], val_loss, 'b.-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.legend()
plt.show()