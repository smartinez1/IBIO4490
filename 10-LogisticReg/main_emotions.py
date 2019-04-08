#!/home/afromero/anaconda3/bin/python
# -*- coding: utf-8 -*-

# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    #
    with open("fer2013.csv") as f:
            content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        #emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        emotion = int(emotion) #all emotion faces
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print ('Accuracy from sk-learn: {0}'.format(clf.score(x_test, y_test)))

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)


    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 1 # smile label
        self.lr = 0.00001 # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr


def train(model):
    auxTr=[]
    auxTe=[]
    auxEp=[]
    #--------------------------------------
    x_train, y_train, x_test, y_test = get_data()
    batch_size = 50 # Change if you want
    epochs = 10000 # Change if you want
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_test)
        loss_test = model.compute_loss(out, y_test)
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_test))
        auxEp.append(i)
        auxTr.append(np.array(loss).mean())
        auxTe.append(loss_test)
        plot(auxEp,auxTr,auxTe,epochs)





def plot(ep,tr,te,maxEpoch): # Add arguments
    # CODE HERE
    # Save a pdf figure with train and test losses
    aux=maxEpoch-1
    if ep[len(ep)-1]==aux:
        fig=plt.figure()

    plt.plot(ep,tr,'m',ep,te,'b')
    plt.title('Train & Test loss vs Epoch')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(('Train','Test'))
    if ep[len(ep)-1]==aux:
        fig.savefig("myGraph.pdf", bbox_inches='tight')
        plt.show()




def test(model):
     _, _, x_test, y_test = get_data()
     out = model.forward(x_test)
     Threshs=np.linspace(-2,2,50)
     prec=[]
     reca=[]
     fMeasure=[]
     ACAs=[]
     for th in Threshs:

         outBin=out>th
         numInstance=len(y_test)
         confMat=np.zeros([2,2])
         ins=0
         tp=0
         tn=0
         fp=0
         fn=0
         for idx in range(numInstance):

            if outBin[idx]==y_test[idx]:

                 if y_test[idx]==1:
                     confMat[0,0]=confMat[0,0]+1
                     ins=ins+1
                     tp=tp+1

                 else:
                     confMat[1,1]=confMat[1,1]+1
                     ins=ins+1
                     tn=tn+1

            else:
                  if y_test[idx]==1:
                      confMat[0,1]=confMat[0,1]+1
                      ins=ins+1
                      fn=fn+1

                  else:
                      confMat[1,0]=confMat[1,0]+1
                      ins=ins+1
                      fp=fp+1

         confMat=confMat/ins
         ACA=confMat[1,1]+confMat[0,0]
         ACAs.append(ACA)
         auxP=(tp)/((tp)+(fp))
         auxR=(tp)/((tp)+(fn))
         FM=(2*auxP*auxR)/(auxP+auxR)
         prec.append(auxP)
         reca.append(auxR)
         fMeasure.append(FM)

     maxF=np.max(fMeasure)
     jdx=0
     while maxF!=fMeasure[jdx]:
         jdx=jdx+1

     PR=plt.figure()
     plt.plot(reca,prec)
     plt.grid(True)
     plt.title('Precision-Recall curve - F1='+str(maxF)+' in threshold='+str(Threshs[jdx])+', ACA='+str(ACAs[jdx]))
     plt.xlabel('Recall')
     plt.ylabel('Precision')
     PR.savefig('Precision-Recall.PNG')
     plt.show()

     with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)
    # YOU CODE HERE
    # Show some qualitative results and the total accuracy for the whole test set


if __name__ == '__main__':
    import argparse
    try:
        print('Dataset Found')
    except FileNotFoundError:

        import os
        print('Downloading Dataset...')
        os.system('wget http://bcv001.uniandes.edu.co/fer2013.zip')
        os.system('unzip fer2013.zip')
        os.system('rm -rf fer2013.zip')
        print('Dataset succesfully downloaded')
   ##-----------------------------------------##

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',action="store_true")
    parser.add_argument('--demo',action="store_true")
    args = parser.parse_args()

    if args.test:
        try:
            with open('model.pickle','rb') as f:
                model=pickle.load(f)

            print('Using Pre-Trained Model')
        except FileNotFoundError:
            model=Model()
            print('Pre-Trained Model not Found, training new Model...')
            train(model)

        test(model)
    elif args.demo:
        import skimage.io as io
        import skimage.color as col
        import skimage.transform as tm
        #import os
        try:
            with open('model.pickle','rb') as f:
                model=pickle.load(f)

            print('Using Pre-Trained Model')
        except FileNotFoundError:
            model=Model()
            print('Pre-Trained Model not Found, training new Model...')
            train(model)

        #if (os.path.exists("./images"))==False:
        #    print('Images folder not found')

        from os import listdir
        from os.path import isfile, join
        Files = [f for f in listdir('images') if isfile(join('images', f))]
        images=np.zeros((len(Files),48,48))
        for idx in range(len(Files)):
            img=io.imread('images/'+Files[idx])
            img=col.rgb2gray(img)
            img=tm.resize(img, (48, 48), anti_aliasing=True)
            images[idx,:,:]=img
            OGimg=io.imread('images/'+Files[idx])
            print('Processing image '+str(idx+1)+' out of '+str(len(Files)))
        results=model.forward(images)

        indices=np.random.permutation(len(Files))

        for idx in range(1,7):
            i=indices[idx]
            if results[i]>0:
                aux='happy'
            else:
                aux='not happy'

            plt.subplot(3,2,idx)
            plt.imshow(images[i,:,:],cmap='gray')
            plt.axis('off')
            plt.title('classified as '+aux)
        plt.show()


    else:
        model = Model()
        train(model)
        test(model)
