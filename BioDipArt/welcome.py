from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
import copy
import tkinter as tk
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt


from os.path import exists
#-------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------GOOGLE COLLAB CODE-----------------------------------------------------

# Label mapping
import json
with open('cat-to-name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load resnet-152 pre-trained network--------------------------------------------------------------------------------------
model = models.resnet152(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
#Let's check the model architecture:
    print(model)
#---------------------------------------------------------------------------------------------------------------------------
# Train a model with a pre-trained network
num_epochs = 10
#load checkpoint
# Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location ='cpu')
    model = models.resnet152()

    # Our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 2
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 512)),
        ('relu', nn.ReLU()),
        # ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(512, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']

    # Get index to class mapping
loaded_model, class_to_idx = load_checkpoint('model/8960_checkpoint.pth')
idx_to_class = {v: k for k, v in class_to_idx.items()}
#-------------------------------------------------------------


#---------------------------------------------------------------------------
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage / 255.

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485) / (0.229)
    imgB = (imgB - 0.456) / (0.224)
    imgC = (imgC - 0.406) / (0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage
#-----------------------------------------------------------------------------------
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
#----------------------------------------------------------------------
def predict(image_path, model, topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file

    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]

    top_idx = np.argsort(pobabilities)[-topk:][::-1]
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class

#print(predict('images/Benign.png', loaded_model))
#-----------------------------------------------------------------------------------------------

#this is first window
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        load = Image.open("images\logo.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=220, y=10)

root = Tk()
app = Window(root)
root.wm_title("Bio DipArt")


#this is second window
def second_win():
#destroing 1st window
  root.destroy()

  top=Tk()
  top.title("Menu")
  top.geometry("1450x740")



  ben_label=Label(top, text='Benign Image', background='#004379', foreground ='white', height=2, width=98, border=4).place(x=55, y=100)
  loads = Image.open("images\Benign.png")
  renders = ImageTk.PhotoImage(loads)
  imgs = Label(top, image=renders)
  imgs.image = renders
  imgs.place(x=50, y=150)

  prediction = Button(top, text='Prediction ', background='#004379', command=n_win_predict, foreground='white', activebackground='#757575', height=2, width=15, border=2)
  prediction.place(x=880, y=180)


  upload  = Button(top, text='Validate ', command=validate, background='#004379', foreground='white', activebackground='#757575', height=2, width=15, border=2)
  upload.place(x=880, y=280)
  about = Button(top, text='About', background='#004379',command=about_win, foreground='white', activebackground='#757575', height=2, width=15, border=2)
  about.place(x=880, y=380)

  icon_b = Image.open("images\icon_b.JPG")
  icon_img = ImageTk.PhotoImage(icon_b)
  imgs_b = Label(top, image=icon_img)
  imgs_b.image=icon_img
  imgs_b.place(x=800, y=460)





#predict func--------------------------------------
#def uploads_img():
 #   global filename
  #  filename = filedialog.askopenfilename(initialdir="/", title="Select The Image",
   #                                       filetypes=(("jpg", "*.jpg"), ("All Files", "*.*")))

    # Open and return file path
    #l1 = Label(root, text="File path: " + filename).pack()
    #t, s = predict(filename, loaded_model, 2)
    #print(t)
    #print(s)
    #type(t)
    #type(s)
    #result = 'benign'
    #print(filename)

# Implement the code to predict the class from an image file

#----------------------------------------------------new window for prediction-----------------------------
#--------img show----
def gtest(p,l12):
    img = mpimg.imread(p)
    imgplot = plt.imshow(img)
    plt.title(label=l12)
    plt.show()
#---------------------


def n_win_predict():

    r=Tk()
    r.geometry("1450x740")
    r.title("PREDICTION")


    def uploads_img():
        global filename
        r.filename = filedialog.askopenfilename(initialdir="/", title="Select The Image",
                                              filetypes=(("jpg", "*.jpg"), ("All Files", "*.*")))




        l1 = Label(r, text="File-Path:   "+r.filename).place(x=200,y=350)

        return r.filename

        result = ' '


    def p_img():

        f = str(uploads_img())

        t,s=predict(f,loaded_model,2)

        if t[0]>t[1]:
         r.result=s[0]

         l1= Label(r, text=r.result,bd=2).place(x=200, y=380)
         gtest(f, l12=r.result)
        elif t[1]>t[0]:
         r.result=s[1]
         l2 = Label(r, text="RESULT:  "+r.result).place(x=200, y=380)
         gtest(f, l12=r.result)

         return r.result

    prediction6 = Button(r, text='PREDICT IMAGE ', background='#004379', command=p_img, foreground='white',
                         activebackground='#757575', height=2, width=15, border=2)
    prediction6.place(x=200, y=500)

#------------------------new window result--------------------------











#------------------------------validate data------------------------------------------
#-----------------img show--------------------
def gtest1(p1,l13):
    img7 = mpimg.imread(p1)
    imgplot = plt.imshow(img7)
    plt.title(label=l13)
    plt.show()
#----------------------------------------------
def validate():
    tops = Tk()
    tops.title("Validate")
    tops.geometry("1450x740")

    def uploads_img_v():
        global filename6
        tops.filename6 = filedialog.askopenfilename(initialdir="/", title="Select The Image",
                                              filetypes=(("jpg", "*.jpg"), ("png", "*.png"),("All Files", "*.*")))

        # Open and return file path


        l1 = Label(tops, text="File-Path:   "+tops.filename6).place(x=200,y=350)
        #l1.pack()
        return tops.filename6
    def p_img_v():

        fv = str(uploads_img_v())
        tv,sv=predict(fv,loaded_model,2)

        if tv[0]>tv[1]:
         tops.resultv=sv[0]

         l1= Label(tops, text=tops.resultv).place(x=200, y=380)
         gtest1(fv, l13=tops.resultv)
        elif tv[1]>tv[0]:
         tops.resultv=sv[1]
         l2 = Label(tops, text="RESULT:  "+tops.resultv).place(x=200, y=380)
         gtest1(fv, l13=tops.resultv)

    file_btn = Button(tops, text='Upload Image For Validation', command=p_img_v, background='#004379', foreground='white',
                        activebackground='#757575', height=2, width=40, border=2)
    file_btn.place(x=200, y=500)





def file_path():
      names = filedialog.askopenfilename(initialdir="/",
                           filetypes=(('png','.png'),("Text File", "*.txt"), ("All Files", "*.*")),
                           title="Choose a file.")


def about_win():
    tops = Tk()
    tops.title("About")
    tops.geometry("1450x740")
    batch_labels = Label(tops, text='BioDipArt is Breast cancer prediction software.', background='#004379', foreground='white', height=5, width=150,
                        border=4).place(x=220, y=80)
    batch_label = Label(tops, text='It take image as an input and predict whether the image uploaded for examining is benign or malignant.', background='white', foreground='#004379', height=5, width=150,
                        border=4).place(x=220, y=180)
    batch_labels = Label(tops, text='As already train model is integrated in a system that identifies its different features and predict the images and gave result as an output.', background='#004379',
                         foreground='white', height=5, width=150,border=4).place(x=220, y=280)
    batch_label = Label(tops,
                        text='In train section ,a user can upload an image file and also gave n_thread,batch_size and epouch values,to see or improve its valid accuracy results.',
                        background='white', foreground='#004379', height=5, width=150,
                        border=4).place(x=220, y=380)
    batch_labels = Label(tops,
                         text='This software is helpful in identifying cancer at early stage and also help the medical professionals to diagnose and treat the deadly diseases e.g. cancer with precision, accuracy and ease.',
                         background='#004379',
                         foreground='white', height=5, width=150, border=4).place(x=220, y=480)
    batch_label = Label(tops,
                        text='The system will help the researchers and pathologists in treating the diseases in more effective manner.',
                        background='white', foreground='#004379', height=5, width=150,
                        border=4).place(x=220, y=580)
#welcome pg button
button1 = Button(root, text='Start ', background='#004379', command=second_win, foreground='white', activebackground='#757575', height=3, width=13, border=2)
button1.place(x=580, y=620)
root.geometry("1450x740")

root.mainloop()
