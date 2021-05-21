# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 07:57:32 2021

@author: Kat Ria
"""


####################
#draw and save#
####################  
from PIL import ImageDraw
import PIL
from tkinter import *
from tkinter.filedialog import asksaveasfile

width = 400
height = 400
center = height//2

current_x, current_y=0,0

def locate_xy(event):
    global current_x, current_y
    current_x, current_y = event.x, event.y

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=30, tag="new")
    draw.line([x1, y1, x2, y2],fill="black",width=30)
    
    
def new_canvas():
    canvas.delete('all')
    draw.rectangle((0,0,width,height), fill=(255,255,255))
    predres.configure(text='Predictions: ')
    canvas.bind('<B1-Motion>',paint)

def save():
    #filename = asksaveasfile(mode = "wb",defaultextension=".png", filetypes=(("PNG file", "*.png"), ("All Files", "*.*")))
    image1.thumbnail((28,28), PIL.Image.ANTIALIAS)
    filename = 'art00.png'
    image1.save(filename)

#################################
    
#################################
import tensorflow
from tensorflow.keras import models
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt

model = models.load_model('cnn_model.h5')

class_names = ['basketball','baseball', 'shark', 'dolphin', 'duck', 'bird', 'van', 'ambulance', 'knee', 'leg']

def results(): 
        
    img = cv2.imread('art00.png')
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28,28))
    dat = np.array(img)
    dat = dat.reshape(1,28,28,1).astype('float32')/255.0
    plt.imshow(img)
    
    prediction = model.predict(dat)[0] 
    index = (-prediction).argsort(axis=0)[:3]
    data = [class_names[x] for x in index]
    #print(data)    
    #predres.configure(text='Predictions: '+str(data))

    predictionv = np.squeeze(model.predict(dat))
    indexv = np.argsort(-predictionv)
    top_3 = indexv[:3]
    top_results = predictionv[indexv]
            
    datv = np.round(100*top_results)[:3]
   
    #print(datv)
    predres.configure(text='Predictions: '+str(data)+'\nProbabilities(%): \t'+(str(datv)))
    print()
    
    for i, (pred, ind) in enumerate(zip(top_results, top_3)):
        bruh = f'{i+1}. {class_names[ind]}: {np.round(100*pred)} %'
        #bruh = f'{i+1}. {class_names[ind]}: {"{:.2f}".format(np.rou(100*pred))} %'
        print(bruh)
    #    predres.configure(text="Predictions: "+(str(bruh)))
        
    #########
    #########

#################################
#################################

root = Tk()
root.title("Drawing Canvas")
root.geometry("400x400")

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

canvas= Canvas(root,background='white')
canvas.grid(row=0, column=0, sticky='nsew')

menubar = Menu(root)
root.config(menu=menubar)
submenu= Menu(menubar, tearoff=0)

menubar.add_cascade(label='File', menu=submenu)
submenu.add_command(label='New Canvas', command=new_canvas)
submenu.add_command(label='save', command=save)
submenu.add_command(label='Predict', command=results)

predres = Label(root, text="Predictions:")
predres.grid(row=2, columnspan=2, sticky=W)

image1 = PIL.Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image1)

canvas.bind("<B1-Motion>", locate_xy)
canvas.bind("<B1-Motion>", paint)

root.mainloop()



