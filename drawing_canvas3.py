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

import tensorflow
from tensorflow.keras import models
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
import win32gui
import PIL.ImageGrab

model = models.load_model('cnn_model.h5')
class_names = ['basketball','baseball', 'shark', 'dolphin', 'duck', 'bird', 'van', 'ambulance', 'knee', 'leg']


width = 400
height = 400
center = height//2

current_x, current_y=0,0

class application(Frame):
    def __init__(self,master):
        super().__init__(master)
        self.master=master
        self.pack()
        self.createWidget()
    
    def createWidget(self):
        self.canvas = Canvas(self,width=width,height=width,bg='white')
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind('<B1-Motion>',self.locate_xy)
        
    def locate_xy(self, event):
        global lastx, lasty
        self.canvas.bind('<B1-Motion>', self.paint)
        lastx, lasty = event.x, event.y

    def paint(self,event):
        global lastx, lasty
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black",width=30, tag="new")
        
        lastx, lasty = 0,0
        
    def new_canvas(self):
        self.canvas.delete("all")
        self.canvas.bind('<B1-Motion>',self.locate_xy)
        predres.configure(text='Predictions: ')
        
    
    def results(self): 
        wind = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(wind)
        image1 = PIL.ImageGrab.grab(rect)
        
        image1.thumbnail((28,28), PIL.Image.ANTIALIAS)
        filename = 'art00.png'
        image1.save(filename)
    
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
    
        predictionv = np.squeeze(model.predict(dat))
        indexv = np.argsort(-predictionv)
        top_3 = indexv[:3]
        top_results = predictionv[indexv]
            
        datv = np.round(100*top_results)[:3]
        predres.configure(text='Predictions: '+str(data)+'\nProbabilities(%): \t'+(str(datv)))
        print()
    
        for i, (pred, ind) in enumerate(zip(top_results, top_3)):
            bruh = f'{i+1}. {class_names[ind]}: {np.round(100*pred)} %'
            #bruh = f'{i+1}. {class_names[ind]}: {"{:.2f}".format(np.rou(100*pred))} %'
            print(bruh)
            
        
        def new_wind():
            newwin = Toplevel(root)
            display = Label(newwin, text="Predictions: "+str(data)+'\nProbabilities(%): \t'+(str(datv)))
            display.pack()

        new_wind()
#################################
#################################
if __name__ == '__main__':
    root = Tk()
    root.title("Drawing Canvas")
    root.geometry("400x400")
    
    app=application(root)

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    canvas= Canvas(root,background='white')
    #canvas.grid(row=0, column=0, sticky='nsew')

    menubar = Menu(root)
    root.config(menu=menubar)
    submenu= Menu(menubar, tearoff=0)

    menubar.add_cascade(label='File', menu=submenu)
    submenu.add_command(label='New Canvas', command=app.new_canvas)
    submenu.add_command(label='Predict', command=app.results)

    predres = Label(root, text="Predictions:")
    #predres.grid(row=2, columnspan=2, sticky=W)
    predres.pack()
    
    image1 = PIL.Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image1)

    #canvas.bind("<B1-Motion>", locate_xy)
    #canvas.bind("<B1-Motion>", paint)

    root.mainloop()



