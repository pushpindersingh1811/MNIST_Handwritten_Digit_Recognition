# GUI

import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageGrab

#Load a model
from keras.models import load_model
model = load_model('MNIST_model.h5')
print('Model has successfully loaded...Go for the APP...')

# create a main window first
root = Tk()
root.resizable(0, 0)
root.title('Handwritten Digit Recognition GUI App.')

#Initialize few variables
lastx, lasty = None, None 
image_number = 0

#create a canvas for drawing
cv = Canvas(root, width = 640, height = 480, bg = 'white')
cv.grid(row = 0, column = 0, pady = 2, sticky = W, columnspan = 2)



def clear_widget():
    global cv
    # To clear a canvas
    cv.delete('all')
    
def activate_event(event):
    global lastx, lasty
    #<Button1>
    cv.bind('<B1-Motion>', draw_line)
    lastx, lasty = event.x, event.y
    
def draw_line(event):
    global lastx, lasty
    x, y = event.x, event.y
    # do the canvas drawing
    cv.create_line((lastx, lasty, x, y), width = 8, fill = 'black', capstyle = ROUND, smooth = TRUE, splinesteps = 12)
    lastx, lasty = x, y
    
def Recognize_Digit():
    global image_number
    prediction = []
    percentage = []
    #image_number = 0
    file_name = f'image_{image_number}.png'
    widget = cv
    
    #get the widget coordinates
    x = root.winfo_rootx()+widget.winfo_x()
    y = root.winfo_rooty()+widget.winfo_y()
    x1 = x+widget.winfo_width()
    y1 = y+widget.winfo_height()
    
    #grb the image, crop it according to my requirment and save it png format
    ImageGrab.grab().crop((x, y, x1, y1)).save(file_name)
    
    #read the image in colour format
    image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    #convert the image into grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Applying the Otsu Thresholding
    ret, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # FindContour helps to find the contour of the image
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
        break

    test_img = th[y:y+h, x:x+w]
    img = cv2.resize(test_img, (28, 28), interpolation = cv2.INTER_AREA)
    img = img.reshape(1, 28, 28, 1)
    model = load_model('MNIST_model.h5')
    prediction = model.predict(img)
    final_prediction = np.argmax(prediction)
    #cv2.putText() method to put text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 0, 0)
    thickness = 1
    cv2.putText(image, 'prediction: {}'.format(str(final_prediction)), (x, y-5), font, font_scale, color, thickness)
    
    # Show the predicted result on  new window
    cv2.imshow('image', image)
    cv2.waitKey(0)
    
#Mechanism to del with en=vent yourself
cv.bind('<Button-1>', activate_event)

#Add buttons and labels
btn_save= Button(text = 'Recognize Digit', command = Recognize_Digit)
btn_save.grid(row = 2, column = 0, pady = 1, padx = 1)
btn_clear = Button(text = 'Clear Widget', command = clear_widget)
btn_clear.grid(row = 2, column = 1, pady = 1, padx = 1)

#mainloop() is used when your application is ready to run
root.mainloop()
    
        
        
    
    
    
    
    

