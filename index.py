
import numpy as np
import pandas as pd
from cv2 import *
import os
import sys
import math
import time
from sklearn import svm
import similarityScore
import pickle
import DataSet
import match
from array import array



# dictionary of labels
rnoMapping = {}
    
def takeSecond(elem):
    return elem[0][1]

def trainsvm(label_result):
    import TrainingSVMScore
    TrainingSVMScore.main()
    label_result.config(text="The training phase was runed.") 
    
def main(input_test_image,label_result,panel2,panel3,panel4,panel5):
       
    test_input_image=input_test_image
    
    filePart="dataset/recognition/"
    train_set, test_set=DataSet.LabelData("recognition")
    max_score=0
    pred_image=''
    all_array=[] 
    
    # all_array.append([])
    
    index_array=0
    kk=0
    for pic in train_set:
        # if kk<=2:
        all_array.append([])
        score=similarityScore.score(test_input_image,pic[0])
        # score=match.main(test_input_image,pic[0],'sift')
        # test=similarityScore.train(test_input_image,pic[0])
        test = np.array([[score]])    
        # load the model from disk
        filename = 'finalized_model1.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

        temp = loaded_model.predict(test)
        pred = temp.astype(np.int64)
        counts = np.bincount(pred)
        pred_label = np.argmax(counts)
        if pred_label==1:
            if score>max_score:
                max_score=score
                pred_image=pic[0]
            all_array[kk].append(pic[0])
            all_array[kk].append(float(score))
            # rnoMapping["path"]=pic[0]
            # rnoMapping["score"]=float(score)
            # print(pic[0])
            # print(score)
            index_array+=1
        else:
            print("0")
        kk+=1
        # else:
            # break
    # take second element for sort
    
    # print(rnoMapping)
    # dtype = [('path', 'S10'), ('score', float)]
    # a = np.array(all_array, dtype=dtype) 
    # np.sort(a, order='score') 
    # all_array.sort(key=takeSecond,reverse=True)
    # sorted(all_array,key=lambda l:l[1], reverse=True)
    print("--------------------------")
    u=0
    for f in all_array:
        if u<10:
            print(f[0])
            print(f[1])
        else:
            break
    st=pred_image.split('/')
    name=st[2]
    
    label_result.config(text="Name of person: "+name) 
    
    train_img = os.listdir(filePart+name+"/train/")
    iu=-1
    for img in train_img:  
        iu+=1
        if iu==0:
            img2 = ImageTk.PhotoImage(Image.open(filePart+name+"/train/"+img).resize((75,75)))
            panel2.configure(image = img2)   
            panel2.image = img2
        elif iu==1:
            img3 = ImageTk.PhotoImage(Image.open(filePart+name+"/train/"+img).resize((75,75)))
            panel3.configure(image = img3)   
            panel3.image = img3
        elif iu==2:
            img4 = ImageTk.PhotoImage(Image.open(filePart+name+"/train/"+img).resize((75,75)))
            panel4.configure(image = img4)   
            panel4.image = img4
        elif iu==3:
            img5 = ImageTk.PhotoImage(Image.open(filePart+name+"/train/"+img).resize((75,75)))
            panel5.configure(image = img5)
            panel5.image = img5
        else:
            break
        
    # print(pred_image)
    # print(max_score)

        
if __name__ == '__main__':
    from tkinter import *
    from functools import partial
    import tkinter
    from tkinter import filedialog
    # from Tkinter import Label,Tk
    from PIL import Image, ImageTk
    top = tkinter.Tk()
    # Code to add widgets will go here...
    top.geometry("400x600")  
    input_name = tkinter.StringVar()
      
    dis1 = Label(top, text = "Face Recognition").grid(row=0, column=0) 
    dis2 = Label(top, text = "Input image").grid(row=1, column=0) 

    #-----------------------------------------------------
    #------------------------------------------------------
    # canvas = Canvas(top, width = 300, height = 300)      
    # canvas.grid(row=4, column=0, columnspan = 3)      
    # img = PhotoImage(file=x)      
    # canvas.create_image(20,20, anchor=NW, image=img)      
    # mainloop()   
    # Select the Imagename from a folder 
    x = filedialog.askopenfilename(title ='"pen') 
    img = ImageTk.PhotoImage(Image.open(x))
    panel = Label(top, image = img)
    panel.grid(row = 10, column=0, columnspan = 3) 
    # top.mainloop()
    #------------------------------------------------------
        
    panel2 = Label(top)
    panel2.grid(row = 19, column=0)
    panel3 = Label(top)
    panel3.grid(row = 19, column=1)
    panel4 = Label(top)
    panel4.grid(row = 20, column=0)
    panel5 = Label(top)
    panel5.grid(row = 20, column=1)
    
    
    
    labelResult = Label(top)
    labelResult.grid(row=16, column=0, columnspan = 3)  
    # labelResult2 = Label(top)
    # labelResult2.grid(row=6, column=0, columnspan = 3) 
    main = partial(main, x,labelResult,panel2,panel3,panel4,panel5)  
    sbmitbtn = Button(top, text = "Recognition",activebackground = "pink", activeforeground = "blue", command=main)
    sbmitbtn.grid(row=12, column=0, columnspan = 3)  
    
    trainsvm1 = partial(trainsvm,labelResult)
    sbmitbtn1 = Button(top, text = "Training SVM",activebackground = "pink", activeforeground = "blue", command=trainsvm1)
    sbmitbtn1.grid(row=14, column=0, columnspan = 3) 

    
    uni1 = Label(top, text = "________________________________________________________________").grid(row=30, column=0, columnspan = 3) 
    uni = Label(top, text = "Design and development at Ferdowsi University of Mashhad").grid(row=31, column=0, columnspan = 3, pady = 5) 
    top.mainloop()