from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 
import time
from PIL import ImageTk, Image  
import tkinter as tk
from tkinter import filedialog
import matplotlib
from matplotlib import pylab as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import nibabel as nib
import torch
import monai
import os



class MainApplication(tk.Frame):
    
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        file_path = ''
        top_frame = Frame(root)
        top_frame.pack()
        self.text = Label(
            top_frame, 
            text='Upload NIFTI file'
        )
        #self.text.pack(padx=5, pady=5, side=tk.LEFT)
        self.text.grid(row = 0, column = 0, sticky = W, padx = 16, pady = 16)
        btn_choose_file = Button(top_frame, text='Choose File', command=self.open_img)    
        btn_choose_file.grid(row=0,column=1,sticky = W, padx = 16, pady = 16)
        

        mid_frame = Frame(root)
        mid_frame.pack()
        self.figure = plt.figure(figsize=(9,3))
        # self.figures = []
        # for i in range(3):
        #     fig = plt.figure(figsize=(2, 2))
        #     self.figures.append(fig)
        #self.fig, self.axs = plt.subplots(3)
        #self.canvas = FigureCanvasTkAgg(self.figures[0])
        
        self.scale_sag = Scale(mid_frame, from_=0, to=250, orient=HORIZONTAL)
        self.scale_sag.grid(row = 1, column = 0, padx = 16)
        self.scale_cor = Scale(mid_frame, from_=0, to=250, orient=HORIZONTAL)
        self.scale_cor.grid(row = 1, column = 1, padx = 16)
        self.scale_ax = Scale(mid_frame, from_=0, to=250, orient=HORIZONTAL)
        self.scale_ax.grid(row = 1, column = 2, padx = 16)

        self.scale_ax.set(95)
        self.scale_sag.set(95)
        self.scale_cor.set(95)

        self.scale_sag.bind("<ButtonRelease-1>", self.update)
        self.scale_cor.bind("<ButtonRelease-1>", self.update)
        self.scale_ax.bind("<ButtonRelease-1>", self.update)


        self.canvas_frame = Frame(root)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas_frame.pack()
        
        self.canvas.get_tk_widget().pack()

        self.bott_frame = Frame(root)
        self.bott_frame.pack()
        btn_predict_structure = Button(self.bott_frame, text='Predict Structure Labels', command=self.predictFunc)
        btn_predict_structure.grid(row=3,column=3)

        btn_download_label = Button(self.bott_frame, text='Download Labels', command=self.open_img)
        btn_download_label.grid(row=4,column=3)
       
    
    # def updateAxial(self, event):
    #     self.plotAxial()
    #     self.drawCanvas()
    # def updateSagittal(self, event):
    #     self.plotSagittal()
    #     self.drawCanvas()
    # def updateCoronal(self, event):
    #     self.plotCoronal()
    #     self.drawCanvas()

    def update(self, event):
        self.plotAxial()
        self.plotSagittal()
        self.plotCoronal()
        self.drawCanvas()

    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename
    
    def drawCanvas(self):
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def loadNifti(self):
        self.image = np.array(nib.load(self.currFile).get_fdata(), dtype=np.float32)

    def plotAxial(self):
        self.canvas.get_tk_widget().pack_forget()   
        #plt.subplot(121) 
        ax = self.image[:, int(self.scale_sag.get()), :]
        ax = np.rot90(ax,axes=(-2,-1)) 
        #self.figures[0].imshow(ax, cmap="gray")
        plt.subplot(131)
        plt.imshow(ax, cmap='gray')

        
    def plotSagittal(self):
        self.canvas.get_tk_widget().pack_forget()    
        #plt.subplot(123) 
        sag = self.image[int(self.scale_cor.get()), :, :]
        sag = np.rot90(sag,axes=(-2,-1)) 
        plt.subplot(132)
        #self.figures[1]
        plt.imshow(sag, cmap="gray")
 

    def plotCoronal(self):
        #plt.subplot(122)
        self.canvas.get_tk_widget().pack_forget()    
        cor = self.image[:, :, int(self.scale_ax.get())]
        cor = np.rot90(cor,axes=(-2,-1))  
        plt.subplot(133)
        #self.figures[2]
        plt.imshow(cor, cmap="gray")

        
    def open_img(self):
        self.currFile = self.openfn()
        self.loadNifti()
        self.plotAxial()
        self.plotCoronal()
        self.plotSagittal()
        self.drawCanvas()

    def getSubvolumeOrigin(self): # HACK FUNCTION FOR TEMPORARY FIX OF MEMORY ISSUE.
        """Get the origin of the cubic subvolume at the centre of the image. This is just a temporary solution to the memory issue."""

        img = self.image

        xlen = img.shape[0]
        ylen = img.shape[1]
        zlen = img.shape[2]
        xmin = int((xlen-96)/2)
        ymin = int((ylen-96)/2)
        zmin = int((zlen-96)/2)
        return (xmin, ymin, zmin)

    def predictFunc(self):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = monai.networks.nets.HighResNet(spatial_dims=3,
                                           in_channels=1,
                                           out_channels=10,
                                           dropout_prob=0.0
                                       )
        
        #best_state_dict = torch.load(os.path.join(out_dir, 'best.pt'))
        cwd = os.getcwd()
        best_state_dict = torch.load(os.path.join(cwd, 'best_long.pt'),map_location=torch.device('cpu'))
        #model.to(device)
        model.load_state_dict(best_state_dict, strict=False)
        model.eval()
        with torch.no_grad():
            xmin, ymin, zmin = self.getSubvolumeOrigin()
            y = self.image[xmin:xmin+96, ymin:ymin+96, zmin:zmin+96]
        
            y = np.expand_dims(y, axis=0)
            y = np.expand_dims(y, axis=0)
            #img = img.to(device)
            yhat = model(torch.tensor(y))
            #yhat = yhat.to('cpu')
        yhat = np.array(yhat)
        yhat = yhat[0, :, :, :, :] # shape (num_labels+1, H, W, D) # get first item in batch (batch size is 1 here)
        yhat = np.argmax(yhat, axis=0)

        plt.imshow(yhat[:, :, 80], cmap="gray")
        plt.subplot(133)
        plt.imshow(yhat[:, 80, :], cmap="gray")
        plt.subplot(132)
        plt.imshow(yhat[80, :, :], cmap="gray")
        plt.subplot(131)

        self.drawCanvas()

        

if __name__ == "__main__": 

    # if os.environ.get('DISPLAY','') == '':
    #     print('no display found. Using :0.0')
    #     os.environ.__setitem__('DISPLAY', ':0.0')

    root = tk.Tk()
    root.title('GUI Test')
    root.geometry("800x500+30+30") 
    #root.grid_rowconfigure(0, weight=1)
    #root.grid_columnconfigure(0, weight=1)
    MainApplication(root)
    root.mainloop()