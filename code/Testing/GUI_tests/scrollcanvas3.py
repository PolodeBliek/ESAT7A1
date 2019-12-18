# from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk


class PipelineScreen(tk.Tk):
    def __init__(self, parent=None):
        tk.Tk.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        backbtn = tk.Button(text='back')
        backbtn.grid(row=0, column=0)
        pipeln = ScrolledCanvas(self)
        pipeln.grid(row=1, column=0)


class ScrolledCanvas(tk.Frame):
    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)
        self.grid(sticky='nesw')
        canv = tk.Canvas(self, relief='sunken')
        canv.config(width=400, height=200)
        canv.config(highlightthickness=0)
        sbarV = tk.Scrollbar(self, orient='vertical')
        sbarH = tk.Scrollbar(self, orient='horizontal')
        sbarV.config(command=canv.yview)
        sbarH.config(command=canv.xview)
        canv.config(yscrollcommand=sbarV.set)
        canv.config(xscrollcommand=sbarH.set)
        sbarV.pack(side='right', fill='y')
        sbarH.pack(side='bottom', fill='x')
        canv.pack(side='left', expand='yes', fill='both')
        self.im = Image.open("imgs/TreeDiagram.png")
        # w, h = self.im.size
        # self.im = self.im.resize((int(w/2), int(h/2)))
        width, height = self.im.size
        canv.config(scrollregion=(0, 0, width, height))
        self.im2 = ImageTk.PhotoImage(self.im)
        self.imgtag = canv.create_image(0, 0, anchor="nw", image=self.im2)


root = PipelineScreen()
root.mainloop()

# ScrolledCanvas().mainloop()