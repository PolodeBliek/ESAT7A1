import tkinter as tk
from tkinter import Image, PhotoImage

root = tk.Tk()

frame = tk.Frame(root, bd=2, relief='sunken')

frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

xscrollbar = tk.Scrollbar(frame, orient='horizontal')
xscrollbar.grid(row=1, column=0, sticky='ew')

yscrollbar = tk.Scrollbar(frame)
yscrollbar.grid(row=0, column=1, sticky='ns')

canvas = tk.Canvas(frame, bd=0, xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set)
canvas.grid(row=0, column=0, sticky='news')
canvas.config(scrollregion=canvas.bbox(tk.ALL))

File = "jpg filepath here"
img = PhotoImage("file=GUI_images/TreeDiagram.png")
canvas.create_image(0,0,image=img, anchor="nw")

xscrollbar.config(command=canvas.xview)
yscrollbar.config(command=canvas.yview)

frame.pack()
root.mainloop()