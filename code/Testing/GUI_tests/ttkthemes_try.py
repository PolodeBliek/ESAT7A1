# vb online
# import tkinter as tk
from tkinter import ttk  # Normal Tkinter.*widgets are not themed!
from ttkthemes import ThemedTk

window = ThemedTk(theme="arc")
f1 = ttk.Frame(window)
f1.pack(side='bottom')
ttk.Button(f1, text="Quit", command=window.destroy).pack()
window.mainloop()
