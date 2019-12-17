import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk, FigureCanvasTk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Process


a2 = np.array([[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]])


def show_images(d: dict):
    """
        Plot all the images in dict
    """
    n = len(d)
    k = int(math.sqrt(n)) + 1
    f = plt.figure(constrained_layout = True, num = f'Results:')
    spec = gridspec.GridSpec(ncols=k, nrows=n//k+1, figure=f)
    for i, key, vals in zip(range(n), d.keys(), d.values()):
        img, map_ = vals
        ax = f.add_subplot(spec[i//k, i % k])
        ax.imshow(img, cmap=map_)
        plt.xticks([])
        plt.yticks([])
        ax.set_xlabel(f"{key}")

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()
    return f


def plot_imgs(img):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    return fig


def save_imgs(img):
    print("image saved")


class WindowWindow(tk.Frame):
    counter = 0
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.button = tk.Button(self, text="Create new window",
                                command=self.create_window)
        self.button.pack(side="top")

    def create_window(self):
        self.counter += 1
        t = tk.Toplevel(self)
        t.wm_title("Window #%s" % self.counter)
        try:
            self.canvas.get_tk_widget().pack_forget()
        except AttributeError:
            pass
        showdict = {"piep": (np.array([[1, 0], [0, 1]]),'gray')}
        savedict = 3
        fig = show_images(showdict)
        canvas = FigureCanvasTkAgg(fig, master=t)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        toolbar = NavigationToolbar2Tk(canvas, t)
        toolbar.update()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        button = tk.Button(master=t, text="Save all images", command=lambda: save_imgs(savedict))
        button.pack(side='bottom')


class APP(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # does the same as 'tk.Tk.__init__(self, *args, **kwargs)'
        self.title("ESAT7A1")  # Object Counting Software
        startbtn = tk.Button(self, text='start', command=self.pipeline)
        startbtn.pack()

        poep = WindowWindow(self)
        poep.pack(side='top')

    def pipeline(self):
        a1 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0]])
        show_dict = {"foto1": (a1, 'gray'), "foto2": (a1, 'viridis')}
        save_dict = a1
        p = Process(target=self.create_plot_window(show_dict, save_dict))
        p.start()
        # p = Process(target=self.print_shit(3))
        # p.start()

    def create_plot_window(self, showdict, savedict):
        t = tk.Toplevel(self)
        t.wm_title("Window two")
        # fig = plot_imgs(showdict)
        fig = show_images(showdict)
        canvas = FigureCanvasTkAgg(fig, master=t)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        toolbar = NavigationToolbar2Tk(canvas, t)
        toolbar.update()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        button = tk.Button(master=t, text="Save all images", command=lambda: save_imgs(savedict))
        button.pack(side='bottom')

    def print_shit(self, pimel):
        print("shit")


if __name__ == '__main__':
    app = APP()
    app.mainloop()


# root = tkinter.Tk()
# root.wm_title("Embedding in Tk")

# fig = Figure(figsize=(5, 4), dpi=100)
# # t = np.arange(0, 3, .01)
# # fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
# ax = fig.add_subplot(111)
# ax.imshow(a1, cmap='gray')

# canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
# canvas.draw()
# canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

# toolbar = NavigationToolbar2Tk(canvas, root)
# toolbar.update()
# canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


# def on_key_press(event):
#     print("you pressed {}".format(event.key))
#     key_press_handler(event, canvas, toolbar)
#
#
# canvas.mpl_connect("key_press_event", on_key_press)


# def _quit():
#     root.quit()     # stops mainloop
#     root.destroy()  # this is necessary on Windows to prevent
#                     # Fatal Python Error: PyEval_RestoreThread: NULL tstate


# button = tkinter.Button(master=root, text="Quit", command=_quit)
# button.pack(side=tkinter.BOTTOM)
#
# tkinter.mainloop()

