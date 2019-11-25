# file to process images:
import Main.whole_process2 as wp
# tkinter imports:
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
# other imports:
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import os
from multiprocessing import Process

LARGE_FONT, LARGER_FONT = ("Verdana", 12), ("Verdana", 18)
ESAT7A1 = os.path.dirname(os.path.abspath(__file__)).replace("code\\GUI", "")


class BROPAS(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # does the same as 'tk.Tk.__init__(self, *args, **kwargs)'
        self.title("BROPAS - Broad Range Object Processing and Analyzing Software")  # Object Counting Software
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.geometry("500x300")  # start dimensions

        # set up the container of all the screens/frames/menus
        container = tk.Frame(self, bg="black")
        container.grid(row=0, column=0, sticky="news")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        # 'collect' all the screens/frames/menus that you want to show
        self.frames = {}
        for F in (MainMenu, CreditsScreen, ScanScreen):
            frame = F(container, self)
            frame.grid(row=0, column=0, sticky="news")

            self.frames[F] = frame

        self.show_frame(MainMenu)  # show the default (start) screen

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class MainMenu(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        # info button
        ib = tk.Button(self, text="info", width=8, relief="groove", borderwidth=2, command=lambda: messagebox.showinfo(None, "Coming soon..."))
        ib.grid(row=0, column=0, sticky="w", padx=10)
        # methode button
        mb = tk.Button(self, text="De methode", width=16, height=2, relief="groove", borderwidth=2, command=lambda: messagebox.showinfo(None, "Coming soon..."))
        mb.grid(row=0, column=0, sticky="nes")

        # start button
        start_button = tk.Button(self, text="Start", width=10, height=3, bg="green", relief="groove", borderwidth=2,
                                 command=lambda: controller.show_frame(ScanScreen), font=LARGER_FONT)
        start_button.grid(row=1, column=0, sticky="s")

        # credits button
        credits_button = tk.Button(self, text="Credits", command=lambda: controller.show_frame(CreditsScreen))
        credits_button.grid(row=2, column=0, sticky="n")


class ScanScreen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.columnconfigure(2, weight=3)
        self.columnconfigure(3, weight=1)
        self.rowconfigure(0, weight=1)

        # via camera button:
        cb = tk.Button(self, text="Via camera", relief="groove", borderwidth=2, font=LARGER_FONT, bg="green", height=2,
                       command=lambda: self.run_with_kinect())
        cb.grid(row=0, column=0, sticky="ew", padx=20, columnspan=2)

        # via existing picture button
        pb = tk.Button(self, text="Bestaande foto", font=LARGER_FONT, relief="groove", bg="green", borderwidth=2, height=2,
                       command=lambda: self.run_on_selected_img())
        pb.grid(row=0, column=2, sticky="ew", padx=20, columnspan=2)

        # low threshold frame:
        low_fr = tk.Frame(self)
        low_fr.grid(row=1, column=0, sticky="e")
        # low label:
        lowlbl = tk.Label(low_fr, text="low threshold:")
        lowlbl.grid(row=0, column=0)
        # low threshold
        self.low_th = tk.IntVar(self); self.low_th.set(10)
        lowent = tk.Entry(low_fr, textvariable=self.low_th, width=3)
        lowent.grid(row=0, column=1)

        # high threshold frame:
        high_fr = tk.Frame(self)
        high_fr.grid(row=2, column=0, sticky="e")
        # high label:
        highlbl = tk.Label(high_fr, text='high threshold:')
        highlbl.grid(row=0, column=0)
        # high threshold
        self.high_th = tk.IntVar(self); self.high_th.set(200)
        highent = tk.Entry(high_fr, textvariable=self.high_th, width=3)
        highent.grid(row=0, column=1)

        # selection menu for gauss_reps
        gaussFrame = tk.Frame(self)
        gaussFrame.grid(row=3, column=0, sticky="e")
        gaussFrame.columnconfigure(0, weight=1)
        # gauss label
        gauss_label = tk.Label(gaussFrame, text="# blurs:")
        gauss_label.grid(row=0, column=0)
        # selection menu
        self.gauss_reps = tk.IntVar(self)
        self.gauss_reps.set(1)  # default value
        gauss_selection = tk.OptionMenu(gaussFrame, self.gauss_reps, *list(range(11)))
        gauss_selection.grid(row=0, column=1)

        # show images checkbox
        self.show_bool = tk.IntVar()
        show_cb = tk.Checkbutton(self, variable=self.show_bool, text="show images", font=LARGE_FONT,
                                  relief="groove", borderwidth=2)
        show_cb.select()  # set 'selected' state as default state
        show_cb.grid(row=1, column=1, sticky="ew", columnspan=2, padx=20)

        # save images checkbox
        self.save_bool = tk.IntVar()
        save_cb = tk.Checkbutton(self, variable=self.save_bool, text="save images", font=LARGE_FONT,
                                  relief="groove", borderwidth=2)
        # save_cb.select()  # set 'selected' state as default state
        save_cb.grid(row=2, column=1, sticky="ew", columnspan=2, padx=20)

        # hold on selected image button
        self.keep_path = tk.StringVar()
        self.keep_bool = tk.IntVar()
        hold_cb = tk.Checkbutton(self, variable=self.keep_bool, text="hold selected image", font=LARGE_FONT,
                                 relief='groove', borderwidth=2)
        hold_cb.grid(row = 3, column=1, sticky="ew", columnspan=2, padx=20)

        # gaussian blur checkbox:
        self.gauss_bool = tk.IntVar()
        gauss_cb = tk.Checkbutton(self, variable=self.gauss_bool, text="gaussian blur")
        gauss_cb.select()
        gauss_cb.grid(row=1, column=3, sticky="w")

        # sobel checkbox:
        self.sobel_bool = tk.IntVar()
        sobel_cb = tk.Checkbutton(self, variable=self.sobel_bool, text="sobel")
        sobel_cb.select()
        sobel_cb.grid(row=2, column=3, sticky="w")

        # hyst checkbox:
        self.hyst_bool = tk.IntVar()
        hyst_cb = tk.Checkbutton(self, variable=self.hyst_bool, text="hysteresis")
        hyst_cb.select()
        hyst_cb.grid(row=3, column=3, sticky="w")

        # fill checkbox:
        self.fill_bool = tk.IntVar()
        fill_cb = tk.Checkbutton(self, variable=self.fill_bool, text="fill")
        fill_cb.select()
        fill_cb.grid(row=4, column=3, sticky="w")

        # count checkbox:
        self.count_bool = tk.IntVar()
        count_cb = tk.Checkbutton(self, variable=self.count_bool, text="count objects")
        count_cb.select()
        count_cb.grid(row=5, column=3, sticky="w")

        # back button
        bb = tk.Button(self, text="back to menu", command=lambda: controller.show_frame(MainMenu))
        bb.grid(row=6, column=3, sticky="e")

    def get_filepath(self):
        filepath = filedialog.askopenfilename()
        return filepath

    def run_on_selected_img(self):
        # get the image
        if self.keep_bool.get() and not self.keep_path.get() == "":
            img_path = self.keep_path.get()
        else:
            img_path = self.get_filepath()
        if img_path == "":  # if no picture selected, return
            print("no picture selected")
            return
        self.keep_path.set(img_path)  # keep updating the selected image path
        image = np.array(Image.open(img_path))  # convert to workable array
        show_dict = {"Selected Image": (image, 'gray')}
        save_dict = {"SelectedImage": (image, f'{ESAT7A1}/Images/input images/')}
        self.color_pipeline(image, show_dict, save_dict)

    def run_with_kinect(self):
        if not wp.is_connected():
            messagebox.showerror("CONNECTION ERROR!",
                                 "You are currently not connected to the kinect.\nPlease connect and try again.")
            return
        color_image, depth_image = wp.kinect_to_pc(1080, 1920, 4)

        color_show_dict = {"Color Image": (color_image, 'gray')}
        color_save_dict = {"ColorImage": (color_image, f'{ESAT7A1}/Images/input images/')}
        self.color_pipeline(color_image, color_show_dict, color_save_dict)

        depth_show_dict = {"Color Image": (depth_image, 'gray')}
        depth_save_dict = {"ColorImage": (depth_image, f'{ESAT7A1}/Images/input images/')}
        self.depth_pipeline(depth_image, depth_show_dict, depth_save_dict)

    def color_pipeline(self, image, show_dict, save_dict):
        # maybe multiprocessing when showing images, otherwise you might have to wait to run depth_pipeline
        gray = wp.grayscale(image)  # grayscaling is necessary to the process
        show_dict.update({"Grayscaled": (gray, 'gray')})
        save_dict.update({"Gray": (gray, f'{ESAT7A1}/Images/grayscaled images/')})  # gray is the reference, don't change it
        if self.gauss_bool.get():
            gauss = wp.gaussian_blur(gray, self.gauss_reps.get())
            show_dict.update({"Gaussian Blur": (gauss, 'gray')})
            save_dict.update({"Gauss": (gauss, f'{ESAT7A1}/Images/blurred images/')})
        else:
            gauss = gray
        if self.sobel_bool.get():
            sobel = wp.sobel(gauss)
            show_dict.update({"Sobel": (sobel, 'gray')})
            save_dict.update({"Sobel": (sobel, f'{ESAT7A1}/Images/sobel images/')})
        else:
            sobel = gauss
        if self.hyst_bool.get():
            hyst = wp.hysteresis(sobel, self.low_th.get(), self.high_th.get())
            show_dict.update({"Hysteresis": (hyst, 'gray')})
            save_dict.update({"Hyst": (hyst, f'{ESAT7A1}/Images/hysteresis images/')})
        else:
            hyst = sobel
        if self.fill_bool.get():
            filled = ndimage.binary_fill_holes(hyst)
            show_dict.update({"Filled": (filled, 'gray')})
            save_dict.update({"Filled": (filled, f'{ESAT7A1}/Images/filled images/')})
        else:
            filled = hyst
        if self.sobel_bool.get():
            sobel2 = wp.sobel(gauss)
            show_dict.update({"Tweede Sobel": (sobel, 'gray')})
            save_dict.update({"Sobel2": (sobel, f'{ESAT7A1}/Images/sobel images/')})
        else:
            sobel2 = filled

        if self.count_bool.get():
            db = wp.detect_objects(filled)
            show_dict.update({"DBSCAN": (db, 'viridis')})
            save_dict.update({"DetectedObjects": (db, f'{ESAT7A1}/Images/object images/')})

        # save and show
        if self.show_bool.get():
            p = Process(target=wp.show_images, args=(show_dict,))
            p.start()
            # p.join()
            # wp.show_images(show_dict)
        if self.save_bool.get():
            wp.save_images(save_dict)

    def depth_pipeline(self, depth_image, show_dict, save_dict):
        pass


class CreditsScreen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="black")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # the menu frame holds the 'back' button and the title
        menuFrame = tk.Frame(self)
        menuFrame.grid(padx=10, pady=10, row=0, column=0, sticky="new")
        menuFrame.columnconfigure(0, weight=1)

        # title
        label = tk.Label(menuFrame, text="Credits", font=LARGE_FONT)
        label.grid(row=0, column=0, sticky="new")

        # 'back' button:
        self.image = tk.PhotoImage(file="GUI_images/back_arrow_icon.png")
        # resize the image to fit on the button
        self.downscaled_image = self.image.subsample(50, 50)
        # set the downscaled image on the button
        back_button = tk.Button(menuFrame, image=self.downscaled_image,
                                command=lambda: controller.show_frame(MainMenu))
        back_button.place(x=0, y=0, relheight=1, anchor="nw")

        # the credits space ('credits' is a built in function, hence the name 'creditss')
        creditss = tk.Text(self, height=2, width=30)
        creditss.grid(row=1, column=0, sticky="news")
        creditss.insert(tk.END, "*Brakke Gantt Chart: Robin")


if __name__ == '__main__':

    app = BROPAS()
    app.mainloop()
