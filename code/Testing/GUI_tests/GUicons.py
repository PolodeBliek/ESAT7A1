# file to process images:
import Main.whole_process2 as wp
# tkinter imports:
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from ttkthemes import ThemedTk
import ctypes
# other imports:
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import os
from multiprocessing import Process

ESAT7A1 = os.path.dirname(os.path.abspath(__file__)).replace("code\\GUI", "")

# Tk._windowingsystem == 'win32':


class BROPAS(ThemedTk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # does the same as 'tk.Tk.__init__(self, *args, **kwargs)'
        self.title("BROPAS - Broad Range Object Processing and Analyzing Software")  # Object Counting Software
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.geometry("525x300")  # start dimensions

        # set up the container of all the screens/frames/menus
        container = ttk.Frame(self)
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


class MainMenu(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1, minsize=100)
        self.rowconfigure(2, weight=1)

        # info button
        ib = ttk.Button(self, text="info", command=lambda: messagebox.showinfo(None, "Coming soon..."), takefocus=False)
        ib.grid(row=0, column=0, sticky="w")
        # methode button
        mb = ttk.Button(self, text="De methode", command=lambda: messagebox.showinfo(None, "Coming soon..."), takefocus=False)
        mb.grid(row=0, column=0, sticky="nes")

        # start button
        self.startbutton = tk.PhotoImage(file="imgs/startbutton3.png")
        self.startbuttonimg = self.startbutton.subsample(13, 13)
        start_button = ttk.Button(self, text="Start", command=lambda: controller.show_frame(ScanScreen),
                                  image=self.startbuttonimg, takefocus=False)
        start_button.grid(row=1, column=0, sticky="news", padx=160, pady=30)

        # credits button
        credits_button = ttk.Button(self, text="Credits", command=lambda: controller.show_frame(CreditsScreen), takefocus=False)
        credits_button.grid(row=2, column=0, sticky="n")


class ScanScreen(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.rowconfigure(1, weight=1, minsize=50)
        mnb = self.makemenu()
        self.keybd_event = ctypes.windll.user32.keybd_event
        self.alt_key = 0x12
        self.key_up = 0x0002

        # via camera button:
        self.kinect = tk.PhotoImage(file="imgs/kinect2.png")
        self.kinectimg = self.kinect.subsample(6, 6)
        cb = ttk.Button(self, text="Via Camera", command=lambda: self.run_with_kinect(), takefocus=False, image=self.kinectimg)
        cb.grid(row=1, column=0, sticky="news", padx=20, columnspan=2, pady=30)

        # via existing picture button
        self.folder = tk.PhotoImage(file="imgs/openfolderblack2.png")
        self.folderimg = self.folder.subsample(12, 12)
        pb = ttk.Button(self, text="Select Image", command=lambda: self.run_on_selected_img(), takefocus=False, image=self.folderimg)
        pb.grid(row=1, column=2, sticky="news", padx=20, columnspan=2, pady=30)

        # col1
        # low threshold frame:
        low_fr = ttk.Frame(self)
        low_fr.grid(row=2, column=0, sticky="e")
        # low label:
        lowlbl = ttk.Label(low_fr, text="low threshold:", takefocus=False)
        lowlbl.grid(row=0, column=0)
        # low threshold
        self.low_th = tk.IntVar(self); self.low_th.set(10)
        lowent = ttk.Entry(low_fr, textvariable=self.low_th, width=3)
        lowent.grid(row=0, column=1)

        # high threshold frame:
        high_fr = ttk.Frame(self)
        high_fr.grid(row=3, column=0, sticky="e")
        # high label:
        highlbl = ttk.Label(high_fr, text='high threshold:', takefocus=False)
        highlbl.grid(row=0, column=0)
        # high threshold
        self.high_th = tk.IntVar(self); self.high_th.set(100)
        highent = ttk.Entry(high_fr, textvariable=self.high_th, width=3)
        highent.grid(row=0, column=1)

        # selection menu for gauss_reps
        gaussFrame = ttk.Frame(self)
        gaussFrame.grid(row=4, column=0, sticky="e")
        gaussFrame.columnconfigure(0, weight=1)
        # gauss label
        gauss_label = ttk.Label(gaussFrame, text="# blurs:", takefocus=False)
        gauss_label.grid(row=0, column=0)
        # selection menu
        self.gauss_reps = tk.IntVar(self)
        gauss_selection = ttk.OptionMenu(gaussFrame, self.gauss_reps, *list(range(-1, 11)))
        self.gauss_reps.set(4)  # default value
        gauss_selection.grid(row=0, column=1)

        # col2
        # show images checkbox
        self.show_bool = tk.IntVar(); self.show_bool.set(1)
        show_cb = ttk.Checkbutton(self, variable=self.show_bool, text="show images", takefocus=False)
        show_cb.grid(row=2, column=1, columnspan=2, padx=20)

        # save images checkbox
        self.save_bool = tk.IntVar(); self.save_bool.set(0)
        save_cb = ttk.Checkbutton(self, variable=self.save_bool, text="save images", takefocus=False)
        save_cb.grid(row=3, column=1, columnspan=2, padx=20)

        # hold on selected image button
        self.keep_path = tk.StringVar()
        self.keep_bool = tk.IntVar(); self.keep_bool.set(0)
        hold_cb = ttk.Checkbutton(self, variable=self.keep_bool, text="hold on image", takefocus=False)
        hold_cb.grid(row=4, column=1, columnspan=2, padx=20)

        # col3
        # gaussian blur checkbox:
        self.gauss_bool = tk.IntVar(); self.gauss_bool.set(1)
        gauss_cb = ttk.Checkbutton(self, variable=self.gauss_bool, text="gaussian blur", takefocus=False)
        # gauss_cb.select()
        gauss_cb.grid(row=2, column=3, sticky="w")

        # sobel checkbox:
        self.sobel_bool = tk.IntVar(); self.sobel_bool.set(1)
        sobel_cb = ttk.Checkbutton(self, variable=self.sobel_bool, text="sobel", takefocus=False)
        sobel_cb.grid(row=3, column=3, sticky="w")

        # hyst checkbox:
        self.hyst_bool = tk.IntVar(); self.hyst_bool.set(1)
        hyst_cb = ttk.Checkbutton(self, variable=self.hyst_bool, text="hysteresis", takefocus=False)
        hyst_cb.grid(row=4, column=3, sticky="w")

        # fill checkbox:
        self.fill_bool = tk.IntVar(); self.fill_bool.set(1)
        fill_cb = ttk.Checkbutton(self, variable=self.fill_bool, text="fill", takefocus=False)
        fill_cb.grid(row=5, column=3, sticky="w")

        # sobel2 checkbox:
        self.senne_bool = tk.IntVar(); self.senne_bool.set(1)
        senne_cb = ttk.Checkbutton(self, variable=self.senne_bool, text="senne count", takefocus=False)
        senne_cb.grid(row=6, column=3, sticky="w")

        # count checkbox:
        self.count_bool = tk.IntVar(); self.count_bool.set(1)
        count_cb = ttk.Checkbutton(self, variable=self.count_bool, text="count objects", takefocus=False)
        count_cb.grid(row=7, column=3, sticky="w")

        # count checkbox:
        self.box_bool = tk.IntVar(); self.box_bool.set(1)
        box_cb = ttk.Checkbutton(self, variable=self.box_bool, text="draw boxes", takefocus=False)
        box_cb.grid(row=8, column=3, sticky="w")

        # back button
        bb = ttk.Button(self, text="back to menu", command=lambda: controller.show_frame(MainMenu), takefocus=False)
        bb.grid(row=9, column=3, sticky="e")

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
        if self.senne_bool.get():
            sobel2 = wp.sobel(filled)
            senne_obj, nb_obj = wp.detect_objects_senne(0, sobel2, 100)
            show_dict.update({f"senne: {nb_obj} objects": (senne_obj, 'gray')})
            save_dict.update({"Sobel2": (senne_obj, f'{ESAT7A1}/Images/sobel images/')})
        else:
            sobel2 = filled

        if self.count_bool.get():
            # hier dan sennes algoritme in steken
            db, nb_objects = wp.detect_objects(filled)
            show_dict.update({"DBSCAN": (db, 'viridis')})
            save_dict.update({"DetectedObjects": (db, f'{ESAT7A1}/Images/object images/')})
        else:
            db = filled
            nb_objects = None
        if self.box_bool.get():
            boxes = wp.draw_boxes(image, db)
            show_dict.update({"boxes": (boxes, 'gray')})
            save_dict.update({"Boxes": (boxes, f'{ESAT7A1}/Images/draw boxes/')})

        # save and show
        if self.show_bool.get():
            p = Process(target=wp.show_images, args=(show_dict, nb_objects))
            p.start()
            # p.join()
            # wp.show_images(show_dict)
        if self.save_bool.get():
            wp.save_images(save_dict)

    def depth_pipeline(self, depth_image, show_dict, save_dict):
        pass

    def traverse_to_menu(self, key=''):
        print("hey")
        if key:
            print("key")
            self.ansi_key = ord(key.upper())
            #   press alt + key
            self.keybd_event(self.alt_key, 0, 0, 0)
            self.keybd_event(self.ansi_key, 0, 0, 0)

            #   release alt + key
            self.keybd_event(self.ansi_key, 0, self.key_up, 0)
            self.keybd_event(self.alt_key, 0, self.key_up, 0)

    def makemenu(self):
        # apply bools
        gauss_applybool = tk.BooleanVar()
        sobel_applybool = tk.BooleanVar()
        hyst_applybool = tk.BooleanVar()
        senne_applybool = tk.BooleanVar()
        db_applybool = tk.BooleanVar()
        boxes_applybool = tk.BooleanVar()

        # show bools
        og_showbool = tk.BooleanVar()
        gray_showbool = tk.BooleanVar()
        gauss_showbool = tk.BooleanVar()
        sobel_showbool = tk.BooleanVar()
        hyst_showbool = tk.BooleanVar()
        senne_showbool = tk.BooleanVar()
        db_showbool = tk.BooleanVar()
        boxes_showbool = tk.BooleanVar()

        # main menu bar
        menubar = tk.Frame(self, relief=tk.RAISED, bd=1, bg='#D1D1D1')  # , relief=tk.RAISED, bd=2
        menubar.grid(row=0, column=0, sticky='ew', columnspan=4)

        # variables button
        variablesbutton = tk.Menubutton(menubar, text='Variables', underline=0, bg='#D1D1D1', bd=2)
        variablesbutton.grid(row=0, column=0, sticky='w')
        variables = tk.Menu(variablesbutton, tearoff=False)
        variables.add_separator()
        variablesbutton.config(menu=variables)

        # apply menu (dropdown)
        applybutton = tk.Menubutton(menubar, text='Apply', underline=0, bg='#D1D1D1', bd=2)
        applybutton.grid(row=0, column=1, sticky="w")
        apply = tk.Menu(applybutton)
        apply.add_checkbutton(label='gaussian blur', variable=gauss_applybool, command=lambda: self.traverse_to_menu('a'))
        apply.add_checkbutton(label='sobel', variable=sobel_applybool, command=lambda: self.traverse_to_menu('a'))
        apply.add_checkbutton(label='hysteresis', variable=hyst_applybool, command=lambda: self.traverse_to_menu('a'))
        apply.add_checkbutton(label='senne', variable=senne_applybool, command=lambda: self.traverse_to_menu('a'))
        apply.add_checkbutton(label='dbscan', variable=db_applybool, command=lambda: self.traverse_to_menu('a'))
        apply.add_checkbutton(label='boxes', variable=boxes_applybool, command=lambda: self.traverse_to_menu('a'))
        applybutton.config(menu=apply)

        # show menu (dropdown)
        showbutton = tk.Menubutton(menubar, text='Show', underline=0, bg='#D1D1D1', bd=2)
        showbutton.grid(row=0, column=2, sticky="w")
        show = tk.Menu(showbutton)
        show.add_checkbutton(label='original image', variable=og_showbool, command=lambda: self.traverse_to_menu('s'))
        show.add_checkbutton(label='grayscale', variable=gray_showbool, command=lambda: self.traverse_to_menu('s'))
        show.add_checkbutton(label='gaussian blur', variable=gauss_showbool, command=lambda: self.traverse_to_menu('s'))
        show.add_checkbutton(label='sobel', variable=sobel_showbool, command=lambda: self.traverse_to_menu('s'))
        show.add_checkbutton(label='hysteresis', variable=hyst_showbool, command=lambda: self.traverse_to_menu('s'))
        show.add_checkbutton(label='senne', variable=senne_showbool, command=lambda: self.traverse_to_menu('s'))
        show.add_checkbutton(label='dbscan', variable=db_showbool, command=lambda: self.traverse_to_menu('s'))
        show.add_checkbutton(label='boxes', variable=boxes_showbool, command=lambda: self.traverse_to_menu('s'))
        showbutton.config(menu=show)

        # help menu
        helpmn = tk.Menubutton(menubar, text='Help', underline=0, bg='#D1D1D1', bd=2)
        helpmn.grid(row=0, column=3, sticky='w')
        help = tk.Menu(helpmn)
        help.add_command(label="I'm stupid")
        helpmn.config(menu=help)

        # seperator = tk.Menubutton(menubar, text='\u22EE')
        # seperator.grid(row=0, column=4)

        # submenu = tk.Menu(apply, tearoff=0)
        # submenu.add_command(label='Spam', command=self.quit, underline=0)
        # submenu.add_command(label='Eggs', command=lambda: print("alleez, het werkt"), underline=0)
        # apply.add_cascade(label='Stuff', menu=submenu, underline=0)
        return menubar


class CreditsScreen(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # 'back' button:
        back_button = ttk.Button(self, text='back to menu', command=lambda: controller.show_frame(MainMenu), takefocus=False)
        back_button.grid(row=0, column=0, sticky='w')

        # the credits space ('credits' is a built in function, hence the name 'creditss')
        creditss = tk.Text(self, height=2, width=30)
        creditss.grid(row=1, column=0, sticky="news")
        creditss.insert(tk.END, "*Brakke Gantt Chart: Robin")


if __name__ == '__main__':

    app = BROPAS()
    app.get_themes()
    app.set_theme('plastik')
    app.mainloop()
