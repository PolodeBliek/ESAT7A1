# file to process images:
import Main.Counting_algorithm as wp
# tkinter imports:
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import messagebox
from tkinter import filedialog
import ctypes
import matplotlib.pyplot as plt

# other imports:
import numpy as np
from scipy import ndimage
import os
import pickle
import copy
import PIL.ImageOps
from PIL import Image
from multiprocessing import Process
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import time

ESAT7A1 = os.path.dirname(os.path.abspath(__file__)).replace("code\\GUI", "")

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
kinect2 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)


class BROPAS(ThemedTk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # does the same as 'tk.Tk.__init__(self, *args, **kwargs)'
        self.title("ESAT7A1")  # Object Counting Software
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.geometry("525x300")  # start dimensions

        self.emptyMenu = tk.Menu(self)

        # set up the container of all the screens/frames/menus
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="news")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        # 'collect' all the screens/frames/menus that you want to show
        self.frames = {}
        for F in (StartScreen, CreditsScreen, ScanScreen):
            frame = F(container, self)
            frame.grid(row=0, column=0, sticky="news")

            self.frames[F] = frame

        self.show_frame(StartScreen)  # show the default (start) screen

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

        if isinstance(self.frames[cont], ScanScreen):
            menubar = frame.makemenu(self)
            self.config(menu=menubar)
        else:
            self.configure(menu=self.emptyMenu)


class StartScreen(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1, minsize=50)
        self.rowconfigure(1, weight=1, minsize=100)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # padframe:
        padfr = tk.Frame(self)
        padfr.grid(row=0, column=0, columnspan=2, sticky='news', pady=20)

        # info button
        infobtn = ttk.Button(self, text="  info  ", command=lambda: messagebox.showinfo(None, "Coming soon..."), takefocus=False)
        infobtn.grid(row=2, column=0, sticky="ne")
        # methode button
        pipelinebtn = ttk.Button(self, text="pipeline", command=lambda: messagebox.showinfo(None, "Coming soon..."), takefocus=False)
        pipelinebtn.grid(row=2, column=1, sticky="nw")

        # start button
        self.startbtnimg = tk.PhotoImage(file="GUI_images/startbutton1.png")
        self.startbtnimg_ = self.startbtnimg.subsample(7, 8)
        start_button = ttk.Button(self, text="Start", command=lambda: controller.show_frame(ScanScreen),
                                  image=self.startbtnimg_, takefocus=False)
        start_button.grid(row=1, column=0, sticky="news", padx=160, pady=30, columnspan=2)


class ScanScreen(tk.Frame):
    # making the scanscreen
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        # self.grid_propagate()

        # for keeping the menu open
        self.keybd_event = ctypes.windll.user32.keybd_event
        self.alt_key = 0x12
        self.key_up = 0x0002

        ## all variables:
        # apply bools
        self.gauss_applybool = tk.BooleanVar(); self.gauss_applybool.set(1)
        self.sobel_applybool = tk.BooleanVar(); self.sobel_applybool.set(1)
        self.hyst_applybool = tk.BooleanVar(); self.hyst_applybool.set(1)
        self.fill_applybool = tk.BooleanVar(); self.fill_applybool.set(1)
        self.senne_applybool = tk.BooleanVar(); self.senne_applybool.set(1)
        self.db_applybool = tk.BooleanVar()
        self.boxes_applybool = tk.BooleanVar()
        self.depth_applybool = tk.BooleanVar()
        self.hasoverlap_applybool = tk.BooleanVar()

        # show bools
        self.show_all_bool = tk.BooleanVar()
        self.colorimg_showbool = tk.BooleanVar(); self.colorimg_showbool.set(1)
        self.depthimg_showbool = tk.BooleanVar()
        self.gray_showbool = tk.BooleanVar()
        self.gauss_showbool = tk.BooleanVar()
        self.sobel_showbool = tk.BooleanVar()
        self.hyst_showbool = tk.BooleanVar()
        self.fill_showbool = tk.BooleanVar()
        self.senne_showbool = tk.BooleanVar(); self.senne_showbool.set(1)
        self.db_showbool = tk.BooleanVar()
        self.boxes_showbool = tk.BooleanVar()
        self.depth_algorithm_showbool = tk.BooleanVar()
        self.hasoverlap_showbool = tk.BooleanVar()

        # save bool
        self.save_bool = tk.BooleanVar()

        # variables (low threshold, high threshold and #gaussian blurs)
        self.low_th = tk.IntVar(self); self.low_th.set(10)
        self.high_th = tk.IntVar(self); self.high_th.set(100)
        self.gauss_reps = tk.IntVar(self); self.gauss_reps.set(4)  # default value

        # make the variable menubar
        self.makevarbar()

        ## the picture buttons n stuff:
        # the frames:
        cameraFrame = tk.Frame(self)
        cameraFrame.grid(row=1, column=0, padx=20, pady=30, sticky='news')
        cameraFrame.rowconfigure(0, weight=1)
        cameraFrame.columnconfigure(0, weight=1)
        selectedimgFrame = tk.Frame(self)
        selectedimgFrame.grid(row=1, column=1, padx=20, pady=30, sticky='news')
        selectedimgFrame.rowconfigure(0, weight=1)
        selectedimgFrame.columnconfigure(0, weight=1)

        # via camera button:
        self.kinect = tk.PhotoImage(file="GUI_images/kinect2.png")
        self.kinectimg = self.kinect.subsample(6, 6)
        cb = ttk.Button(cameraFrame, command=lambda: self.run_with_kinect(), takefocus=False, image=self.kinectimg)
        cb.grid(row=0, column=0, sticky="news")

        # via existing picture button
        self.folder = tk.PhotoImage(file="GUI_images/openfolderblack2.png")
        self.folderimg = self.folder.subsample(12, 12)
        pb = ttk.Button(selectedimgFrame, command=lambda: self.run_on_selected_img(), takefocus=False, image=self.folderimg)
        pb.grid(row=0, column=0, sticky="news")

        # initialize the depth camera
        self.initialize = tk.BooleanVar()
        initialize_cb = ttk.Checkbutton(cameraFrame, variable=self.initialize, text="Initialize depth camera                  ", takefocus=False)
        # depth_cb.select()
        initialize_cb.grid(row=1, column=0, sticky="ew")

        # hold on selected image button
        self.hold_bool = tk.BooleanVar()
        self.color_hold_path = tk.StringVar()
        self.depth_hold_path = tk.StringVar()
        hold_cb = ttk.Checkbutton(selectedimgFrame, variable=self.hold_bool, text="Hold on selected image                   ", takefocus=False)
        hold_cb.grid(row=1, column=0, sticky="ew", columnspan=2)

    def traverse_to_menu(self, key=''):
        if key:
            ansi_key = ord(key.upper())
            #   press alt + key
            self.keybd_event(self.alt_key, 0, 0, 0)
            self.keybd_event(ansi_key, 0, 0, 0)

            #   release alt + key
            self.keybd_event(ansi_key, 0, self.key_up, 0)
            self.keybd_event(self.alt_key, 0, self.key_up, 0)

    def makemenu(self, controller):

        # main menu bar
        menubar = tk.Menu(controller)  # , relief=tk.RAISED, bd=2
        controller.config(menu=menubar)

        # back button
        menubar.add_command(label='Back', command=lambda: controller.show_frame(StartScreen), underline=0)

        # apply menu (dropdown)
        applymenu = tk.Menu(menubar)  # , tearoff=0
        applymenu.add_checkbutton(label='gaussian blur', variable=self.gauss_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_checkbutton(label='sobel', variable=self.sobel_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_checkbutton(label='hysteresis', variable=self.hyst_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_checkbutton(label='fill', variable=self.fill_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_separator()
        applymenu.add_checkbutton(label='from scratch', variable=self.senne_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_checkbutton(label='dbscan', variable=self.db_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_separator()
        applymenu.add_checkbutton(label='draw boxes', variable=self.boxes_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_separator()
        applymenu.add_checkbutton(label='depth', variable=self.depth_applybool, command=lambda: self.traverse_to_menu('a'))
        applymenu.add_checkbutton(label='has overlap', variable=self.hasoverlap_applybool, command=lambda: self.traverse_to_menu('a'))
        menubar.add_cascade(menu=applymenu, label="Apply", underline=0)

        # show menu (dropdown)
        showmenu = tk.Menu(menubar)  # , tearoff=0
        showmenu.add_checkbutton(label='color image', variable=self.colorimg_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_checkbutton(label='depth image', variable=self.depthimg_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_separator()
        showmenu.add_checkbutton(label='grayscale', variable=self.gray_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_checkbutton(label='gaussian blur', variable=self.gauss_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_checkbutton(label='sobel', variable=self.sobel_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_checkbutton(label='hysteresis', variable=self.hyst_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_checkbutton(label='fill', variable=self.fill_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_separator()
        showmenu.add_checkbutton(label='from scratch', variable=self.senne_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_checkbutton(label='dbscan', variable=self.db_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_separator()
        showmenu.add_checkbutton(label='draw boxes', variable=self.boxes_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_separator()
        showmenu.add_checkbutton(label='depth', variable=self.depth_algorithm_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_checkbutton(label='has overlap', variable=self.hasoverlap_showbool, command=lambda: self.traverse_to_menu('s'))
        showmenu.add_separator()
        showmenu.add_checkbutton(label='show all', variable=self.show_all_bool, command=lambda: self.traverse_to_menu('s'))
        menubar.add_cascade(menu=showmenu, label="Show", underline=0)

        # help menu
        menubar.add_command(label="Help", underline=0, command=lambda: print('coming soon'))

        # submenu = tk.Menu(apply)  # , tearoff=0
        # submenu.add_command(label='Spam', command=self.quit, underline=0)
        # submenu.add_command(label='Eggs', command=lambda: print("alleez, het werkt"), underline=0)
        # apply.add_cascade(label='Stuff', menu=submenu, underline=0)
        return menubar

    def makevarbar(self):

        def on_enter(name_):
            if name_ == 'low':
                lowlbl['background'] = 'white'
            if name_ == 'high':
                highlbl['background'] = 'white'
            if name_ == 'gauss':
                gauss_label['background'] = 'white'

        def on_leave(name_):
            if name_ == 'low':
                lowlbl['background'] = 'SystemButtonFace'
            if name_ == 'high':
                highlbl['background'] = 'SystemButtonFace'
            if name_ == 'gauss':
                gauss_label['background'] = 'SystemButtonFace'

        varFrame = tk.Frame(self, bd=2, relief='groove')
        varFrame.grid(row=0, column=0, columnspan=2, sticky='ew')

        # varlabel:
        varlbl = tk.Button(varFrame, text='VARIABLES: ', bg='#D1D1D1', relief='flat', bd=1)
        varlbl.grid(row=0, column=0, sticky='ns')

        # low threshold frame:
        low_fr = tk.Frame(varFrame)
        low_fr.grid(row=0, column=1, sticky="nes")
        # low label:
        lowlbl = tk.Label(low_fr, text=" Low threshold: ")
        lowlbl.grid(row=0, column=0)
        lowent = ttk.Entry(low_fr, textvariable=self.low_th, width=3)
        lowent.grid(row=0, column=1)
        lowlbl.bind('<Enter>', lambda e: on_enter('low'))
        lowlbl.bind('<Leave>', lambda e: on_leave('low'))

        sep1 = tk.Menubutton(varFrame, text='\u22EE')
        sep1.grid(row=0, column=2)

        # high threshold frame:
        high_fr = tk.Frame(varFrame)
        high_fr.grid(row=0, column=3, sticky="nes")
        # high label:
        highlbl = tk.Label(high_fr, text="High threshold: ")
        highlbl.grid(row=0, column=0)
        # high threshold
        highent = ttk.Entry(high_fr, textvariable=self.high_th, width=3)
        highent.grid(row=0, column=1)
        highlbl.bind('<Enter>', lambda e: on_enter('high'))
        highlbl.bind('<Leave>', lambda e: on_leave('high'))

        sep2 = tk.Menubutton(varFrame, text='\u22EE')
        sep2.grid(row=0, column=4)

        # selection menu for gauss_reps
        gaussFrame = tk.Frame(varFrame)
        gaussFrame.grid(row=0, column=5, sticky="nes")
        gaussFrame.columnconfigure(0, weight=1)
        # gauss label
        gauss_label = ttk.Label(gaussFrame, text="Times blur: ")
        gauss_label.grid(row=0, column=0)
        # selection menu
        gauss_selection = ttk.OptionMenu(gaussFrame, self.gauss_reps, *list(range(11)))
        gauss_selection.grid(row=0, column=1)
        gauss_label.bind('<Enter>', lambda e: on_enter('gauss'))
        gauss_label.bind('<Leave>', lambda e: on_leave('gauss'))

    # the actual functions/pipelines
    def run_on_selected_img(self):
        # get the image
        if self.hold_bool.get() and not self.color_hold_path.get() == "":
            color_path = self.color_hold_path.get()
        else:
            color_path = filedialog.askopenfilename()
        if color_path == "":  # if no picture selected, return
            messagebox.showinfo("No picture selected", "Pipeline hasn't been executed, please select a picture and try again.")
            return
        self.color_hold_path.set(color_path)  # keep updating the selected image path
        color_image = np.array(Image.open(color_path))  # convert to workable array

        show_dict = {}
        if self.colorimg_showbool.get() or self.show_all_bool.get():
            show_dict.update({"Selected Color Image": (color_image, 'gray')})
        save_dict = {"SelectedColorImage": (color_image, f'{ESAT7A1}/Images/input images/')}

        # do the same for depth if depth apply is selected
        if self.depth_applybool.get():
            if self.hold_bool.get() and not self.color_hold_path.get() == "":
                depth_path = self.color_hold_path.get()
            else:
                depth_path = filedialog.askopenfilename()

            if depth_path == "":  # if no picture selected, return
                print("No picture selected")
                return

            self.depth_hold_path.set(depth_path)
            depth_image = np.array(Image.open(depth_path))  # convert to workable array

            if self.depthimg_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Selected Depth Image": (depth_image, 'gray')})

        if not self.depth_applybool.get():
            self.color_pipeline(color_image, show_dict, save_dict)
        else:
            self.depth_pipeline(depth_image, color_image, show_dict, save_dict)

    def run_with_kinect(self):
        if not wp.is_connected():
            messagebox.showerror("CONNECTION ERROR!",
                                 "You are currently not connected to the kinect.\nPlease connect and try again.")
            return
        if self.initialize.get():
            wp.kinect_to_pc(kinect, kinect2, True)
            messagebox.showinfo("Info", 'Camera is initialized')
            return
        else:
            color_image, depth_image = wp.kinect_to_pc(kinect, kinect2, False)

        color_image = color_image[90:990, 310:1610]  # cropping
        depth_image = depth_image

        color_show_dict = {}
        if self.colorimg_showbool.get() or self.show_all_bool.get():
            color_show_dict.update({"Color Image": (color_image, 'gray')})
        color_save_dict = {"ColorImage": (color_image, f'{ESAT7A1}/Images/input images/')}

        if not self.initialize.get():
            if not self.depth_applybool.get():
                self.color_pipeline(color_image, color_show_dict, color_save_dict)

            if self.depth_applybool.get():
                depth_show_dict = {"Color Image": (color_image, 'viridis')}
                # depth_show_dict = {"Color Image": (color_image, 'viridis')}  # wordt er later i/d pipeline ingezet
                depth_save_dict = {"Color Image": (color_image, f'{ESAT7A1}/Images/input images/')}
                # depth_save_dict.update({"Depth Image": (depth_image, f'{ESAT7A1}/Images/input images/')})  # idk
                self.depth_pipeline(depth_image, color_image, depth_show_dict, depth_save_dict)

    def color_pipeline(self, image, show_dict, save_dict):
        # maybe multiprocessing when showing images, otherwise you might have to wait to run depth_pipeline
        gray = wp.grayscale(image)  # grayscaling is necessary to the process
        if self.gray_showbool.get() or self.show_all_bool.get():
            show_dict.update({"Grayscaled": (gray, 'gray')})
        save_dict.update({"Gray": (gray, f'{ESAT7A1}/Images/grayscaled images/')})  # gray is the reference for saving
        if self.gauss_applybool.get():
            gauss = wp.gaussian_blur(gray, self.gauss_reps.get())
            if self.gauss_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Gaussian Blur": (gauss, 'gray')})
            save_dict.update({"Gauss": (gauss, f'{ESAT7A1}/Images/blurred images/')})
        else:
            gauss = gray
        if self.sobel_applybool.get():
            sobel = wp.sobel(gauss)
            if self.sobel_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Sobel": (sobel, 'gray')})
            save_dict.update({"Sobel": (sobel, f'{ESAT7A1}/Images/sobel images/')})
        else:
            sobel = gauss
        if self.hyst_applybool.get():
            hyst = wp.hysteresis(sobel, self.low_th.get(), self.high_th.get())
            if self.hyst_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Hysteresis": (hyst, 'gray')})
            save_dict.update({"Hyst": (hyst, f'{ESAT7A1}/Images/hysteresis images/')})
        else:
            hyst = sobel
        if self.fill_applybool.get():
            filled = ndimage.binary_fill_holes(hyst).astype(np.uint8)
            if self.fill_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Filled": (filled, 'gray')})
            save_dict.update({"Filled": (filled, f'{ESAT7A1}/Images/filled images/')})
        else:
            filled = hyst
        if self.senne_applybool.get():
            time_scratch = time.time()
            filled_img = PIL.Image.fromarray(filled)
            (width, height) = filled_img.size
            filled_img = np.array(filled_img.resize((int(width/5), int(height/5))))  # /5 en /5
            filled_img = np.array(filled_img)
            sobel2 = wp.sobel(filled_img)
            # # plt.imshow(filled)
            # plt.imshow(sobel2)
            # plt.show()
            # return
            senne_obj, nb_obj = wp.object_counting_from_scratch(sobel2, 100)
            if self.senne_showbool.get() or self.show_all_bool.get():
                show_dict.update(
                    {f"BROPAS: \n{nb_obj} objects, time: {round(time.time() - time_scratch, 2)}s": (senne_obj, 'viridis')})
            save_dict.update({"BROPAS": (senne_obj, f'{ESAT7A1}/Images/sobel images/')})
        else:
            senne_obj = filled

        if self.db_applybool.get():
            time_db = time.time()
            db, nb_objects = wp.db_scan(filled)
            if self.db_showbool.get() or self.show_all_bool.get():
                show_dict.update(
                    {f"DBSCAN: \n{nb_objects} objects, time: {round(time.time() - time_db, 2)}s": (db, 'viridis')})
            save_dict.update({"DetectedObjects": (db, f'{ESAT7A1}/Images/object images/')})
        else:
            db = filled
            nb_objects = None
        if self.boxes_applybool.get():
            boxes = wp.draw_boxes(image, db)
            if self.boxes_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Boxes": (boxes, 'gray')})
            save_dict.update({"Boxes": (boxes, f'{ESAT7A1}/Images/draw boxes/')})

        # save and show
        if len(show_dict):  # als de lengte van show_dict groter is dan 0
            p = Process(target=wp.show_images, args=(show_dict,))
            p.start()
            # p.join()
            # wp.show_images(show_dict)
        if self.save_bool.get():
            wp.save_images(save_dict)

    def depth_pipeline(self, depth_image, image,  show_dict, save_dict):
        gray = wp.grayscale(image)  # grayscaling is necessary to the process
        if self.gray_showbool.get() or self.show_all_bool.get():
            show_dict.update({"Grayscaled": (gray, 'gray')})
        save_dict.update({"Gray": (gray, f'{ESAT7A1}/Images/grayscaled images/')})  # gray is the reference for saving
        if self.gauss_applybool.get():
            gauss = wp.gaussian_blur(gray, self.gauss_reps.get())
            if self.gauss_showbool.get()or self.show_all_bool.get() :
                show_dict.update({"Gaussian Blur": (gauss, 'gray')})
            save_dict.update({"Gauss": (gauss, f'{ESAT7A1}/Images/blurred images/')})
        else:
            gauss = gray
        if self.sobel_applybool.get():
            sobel = wp.sobel(gauss)
            if self.sobel_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Sobel": (sobel, 'gray')})
            save_dict.update({"Sobel": (sobel, f'{ESAT7A1}/Images/sobel images/')})
        else:
            sobel = gauss
        if self.hyst_applybool.get():
            hyst = wp.hysteresis(sobel, self.low_th.get(), self.high_th.get())
            if self.hyst_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Hysteresis": (hyst, 'gray')})
            save_dict.update({"Hyst": (hyst, f'{ESAT7A1}/Images/hysteresis images/')})
        else:
            hyst = sobel
        if self.fill_applybool.get():
            filled = ndimage.binary_fill_holes(hyst)
            if self.fill_showbool.get() or self.show_all_bool.get():
                show_dict.update({"Filled": (filled, 'gray')})
            save_dict.update({"Filled": (filled, f'{ESAT7A1}/Images/filled images/')})
        else:
            filled = hyst

        if self.db_applybool.get():
            db, nb_objects = wp.db_scan(filled)
        else:
            db = filled
            # nb_objects = 0

        filled = PIL.Image.fromarray(filled)
        (width, height) = filled.size
        filled = np.array(filled.resize((int(width / 5), int(height / 5))))  # Go faster, lose image quality
        sobel2 = wp.sobel(filled)
        senne_obj, nb_obj = wp.object_counting_from_scratch(sobel2, 100)
        show_dict.update({f"BROPAS: {nb_obj} objects": (senne_obj, 'gray')})
        save_dict.update({"BROPAS": (senne_obj, f'{ESAT7A1}/Images/sobel images/')})

        if self.boxes_applybool.get():
            boxes = wp.draw_boxes(image, db)
            if self.boxes_showbool.get() or self.show_all_bool.get():
                show_dict.update({"boxes": (boxes, 'gray')})
            save_dict.update({"Boxes": (boxes, f'{ESAT7A1}/Images/draw boxes/')})

        if self.hasoverlap_applybool.get():
            overlap, overlapFrame = wp.has_overlapping_objects(depth_image)
            if overlap:
                depth_frame, resultFrame, resultFrame2, result, remainingResult = wp.depth_general(depth_image, nb_obj)
                nb_objects = remainingResult
                if self.hasoverlap_showbool.get() or self.show_all_bool.get():
                    show_dict.update({"Depth frame": (depth_frame, 'viridis')})
                    show_dict.update({"Overlap visualized": (overlapFrame, 'viridis')})
                    show_dict.update({"Depth result": (resultFrame, 'viridis')})
                    show_dict.update({f"Depth final result: \n{nb_objects} objects detected": (resultFrame2, 'viridis')})

                    save_dict.update({"Depth frame": (depth_frame, f'{ESAT7A1}/Images/depth frame/')})
                    save_dict.update({"Overlap visualized": (overlapFrame, f'{ESAT7A1}/Images/overlap/')})
                    save_dict.update({"Depth result": (resultFrame, f'{ESAT7A1}/Images/resulting depth/')})
                    save_dict.update({"Remaining depth": (resultFrame2, f'{ESAT7A1}/Images/remaining/')})

        # save and show
        if len(show_dict):
            p = Process(target=wp.show_images, args=(show_dict,))
            p.start()
            # p.join()
            # wp.show_images(show_dict)
        if self.save_bool.get():
            wp.save_images(save_dict)


class CreditsScreen(ttk.Frame):

    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # 'back' button:
        back_button = ttk.Button(self, text='back to menu', command=lambda: controller.show_frame(StartScreen), takefocus=False)
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
