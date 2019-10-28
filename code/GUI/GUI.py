# checkboxes not activated yet

import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import ntpath

from Main.whole_process import process_image
from Main.whole_process import detect_objects
from Main.whole_process import kinect_to_pc

# import os
# currentDir = os.path.dirname(os.path.abspath(__file__)).replace("code\\GUI", "")
# print(currentDir)


LARGE_FONT, LARGER_FONT = ("Verdana", 12), ("Verdana", 18)


class GUIVision(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # does the same as 'tk.Tk.__init__(self, *args, **kwargs)'
        self.title("BROPAS")
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
        for F in (HomeScreen, CreditsScreen, ScanScreen):
            frame = F(container, self)
            frame.grid(row=0, column=0, sticky="news")

            self.frames[F] = frame

        self.show_frame(HomeScreen)  # show the default (start) screen

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class HomeScreen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="green")
        self.columnconfigure(0, weight=1)

        # add background image
        self.background_image = tk.PhotoImage(file="GUI_images/Sobel_foto.png")
        background_label = tk.Label(self, image=self.background_image, bg="black")
        background_label.place(x=0, y=0, relwidth=1, relheight=1, anchor="nw")

        # menuFrame is the frame that contains the title and 'back' button
        menuFrame = tk.Frame(self)
        menuFrame.grid(padx=10, pady=10, row=0, column=0, sticky="new")
        menuFrame.columnconfigure(0, weight=1)

        # title label:
        label = tk.Label(menuFrame, text="Automatic counting of objects in an image", font=LARGE_FONT)
        label.grid(row=0, column=0, sticky="new")

        # information button:
        self.image = tk.PhotoImage(file="GUI_images/information_button2.png")
        # resize the image to fit on the button
        self.downscaled_image = self.image.subsample(30, 30)
        # set image on the button
        information_button = tk.Button(menuFrame, image=self.downscaled_image)
        information_button.place(x=0, y=0, anchor="nw")

        # start button
        start_button = tk.Button(self, text="Start", width=8, height=4,
                                 command=lambda: controller.show_frame(ScanScreen), font=LARGER_FONT)
        start_button.grid(row=1, column=0, sticky="n")

        # credits button
        credits_button = tk.Button(self, text="Credits", command=lambda: controller.show_frame(CreditsScreen))
        credits_button.grid(row=2, column=0, sticky="n")


class ScanScreen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)

        # add background image
        self.background_image = tk.PhotoImage(file="GUI_images/Sobel_foto.png")
        background_label = tk.Label(self, image=self.background_image, bg="black")
        background_label.place(x=0, y=0, relwidth=1, relheight=1, anchor="nw")

        # menuFrame is the frame that contains the title and 'back' button
        menuFrame = tk.Frame(self)
        menuFrame.grid(padx=10, pady=10, row=0, column=0, sticky="new")
        menuFrame.columnconfigure(0, weight=1)

        # title label:
        label = tk.Label(menuFrame, text="Selection Menu", font=LARGE_FONT)
        label.grid(row=0, column=0, sticky="new")

        # 'back' button:
        self.image = tk.PhotoImage(file="GUI_images/back_arrow_icon.png")
        # resize the image to fit on the button
        self.downscaled_image = self.image.subsample(50, 50)
        # set image on the button
        back_button = tk.Button(menuFrame, image=self.downscaled_image, command=lambda: controller.show_frame(HomeScreen))
        back_button.place(x=0, y=0, relheight=1, anchor="nw")

        # new picture button:
        scan_button = tk.Button(self, text="Take new picture", relief="groove", borderwidth=2, font=LARGER_FONT)
                                # command=lambda: messagebox.showinfo(None, "Coming soon..."))
        scan_button.grid(row=1, column=0, sticky="news", padx=20)
        scan_button.bind("<Button-1>", lambda e: self.run_on_new_picture())

        # existing picture button
        selected_img_button = tk.Button(self, text="Select existing picture", font=LARGER_FONT, relief="groove", borderwidth=2)
        selected_img_button.grid(row=2, column=0, sticky="news", padx=20, pady=5)
        # selected_img_button.bind("<Button-1>", lambda e: print("Starting sobel..."), add="+")
        selected_img_button.bind("<Button-1>", lambda e: self.run_on_selected_img(), add="+")

        # checkboxes for showing results:
        checkboxFrame = tk.Frame(self)
        checkboxFrame.grid(row=3, column=0, sticky="news", padx=60, pady=5)
        checkboxFrame.columnconfigure(0, weight=1)
        checkboxFrame.columnconfigure(1, weight=1)

        self.show_steps_bool = tk.IntVar()
        checkbox = tk.Checkbutton(checkboxFrame, variable=self.show_steps_bool, text="Show steps", font=LARGE_FONT,
                                  relief="groove", borderwidth=2, command=lambda: self.show_steps())
        checkbox.select()  # set 'selected' state as default state
        checkbox.grid(row=0, column=0, sticky="news")

        self.show_results_bool = tk.IntVar()
        checkbox = tk.Checkbutton(checkboxFrame, variable=self.show_results_bool, text="Show results", font=LARGE_FONT,
                                  relief="groove", borderwidth=2, command=lambda: self.show_results())
        checkbox.select()  # set 'selected' state as default state
        checkbox.grid(row=0, column=1, sticky="news")

    def show_steps(self):
        if self.show_steps_bool.get():  # show_results is a boolean
            print("show_steps_activated")
            # whole_process.show_results = True
        else:
            print("show steps deactivated")
            # whole_process.show_results = False

    def show_results(self):
        if self.show_results_bool.get():  # show_results is a boolean
            print("show results activated")
            # whole_process.show_results = True
        else:
            print("show results deactivated")
            # whole_process.show_results = False

    def get_filepath(self):
        filepath = filedialog.askopenfilename()
        # print(f"file path = {filepath}")
        return filepath

    def run_on_selected_img(self):
        img_path = self.get_filepath()
        # img_name = ntpath.basename(img_path)
        matrix = process_image(img_path)
        detect_objects(matrix)

    def run_on_new_picture(self):
        colorPicture, depthPicture = kinect_to_pc(1080, 1920, 4)
        matrix = process_image(colorPicture)
        detect_objects(matrix)


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
                                command=lambda: controller.show_frame(HomeScreen))
        back_button.place(x=0, y=0, relheight=1, anchor="nw")

        # the credits space ('credits' is a built in function, hence the name 'creditss')
        creditss = tk.Text(self, height=2, width=30)
        creditss.grid(row=1, column=0, sticky="news")
        creditss.insert(tk.END, "*Brakke Gantt Chart: Robin")


if __name__ == '__main__':

    app = GUIVision()
    app.mainloop()
