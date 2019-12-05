from tkinter import ttk
from ttkthemes import ThemedTk


class BROPAS(ThemedTk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # does the same as 'tk.Tk.__init__(self, *args, **kwargs)'
        self.geometry("525x300")  # start dimensions

        # set up the container of all the screens/frames/menus
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="news")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        # 'collect' all the screens/frames/menus that you want to show
        self.frames = {}
        for F in (MainMenu,):
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
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        # info button
        ib = ttk.Button(self, text="info")
        ib.grid(row=0, column=0, sticky="w", padx=10)
        # methode button
        mb = ttk.Button(self, text="De methode")
        mb.grid(row=0, column=1, sticky="e")


if __name__ == '__main__':

    app = BROPAS()
    app.get_themes()
    app.set_theme('plastik')
    app.mainloop()
