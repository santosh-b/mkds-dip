from tkinter import *
from PIL import Image, ImageTk
import os
import time


class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.label = Label(compound="top")
        self.label.pack(side="top", padx=8, pady=8)
        self._img = None
        self._update_img(10)

    def update_buffer(self, img):
        # Preprocess the input image and update the buffer accordingly
        self._img = img

    def _update_img(self, delay, event=None):
        if self._img:
            self._img_buffer = ImageTk.PhotoImage(self._img)
            self.label.configure(image=self._img_buffer)
        self.after(delay, self._update_img, 10)