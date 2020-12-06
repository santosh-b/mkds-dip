from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import collections
import threading
from scipy.signal import convolve2d
import time

class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.label = Label(compound="top")
        self.label.pack(side="top", padx=8, pady=8)
        self._img = None
        self._rawimg = None
        self._update_img(10)
        self.c = collections.deque(maxlen=100)
        self.c.append(0)
        self.stats = None
        self.data = None
        self.conf = None
        self.view = None
        threading.Thread(target=self.update_graph).start()
        threading.Thread(target=self.update_view).start()

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def update_view(self):
        filter = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        while True:
            if self._rawimg is not None:
                self.view = Image.fromarray(convolve2d(np.mean(self._rawimg, axis=-1)[:192,:], filter)).resize((50,50))
                time.sleep(.1)

    def update_buffer(self, img, data, preds, frame):
        # Preprocess the input image and update the buffer accordingly
        self._img = img
        self._rawimg = img.copy()
        draw = ImageDraw.Draw(self._img, 'RGBA')
        draw.rectangle([32,180,223,185], fill=(0,0,0,127), width=1, outline='black')
        w = 16
        h = 16
        if frame%20==0 and preds is not None:
            self.conf = np.max(preds)
        if self.conf is not None and preds is not None:
            draw.ellipse((3, 250, 3+89*.5, 250+89*.5), (255,255,0,127), 'black')  # made this a little smaller..
            draw.ellipse((3+25*.5, 250+20*.5, 3+35*.5, 250+30*.5), 'black', 'black',)
            draw.ellipse((3+50*.5, 250+20*.5, 3+60*.5, 250+30*.5), 'black', 'black')
            if self.conf>0.85:
                draw.arc((1+20*.5, 250+40*.5, 3+70*.5, 250+70*.5), 0, 180, 'black')
            elif self.conf<0.3:
                draw.arc((1 + 20 * .5, 255 + 40 * .5, 3 + 70 * .5, 255 + 70 * .5), 180, 360, 'black')
            else:
                draw.line((1 + 20 * .5, 280, 3 + 70 * .5, 280), 'black')
        if preds is not None:
            pos = sum([x*p_x for x, p_x in enumerate(preds)])
            draw.polygon([(30+15*pos,180),(30+w/2+15*pos,180-h),(30+w+15*pos,180)], fill=(255,0,0,127), outline='black')
        if self.stats:
            self._img.paste(self.stats, (0, 192))
        if self.view is not None:
            self._img.paste(self.view, (50, 192))
        self.data = data

    def update_graph(self):
        while True:
            conf = plt.figure(figsize=(.5,.5))
            ax = conf.add_axes((0,0,1,1))

            ax.plot(self.c, color='white')
            ax.set_facecolor('black')
            ax.patch.set_edgecolor('black')

            ax.patch.set_linewidth('1')
            #draw = ImageDraw.Draw(img)
            #draw.rectangle((0,192,258,386), fill='gray')
            self.stats = self.fig2img(conf)
            plt.close(conf)
            if self.data:
                self.c.append(-(self.c[-1] - self.data)/100)
            time.sleep(.1)

    def _update_img(self, delay, event=None):
        if self._img:
            self._img_buffer = ImageTk.PhotoImage(self._img)
            self.label.configure(image=self._img_buffer)
        self.after(delay, self._update_img, 10)