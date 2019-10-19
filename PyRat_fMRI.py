# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:35:06 2018

@author: nikoloz
"""

# do all imports
from tkinter import Tk, Frame, Button, Label, LabelFrame, Entry, Canvas
from tkinter import Checkbutton, filedialog, END, W, IntVar
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter
from models import two_gamma_hrf, plot_hrf, block, events, hrf_convolve
from utilities import mat_pearsonr
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as sps
import nibabel as nib
import seaborn as sns

# plotting settings
mpl.use("TkAgg")
sns.set(style='ticks', context='notebook', font_scale=1.5)
cp = sns.color_palette()

# Define class


class PyRat:

    def __init__(self, master):

        # initiate master frame
        self.master = master

        # initiate main frame (master's child)
        self.frame = Frame(master)
        self.frame.pack()

        # initiate class variables
        self.filename = 'not loaded'        # name of NIFTI file
        self.data = np.empty(0)             # holds 4D data matrix
        self.x = 0                          # X dimension
        self.y = 0                          # Y dimension
        self.z = 0                          # Z dimension (slices)
        self.t = 0                          # time-points (volumes)
        self.dunit = 'mm'                   # distance unit
        self.tunit = 's'                    # time unit
        self.xdim = 1.                      # X resolution
        self.ydim = 1.                      # Y resolution
        self.zdim = 1.                      # Z resolution
        self.TR = 2.                        # temporal resolution
        self.hrf_conv = IntVar()            # whether to convolve with HRF
        # text reporting current status
        self.st_text = 'waiting to load the 4D fMRI dataset'
        # call function for widget creation
        self.create_widgets()

    def create_widgets(self):

        WIDTH = 45

        # quit frame
        quit_pyrat = Button(self.frame, text="QUIT", fg="red",
                            command=self.master.destroy, width=WIDTH)
        quit_pyrat.grid(row=0, column=0)

        # status display frame
        self.status_frame()

        # data and options frame
        opt_frame = LabelFrame(self.frame, text='DATA:')
        opt_frame.grid(row=2, column=0, columnspan=3)

        select_file_button = Button(opt_frame,
                                    text='Select 4D NIFTI file',
                                    command=self.select_file, width=WIDTH)
        select_file_button.grid(row=0, column=0, columnspan=2)

        params = Canvas(opt_frame, width=320, height=60, bg='white')
        params.create_text(160, 10, text='file name: ' + self.filename)
        params.create_text(160, 30,
                           text='x={0}, y={1}, z={2}, t={3}'.format(
                                self.x, self.y, self.z, self.t))
        Sres = 'Voxel: {0:.2f}x{1:.2f}x{2:.2f} {3}'.format(
                self.xdim, self.ydim, self.zdim, self.dunit)
        Tres = 'TR: {0:.2f} {1}'.format(self.TR, self.tunit)
        params.create_text(160, 50,
                           text=Sres + ', ' + Tres)
        params.grid(row=1, column=0, columnspan=2)

        show_data = Button(opt_frame, text='Show data',
                           command=self.plot_mosaic, width=WIDTH)
        show_data.grid(row=3, column=0, columnspan=2)

        detrend_button = Button(opt_frame, text='Linear detrend',
                                command=self.linear_detrend, width=WIDTH)
        detrend_button.grid(row=4, column=0, columnspan=2)

        Label(opt_frame, text='FWHM (mm)', width=WIDTH//3).grid(
              row=5, column=0, sticky=W)
        self.FWHM = Entry(opt_frame, width=(2*WIDTH)//3)
        self.FWHM.grid(row=5, column=1)
        self.FWHM.delete(0, END)
        self.FWHM.insert(0, '3.0')

        smooth_button = Button(opt_frame, text='Smooth',
                               command=self.smooth, width=WIDTH)
        smooth_button.grid(row=6, column=0, columnspan=2)

        # model frame
        mod_frame = LabelFrame(self.frame, text='MODEL (in seconds):',
                               width=WIDTH)
        mod_frame.grid(row=4, column=0)

        Label(mod_frame, text='baseline', width=15).grid(
              row=0, column=0, sticky=W)
        self.base = Entry(mod_frame, width=30)
        self.base.grid(row=0, column=1)
        self.base.delete(0, END)
        self.base.insert(0, '18')

        Label(mod_frame, text='on', width=15).grid(row=1, column=0, sticky=W)
        self.on = Entry(mod_frame, width=30)
        self.on.grid(row=1, column=1)
        self.on.delete(0, END)
        self.on.insert(0, '12')

        Label(mod_frame, text='off', width=15).grid(row=2, column=0, sticky=W)
        self.off = Entry(mod_frame, width=30)
        self.off.grid(row=2, column=1)
        self.off.delete(0, END)
        self.off.insert(0, '18')

        Label(mod_frame, text='cycles', width=15).grid(
              row=3, column=0, sticky=W)
        self.cycles = Entry(mod_frame, width=30)
        self.cycles.grid(row=3, column=1)
        self.cycles.delete(0, END)
        self.cycles.insert(0, '8')

        convolve = Checkbutton(mod_frame, text="Convolve with HRF",
                               variable=self.hrf_conv)
        convolve.grid(row=4, column=0, columnspan=2)
        self.hrf_conv.set(1)

        show_boxcar = Button(mod_frame, text='Build model',
                             command=self.build_model, width=45)
        show_boxcar.grid(row=5, column=0, columnspan=2)

        # analysis frame
        ana_frame = LabelFrame(self.frame, text='ANALYSIS:', width=WIDTH)
        ana_frame.grid(row=5, column=0)

        corr_button = Button(ana_frame, text='Compute linear correlation',
                             command=self.correlate, width=WIDTH)
        corr_button.grid(row=0, column=0, columnspan=2)

        Label(ana_frame, text='threshold: |R| > ').grid(
              row=1, column=0, sticky=W)
        self.thr = Entry(ana_frame, width=2*(WIDTH//3))
        self.thr.grid(row=1, column=1)
        self.thr.delete(0, END)
        self.thr.insert(0, '0.3')

        overlay_button = Button(ana_frame, text='Overlay thresholded map',
                                command=self.overlay, width=WIDTH)
        overlay_button.grid(row=2, column=0, columnspan=2)

    def status_frame(self):

        # create the frame
        st_frame = LabelFrame(self.frame, text='STATUS')
        st_frame.grid(row=1, column=0)

        # create canvas with text
        status = Canvas(st_frame, width=320, height=30, bg='white')
        status.create_text(160, 15, text=self.st_text)
        status.grid(row=0, column=0)

    def select_file(self):

        # read the file path
        self.filepath = filedialog.askopenfilename()
        # extract the filename only
        self.filename = self.filepath.split("/")[-1]

        # update status frame
        self.st_text = 'file selected: ' + self.filename
        self.status_frame()

        # call function for loading the EPI data
        self.load_epi()

    def load_epi(self):

        # load 4d data
        epi = nib.load(self.filepath)
        # print dimensions (x, y, slices, volumes)
        self.x, self.y, self.z, self.t = epi.shape
        # load nifti header
        hdr = epi.header
        # print units of distance and time
        self.dunit, self.tunit = hdr.get_xyzt_units()
        # get spatial and temporal resolutions
        self.xdim = hdr['pixdim'][1]
        self.ydim = hdr['pixdim'][2]
        self.zdim = hdr['pixdim'][3]
        self.TR = hdr['pixdim'][4]
        # convert to numpy array
        self.data = epi.get_data()

        # update data frame (to display parameters)
        self.create_widgets()
        # show mean functional volume
        self.plot_mosaic()

        # update status frame
        self.st_text = 'file selected: ' + self.filename
        self.status_frame()

    def plot_mosaic(self):

        ncols = 5

        # add enough slices to make a multiple of ncols
        extra_z = ncols - (self.z % ncols)
        data_extra = np.pad(self.data,
                            [(0, 0), (0, 0), (0, extra_z), (0, 0)],
                            mode='constant', constant_values=0)
        new_z = self.z + extra_z
        # reshape to 3D (mosaic)
        vol_mosaics = []
        for t in range(self.t):
            vol = data_extra[:, :, :, t]
            slices = tuple([vol[:, :, i].T for i in range(new_z)])
            stack = np.hstack(slices)
            substacks = tuple(np.split(stack, ncols, axis=1))
            vol_mosaics.append(np.vstack(substacks))

        self.mosaic = np.dstack(tuple(vol_mosaics))
        self.xm, self.ym, self.t = self.mosaic.shape
        self.mean_func = np.mean(self.mosaic, axis=-1)

        fig = plt.figure(figsize=(self.ym/60, self.xm/60))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.mean_func, cmap='Greys_r')
        plt.tight_layout()
        fig.text(0.025, 0.1, 'R', fontsize='large', color='0.5', ha='center')
        fig.text(0.975, 0.1, 'L', fontsize='large', color='0.5', ha='center')
        fig.canvas.set_window_title('Mean volume')

        self.st_text = 'first volume shown'
        self.status_frame()
        plt.show()

    def linear_detrend(self):

        mean_volume = np.mean(self.data, axis=-1, keepdims=True)
        detrended = detrend(self.data, axis=-1, type='linear')
        self.data = detrended + mean_volume
        self.plot_mosaic()

        self.st_text = 'Detrending completed'
        self.status_frame()

    def smooth(self):

        FWHM = float(self.FWHM.get())
        if FWHM <= 0:
            self.st_text = 'Smoothing requires a positive FWHM value'
            self.status_frame()
            return
        else:
            # mm to voxels
            s = FWHM/2.355
            # sigma to voxels
            s1, s2, s3 = s/self.xdim, s/self.ydim, s/self.zdim
            # Perform 3D gaussian smoothing
            self.data = gaussian_filter(self.data, [s1, s2, s3, 0])
            self.plot_mosaic()

            self.st_text = 'Smoothing completed'
            self.status_frame()

    def build_model(self):

        b = int(self.base.get())
        on = int(self.on.get())
        off = int(self.off.get())
        c = int(self.cycles.get())

        self.model = block(base=b, on=on, off=off,
                           cycles=c, TR=self.TR)

        check = self.hrf_conv.get()
        if check == 1:
            HRF = two_gamma_hrf(TR=self.TR)
            self.model = hrf_convolve(self.model, hrf=HRF)

        mod_fig = plt.figure(figsize=(8, 4))
        seconds = self.TR*self.t
        time = np.arange(0, seconds, self.TR)
        plt.plot(time, self.model, lw=3, label='model', color=cp[3])
        plt.ylim(-0.3, 1.3)
        plt.yticks([0, 0.5, 1])
        step = 60
        if seconds > 300:
            step = 120
        elif seconds > 600:
            step = 240
        plt.xticks(np.arange(0, seconds+1, step),
                   np.arange(0, (seconds/60)+0.1, step//60))
        plt.xlim(0, seconds+1)
        plt.xlabel('Time (s)')
        plt.legend(loc=1)
        sns.despine()
        plt.tight_layout()
        for i in range(c):
            plt.axvspan(b-1 + on*i + off*i, b-1 + on*(i+1) + off*i,
                        facecolor=cp[3], alpha=0.2)
        mod_fig.canvas.set_window_title('Model')
        plt.show()

    def correlate(self):

        mos = self.mosaic
        mos2D = mos.reshape((self.xm*self.ym, self.t))
        model2D = self.model.reshape((-1, self.t))
        R2D = mat_pearsonr(mos2D, model2D)
        self.R = R2D.reshape((self.xm, self.ym))

        fig = plt.figure(figsize=(self.ym/60, self.xm/60))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.mean_func, cmap='Greys_r')
        plt.imshow(self.R, cmap='RdBu_r', alpha=1.0,
                   vmin=-0.8, vmax=0.8)
        plt.title("Pearson's R")
        plt.tight_layout()
        plt.colorbar(fraction=0.065, pad=0.01, shrink=0.5,
                     ticks=[-0.8, -0.4, 0.0, 0.4, 0.8])
        fig.text(0.025, 0.1, 'R', fontsize='large', color='0.5', ha='center')
        fig.text(0.975, 0.1, 'L', fontsize='large', color='0.5', ha='center')
        fig.canvas.set_window_title("Pearson's R")
        plt.show()

        self.st_text = 'correlation map shown'
        self.status_frame()

    def overlay(self):

        rthr = float(self.thr.get())
        R_thr = np.ma.masked_where(np.abs(self.R) <= rthr, self.R)

        fig = plt.figure(figsize=(self.ym/60, self.xm/60))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.mean_func, cmap='Greys_r')
        plt.imshow(R_thr, cmap='RdBu_r', alpha=1.0,
                   vmin=-0.8, vmax=0.8)
        plt.title("Thresholded correlation: |R| > {0:.2f}".format(rthr))
        plt.tight_layout()
        plt.colorbar(fraction=0.065, pad=0.01, shrink=0.5,
                     ticks=[-0.8, -0.4, 0.0, 0.4, 0.8])
        fig.text(0.025, 0.1, 'R', fontsize='large', color='0.5', ha='center')
        fig.text(0.975, 0.1, 'L', fontsize='large', color='0.5', ha='center')
        fig.canvas.set_window_title("Thresholded Pearson's R")
        plt.show()

        self.st_text = 'Thresholded correlation overlay shown'
        self.status_frame()


root = Tk()
root.wm_title("PyRat fMRI")
PyRat(root)
root.mainloop()
