from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import subprocess
from matplotlib import pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
plt.style.use('classic')
import pymongo

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import stats
from astropy.time import Time
import astropy.coordinates as c
from astropy.wcs import WCS
from astropy.table import Table, Column
from astropy.wcs.utils import proj_plane_pixel_scales
import ephem
import ccdproc
import photutils
import sep

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

from .utils import pre_condition, post_condition


##-----------------------------------------------------------------------------
## Primitive: RenderJPEG
##-----------------------------------------------------------------------------
class RenderJPEG(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        checks = [pre_condition(self, 'Skip image is not set',
                                not self.action.args.skip),
                 ]
        return np.all(checks)

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = []
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        im = self.action.args.kd.pixeldata[0].data

        plt.rcParams.update({'font.size': 24})
        binning = self.cfg['jpeg'].getint('binning', 1)
        vmin = np.percentile(im, self.cfg['jpeg'].getfloat('vmin_percent', 0.5))
        vmax = np.percentile(im, self.cfg['jpeg'].getfloat('vmax_percent', 99))
        dpi = self.cfg['jpeg'].getint('dpi', 72)
        nx, ny = im.shape
        sx = nx/dpi/binning
        sy = ny/dpi/binning
        FWHM_pix = self.action.args.fwhm if self.action.args.fwhm is not None else 8
        ap_radius = self.cfg['Photometry'].getfloat('aperture_radius', 2)*FWHM_pix

        if self.action.args.wcs is not None:
            pixel_scale = np.mean(proj_plane_pixel_scales(self.action.args.wcs))*60*60
        else:
            pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1)

        fig = plt.figure(figsize=(2*sx, 1*sy), dpi=dpi)

        plotpos = [ [ [0.010, 0.010, 0.550, 0.965], [0.565, 0.775, 0.375, 0.200] ],
                    [ None                        , [0.565, 0.340, 0.375, 0.400] ],
                    [ None                        , [0.565, 0.035, 0.375, 0.270] ],
                  ]

        ##-------------------------------------------------------------------------
        # Show JPEG of Image
        self.log.debug(f'  Rendering JPEG of image')
        jpeg_axes = plt.axes(plotpos[0][0])
        jpeg_axes.imshow(im, cmap=plt.cm.gray_r, vmin=vmin, vmax=vmax)
        jpeg_axes.set_xticks([])
        jpeg_axes.set_yticks([])
        titlestr = f'{self.action.args.fitsfile}: '

        ##-------------------------------------------------------------------------
        # Overlay Extracted (blue)
        if self.cfg['jpeg'].getboolean('overplot_extracted', False) is True\
                and self.action.args.objects is not None:
            titlestr += f'blue=extracted({len(self.action.args.objects)}) '
            radius = ap_radius/binning
            for star in self.action.args.objects:
                if star['x'] > 0 and star['x'] < nx and star['y'] > 0 and star['y'] < ny:
                    x = star['x']
                    y = star['y']
                    jpeg_axes.plot([x, x], [y+radius, y+2.5*radius], 'b', alpha=0.7)
                    jpeg_axes.plot([x, x], [y-radius, y-2.5*radius], 'b', alpha=0.7)
                    jpeg_axes.plot([x-radius, x-2.5*radius], [y, y], 'b', alpha=0.7)
                    jpeg_axes.plot([x+radius, x+2.5*radius], [y, y], 'b', alpha=0.7)

        ##-------------------------------------------------------------------------
        # Overlay Calibrators (red)
        if self.cfg['jpeg'].getboolean('overplot_calibrators', False) is True\
                and self.action.args.associated_calibrators is not None\
                and self.action.args.wcs is not None:
            self.log.debug(f'  Overlay measured calibration stars')
            calibrators = self.action.args.associated_calibrators
            titlestr += f'red=calibrators({len(calibrators)}) '
            radius = ap_radius/binning
            for entry in calibrators:
                if entry['apflux'] > 0:
                    xy = (entry['x'], entry['y'])
                    c = plt.Circle(xy, radius=radius,
                                   edgecolor='r', facecolor='none', alpha=0.5)
                    jpeg_axes.add_artist(c)
                    c = plt.Circle(xy, radius=int(np.ceil(1.5*radius)),
                                   edgecolor='r', facecolor='none', alpha=0.5)
                    jpeg_axes.add_artist(c)
                    c = plt.Circle(xy, radius=int(np.ceil(2.0*radius)),
                                   edgecolor='r', facecolor='none', alpha=0.5)
                    jpeg_axes.add_artist(c)

        ##-------------------------------------------------------------------------
        # Overlay Catalog (green)
        if self.cfg['jpeg'].getboolean('overplot_catalog', False) is True\
                and self.action.args.calibration_catalog is not None\
                and self.action.args.wcs is not None:
            self.log.debug(f'  Overlay stars from catalog')
            catalog = self.action.args.calibration_catalog
            titlestr += f'green=catalog({len(catalog)}) '
            radius = ap_radius/binning
            for entry in catalog:
                x, y = self.action.args.wcs.all_world2pix(entry['raMean'],
                                                          entry['decMean'], 1)
                jpeg_axes.plot([x, x], [y+radius, y+2.5*radius], 'g', alpha=0.7)
                jpeg_axes.plot([x, x], [y-radius, y-2.5*radius], 'g', alpha=0.7)
                jpeg_axes.plot([x-radius, x-2.5*radius], [y, y], 'g', alpha=0.7)
                jpeg_axes.plot([x+radius, x+2.5*radius], [y, y], 'g', alpha=0.7)


        ##-------------------------------------------------------------------------
        # Overlay Pointing
        if self.cfg['jpeg'].getboolean('overplot_pointing', False) is True\
                and self.action.args.header_pointing is not None\
                and self.action.args.wcs is not None\
                and self.action.args.wcs_pointing is not None:
            self.log.debug(f'  Overlay pointing')
            radius = self.cfg['jpeg'].getfloat('pointing_radius', 40)
            x, y = self.action.args.wcs.all_world2pix(
                               self.action.args.header_pointing.ra.degree,
                               self.action.args.header_pointing.dec.degree, 1)
            jpeg_axes.plot([nx/2-3*radius,nx/2+3*radius], [ny/2,ny/2], 'k-',
                           alpha=0.5)
            jpeg_axes.plot([nx/2, nx/2], [ny/2-3*radius,ny/2+3*radius], 'k-',
                           alpha=0.5)
            # Draw crosshair on target
            c = plt.Circle((x, y), radius=radius, edgecolor='g', alpha=0.7,
                           facecolor='none')
            jpeg_axes.add_artist(c)
            jpeg_axes.plot([x, x], [y+0.6*radius, y+1.4*radius], 'g', alpha=0.7)
            jpeg_axes.plot([x, x], [y-0.6*radius, y-1.4*radius], 'g', alpha=0.7)
            jpeg_axes.plot([x-0.6*radius, x-1.4*radius], [y, y], 'g', alpha=0.7)
            jpeg_axes.plot([x+0.6*radius, x+1.4*radius], [y, y], 'g', alpha=0.7)

        jpeg_axes.set_xlim(0,nx)
        jpeg_axes.set_ylim(0,ny)

        ##-------------------------------------------------------------------------
        ## Histogram of Pixel Values (for flats)
        if self.action.args.calibration_catalog is None:
            self.log.debug(f'  Generating histogram of pixel values')
            pixel_axes = plt.axes([0.565, 0.540, 0.375, 0.430])
            mean, med, std = stats.sigma_clipped_stats(im)
            p1, p99 = np.percentile(im, 1), np.percentile(im, 99)
            pixel_axes.set_title(f"Histogram of Pixel Values (median = {med:.0f})")
            npix, bins, p = pixel_axes.hist(im.ravel(), color='b', alpha=0.5,
                                            bins=np.linspace(p1,p99,50))
            pixel_axes.plot([med, med], [0,max(npix)*1.2], 'r-', alpha=0.5)
            pixel_axes.set_xlabel('e-/s')
            pixel_axes.set_ylabel('N Pix')
            pixel_axes.set_ylim(0,max(npix)*1.2)

        ##-------------------------------------------------------------------------
        # Plot histogram of FWHM
        if self.action.args.objects is not None:
            self.log.debug(f'  Generating histogram of FWHM values')
            fwhm_axes = plt.axes(plotpos[0][1])
            minfwhm = 1
            maxfwhm = 7
            avg_fwhm = np.median(self.action.args.objects['FWHM'])*pixel_scale
            fwhm_axes.set_title(f"FWHM = {avg_fwhm:.1f} arcsec")
            nstars, bins, p = fwhm_axes.hist(self.action.args.objects['FWHM']*pixel_scale,
                                             bins=np.arange(minfwhm,maxfwhm,0.25),
                                             color='b', alpha=0.5)
            fwhm_axes.plot([avg_fwhm, avg_fwhm], [0,max(nstars)*1.2], 'r-', alpha=0.5)
            fwhm_axes.set_ylabel('N stars')
            fwhm_axes.set_ylim(0,max(nstars)*1.2)
        if self.action.args.associated_calibrators is not None:
            nstars, bins, p = fwhm_axes.hist(self.action.args.associated_calibrators['FWHM']*pixel_scale,
                                             bins=np.arange(minfwhm,maxfwhm,0.25),
                                             color='r', alpha=0.5)
            fwhm_axes.plot([avg_fwhm, avg_fwhm], [0,max(nstars)*1.2], 'r-', alpha=0.5)

        ##-------------------------------------------------------------------------
        # Plot FWHM vs. Flux
#         if self.action.args.objects is not None:
#             self.log.debug(f'  Generating plot of FWHM values vs. flux')
#             avg_fwhm = np.median(self.action.args.objects['FWHM'])*pixel_scale
#             fwhmmag_axes = plt.axes(plotpos[1][1])
#             fwhmmag_axes.plot(self.action.args.objects['FWHM']*pixel_scale,
#                               self.action.args.objects['apflux'].data, 'bo',
#                               mec='none', alpha=0.3)
#             fwhmmag_axes.plot([avg_fwhm, avg_fwhm],
#                               [1,max(self.action.args.objects['apflux'].data)*1.5],
#                               'r-', alpha=0.5)
#             fwhmmag_axes.set_xlabel("FWHM (arcsec)")
#             fwhmmag_axes.set_xlim(minfwhm, maxfwhm)
#             fwhmmag_axes.set_ylabel(f"Flux (e-/s)")
#             fwhmmag_axes.set_yscale("log")
#         if self.action.args.associated_calibrators is not None:
#             fwhmmag_axes.plot(self.action.args.associated_calibrators['FWHM']*pixel_scale,
#                               self.action.args.associated_calibrators['apflux'].data,
#                               'ro', mec='none', alpha=0.3)

        ##-------------------------------------------------------------------------
        # Plot instrumental mags
#         if self.action.args.associated_calibrators is not None:
#             flux_axes = plt.axes(plotpos[2][1])
#             self.log.debug(f'  Generating plot of flux vs catalog magnitude')
#             flux_axes.plot(self.action.args.associated_calibrators['mag'],
#                            self.action.args.associated_calibrators['apflux'], 'bo',
#                            label='photutils', mec=None, alpha=0.6)
#             flux_axes.set_xlim(min(self.action.args.associated_calibrators['mag']),
#                                max(self.action.args.associated_calibrators['mag']))
#             flux_axes.set_yscale('log')
#             flux_axes.set_ylabel('Measured Flux (e-/s)')
#             plt.grid()
#             plt.legend(loc='best')


        ##-------------------------------------------------------------------------
        # Plot instrumental mag diffs
        if self.action.args.associated_calibrators is not None\
            and self.action.args.zero_point_f0 is not None:
            diff_axes = plt.axes(plotpos[1][1])
            self.log.debug(f'  Generating plot of flux residual')
            deltamag = self.action.args.associated_calibrators['mag']\
                       - self.action.args.associated_calibrators['instmag']
            mean, med, std = stats.sigma_clipped_stats(deltamag)
            xmin = min(self.action.args.associated_calibrators['mag'])
            xmax = max(self.action.args.associated_calibrators['mag'])
            diff_axes.plot(self.action.args.associated_calibrators['mag'],
                           deltamag, 'bo',
                           label=f'Individual Zero Points',
                           mec=None, alpha=0.6)
            zp = self.action.args.zero_point
            label = f'Zero Point = {zp:.2f} (throughput={self.action.args.throughput:.3f})'
            diff_axes.plot([xmin, xmax], [zp,zp], 'k-', mec=None, alpha=0.6,
                           label=label)
            diff_axes.plot([xmin, xmax], [zp+0.1,zp+0.1], 'r-', mec=None, alpha=0.6)
            diff_axes.plot([xmin, xmax], [zp-0.1,zp-0.1], 'r-', mec=None, alpha=0.6,
                           label=f"+/-0.1 magnitude error (StdDev={std:.2f})")
            diff_axes.set_xlabel('Catalog Magnitude')
            diff_axes.set_xlim(xmin, xmax)
            diff_axes.set_ylabel('Catalog - Instrumental Mag')
            diff_axes.set_ylim(zp-7*std, zp+7*std)
            plt.legend(loc='best')
            plt.grid()


        ##-------------------------------------------------------------------------
        # Plot SNR
        if self.action.args.associated_calibrators is not None:
            snr_axes = plt.axes(plotpos[2][1])
            self.log.debug(f'  Generating plot of SNR')
            xmin = min(self.action.args.associated_calibrators['mag'])
            xmax = max(self.action.args.associated_calibrators['mag'])
            snr_axes.plot(self.action.args.associated_calibrators['mag'],
                          self.action.args.associated_calibrators['snr'], 'bo',
                          label=f'SNR',
                          mec=None, alpha=0.6)
            snr_axes.set_xlabel('Catalog Magnitude')
            snr_axes.set_xlim(xmin, xmax)
            snr_axes.set_ylabel('SNR')
            snr_axes.set_ylim(1, np.percentile(self.action.args.associated_calibrators['snr'], 99))
            snr_axes.set_yscale("log")
#             plt.legend(loc='best')
            plt.grid()


        jpeg_axes.set_title(titlestr)
        reportfilename = f'{self.action.args.fitsfile.split(".")[0]}.jpg'
        self.action.args.jpegfile = Path('/var/www/plots/V20/') / reportfilename
        plt.savefig(self.action.args.jpegfile, dpi=dpi)


        return self.action.args


if __name__ == '__main__':
    pass
