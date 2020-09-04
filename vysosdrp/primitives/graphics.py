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

        if self.action.args.wcs is not None:
            pixel_scale = np.mean(proj_plane_pixel_scales(self.action.args.wcs))*60*60
        else:
            pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1)

        fig = plt.figure(figsize=(2*sx, 1*sy), dpi=dpi)

        plotpos = [ [ [0.010, 0.010, 0.550, 0.965], [0.565, 0.775, 0.375, 0.200] ],
                    [ None                        , [0.565, 0.540, 0.375, 0.200] ],
                    [ None                        , [0.565, 0.265, 0.375, 0.240] ],
                    [ None                        , [0.565, 0.035, 0.375, 0.210] ],
                  ]

        ##-------------------------------------------------------------------------
        # Show JPEG of Image
        jpeg_axes = plt.axes(plotpos[0][0])
        jpeg_axes.imshow(im, cmap=plt.cm.gray_r, vmin=vmin, vmax=vmax)
        jpeg_axes.set_xticks([])
        jpeg_axes.set_yticks([])
        titlestr = f'{self.action.args.fitsfile}: '

        ##-------------------------------------------------------------------------
        # Overlay Extracted (blue)
        if self.cfg['jpeg'].getboolean('overplot_extracted', False) is True and self.action.args.objects is not None:
            titlestr += f'blue=extracted({len(self.action.args.objects)}) '
#             radius = self.cfg['jpeg'].getfloat('extracted_radius', 6)
            radius = int(2.0*self.action.args.fwhm)/binning
            for star in self.action.args.objects:
                if star['x'] > 0 and star['x'] < nx and star['y'] > 0 and star['y'] < ny:
                    x = star['x']
                    y = star['y']
                    jpeg_axes.plot([x, x], [y+radius, y+2.5*radius], 'b', alpha=0.7)
                    jpeg_axes.plot([x, x], [y-radius, y-2.5*radius], 'b', alpha=0.7)
                    jpeg_axes.plot([x-radius, x-2.5*radius], [y, y], 'b', alpha=0.7)
                    jpeg_axes.plot([x+radius, x+2.5*radius], [y, y], 'b', alpha=0.7)
#                     c = plt.Circle((star['x'], star['y']), radius=radius,
#                                    edgecolor='b', facecolor='none')
#                     jpeg_axes.add_artist(c)

        ##-------------------------------------------------------------------------
        # Overlay Catalog (green)
        if self.cfg['jpeg'].getboolean('overplot_catalog', False) is True and self.action.args.catalog is not None and self.action.args.wcs is not None:
            titlestr += f'green=catalog({len(self.action.args.catalog)}) '
#             radius = self.cfg['jpeg'].getfloat('catalog_radius', 6)
            radius = int(2.0*self.action.args.fwhm)/binning
            x, y = self.action.args.wcs.all_world2pix(self.action.args.catalog['RA'], self.action.args.catalog['DEC'], 1)
            for xy in zip(x, y):
                if xy[0] > 0 and xy[0] < nx and xy[1] > 0 and xy[1] < ny:
                    x, y = xy
                    jpeg_axes.plot([x, x], [y+radius, y+2.5*radius], 'g', alpha=0.7)
                    jpeg_axes.plot([x, x], [y-radius, y-2.5*radius], 'g', alpha=0.7)
                    jpeg_axes.plot([x-radius, x-2.5*radius], [y, y], 'g', alpha=0.7)
                    jpeg_axes.plot([x+radius, x+2.5*radius], [y, y], 'g', alpha=0.7)
#                     c = plt.Circle(xy, radius=radius, edgecolor='g', facecolor='none')
#                     jpeg_axes.add_artist(c)

        ##-------------------------------------------------------------------------
        # Overlay Associated (red)
        if self.cfg['jpeg'].getboolean('overplot_associated', False) is True and self.action.args.associated is not None and self.action.args.wcs is not None:
            titlestr += f'red=associated({len(self.action.args.associated)}) '
#             radius = self.cfg['jpeg'].getfloat('associated_radius', 6)
            radius = int(2.0*self.action.args.fwhm)/binning
            for entry in self.action.args.associated:
                xy = (entry['x'], entry['y'])
                c = plt.Circle(xy, radius=radius, edgecolor='r', facecolor='none', alpha=0.5)
                jpeg_axes.add_artist(c)
                c = plt.Circle(xy, radius=int(np.ceil(1.5*radius)), edgecolor='r', facecolor='none', alpha=0.5)
                jpeg_axes.add_artist(c)
                c = plt.Circle(xy, radius=int(np.ceil(2.0*radius)), edgecolor='r', facecolor='none', alpha=0.5)
                jpeg_axes.add_artist(c)

        ##-------------------------------------------------------------------------
        # Overlay Pointing
        if self.cfg['jpeg'].getboolean('overplot_pointing', False) is True\
            and self.action.args.header_pointing is not None\
            and self.action.args.wcs is not None\
            and self.action.args.wcs_pointing is not None:
            radius = self.cfg['jpeg'].getfloat('pointing_radius', 40)
            x, y = self.action.args.wcs.all_world2pix(self.action.args.header_pointing.ra.degree,
                                     self.action.args.header_pointing.dec.degree, 1)
            jpeg_axes.plot([nx/2-3*radius,nx/2+3*radius], [ny/2,ny/2], 'k-', alpha=0.5)
            jpeg_axes.plot([nx/2, nx/2], [ny/2-3*radius,ny/2+3*radius], 'k-', alpha=0.5)
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
        if self.action.args.objects is None:
#             pixel_axes = plt.axes(plotpos[0][1])
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
        if self.action.args.associated is not None:
            nstars, bins, p = fwhm_axes.hist(self.action.args.associated['FWHM']*pixel_scale,
                                             bins=np.arange(minfwhm,maxfwhm,0.25),
                                             color='r', alpha=0.5)
            fwhm_axes.plot([avg_fwhm, avg_fwhm], [0,max(nstars)*1.2], 'r-', alpha=0.5)

        ##-------------------------------------------------------------------------
        # Plot histogram of Sky Background
    #     if self.action.args.objects is not None:
    #         sky_axes = plt.axes(plotpos[0][1])
    #         avg_sky = np.median(self.action.args.objects['sky'].value)
    #         sky_axes.set_title(f"Sky Background = {avg_sky:.1f} e-/pix")
    #         lowsky = np.percentile(self.action.args.objects['sky'].value, 1)
    #         highsky = np.percentile(self.action.args.objects['sky'].value, 99)
    #         nstars, bins, p = sky_axes.hist(self.action.args.objects['sky'].value,
    #                                         bins=np.linspace(lowsky, highsky, 20),
    #                                         color='b', alpha=0.5)
    #         sky_axes.plot([avg_sky, avg_sky], [0,max(nstars)*1.2], 'r-', alpha=0.5)
    #         sky_axes.set_ylabel('N stars')
    #         sky_axes.set_ylim(0,max(nstars)*1.2)
    #         sky_axes.set_xlabel("Sky Background (e-/pix)")

        ##-------------------------------------------------------------------------
        # Plot FWHM vs. Flux
        if self.action.args.objects is not None:
            avg_fwhm = np.median(self.action.args.objects['FWHM'])*pixel_scale
            fwhmmag_axes = plt.axes(plotpos[1][1])
            fwhmmag_axes.plot(self.action.args.objects['FWHM']*pixel_scale, self.action.args.objects['flux2'], 'bo',
                              mec='none', alpha=0.3)
            fwhmmag_axes.plot([avg_fwhm, avg_fwhm], [1,max(self.action.args.objects['flux2'].value)*1.5],
                              'r-', alpha=0.5)
            fwhmmag_axes.set_xlabel("FWHM (arcsec)")
            fwhmmag_axes.set_xlim(minfwhm, maxfwhm)
            fwhmmag_axes.set_ylabel(f"Flux (e-/s)")
            fwhmmag_axes.set_yscale("log")
        if self.action.args.associated is not None:
            fwhmmag_axes.plot(self.action.args.associated['FWHM']*pixel_scale, self.action.args.associated['flux2'], 'ro',
                              mec='none', alpha=0.3)

        ##-------------------------------------------------------------------------
        # Plot instrumental mags
        if self.action.args.associated is not None:
            flux_axes = plt.axes(plotpos[2][1])
#             flux_axes.plot(self.action.args.associated['mag'], self.action.args.associated['flux'], 'go',
#                           label='Source Extractor', mec=None, alpha=0.6)
            flux_axes.plot(self.action.args.associated['mag'], self.action.args.associated['flux2'], 'bo',
                          label='photutils', mec=None, alpha=0.6)
            flux_axes.set_xlim(min(self.action.args.associated['mag']), max(self.action.args.associated['mag']))

            flux_axes.set_yscale('log')
            flux_axes.invert_yaxis()
            flux_axes.set_ylabel('Measured Flux (e-/s)')
            plt.grid()
            if self.action.args.zero_point_fit is not None:
                label = f'throughput={self.action.args.zero_point_fit.slope.value:.3g} e-/photon'
                flux_axes.plot(self.action.args.associated['mag'],
                              self.action.args.zero_point_fit(self.action.args.associated['catflux']), 'r-',
                              label=label)
            plt.legend(loc='best')

        ##-------------------------------------------------------------------------
        # Plot instrumental mag diffs
        if self.action.args.associated is not None and self.action.args.zero_point_fit is not None:
            diff_axes = plt.axes(plotpos[3][1])
            ratio = self.action.args.associated['flux2'] / self.action.args.zero_point_fit(self.action.args.associated['catflux'])
            deltamag = -2.512*np.log10(ratio)
            mean, med, std = stats.sigma_clipped_stats(deltamag)
            p1, p99 = np.percentile(deltamag, 1), np.percentile(deltamag, 99)

            diff_axes.plot(self.action.args.associated['mag'], deltamag, 'bo',
                           label=f'Delta Mag (StdDev={std:.2f}, p1={p1:.2f}, p99={p99:.2f})',
                           mec=None, alpha=0.6)
            diff_axes.plot([min(self.action.args.associated['mag']),
                            max(self.action.args.associated['mag'])],
                           [0,0], 'k-', mec=None, alpha=0.6,
                           )
            diff_axes.plot([min(self.action.args.associated['mag']),
                            max(self.action.args.associated['mag'])],
                           [0.1,0.1], 'r-', mec=None, alpha=0.6,
                           label=f"+/-0.1 magnitude error",
                           )
            diff_axes.plot([min(self.action.args.associated['mag']),
                            max(self.action.args.associated['mag'])],
                           [-0.1,-0.1], 'r-', mec=None, alpha=0.6,
                           )
            diff_axes.set_xlabel('Catalog Magnitude')
            diff_axes.set_xlim(min(self.action.args.associated['mag']), max(self.action.args.associated['mag']))

            diff_axes.set_ylim(min([p1,-0.2]),max([p99,+0.2]))
            diff_axes.set_ylabel('-2.512*log10(Measured/Fitted)')
            plt.legend(loc='best')
            plt.grid()


        jpeg_axes.set_title(titlestr)
        reportfilename = f'{self.action.args.fitsfile.split(".")[0]}.jpg'
        self.action.args.jpegfile = Path('/var/www/plots/V20/') / reportfilename
        plt.savefig(self.action.args.jpegfile, dpi=dpi)


        return self.action.args
