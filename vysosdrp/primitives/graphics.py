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
        wcs = self.action.args.wcs
        fitsfile = self.action.args.fitsfile
        objects = self.action.args.objects
        catalog = self.action.args.catalog
        associated = self.action.args.associated
        header_pointing = self.action.args.header_pointing
        wcs_pointing = self.action.args.wcs_pointing
        zero_point_fit = self.action.args.zero_point_fit

        plt.rcParams.update({'font.size': 24})
        binning = self.cfg['jpeg'].getint('binning', 1)
        vmin = np.percentile(im, self.cfg['jpeg'].getfloat('vmin_percent', 0.5))
        vmax = np.percentile(im, self.cfg['jpeg'].getfloat('vmax_percent', 99))
        dpi = self.cfg['jpeg'].getint('dpi', 72)
        nx, ny = im.shape
        sx = nx/dpi/binning
        sy = ny/dpi/binning

        if wcs is not None:
            pixel_scale = np.mean(proj_plane_pixel_scales(wcs))*60*60
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
        titlestr = f'{fitsfile}: '

        ##-------------------------------------------------------------------------
        # Overlay Extracted (green)
        if self.cfg['jpeg'].getboolean('overplot_extracted', False) is True and objects is not None:
            titlestr += 'green=extracted '
            radius = self.cfg['jpeg'].getfloat('extracted_radius', 6)
            for star in objects:
                if star['x'] > 0 and star['x'] < nx and star['y'] > 0 and star['y'] < ny:
                    c = plt.Circle((star['x'], star['y']), radius=radius,
                                   edgecolor='g', facecolor='none')
                    jpeg_axes.add_artist(c)

        ##-------------------------------------------------------------------------
        # Overlay Catalog (blue)
        if self.cfg['jpeg'].getboolean('overplot_catalog', False) is True and catalog is not None and wcs is not None:
            titlestr += 'blue=catalog '
            radius = self.cfg['jpeg'].getfloat('catalog_radius', 6)
            x, y = wcs.all_world2pix(catalog['RA'], catalog['DEC'], 1)
            for xy in zip(x, y):
                if xy[0] > 0 and xy[0] < nx and xy[1] > 0 and xy[1] < ny:
                    c = plt.Circle(xy, radius=radius, edgecolor='b', facecolor='none')
                    jpeg_axes.add_artist(c)

        ##-------------------------------------------------------------------------
        # Overlay Associated (red)
        if self.cfg['jpeg'].getboolean('overplot_associated', False) is True and associated is not None and wcs is not None:
            titlestr += 'red=associated '
            radius = self.cfg['jpeg'].getfloat('associated_radius', 6)
            for entry in associated:
                xy = (entry['x'], entry['y'])
                c = plt.Circle(xy, radius=radius, edgecolor='r', facecolor='none')
                jpeg_axes.add_artist(c)

        ##-------------------------------------------------------------------------
        # Overlay Pointing
        if self.cfg['jpeg'].getboolean('overplot_pointing', False) is True\
            and header_pointing is not None\
            and wcs_pointing is not None:
            radius = self.cfg['jpeg'].getfloat('pointing_radius', 6)
            x, y = wcs.all_world2pix(header_pointing.ra.degree,
                                     header_pointing.dec.degree, 1)
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
        # Plot histogram of FWHM
        if objects is not None:
            fwhm_axes = plt.axes(plotpos[0][1])
            minfwhm = 1
            maxfwhm = 7
            avg_fwhm = np.median(objects['FWHM'])*pixel_scale
            fwhm_axes.set_title(f"FWHM = {avg_fwhm:.1f} arcsec")
            nstars, bins, p = fwhm_axes.hist(objects['FWHM']*pixel_scale,
                                             bins=np.arange(minfwhm,maxfwhm,0.25),
                                             color='g', alpha=0.5)
            fwhm_axes.plot([avg_fwhm, avg_fwhm], [0,max(nstars)*1.2], 'r-', alpha=0.5)
            fwhm_axes.set_ylabel('N stars')
            fwhm_axes.set_ylim(0,max(nstars)*1.2)
        if associated is not None:
            nstars, bins, p = fwhm_axes.hist(associated['FWHM']*pixel_scale,
                                             bins=np.arange(minfwhm,maxfwhm,0.25),
                                             color='r', alpha=0.5)
            fwhm_axes.plot([avg_fwhm, avg_fwhm], [0,max(nstars)*1.2], 'r-', alpha=0.5)

        ##-------------------------------------------------------------------------
        # Plot histogram of Sky Background
    #     if objects is not None:
    #         sky_axes = plt.axes(plotpos[0][1])
    #         avg_sky = np.median(objects['sky'].value)
    #         sky_axes.set_title(f"Sky Background = {avg_sky:.1f} e-/pix")
    #         lowsky = np.percentile(objects['sky'].value, 1)
    #         highsky = np.percentile(objects['sky'].value, 99)
    #         nstars, bins, p = sky_axes.hist(objects['sky'].value,
    #                                         bins=np.linspace(lowsky, highsky, 20),
    #                                         color='g', alpha=0.5)
    #         sky_axes.plot([avg_sky, avg_sky], [0,max(nstars)*1.2], 'r-', alpha=0.5)
    #         sky_axes.set_ylabel('N stars')
    #         sky_axes.set_ylim(0,max(nstars)*1.2)
    #         sky_axes.set_xlabel("Sky Background (e-/pix)")

        ##-------------------------------------------------------------------------
        # Plot FWHM vs. Flux
        if objects is not None:
            avg_fwhm = np.median(objects['FWHM'])*pixel_scale
            fwhmmag_axes = plt.axes(plotpos[1][1])
            fwhmmag_axes.plot(objects['FWHM']*pixel_scale, objects['flux2'], 'go',
                              mec='none', alpha=0.3)
            fwhmmag_axes.plot([avg_fwhm, avg_fwhm], [1,max(objects['flux2'].value)*1.5],
                              'r-', alpha=0.5)
            fwhmmag_axes.set_xlabel("FWHM (arcsec)")
            fwhmmag_axes.set_xlim(minfwhm, maxfwhm)
            fwhmmag_axes.set_ylabel(f"Flux (e-/s)")
            fwhmmag_axes.set_yscale("log")
        if associated is not None:
            fwhmmag_axes.plot(associated['FWHM']*pixel_scale, associated['flux2'], 'ro',
                              mec='none', alpha=0.3)

        ##-------------------------------------------------------------------------
        # Plot instrumental mags
        if associated is not None:
            flux_axes = plt.axes(plotpos[2][1])
#             flux_axes.plot(associated['catflux'], associated['flux'], 'go',
#                           label='Source Extractor', mec=None, alpha=0.6)
#             flux_axes.plot(associated['catflux'], associated['flux2'], 'bo',
#                           label='photutils', mec=None, alpha=0.6)
#             flux_axes.set_xscale('log')
#             flux_axes.invert_xaxis()

            flux_axes.plot(associated['mag'], associated['flux'], 'go',
                          label='Source Extractor', mec=None, alpha=0.6)
            flux_axes.plot(associated['mag'], associated['flux2'], 'bo',
                          label='photutils', mec=None, alpha=0.6)
            flux_axes.set_xlim(min(associated['mag']), max(associated['mag']))

            flux_axes.set_yscale('log')
            flux_axes.invert_yaxis()
            flux_axes.set_ylabel('Measured Flux (e-/s)')
            plt.grid()
            if zero_point_fit is not None:
                label = f'throughput={zero_point_fit.slope.value:.3g} e-/photon'
                flux_axes.plot(associated['mag'],
                              zero_point_fit(associated['catflux']), 'r-',
                              label=label)
#                 flux_axes.plot(associated['catflux'],
#                               zero_point_fit(associated['catflux']), 'r-',
#                               label=label)
            plt.legend(loc='best')

        ##-------------------------------------------------------------------------
        # Plot instrumental mag diffs
        if associated is not None and zero_point_fit is not None:
            diff_axes = plt.axes(plotpos[3][1])
            ratio = associated['flux2'] / zero_point_fit(associated['catflux'])
            deltamag = -2.512*np.log10(ratio)
            mean, med, std = stats.sigma_clipped_stats(deltamag)
            p1, p99 = np.percentile(deltamag, 1), np.percentile(deltamag, 99)
#             diff_axes.plot(associated['catflux'], deltamag, 'bo',
#                            label='Delta Mag (StdDev={std:.2f}, p1={p1:.2f}, p99={p99:.2f})',
#                            mec=None, alpha=0.6)
#             diff_axes.plot([min(associated['catflux']), max(associated['catflux'])], [0,0], 'k-',
#                            mec=None, alpha=0.6)
#             diff_axes.set_xlabel('Estimated Catalog Flux (photons/s)')
#             diff_axes.set_xscale('log')
#             diff_axes.invert_xaxis()

            diff_axes.plot(associated['mag'], deltamag, 'bo',
                           label=f'Delta Mag (StdDev={std:.2f}, p1={p1:.2f}, p99={p99:.2f})',
                           mec=None, alpha=0.6)
            diff_axes.plot([min(associated['mag']), max(associated['mag'])], [0,0], 'k-',
                           mec=None, alpha=0.6)
            diff_axes.plot([min(associated['mag']), max(associated['mag'])], [0.1,0.1], 'r-',
                           mec=None, alpha=0.6)
            diff_axes.plot([min(associated['mag']), max(associated['mag'])], [-0.1,-0.1], 'r-',
                           mec=None, alpha=0.6)
            diff_axes.set_xlabel('Catalog Magnitude')
            diff_axes.set_xlim(min(associated['mag']), max(associated['mag']))

            diff_axes.set_ylim(p1,p99)
            diff_axes.set_ylabel('-2.512*log10(Measured/Fitted)')
            plt.legend(loc='best')
            plt.grid()


        jpeg_axes.set_title(titlestr)
        reportfilename = f'{fitsfile.split(".")[0]}.jpg'
        self.action.args.jpegfile = Path('/var/www/plots/V20/') / reportfilename
        plt.savefig(self.action.args.jpegfile, dpi=dpi)


        return self.action.args
