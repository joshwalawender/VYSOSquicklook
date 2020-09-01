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

from .__init__ import pre_condition, post_condition


##-----------------------------------------------------------------------------
## Function: generate_report
##-----------------------------------------------------------------------------
def generate_report(im, wcs, fitsfile=None, cfg=None, fwhm=None,
                    objects=None, catalog=None, associated=None,
                    header_pointing=None, wcs_pointing=None,
                    zero_point_fit=None, f0=None,
                    ):
    '''Generate an report on the image. Contains the image itself with overlays
    and analysis plots.
    '''
    plt.rcParams.update({'font.size': 24})
    binning = cfg['jpeg'].getint('binning', 1)
    vmin = np.percentile(im, cfg['jpeg'].getfloat('vmin_percent', 0.5))
    vmax = np.percentile(im, cfg['jpeg'].getfloat('vmax_percent', 99))
    dpi = cfg['jpeg'].getint('dpi', 72)
    nx, ny = im.shape
    sx = nx/dpi/binning
    sy = ny/dpi/binning

    if wcs is not None:
        pixel_scale = np.mean(proj_plane_pixel_scales(wcs))*60*60
    else:
        pixel_scale = cfg['Telescope'].getfloat('pixel_scale', 1)

    fig = plt.figure(figsize=(2*sx, 1*sy), dpi=dpi)

    plotpos = [ [ [0.010, 0.010, 0.550, 0.965], [0.565, 0.775, 0.375, 0.200] ],
                [ None                        , [0.565, 0.540, 0.375, 0.200] ],
                [ None                        , [0.565, 0.265, 0.375, 0.240] ],
                [ None                        , [0.565, 0.025, 0.375, 0.220] ],
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
    if cfg['jpeg'].getboolean('overplot_extracted', False) is True and objects is not None:
        titlestr += 'green=extracted '
        radius = cfg['jpeg'].getfloat('extracted_radius', 6)
        for star in objects:
            if star['x'] > 0 and star['x'] < nx and star['y'] > 0 and star['y'] < ny:
                c = plt.Circle((star['x'], star['y']), radius=radius,
                               edgecolor='g', facecolor='none')
                jpeg_axes.add_artist(c)

    ##-------------------------------------------------------------------------
    # Overlay Catalog (blue)
    if cfg['jpeg'].getboolean('overplot_catalog', False) is True and catalog is not None:
        titlestr += 'blue=catalog '
        radius = cfg['jpeg'].getfloat('catalog_radius', 6)
        x, y = wcs.all_world2pix(catalog['RA'], catalog['DEC'], 1)
        for xy in zip(x, y):
            if xy[0] > 0 and xy[0] < nx and xy[1] > 0 and xy[1] < ny:
                c = plt.Circle(xy, radius=radius, edgecolor='b', facecolor='none')
                jpeg_axes.add_artist(c)

    ##-------------------------------------------------------------------------
    # Overlay Associated (red)
    if cfg['jpeg'].getboolean('overplot_associated', False) is True and associated is not None:
        titlestr += 'red=associated '
        radius = cfg['jpeg'].getfloat('associated_radius', 6)
        for entry in associated:
            xy = (entry['x'], entry['y'])
            c = plt.Circle(xy, radius=radius, edgecolor='r', facecolor='none')
            jpeg_axes.add_artist(c)

    ##-------------------------------------------------------------------------
    # Overlay Pointing
    if cfg['jpeg'].getboolean('overplot_pointing', False) is True\
        and header_pointing is not None\
        and wcs_pointing is not None:
        radius = cfg['jpeg'].getfloat('pointing_radius', 6)
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
        flux_axes.plot(associated['catflux'], associated['flux'], 'go',
                      label='Source Extractor', mec=None, alpha=0.6)
        flux_axes.plot(associated['catflux'], associated['flux2'], 'bo',
                      label='photutils', mec=None, alpha=0.6)
        flux_axes.set_xscale('log')
        flux_axes.set_yscale('log')
        flux_axes.invert_xaxis()
        flux_axes.invert_yaxis()
        flux_axes.set_xlabel('Estimated Catalog Flux (photons/s)')
        flux_axes.set_ylabel('Measured Flux (e-/s)')
        plt.grid()
        if zero_point_fit is not None:
            label = f'throughput={zero_point_fit.slope.value:.3g} e-/photon'
            flux_axes.plot(associated['catflux'],
                          zero_point_fit(associated['catflux']), 'r-',
                          label=label)
        plt.legend(loc='best')

#         if f0 is not None:
#             mag_axes = flux_axes.twiny()
#             mag_axes.set_xlabel('Catalog Magnitude')
#             minmag = np.floor(2.512*np.log10(f0/max(associated['catflux'])))
#             maxmag = np.ceil(2.512*np.log10(f0/min(associated['catflux'])))
#             mags = np.arange(minmag,maxmag,1)
#             f = f0 * 10**(-mags/2.512)
#             mag_axes.set_xticks(f)
#             mag_axes.set_xticklabels([f"{m:.0f}" for m in mags])
#             mag_axes.set_xlabel(f"{maxmag:.0f} {min(associated['catflux']):.2g}")

    ##-------------------------------------------------------------------------
    # Plot instrumental mag diffs
    if associated is not None and zero_point_fit is not None:
        diff_axes = plt.axes(plotpos[3][1])
        diffs = associated['flux2'] - zero_point_fit(associated['catflux'])
        mean, med, std = stats.sigma_clipped_stats(diffs)
        flux_axes.plot(associated['catflux'], diffs, 'bo',
                      label='diffs', mec=None, alpha=0.6)
        flux_axes.set_title(f"StdDev = {std:.2g}")
        flux_axes.set_ylabel('Measured Flux - Fitted Flux (e-/s)')
        flux_axes.set_xlabel('Estimated Catalog Flux (photons/s)')
        flux_axes.set_xscale('log')
        flux_axes.set_yscale('log')

    jpeg_axes.set_title(titlestr)
    reportfilename = f'{fitsfile.split(".")[0]}.jpg'
    reportfile = Path('/var/www/plots/V20/') / reportfilename
    plt.savefig(reportfile, dpi=dpi)
    return reportfile


##-----------------------------------------------------------------------------
## Primitive: CopyDataLocally
##-----------------------------------------------------------------------------
class CopyDataLocally(BasePrimitive):
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
                  pre_condition(self, 'Found expected image path',
                                self.action.args.fitsfilepath.parts[:5] == ['/', 'Users', 'vysosuser', 'V20Data', 'Images']),
                  pre_condition(self, 'Found UT date in directory name',
                                re.match('\d{8}UT', self.action.args.fitsfilepath.parts[-2])),
                  pre_condition(self, 'copy_local setting is configured',
                                self.cfg['Telescope'].get('copy_local', None) is not None),
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

        # Try to determine date string from path to file
        fitsfile = self.action.args.fitsfilepath
        date_string = fitsfile.parts[-2]

        # Look for log file
        logfile = fitsfile.parent.parent.parent / 'Logs' / fitsfile.parts[-2] / f"{fitsfile.stem}.log"

        destinations = self.cfg['Telescope'].get('copy_local', None).split(',')
        success = [False] * len(destinations)
        for destination in destinations:
            destination = Path(destination).expanduser()
            self.log.debug(f'  Destination: {destination}')
            image_destination = destination.joinpath('Images', '2020', date_string, fitsfile.name)
            if image_destination.parent.exists() is False:
                image_destination.parent.mkdir(parents=True)
            image_destination_fz = destination.joinpath('Images', '2020', date_string, f'{fitsfile.name}.fz')
            self.log.debug(f'  Image Destination: {image_destination}')
            log_destination = destination.joinpath('Logs', '2020', date_string)
            if log_destination.parent.exists() is False:
                log_destination.parent.mkdir(parents=True)
            self.log.debug(f'  Log Destination: {log_destination}')

            if image_destination.exists() == False and image_destination_fz.exists() == False:
                self.log.info(f'Writing fits file to {image_destination}')
                with fits.open(fitsfile, checksum=True) as hdul:
                    hdul[0].add_checksum()
                    hdul.writeto(image_destination, checksum=True)
            if image_destination.exists() == True and image_destination_fz.exists() == False:
                self.log.info(f'Compressing fits file at {image_destination}')
                subprocess.call(['fpack', image_destination])

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: MoonInfo
##-----------------------------------------------------------------------------
class MoonInfo(BasePrimitive):
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

        lat=c.Latitude(self.action.args.kd.get('SITELAT'), unit=u.degree)
        lon=c.Longitude(self.action.args.kd.get('SITELONG'), unit=u.degree)
        height=float(self.action.args.kd.get('ALT-OBS')) * u.meter
        loc = c.EarthLocation(lon, lat, height)
        temperature=float(self.action.args.kd.get('AMBTEMP'))*u.Celsius
        pressure=self.cfg['Telescope'].getfloat('pressure', 700)*u.mbar
        altazframe = c.AltAz(location=loc, obstime=self.action.args.kd.obstime(),
                             temperature=temperature,
                             pressure=pressure)
        moon = c.get_moon(Time(self.action.args.kd.obstime()), location=loc)
        sun = c.get_sun(Time(self.action.args.kd.obstime()))

        moon_alt = ((moon.transform_to(altazframe).alt).to(u.degree)).value
        moon_separation = (moon.separation(self.action.args.header_pointing).to(u.degree)).value\
                    if self.action.args.header_pointing is not None else None

        # Moon illumination formula from Meeus, “Astronomical 
        # Algorithms". Formulae 46.1 and 46.2 in the 1991 edition, 
        # using the approximation cos(psi) \approx -cos(i). Error 
        # should be no more than 0.0014 (p. 316). 
        moon_illum = 50*(1 - np.sin(sun.dec.radian)*np.sin(moon.dec.radian)\
                     - np.cos(sun.dec.radian)*np.cos(moon.dec.radian)\
                     * np.cos(sun.ra.radian-moon.ra.radian))

        self.action.args.moon_alt = moon_alt
        self.action.args.moon_separation = moon_separation
        self.action.args.moon_illum = moon_illum

        return self.action.args


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

        self.action.args.jpegfile = generate_report(
                              self.action.args.kd.pixeldata[0].data,
                              self.action.args.wcs,
                              fitsfile=self.action.args.fitsfile,
                              cfg=self.cfg,
                              objects=self.action.args.objects,
                              catalog=self.action.args.catalog,
                              associated=self.action.args.associated,
                              header_pointing=self.action.args.header_pointing,
                              wcs_pointing=self.action.args.wcs_pointing,
                              zero_point_fit=self.action.args.zero_point_fit,
                              f0=self.action.args.f0,
                              )

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: Record
##-----------------------------------------------------------------------------
class Record(BasePrimitive):
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
                  pre_condition(self, 'norecord configuration is not set',
                                not self.cfg['Telescope'].getboolean('norecord', False)),
                  pre_condition(self, 'connected to mongoDB',
                                self.action.args.images is not None),
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

        # Comple image info to store
        self.image_info = {
            'filename': self.action.args.fitsfile,
            'telescope': self.action.args.kd.instrument,
            'compressed': Path(self.action.args.kd.fitsfilename).suffix == '.fz',
            }
        # From Header
        if self.action.args.kd.get('OBJECT', None) is not None:
            self.image_info['target name'] = self.action.args.kd.get('OBJECT')
        if self.action.args.kd.get('EXPTIME', None) is not None:
            self.image_info['exptime'] = self.action.args.kd.get('EXPTIME')
        if self.action.args.kd.obstime() is not None:
            self.image_info['date'] = self.action.args.kd.obstime()
        if self.action.args.kd.get('FILTER', None) is not None:
            self.image_info['filter'] = self.action.args.kd.get('FILTER')
        if self.action.args.kd.get('AZIMUTH', None) is not None:
            self.image_info['az'] = float(self.action.args.kd.get('AZIMUTH'))
        if self.action.args.kd.get('ALTITUDE', None) is not None:
            self.image_info['alt'] = float(self.action.args.kd.get('ALTITUDE'))
        if self.action.args.kd.get('AIRMASS', None) is not None:
            self.image_info['airmass'] = float(self.action.args.kd.get('AIRMASS'))
        # From Science Image Analysis
        if hasattr(self.action.args, 'header_pointing'):
            if self.action.args.header_pointing is not None:
                self.image_info['header_RA'] = self.action.args.header_pointing.ra.deg
                self.image_info['header_DEC'] = self.action.args.header_pointing.dec.deg
        if hasattr(self.action.args, 'moon_alt'):
            if self.action.args.moon_alt is not None:
                self.image_info['moon_alt'] = self.action.args.moon_alt
        if hasattr(self.action.args, 'moon_separation'):
            if self.action.args.moon_separation is not None:
                self.image_info['moon_separation'] = self.action.args.moon_separation
        if hasattr(self.action.args, 'moon_illum'):
            if self.action.args.moon_illum is not None:
                self.image_info['moon_illumination'] = self.action.args.moon_illum
        if hasattr(self.action.args, 'fwhm'):
            if self.action.args.fwhm is not None:
                self.image_info['FWHM_pix'] = self.action.args.fwhm
        if hasattr(self.action.args, 'ellipticity'):
            if self.action.args.ellipticity is not None:
                self.image_info['ellipticity'] = self.action.args.ellipticity
        if hasattr(self.action.args, 'n_objects'):
            if self.action.args.n_objects is not None:
                self.image_info['n_stars'] = self.action.args.n_objects
        if hasattr(self.action.args, 'zero_point_fit'):
            if self.action.args.zero_point_fit is not None:
                self.image_info['zero point'] = self.action.args.zero_point_fit.slope.value
        if hasattr(self.action.args, 'sky_background'):
            if self.action.args.sky_background is not None:
                self.image_info['sky background'] = self.action.args.sky_background
        if hasattr(self.action.args, 'perr'):
            if self.action.args.perr is not None and not np.isnan(self.action.args.perr):
                self.image_info['perr_arcmin'] = self.action.args.perr.to(u.arcmin).value
        if hasattr(self.action.args, 'wcs'):
            if self.action.args.wcs is not None:
                self.image_info['wcs'] = str(self.action.args.wcs.to_header()).strip()
        if hasattr(self.action.args, 'jpegfile'):
            if self.action.args.jpegfile is not None:
                self.image_info['jpegs'] = [f"{self.action.args.jpegfile.name}"]
        # From Flat Image Analysis
        if hasattr(self.action.args, 'image_stats'):
            if self.action.args.image_stats is not None:
                self.image_info['sky background'] = self.action.args.image_stats[1]

        # Log this info
        for key in self.image_info.keys():
            self.log.debug(f'  {key}: {self.image_info[key]}')

        # Remove old entries for this image file
        deletion = self.action.args.images.delete_many( {'filename': self.action.args.kd.fitsfilename} )
        self.log.debug(f'  Deleted {deletion.deleted_count} previous entries for {self.action.args.kd.fitsfilename}')

        # Save new entry for this image file
        self.log.debug('Adding image info to mongo database')
        ## Save document
        try:
            inserted_id = self.action.args.images.insert_one(self.image_info).inserted_id
            self.log.debug(f"  Inserted document id: {inserted_id}")
        except:
            e = sys.exc_info()[0]
            self.log.error('Failed to add new document')
            self.log.error(e)
        self.action.args.mongoclient.close()

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: UpdateDirectory
##-----------------------------------------------------------------------------
class UpdateDirectory(BasePrimitive):
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

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        checks = []
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

        newdir = Path(self.action.args.input).expanduser()
        if newdir.exists() is False:
            self.log.info(f"  Creating directory {newdir}")
            newdir.mkdir(parents=True)
        self.log.info(f"  Updating directory to {newdir}")

        self.context.data_set.remove_all()
        self.context.data_set.change_directory(f"{newdir}")

        files = [f.name for f in newdir.glob('*.fts')]
        self.log.info(f"  Ingesting {len(files)} files")
        self.log.debug(files)
        for file in files:
            self.log.debug(f"  Appending {file}")
            self.context.data_set.append_item(file)

        self.context.data_set.start_monitor()

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: SetOverwrite
##-----------------------------------------------------------------------------
class SetOverwrite(BasePrimitive):
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
        checks = []
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
        self.log.info(f"  Setting overwrite to True")
        self.cfg.set('Telescope', 'overwrite', value='True')

        return self.action.args



# class Template(BasePrimitive):
#     """
#     This is a template for primitives, which is usually an action.
# 
#     The methods in the base class can be overloaded:
#     - _pre_condition
#     - _post_condition
#     - _perform
#     - apply
#     - __call__
#     """
# 
#     def __init__(self, action, context):
#         BasePrimitive.__init__(self, action, context)
#         self.log = context.pipeline_logger
#         self.cfg = self.context.config.instrument
# 
#     def _pre_condition(self):
#         """Check for conditions necessary to run this process"""
#         checks = []
#         return np.all(checks)
# 
#     def _post_condition(self):
#         """Check for conditions necessary to verify that the process run correctly"""
#         checks = []
#         return np.all(checks)
# 
#     def _perform(self):
#         """
#         Returns an Argument() with the parameters that depends on this operation.
#         """
#         self.log.info(f"Running {self.__class__.__name__} action")
# 
#         return self.action.args

