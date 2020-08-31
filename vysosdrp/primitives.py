from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import subprocess
from matplotlib import pyplot as plt

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import stats
from astropy.time import Time
import astropy.coordinates as c
from astropy.wcs import WCS
from astropy.table import Table, Column
import ccdproc
import photutils
import sep

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

from .tools import get_catalog, get_moon_info, sigma_clipping_line_fit
from .vysos_plotting import generate_report


##-----------------------------------------------------------------------------
## Primitive: ReadFITS
##-----------------------------------------------------------------------------
class ReadFITS(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.log.info(f"Initializing {self.__class__.__name__}")
        self.cfg = self.context.config.instrument
        # initialize values in the args for future use
        self.action.args.db_entry = None
        self.action.args.kd = None
        self.action.args.objects = None
        self.action.args.wcs_pointing = None
        self.action.args.perr = np.nan
        self.action.args.wcs = None
        self.action.args.catalog = None
        self.action.args.skip = False
        self.action.args.fwhm = np.nan
        self.action.args.ellipticity = np.nan
        self.action.args.zero_point = np.nan
        self.action.args.fitsfilepath = Path(self.action.args.name).expanduser().absolute()
        self.action.args.associated = None
        self.action.args.zero_point_fit = None
        self.action.args.f0 = None

        # If we are reading a compressed file, use the uncompressed version of
        # the name for the database
        if self.action.args.fitsfilepath.suffix == '.fz':
            self.action.args.fitsfile = '.'.join(self.action.args.fitsfilepath.name.split('.')[:-1])
        else:
            self.action.args.fitsfile = self.action.args.fitsfilepath.name

        ## Connect to mongo
        try:
            import pymongo
            self.log.debug('Connecting to mongo db')
            self.action.args.mongoclient = pymongo.MongoClient('localhost', 27017)
            self.action.args.images = self.action.args.mongoclient.vysos['images']
        except:
            self.log.error('Could not connect to mongo db')
            self.action.args.mongoclient = None
            self.action.args.images = None


    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

        # Check that fits file exists
        if self.action.args.fitsfilepath.exists():
            self.log.info(f"  File: {self.action.args.fitsfilepath}")
        else:
            self.log.info(f"  Could not find file: {self.action.args.fitsfilepath}")
            some_pre_condition = False

        if self.action.args.images is not None:
            already_processed = [d for d in self.action.args.images.find( {'filename': self.action.args.fitsfile} )]
            if len(already_processed) != 0\
               and self.cfg['Telescope'].getboolean('overwrite', False) is False:
                self.log.info(f"overwrite is {self.cfg['Telescope'].getboolean('overwrite')}")
                self.log.info('  File is already in the database, skipping further processing')
                self.action.args.skip = True
            if len(already_processed) != 0:
                self.action.args.db_entry = already_processed[0]
                if self.action.args.db_entry.get('wcs', None) is not None:
                    self.action.args.wcs = WCS(self.action.args.db_entry.get('wcs'))
                    self.log.info('Found Previously Solved WCS')

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = self.action.args.kd is not None
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
            self.action.args.skip = True
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        # Read FITS file
        self.action.args.kd = fits_reader(self.action.args.fitsfilepath,
                                          datatype=VYSOS20)

        # Read some header info
        self.action.args.obstime = datetime.strptime(self.action.args.kd.get('DATE-OBS'),
                                                     '%Y-%m-%dT%H:%M:%S')
        self.action.args.header_pointing = c.SkyCoord(self.action.args.kd.get('RA'),
                                                      self.action.args.kd.get('DEC'),
                                                      frame='fk5',
                                                      unit=(u.hourangle, u.deg))
        return self.action.args


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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip

        if self.action.args.fitsfilepath.parts[:5] != ['/', 'Users', 'vysosuser', 'V20Data', 'Images']:
            some_pre_condition = False

        if not re.match('\d{8}UT', self.action.args.fitsfilepath.parts[-2]):
            some_pre_condition = False

        # Check if a destination is set in the config file
        if self.cfg['Telescope'].get('copy_local', None) is None:
            some_pre_condition = False

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        alt, sep, illum = get_moon_info(
                  lat=c.Latitude(self.action.args.kd.get('SITELAT'), unit=u.degree),
                  lon=c.Longitude(self.action.args.kd.get('SITELONG'), unit=u.degree),
                  height=float(self.action.args.kd.get('ALT-OBS')) * u.meter,
                  temperature=float(self.action.args.kd.get('AMBTEMP'))*u.Celsius,
                  pressure=self.cfg['Telescope'].getfloat('pressure', 700)*u.mbar,
                  time=self.action.args.obstime,
                  pointing=self.action.args.header_pointing)
        self.action.args.moon_alt = alt
        self.action.args.moon_separation = sep
        self.action.args.moon_illum = illum

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: GainCorrect
##-----------------------------------------------------------------------------
class GainCorrect(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        gain = self.action.args.kd.get('GAIN', None)
        if gain is not None: self.log.debug(f'Got gain from header: {gain}')
        if gain is None:
            gain = self.cfg['Telescope'].getfloat('gain', None)
            self.log.debug(f'Got gain from config: {gain}')
            self.action.args.kd.headers.append(fits.Header( {'GAIN': gain} ))

        for i,pd in enumerate(self.action.args.kd.pixeldata):
            self.log.debug('Gain correcting pixeldata')
            self.action.args.kd.pixeldata[i] = ccdproc.gain_correct(pd, gain,
                                               gain_unit=u.electron/u.adu)

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: CreateDeviation
##-----------------------------------------------------------------------------
class CreateDeviation(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        header_keywords = ['RN', 'READNOISE']
        for kw in header_keywords:
            rn = self.action.args.kd.get(kw, None)
            if rn is not None: break
        if rn is not None: self.log.debug(f'Got read noise from header: {rn}')

        if rn is None:
            rn = self.cfg['Telescope'].getfloat('RN', None)
            self.log.debug(f'Got read noise from config: {rn}')
            self.action.args.kd.headers.append(fits.Header( {'READNOISE': rn} ))

        for i,pd in enumerate(self.action.args.kd.pixeldata):
            if pd.unit == u.electron:
                self.action.args.kd.pixeldata[i] = ccdproc.create_deviation(pd,
                                   readnoise=rn*u.electron)
            elif pd.unit == u.adu:
                self.action.args.kd.pixeldata[i] = ccdproc.create_deviation(pd,
                                   gain=float(self.action.args.kd.get('GAIN')),
                                   readnoise=rn*u.electron)
            else:
                self.log.error('Could not estimate uncertainty')

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: MakeSourceMask
##-----------------------------------------------------------------------------
class MakeSourceMask(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context, snr=5, npixels=5):
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.snr = snr
        self.npixels = npixels

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip
        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        self.action.args.source_mask = [None] * len(self.action.args.kd.pixeldata)
        for i,pd in enumerate(self.action.args.kd.pixeldata):
            source_mask = photutils.make_source_mask(pd, self.snr, self.npixels)
            self.action.args.source_mask[i] = source_mask

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: SubtractBackground
##-----------------------------------------------------------------------------
class SubtractBackground(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip
        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
        box_size = self.cfg['Extract'].getint('background_box_size', 128)

        self.action.args.background = [None] * len(self.action.args.kd.pixeldata)
        for i,pd in enumerate(self.action.args.kd.pixeldata):
            bkg = photutils.Background2D(pd, box_size=box_size,
                                         mask=self.action.args.source_mask[i],
                                         sigma_clip=stats.SigmaClip())
            self.action.args.background[i] = bkg
#             self.action.args.kd.pixeldata[i].data -= bkg.background
        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: ExtractStars
##-----------------------------------------------------------------------------
class ExtractStars(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip
        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        exptime = float(self.action.args.kd.get('EXPTIME'))

        # Photutils StarFinder
#         extract_fwhm = self.cfg['Extract'].getfloat('extract_fwhm', 5)
#         thresh = self.cfg['Extract'].getint('extract_threshold', 9)
#         pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1)
#         pd = self.action.args.kd.pixeldata[0]
#         mean, median, std = stats.sigma_clipped_stats(pd.data, sigma=3.0)
#         # daofind = photutils.DAOStarFinder(fwhm=extract_fwhm, threshold=thresh*std)  
#         starfind = photutils.IRAFStarFinder(thresh*std, extract_fwhm)  
#         sources = starfind(pd.data)
#         self.log.info(f'  Found {len(sources):d} stars')
#         self.log.info(sources.keys())
# 
#         self.action.args.objects = sources
#         self.action.args.objects.sort('flux')
#         self.action.args.objects.reverse()
#         self.action.args.n_objects = len(sources)
#         if self.action.args.n_objects == 0:
#             self.log.warning('No stars found')
#             self.action.args.fwhm = np.nan
#             self.action.args.ellipticity = np.nan
#         else:
#             FWHM_pix = np.median(sources['fwhm'])
#             roundness = np.median(sources['roundness'])
#             self.log.info(f'  Median FWHM = {FWHM_pix:.1f} pix ({FWHM_pix*pixel_scale:.2f} arcsec)')
#             self.log.info(f'  roundness = {roundness:.2f}')
#             self.action.args.fwhm = FWHM_pix
#             self.action.args.ellipticity = np.nan

        # Source Extractor
        pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1)
        thresh = self.cfg['Extract'].getint('extract_threshold', 9)
        minarea = self.cfg['Extract'].getint('extract_minarea', 7)
        mina = self.cfg['Extract'].getfloat('fwhm_mina', 1)
        minb = self.cfg['Extract'].getfloat('fwhm_minb', 1)

        bsub = self.action.args.kd.pixeldata[0].data - self.action.args.background[0].background
        objects = sep.extract(bsub,
                              err=self.action.args.kd.pixeldata[0].uncertainty.array,
                              mask=self.action.args.kd.pixeldata[0].mask,
                              thresh=float(thresh), minarea=minarea)
        t = Table(objects)
        t['flux'] /= exptime

        ny, nx = bsub.shape
        r = np.sqrt((t['x']-nx/2.)**2 + (t['y']-ny/2.)**2)
        t.add_column(Column(data=r.data, name='r', dtype=np.float))

        coef = 2*np.sqrt(2*np.log(2))
        fwhm = np.sqrt((coef*t['a'])**2 + (coef*t['b'])**2)
        t.add_column(Column(data=fwhm.data, name='FWHM', dtype=np.float))

        ellipticities = t['a']/t['b']
        t.add_column(Column(data=ellipticities.data, name='ellipticity', dtype=np.float))

        filtered = (t['a'] < mina) | (t['b'] < minb) | (t['flag'] > 0)
        self.log.debug(f'  Removing {np.sum(filtered):d}/{len(filtered):d}'\
                       f' extractions from FWHM calculation')
        self.action.args.n_objects = len(t[~filtered])
        self.log.info(f'  Found {self.action.args.n_objects:d} stars')

        self.action.args.objects = t[~filtered]
        self.action.args.objects.sort('flux')
        self.action.args.objects.reverse()

        if self.action.args.n_objects == 0:
            self.log.warning('No stars found')
            self.action.args.fwhm = np.nan
            self.action.args.ellipticity = np.nan
        else:
            FWHM_pix = np.median(t['FWHM'][~filtered])
            ellipticity = np.median(t['ellipticity'][~filtered])
            self.log.info(f'  Median FWHM = {FWHM_pix:.1f} pix ({FWHM_pix*pixel_scale:.2f} arcsec)')
            self.log.info(f'  ellipticity = {ellipticity:.2f}')
            self.action.args.fwhm = FWHM_pix
            self.action.args.ellipticity = ellipticity

        ## Do second photometry measurement
        positions = [(det['x'], det['y']) for det in self.action.args.objects]
        ap_radius = int(2.0*FWHM_pix)
        star_apertures = photutils.CircularAperture(positions, ap_radius)
        sky_apertures = photutils.CircularAnnulus(positions,
                                                  r_in=ap_radius+2,
                                                  r_out=ap_radius+6)
        phot_table = photutils.aperture_photometry(
                               self.action.args.kd.pixeldata[0],
                               [star_apertures, sky_apertures])
        phot_table['sky'] = phot_table['aperture_sum_1'] / sky_apertures.area()
        med_sky = np.median(phot_table['sky'])
        self.log.info(f'  Median Sky = {med_sky.value:.0f} e-/pix')
        self.action.args.sky_background = med_sky.value
        self.action.args.objects.add_column(phot_table['sky'])
        bkg_sum = phot_table['aperture_sum_1'] / sky_apertures.area() * star_apertures.area()
        final_sum = phot_table['aperture_sum_0'] - bkg_sum
        phot_table['flux2'] = final_sum/exptime
        self.action.args.objects.add_column(phot_table['flux2'])

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: SolveAstrometry
##-----------------------------------------------------------------------------
class SolveAstrometry(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip

        if (self.action.args.wcs is not None) and (self.cfg['Telescope'].getboolean('force_solve', False) is False):
            self.log.info('Found existing wcs, skipping solve')
            nx, ny = self.action.args.kd.pixeldata[0].data.shape
            r, d = self.action.args.wcs.all_pix2world([nx/2.], [ny/2.], 1)
            self.action.args.wcs_pointing = c.SkyCoord(r[0], d[0], frame='fk5',
                                              equinox='J2000',
                                              unit=(u.deg, u.deg),
                                              obstime=self.action.args.obstime)
            self.action.args.perr = self.action.args.wcs_pointing.separation(
                                         self.action.args.header_pointing)
            self.log.info(f'Pointing error = {self.action.args.perr.to(u.arcmin):.1f}')
            some_pre_condition = False

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
        from astroquery.exceptions import TimeoutError as astrometryTimeout
        from astroquery.astrometry_net import AstrometryNet

        ast = AstrometryNet()
        ast.api_key = self.cfg['astrometry'].get('api_key', None)
        solve_timeout = self.cfg['astrometry'].getint('solve_timeout', 90)
        nx, ny = self.action.args.kd.pixeldata[0].data.shape

        if self.action.args.n_objects >= 100:
            stars = self.action.args.objects[:100]
        else:
            stars = self.action.args.objects

        try:
            self.log.debug(f"  Running astrometry.net solve")
            wcs_header = ast.solve_from_source_list(stars['x'], stars['y'],
                                                    nx, ny,
                                                    solve_timeout=solve_timeout,
                                                    center_dec=self.action.args.header_pointing.dec.deg,
                                                    center_ra=self.action.args.header_pointing.ra.deg,
                                                    radius=0.8,
                                                    scale_est=0.44,
                                                    scale_err=0.02,
                                                    scale_units='arcsecperpix',
                                                    tweak_order=2,
                                                    )
        except astrometryTimeout as e:
            self.log.warning('Astrometry solve timed out')
            return self.action.args
        except Exception as e:
            self.log.warning('Astrometry solve failed')
            self.log.warning(e)
            return self.action.args

        if wcs_header == {}:
            self.log.info(f"  Solve failed")
            return self.action.args

        self.log.info(f"  Solve complete")
        self.action.args.wcs_header = wcs_header
        self.action.args.wcs = WCS(wcs_header)

        # Determine Pointing Error
#         pixel_scale = np.mean(proj_plane_pixel_scales(self.action.args.wcs))*60*60
        r, d = self.action.args.wcs.all_pix2world([nx/2.], [ny/2.], 1)
        self.action.args.wcs_pointing = c.SkyCoord(r[0], d[0], frame='fk5',
                                          equinox='J2000',
                                          unit=(u.deg, u.deg),
                                          obstime=self.action.args.obstime)
        self.action.args.perr = self.action.args.wcs_pointing.separation(
                                     self.action.args.header_pointing)
        self.log.info(f'Pointing error = {self.action.args.perr.to(u.arcmin):.1f}')

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: GetCatalogStars
##-----------------------------------------------------------------------------
class GetCatalogStars(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip and self.action.args.wcs is not None
        if self.cfg['jpeg'].get('catalog', None) not in ['Gaia', 'UCAC4']:
            self.log.debug(f"Only support Gaia and UCAC4 catalog")
            some_pre_condition = False
        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        catalogname = self.cfg['jpeg'].get('catalog')
        maglimit = self.cfg['jpeg'].get('catalog_maglimit')

        fp = self.action.args.wcs.calc_footprint(axes=self.action.args.kd.pixeldata[0].data.shape)
        dra = fp[:,0].max() - fp[:,0].min()
        ddec = fp[:,1].max() - fp[:,1].min()
        radius = np.sqrt((dra*np.cos(fp[:,1].mean()*np.pi/180.))**2 + ddec**2)/2.

        if self.action.args.wcs_pointing is not None:
            self.log.debug('Using WCS pointing for catalog query')
            pointing = self.action.args.wcs_pointing
        else:
            self.log.warning('Using header pointing for catalog query')
            pointing = self.action.args.header_pointing

        self.log.info(f"Retrieving {catalogname} entries (magnitude < {maglimit})")
        self.action.args.catalog = get_catalog(pointing, radius, catalog=catalogname, maglimit=maglimit)
        self.log.info(f"  Found {len(self.action.args.catalog)} catalog entries")

        return self.action.args



##-----------------------------------------------------------------------------
## Primitive: AssociateCatalogStars
##-----------------------------------------------------------------------------
class AssociateCatalogStars(BasePrimitive):
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip

        min_stars = 20

        if self.action.args.objects is None:
            some_pre_condition = False
        elif len(self.action.args.objects) < min_stars:
            some_pre_condition = False

        if self.action.args.catalog is None:
            some_pre_condition = False
        elif len(self.action.args.catalog) < min_stars:
            some_pre_condition = False

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        pixel_scale = self.cfg['Telescope'].getfloat('pixel_scale', 1) * u.arcsec/u.pix
        assoc_radius = self.cfg['Extract'].getfloat('accoc_radius', 1)\
                       * self.action.args.fwhm*u.pix\
                       * pixel_scale
        associated = Table(names=('RA', 'DEC', 'x', 'y', 'assoc_distance', 'mag', 'catflux', 'flux', 'flux2', 'instmag', 'FWHM'),
                           dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8') )

        catalog_coords = c.SkyCoord(self.action.args.catalog['RA'], self.action.args.catalog['DEC'])

        band = 4760 # Jy (for i band)
        dl = 0.16 # dl/l (for i band)
        d_telescope = self.cfg['Telescope'].getfloat('d_primary_mm', 508)
        d_obstruction = self.cfg['Telescope'].getfloat('d_obstruction_mm', 127)
        A = 3.14*(d_telescope/2/1000)**2 - 3.14*(d_obstruction/2/1000)**2 # m^2
        # 1 Jy = 1.51e7 photons sec^-1 m^-2 (dlambda/lambda)^-1
        # https://archive.is/20121204144725/http://www.astro.utoronto.ca/~patton/astro/mags.html#selection-587.2-587.19
        self.action.args.f0 = band * 1.51e7 * A * dl # photons / sec

        self.log.info(self.action.args.objects.keys())
        for detected in self.action.args.objects:
            ra_deg, dec_deg = self.action.args.wcs.all_pix2world(detected['x'], detected['y'], 1)
            detected_coord = c.SkyCoord(ra_deg, dec_deg, frame='fk5', unit=(u.deg, u.deg))
            idx, d2d, d3d = detected_coord.match_to_catalog_sky(catalog_coords)
            if d2d[0].to(u.arcsec) < assoc_radius.to(u.arcsec):
                associated.add_row( {'RA': self.action.args.catalog[idx]['RA'],
                                     'DEC': self.action.args.catalog[idx]['DEC'],
                                     'x': detected['x'],
                                     'y': detected['y'],
                                     'assoc_distance': d2d[0].to(u.arcsec).value, 
                                     'mag': self.action.args.catalog[idx]['mag'],
                                     'catflux': self.action.args.f0 * 10**(-self.action.args.catalog[idx]['mag']/2.512), # phot/sec
                                     'flux': detected['flux'],
                                     'instmag': -2.512*np.log(detected['flux']),
                                     'flux2': detected['flux2'],
                                     'FWHM': detected['FWHM'],
                                     } )
        self.action.args.associated = associated if len(associated) > 0 else None
        self.log.info(f'  Associated {len(associated)} catalogs stars')

        fitted_line = sigma_clipping_line_fit(associated['catflux'], associated['flux2'], intercept_fixed=True)

#         from astropy.modeling import models, fitting
#         fit = fitting.LinearLSQFitter()
#         line_init = models.Linear1D(slope=1, intercept=0)
#         line_init.intercept.fixed = True
#         fitted_line = fit(line_init, associated['catflux'], associated['flux2'])

        self.log.info(f"  Slope (e-/photon) = {fitted_line.slope.value:.3g}")
        self.action.args.zero_point_fit = fitted_line

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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = not self.action.args.skip
        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = (not self.action.args.skip)\
                         and (not self.cfg['Telescope'].getboolean('norecord', False))\
                         and (self.action.args.images is not None)

        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
            self.log.info('Done')
            print()
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        self.log.info('Done')
        print()

        return some_post_condition

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
            'target name': self.action.args.kd.get('OBJECT'),
            'exptime': float(self.action.args.kd.get('EXPTIME')),
            'date': datetime.strptime(self.action.args.kd.get('DATE-OBS'), '%Y-%m-%dT%H:%M:%S'),
            'filter': self.action.args.kd.get('FILTER'),
            'az': float(self.action.args.kd.get('AZIMUTH')),
            'alt': float(self.action.args.kd.get('ALTITUDE')),
            'airmass': float(self.action.args.kd.get('AIRMASS')),
            'header_RA': self.action.args.header_pointing.ra.deg,
            'header_DEC': self.action.args.header_pointing.dec.deg,
            'moon_alt': self.action.args.moon_alt,
            'moon_separation': self.action.args.moon_separation,
            'moon_illumination': self.action.args.moon_illum,
            'FWHM_pix': self.action.args.fwhm,
            'ellipticity': self.action.args.ellipticity,
            'n_stars': self.action.args.n_objects,
            'analyzed': True,
            'SIDREversion': 'n/a',
            }
        if self.action.args.zero_point_fit is not None:
            self.image_info['zero point'] = self.action.args.zero_point_fit.slope.value
        if self.action.args.sky_background is not None:
            self.image_info['sky background'] = self.action.args.sky_background
        if self.action.args.perr is not None and not np.isnan(self.action.args.perr):
            self.image_info['perr_arcmin'] = self.action.args.perr.to(u.arcmin).value
        if self.action.args.wcs is not None:
            self.image_info['wcs'] = str(self.action.args.wcs.to_header()).strip()
        if self.action.args.jpegfile is not None:
            self.image_info['jpegs'] = [f"{self.action.args.jpegfile.name}"]
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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True
        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

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
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.log = context.pipeline_logger
        self.cfg = self.context.config.instrument

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True
        if some_pre_condition is True:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
        return some_pre_condition

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True
        if some_post_condition is True:
            self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
        else:
            self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
        return some_post_condition

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
#         """
#         Constructor
#         """
#         BasePrimitive.__init__(self, action, context)
#         # to use the pipeline logger instead of the framework logger, use this:
#         self.log = context.pipeline_logger
#         self.cfg = self.context.config.instrument
# 
#     def _pre_condition(self):
#         """Check for conditions necessary to run this process"""
#         some_pre_condition = not self.action.args.skip
#         if some_pre_condition is True:
#             self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
#         else:
#             self.log.warning(f"Precondition for {self.__class__.__name__} failed")
#         return some_pre_condition
# 
#     def _post_condition(self):
#         """Check for conditions necessary to verify that the process run correctly"""
#         some_post_condition = True
#         if some_post_condition is True:
#             self.log.debug(f"Postcondition for {self.__class__.__name__} satisfied")
#         else:
#             self.log.debug(f"Postcondition for {self.__class__.__name__} failed")
#         return some_post_condition
# 
#     def _perform(self):
#         """
#         Returns an Argument() with the parameters that depends on this operation.
#         """
#         self.log.info(f"Running {self.__class__.__name__} action")
# 
#         return self.action.args

