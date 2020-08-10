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
from astroquery.exceptions import TimeoutError as astrometryTimeout
from astroquery.astrometry_net import AstrometryNet
import ccdproc
import photutils
import sep

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments


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
        self.cfg = self.context.config.instrument
        # initialize to None
        self.action.args.objects = None
        self.action.args.wcs_pointing = None
        self.action.args.perr = np.nan
        self.action.args.wcs = None
        self.action.args.catalog = None

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

        try:
            import pymongo
            self.log.debug('Connecting to mongo db at 192.168.1.101')
            self.mongoclient = pymongo.MongoClient('192.168.1.101', 27017)
            self.images = self.mongoclient.vysos['images']
            self.mongoclient.close()
        except:
            self.log.error('Could not connect to mongo db')
            some_pre_condition = False

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
        fitsfile = Path(self.action.args.name).expanduser()
        if fitsfile.exists():
            self.log.info(f"  File: {fitsfile}")
        else:
            self.log.info(f"  Could not find file: {fitsfile}")
            return False

        # If we are reading a compressed file, use the uncompressed version of
        # the name for the database
        if fitsfile.suffix == '.fz':
            fitsfile_db = '.'.join(fitsfile.name.split('.')[:-1])
        else:
            fitsfile_db = fitsfile.name

        # Check if this exists in the database already
        already_processed = [d for d in self.images.find( {'filename': fitsfile_db} )]
        self.action.args.skip = False
        if len(already_processed) != 0 and self.cfg['VYSOS20'].getboolean('overwrite', False) is False:
            self.log.info('  File is already in the database, skipping further processing')
            self.action.args.skip = True

        # Read FITS file
        self.action.args.kd = fits_reader(fitsfile, datatype=VYSOS20)

        # If we are reading a compressed file, use the uncompressed version of
        # the name for the database
        self.action.args.kd.fitsfile = fitsfile_db
        self.action.args.kd.fitsfilepath = fitsfile

        # Read some header info
        self.action.args.obstime = datetime.strptime(self.action.args.kd.get('DATE-OBS'), '%Y-%m-%dT%H:%M:%S')

        rawRA = self.action.args.kd.get('RA')
        rawDEC = self.action.args.kd.get('DEC')
        self.action.args.header_pointing = c.SkyCoord(rawRA, rawDEC, frame='fk5',
                                                      unit=(u.hourangle, u.deg))

        try:
            self.action.args.wcs = WCS(self.action.args.kd.header[0])
        except:
            self.log.warning('Unable to read WCS from header')
            pass

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

        # Try to determine date string from path to file
        fitsfile = Path(self.action.args.kd.fitsfilepath)
        date_string = fitsfile.parts[-2]
        if not re.match('\d{8}UT', date_string):
            some_pre_condition = False

        # Check if a destination is set in the config file
        
        if self.cfg['VYSOS20'].get('copy_local', None) is None:
            some_pre_condition = False

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        # Try to determine date string from path to file
        fitsfile = Path(self.action.args.kd.fitsfilepath)
        date_string = fitsfile.parts[-2]

        # Look for log file
        logfile = fitsfile.parent.parent.parent / 'Logs' / fitsfile.parts[-2] / f"{fitsfile.stem}.log"

        destinations = self.cfg['VYSOS20'].get('copy_local', None).split(',')
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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        self.lat = c.Latitude(self.action.args.kd.get('SITELAT'), unit=u.degree)
        self.lon = c.Longitude(self.action.args.kd.get('SITELONG'), unit=u.degree)
        self.height = float(self.action.args.kd.get('ALT-OBS')) * u.meter
        self.loc = c.EarthLocation(self.lon, self.lat, self.height)
        tmperature = float(self.action.args.kd.get('AMBTEMP'))*u.Celsius
        press = self.cfg['VYSOS20'].getfloat('pressure', 700)*u.mbar
        self.action.args.altazframe = c.AltAz(location=self.loc,
                          obstime=self.action.args.obstime,
                          temperature=tmperature,
                          pressure=press)
        self.action.args.moon = c.get_moon(Time(self.action.args.obstime),
                                           location=self.loc)

        self.action.args.sun = c.get_sun(Time(self.action.args.obstime))

        self.action.args.moon_alt = ((self.action.args.moon.transform_to(self.action.args.altazframe).alt).to(u.degree)).value
        self.action.args.moon_separation = (self.action.args.moon.separation(self.action.args.header_pointing).to(u.degree)).value

        # Moon illumination formula from Meeus, “Astronomical 
        # Algorithms". Formulae 46.1 and 46.2 in the 1991 edition, 
        # using the approximation cos(psi) \approx -cos(i). Error 
        # should be no more than 0.0014 (p. 316). 
        dec_sun = self.action.args.sun.dec.radian
        dec_moon = self.action.args.moon.dec.radian
        ra_sun = self.action.args.sun.ra.radian
        ra_moon = self.action.args.moon.ra.radian
        self.action.args.moon_illum = 50*(1 - np.sin(dec_sun)*np.sin(dec_moon)\
                                      - np.cos(dec_sun)*np.cos(dec_moon)\
                                      * np.cos(ra_sun-ra_moon))

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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        gain = self.action.args.kd.get('GAIN', None)
        if gain is not None: self.log.debug(f'Got gain from header: {gain}')
        if gain is None:
            gain = self.cfg['VYSOS20'].getfloat('gain', None)
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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

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
            rn = self.cfg['VYSOS20'].getfloat('RN', None)
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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

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
            self.action.args.kd.pixeldata[i].data -= bkg.background
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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
        pixel_scale = self.cfg['VYSOS20'].getfloat('pixel_scale', 1)
        thresh = self.cfg['Extract'].getint('extract_threshold', 9)
        minarea = self.cfg['Extract'].getint('extract_minarea', 7)
        mina = self.cfg['Extract'].getint('fwhm_mina', 1)
        minb = self.cfg['Extract'].getint('fwhm_minb', 1)

        self.action.args.fwhm = np.nan
        self.action.args.ellipticity = np.nan

        pd = self.action.args.kd.pixeldata[0]
        objects = sep.extract(pd.data, err=pd.uncertainty.array,
                              mask=pd.mask,
                              thresh=float(thresh), minarea=minarea)
        t = Table(objects)

        ny, nx = pd.shape
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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
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
                                                    solve_timeout=solve_timeout)
            self.log.debug(f"  done")
        except astrometryTimeout as e:
            self.log.warning('Astrometry solve timed out')
            return self.action.args

        # Determine Pointing
        self.action.args.wcs = WCS(wcs_header)
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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        catalogname = self.cfg['jpeg'].get('catalog')
        catalog = {'UCAC4': 'I/322A', 'Gaia': 'I/345/gaia2'}[catalogname]
        from astroquery.vizier import Vizier
        v = Vizier(columns=['_RAJ2000', '_DEJ2000', 'rmag'],
                   column_filters={"rmag":">0"})
        v.ROW_LIMIT = 1e5

        fp = self.action.args.wcs.calc_footprint(axes=self.action.args.kd.pixeldata[0].data.shape)
        dra = fp[:,0].max() - fp[:,0].min()
        ddec = fp[:,1].max() - fp[:,1].min()
        radius = np.sqrt((dra*np.cos(fp[:,1].mean()*np.pi/180.))**2 + ddec**2)/2.

        if self.action.args.wcs_pointing is not None:
            pointing = self.action.args.wcs_pointing
        else:
            pointing = self.action.args.header_pointing

        self.action.args.catalog = v.query_region(pointing, catalog=catalog,
                                                  radius=c.Angle(radius, "deg"))[0]
        self.log.info(f"Retrieved {len(self.action.args.catalog)} catalog entries")
        self.log.info(self.action.args.catalog.keys())

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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        im = self.action.args.kd.pixeldata[0].data
        binning = self.cfg['jpeg'].getint('binning', 1)
        jpegfilename = f'{self.action.args.kd.fitsfile.split(".")[0]}.jpg'
        vmin = np.percentile(im, self.cfg['jpeg'].getfloat('vmin_percent', 0.5))
        vmax = np.percentile(im, self.cfg['jpeg'].getfloat('vmax_percent', 99))
        dpi = self.cfg['jpeg'].getint('dpi', 72)
        nx, ny = im.shape
        sx = nx/dpi/binning
        sy = ny/dpi/binning
        fig = plt.figure(figsize=(sx, sy), dpi=dpi)
        ax = fig.gca()
        mdata = np.ma.MaskedArray(im)
        palette = plt.cm.gray
        palette.set_bad('r', 1.0)
        plt.imshow(mdata, cmap=palette, vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])

        if self.cfg['jpeg'].getboolean('overplot_extracted', False) is True and self.action.args.objects is not None:
            self.log.info('  Overlaying extracted stars')
            radius = self.cfg['jpeg'].getfloat('extracted_radius', 6)
            for star in self.action.args.objects:
                if star['x'] > 0 and star['x'] < nx and star['y'] > 0 and star['y'] < ny:
                    c = plt.Circle((star['x'], star['y']), radius=radius,
                                   edgecolor='r', facecolor='none')
                    ax.add_artist(c)

        if self.cfg['jpeg'].getboolean('overplot_catalog', False) is True and self.action.args.catalog is not None:
            self.log.info('  Overlaying catalog stars')
            radius = self.cfg['jpeg'].getfloat('catalog_radius', 6)
            x, y = self.action.args.wcs.all_world2pix(self.action.args.catalog['_RAJ2000'],
                                                      self.action.args.catalog['_DEJ2000'], 1)
            for xy in zip(x, y):
                if x > 0 and x < nx and y > 0 and y < ny:
                    c = plt.Circle(xy, radius=radius, edgecolor='b', facecolor='none')
                    ax.add_artist(c)

        if self.cfg['jpeg'].getboolean('overplot_pointing', False) is True\
            and self.action.args.header_pointing is not None\
            and self.action.args.wcs_pointing is not None:
            x, y = self.action.args.wcs.all_world2pix(self.action.args.header_pointing.ra.degree,
                                                      self.action.args.header_pointing.dec.degree, 1)
            plt.plot([0,nx], [ny/2,ny/2], 'y-', alpha=0.7)
            plt.plot([nx/2, nx/2], [0,ny], 'y-', alpha=0.7)
            # Draw crosshair on target
            radius = self.cfg['jpeg'].getfloat('pointing_radius', 6)
            c = plt.Circle((x, y), radius=radius, edgecolor='g', alpha=0.7,
                           facecolor='none')
            ax.add_artist(c)
            plt.plot([x, x], [y+0.6*radius, y+1.4*radius], 'g', alpha=0.7)
            plt.plot([x, x], [y-0.6*radius, y-1.4*radius], 'g', alpha=0.7)
            plt.plot([x-0.6*radius, x-1.4*radius], [y, y], 'g', alpha=0.7)
            plt.plot([x+0.6*radius, x+1.4*radius], [y, y], 'g', alpha=0.7)


        self.action.args.jpegfile = Path('/var/www/plots/V20/') / jpegfilename
        self.log.info(f'  Rendering: {self.action.args.jpegfile}')
        
        plt.savefig(self.action.args.jpegfile, dpi=dpi)

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
        some_pre_condition = (not self.action.args.skip) and (not self.cfg['VYSOS20'].getboolean('norecord', False))

        try:
            import pymongo
            self.log.debug('Connecting to mongo db at 192.168.1.101')
            self.mongoclient = pymongo.MongoClient('192.168.1.101', 27017)
            self.images = self.mongoclient.vysos['images']
        except:
            self.log.error('Could not connect to mongo db')
            some_pre_condition = False

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        # Comple image info to store
        image_info = {'filename': self.action.args.kd.fitsfile,
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
        if self.action.args.perr is not None:
            image_info['perr_arcmin'] = self.action.args.perr.to(u.arcmin).value
        if self.action.args.jpegfile is not None:
            image_info['jpegs'] = [f"{self.action.args.jpegfile.name}"]

        for key in image_info.keys():
            self.log.debug(f'  {key}: {image_info[key]}')

        # Remove old entries for this image file
        deletion = self.images.delete_many( {'filename': self.action.args.kd.fitsfilename} )
        self.log.debug(f'  Deleted {deletion.deleted_count} previous entries for {self.action.args.kd.fitsfilename}')

        # Save new entry for this image file
        self.log.debug('Adding image info to mongo database')
        ## Save document
        try:
            inserted_id = self.images.insert_one(image_info).inserted_id
            self.log.debug(f"  Inserted document id: {inserted_id}")
        except:
            e = sys.exc_info()[0]
            self.log.error('Failed to add new document')
            self.log.error(e)
        self.mongoclient.close()

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

        if some_pre_condition:
            self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            self.log.warning(f"Precondition for {self.__class__.__name__} failed")
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
            return True
        else:
            self.log.warning(f"Postcondition for {self.__class__.__name__} failed")
            return False

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
# 
#         if some_pre_condition:
#             self.log.debug(f"Precondition for {self.__class__.__name__} is satisfied")
#             return True
#         else:
#             return False
# 
#     def _post_condition(self):
#         """Check for conditions necessary to verify that the process run correctly"""
#         some_post_condition = True
# 
#         if some_post_condition:
#             self.log.debug(f"Postcondition for {self.__class__.__name__} is satisfied")
#             return True
#         else:
#             return False
# 
#     def _perform(self):
#         """
#         Returns an Argument() with the parameters that depends on this operation.
#         """
#         self.log.info(f"Running {self.__class__.__name__} action")
# 
#         return self.action.args

