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
## Primitive: CreateBackground
##-----------------------------------------------------------------------------
class CreateBackground(BasePrimitive):
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

