from pathlib import Path
from datetime import datetime, timedelta
import sys
import logging
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

from .utils import pre_condition, post_condition


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
        BasePrimitive.__init__(self, action, context)
        self.log = context.pipeline_logger
        self.log.info("")
        self.cfg = self.context.config.instrument
        # initialize values in the args for general use
        self.action.args.db_entry = None
        self.action.args.kd = None
        self.action.args.skip = False
        self.action.args.fitsfilepath = Path(self.action.args.name).expanduser().absolute()
        # initialize values in the args for use with science frames
        self.action.args.background = None
        self.action.args.objects = None
        self.action.args.header_pointing = None
        self.action.args.wcs_pointing = None
        self.action.args.perr = np.nan
        self.action.args.wcs = None
        self.action.args.catalog = None
        self.action.args.fwhm = None
        self.action.args.ellipticity = None
        self.action.args.zero_point = None
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
        checks = [pre_condition(self, 'FITS file exists',
                                self.action.args.fitsfilepath.exists()),
                 ]
        return np.all(checks)

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        checks = [post_condition(self, 'FITS file was read',
                                 self.action.args.kd is not None),
                 ]
        return np.all(checks)

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")

        if self.action.args.images is not None:
            already_processed = [d for d in self.action.args.images.find( {'filename': self.action.args.fitsfile} )]
            if len(already_processed) != 0\
               and self.cfg['Telescope'].getboolean('overwrite', False) is False:
                self.log.info(f"overwrite is {self.cfg['Telescope'].getboolean('overwrite')}")
                self.log.info('  File is already in the database, skipping further processing')
                self.action.args.skip = True

        # Read FITS file
        self.log.info(f'  Reading: {self.action.args.fitsfile}')
        self.action.args.kd = fits_reader(self.action.args.fitsfilepath,
                                          datatype=VYSOS20)

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: PrepareScience
##-----------------------------------------------------------------------------
class PrepareScience(BasePrimitive):
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

        # Extract header pointing
        try:
            self.action.args.header_pointing = c.SkyCoord(self.action.args.kd.get('RA'),
                                                          self.action.args.kd.get('DEC'),
                                                          frame='fk5',
                                                          unit=(u.hourangle, u.deg))
        except:
            self.action.args.header_pointing = None

        # Read previously solved WCS if available
        if self.action.args.images is not None:
            already_processed = [d for d in self.action.args.images.find( {'filename': self.action.args.fitsfile} )]
            if len(already_processed) != 0:
                self.log.info(f'  Found {len(already_processed)} database entries for this file')
                self.action.args.db_entry = already_processed[0]
                if self.action.args.db_entry.get('wcs', None) is None:
                    self.log.info('  Database entry does not contain WCS')
                else:
                    self.action.args.wcs = WCS(self.action.args.db_entry.get('wcs'))
                    self.log.info('  Found Previously Solved WCS')
                    nx, ny = self.action.args.kd.pixeldata[0].data.shape
                    r, d = self.action.args.wcs.all_pix2world([nx/2.], [ny/2.], 1)
                    self.action.args.wcs_pointing = c.SkyCoord(r[0], d[0], frame='fk5',
                                                      equinox='J2000',
                                                      unit=(u.deg, u.deg),
                                                      obstime=self.action.args.kd.obstime())
                    self.action.args.perr = self.action.args.wcs_pointing.separation(
                                                 self.action.args.header_pointing)
                    self.log.info(f'  Pointing error = {self.action.args.perr.to(u.arcmin):.1f}')

        # Find master bias file
        master_directory = self.cfg['Calibrations'].get('DirectoryForMasters', None)
        if master_directory is not None:
            master_directory = Path(master_directory)
        if master_directory.exists() is True:
            # look for bias from this UT date
            date_string = self.action.args.kd.obstime().strftime('%Y%m%dUT')
            bias_file = master_directory.joinpath(f'MasterBias_{date_string}.fits')
            if bias_file.exists() is True:
                self.log.info(f"  Found master bias file: {bias_file.name}")
                self.action.args.master_bias_file = bias_file
            else:
                # Look for bias within 10 days
                count = 0
                while bias_file.exists() is False and count <= 10:
                    count += 1
                    # Days before
                    date_string = (self.action.args.kd.obstime()-timedelta(count)).strftime('%Y%m%dUT')
                    bias_file = master_directory.joinpath(f'MasterBias_{date_string}.fits')
                    if bias_file.exists() is True:
                        self.log.info(f"  Found master bias file: {bias_file.name}")
                        self.action.args.master_bias_file = bias_file
                    # Days after
                    date_string = (self.action.args.kd.obstime()+timedelta(count)).strftime('%Y%m%dUT')
                    bias_file = master_directory.joinpath(f'MasterBias_{date_string}.fits')
                    if bias_file.exists() is True:
                        self.log.info(f"  Found master bias file: {bias_file.name}")
                        self.action.args.master_bias_file = bias_file
                if bias_file.exists() is False:
                    self.log.info(f"  Failed to find master bias file")
                    self.action.args.master_bias_file = None

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
## Primitive: SaveToList
##-----------------------------------------------------------------------------
class SaveToList(BasePrimitive):
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

        if self.action.args.kd.type() == 'BIAS':
            self.context.biases.append(self.action.args.kd)
            if len(self.context.biases) >= self.cfg['Calibrations'].getint('MaxBiasFrames', 9):
                self.context.biases.pop(0)
            self.log.info(f'  This is bias number {len(self.context.biases)}')
        elif self.action.args.kd.type() == 'DARK':
            self.context.darks.append(self.action.args.kd)
            self.log.info(f'  This is dark number {len(self.context.darks)}')
        elif self.action.args.kd.type() == 'FLAT':
            filter = self.action.args.kd.filter()
            if filter in self.context.flats.keys():
                self.context.flats[filter].append(self.action.args.kd)
            else:
                self.context.flats[filter] = [self.action.args.kd]
            self.log.info(f'  This is {filter} flat number {len(self.context.flats[filter])}')

        return self.action.args


##-----------------------------------------------------------------------------
## Primitive: StackBiases
##-----------------------------------------------------------------------------
class StackBiases(BasePrimitive):
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
        self.biases = context.biases

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        checks = [pre_condition(self, 'Have min number of biases',
                                len(self.biases) >= self.cfg['Calibrations'].getint('MinBiasFrames', 3)),
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

        filedate = max([b.obstime() for b in self.biases])
        filedate_str = filedate.strftime('%Y%m%dUT')
        calibrated_biases = [b.pixeldata[0] for b in self.biases]
        combined_bias = ccdproc.combine(calibrated_biases,
                                        method='average',
                                        sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                        sigma_clip_func=np.ma.median, sigma_clip_dev_func=stats.mad_std,
                                       )
        self.log.info(f"  Combined.")
        combined_bias.meta['combined'] = True
        combined_bias.meta['ncomb'] = len(self.biases)
        combined_bias_filename = f'MasterBias_{filedate_str}.fits'
        combined_bias_filepath = Path(self.cfg['Calibrations'].get('DirectoryForMasters'))
        combined_bias_file = combined_bias_filepath.joinpath(combined_bias_filename)
        if combined_bias_file.exists() is True:
            self.log.debug(f"  Deleting existing: {combined_bias_file}")
            combined_bias_file.unlink()
        self.log.info(f"  Saving: {combined_bias_file}")
        combined_bias.write(combined_bias_filepath.joinpath(combined_bias_filename))

        return self.action.args
