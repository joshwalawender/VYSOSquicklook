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

        if self.action.args.flats is not True and self.action.args.cals is not True:
            self.log.info(f'  Changing UT date, so clearing lists of flats and cals')
            self.context.biases = list()
            self.context.darks = list()
            self.context.flats = dict()

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

