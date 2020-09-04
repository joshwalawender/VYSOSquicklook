from pathlib import Path
from datetime import datetime, timedelta
import logging
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


##-----------------------------------------------------------------------------
## find_master
##-----------------------------------------------------------------------------
def build_master_file_name(kd, master_type, date_string):
    if master_type.lower() in ['bias', 'zero']:
        master_file_name = f"MasterBias_{date_string}.fits"
    elif master_type.lower() in ['dark']:
        exptime = int(kd.exptime())
        master_file_name = f"MasterDark_{exptime:03d}s_{date_string}.fits"
    elif master_type.lower() in ['flat']:
        master_file_name = f"MasterFlat_{kd.filter()}_{date_string}.fits"
    else:
        master_file_name = None
    return master_file_name


def find_master(master_directory, master_type, kd):
    # Find master bias file
    if master_directory is not None:
        master_directory = Path(master_directory)
    else:
        return None
    if master_directory.exists() is False:
        return None

    # Build expected file name
    date_string = kd.obstime().strftime('%Y%m%dUT')
    master_file_name = build_master_file_name(kd, master_type, date_string)
    master_file = master_directory.joinpath(master_file_name)

    # Go hunting for the files
    if master_file.exists() is True:
        return master_file
    else:
        # Look for bias within 10 days
        count = 0
        while master_file.exists() is False and count <= 10:
            count += 1
            # Days before
            date_string = (kd.obstime()-timedelta(count)).strftime('%Y%m%dUT')
            master_file_name = build_master_file_name(kd, master_type, date_string)
            master_file = master_directory.joinpath(master_file_name)
            if master_file.exists() is True:
                return master_file
            # Days after
            date_string = (kd.obstime()+timedelta(count)).strftime('%Y%m%dUT')
            master_file_name = build_master_file_name(kd, master_type, date_string)
            master_file = master_directory.joinpath(master_file_name)
            if master_file.exists() is True:
                return master_file
        if master_file.exists() is False:
            return None
        return master_file


##-----------------------------------------------------------------------------
## Evaluate pre and post conditions
##-----------------------------------------------------------------------------
def pre_condition(primitive, name, condition,
                  fail_level=logging.DEBUG,
                  success_level=logging.DEBUG):
    if condition is True:
        primitive.log.log(success_level,
            f'Precondition for {primitive.__class__.__name__} "{name}" satisfied')
    else:
        primitive.log.log(fail_level,
            f'Precondition for {primitive.__class__.__name__} "{name}" failed')
    return condition


def post_condition(primitive, name, condition,
                   fail_level=logging.WARNING,
                   success_level=logging.DEBUG):
    if condition is True:
        primitive.log.log(success_level,
            f'Postcondition for {primitive.__class__.__name__} "{name}" satisfied')
    else:
        primitive.log.log(fail_level,
            f'Postcondition for {primitive.__class__.__name__} "{name}" failed')
    return condition


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
        self.log.debug('  Adding image info to mongo database')
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

