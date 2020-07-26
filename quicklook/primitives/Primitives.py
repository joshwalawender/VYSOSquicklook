from pathlib import Path
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import stats
import astropy.coordinates as c
from astropy.table import Table, Column
import ccdproc
import photutils
import sep

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

# MODIFY the name of this class and make sure that this module/file is imported in the pipelines definition file that
# has been created in the pipeline directory


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

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

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
            self.log.info(f"File: {fitsfile}")
        else:
            self.log.info(f"Could not find file: {fitsfile}")
            return False

        # Read FITS file
        self.action.args.kd = fits_reader(fitsfile, datatype=VYSOS20)

        # Read some header info
        rawRA = self.action.args.kd.get('RA')
        rawDEC = self.action.args.kd.get('DEC')
        self.action.args.header_pointing = c.SkyCoord(rawRA, rawDEC, frame='fk5',
                                                      unit=(u.hourangle, u.deg))

        return self.action.args


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

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

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
            gain = self.context.config.instrument['VYSOS20'].getfloat('gain', None)
            self.log.debug(f'Got gain from config: {gain}')
            self.action.args.kd.headers.append(fits.Header( {'GAIN': gain} ))

        for i,pd in enumerate(self.action.args.kd.pixeldata):
            self.log.debug('Gain correcting pixeldata')
            self.action.args.kd.pixeldata[i] = ccdproc.gain_correct(pd, gain,
                                               gain_unit=u.electron/u.adu)

        return self.action.args


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

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

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
            rn = self.context.config.instrument['VYSOS20'].getfloat('RN', None)
            self.log.debug(f'Got read noise from config: {rn}')
            self.action.args.kd.headers.append(fits.Header( {'READNOISE': rn} ))

        for i,pd in enumerate(self.action.args.kd.pixeldata):
            self.log.info('Estimating uncertainty in pixeldata')
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
        some_pre_condition = True

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

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

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
        box_size = self.context.config.instrument['VYSOS20'].getint('background_box_size', 128)

        self.action.args.background = [None] * len(self.action.args.kd.pixeldata)
        for i,pd in enumerate(self.action.args.kd.pixeldata):
            bkg = photutils.Background2D(pd, box_size=box_size,
                                         mask=self.action.args.source_mask[i],
                                         sigma_clip=stats.SigmaClip())
            self.action.args.background[i] = bkg
            self.action.args.kd.pixeldata[i].data -= bkg.background
        return self.action.args


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

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

        if some_pre_condition:
            self.log.debug("Precondition for ExtractStars is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.log.debug("Postcondition for ExtractStars is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.log.info(f"Running {self.__class__.__name__} action")
        inst_cfg = self.context.config.instrument['VYSOS20']
        pixel_scale = inst_cfg.getfloat('pixel_scale', 1)
        thresh = inst_cfg.getint('extract_threshold', 9)
        minarea = inst_cfg.getint('extract_minarea', 7)
        mina = inst_cfg.getint('fwhm_mina', 1)
        minb = inst_cfg.getint('fwhm_minb', 1)

        self.action.args.objects = [None] * len(self.action.args.kd.pixeldata)
        self.action.args.fwhm = [None] * len(self.action.args.kd.pixeldata)
        self.action.args.ellipticity = [None] * len(self.action.args.kd.pixeldata)
        for i,pd in enumerate(self.action.args.kd.pixeldata):
            objects = sep.extract(pd.data, err=pd.uncertainty.array,
                                  mask=pd.mask,
                                  thresh=float(thresh), minarea=minarea)
            t = Table(objects)
            self.log.info(f'  Found {len(t):d} sources in extension {i}')

            ny, nx = pd.shape
            r = np.sqrt((t['x']-nx/2.)**2 + (t['y']-ny/2.)**2)
            t.add_column(Column(data=r.data, name='r', dtype=np.float))

            coef = 2*np.sqrt(2*np.log(2))
            fwhm = np.sqrt((coef*t['a'])**2 + (coef*t['b'])**2)
            t.add_column(Column(data=fwhm.data, name='FWHM', dtype=np.float))

            ellipticities = t['a']/t['b']
            t.add_column(Column(data=ellipticities.data, name='ellipticity', dtype=np.float))

            self.action.args.objects[i] = t

            self.log.info('  Determining typical FWHM')        
            filtered = (t['a'] < mina) | (t['b'] < minb) | (t['flag'] > 0)
            self.log.debug(f'  Removing {np.sum(filtered):d}/{len(filtered):d}'\
                          f' extractions from FWHM calculation')
            FWHM_pix = np.median(t['FWHM'][~filtered])
            ellipticity = np.median(t['ellipticity'][~filtered])
            self.log.info(f'  FWHM = {FWHM_pix:.1f} pix')
            self.log.info(f'  FWHM = {FWHM_pix*pixel_scale:.2f} arcsec')
            self.log.info(f'  ellipticity = {ellipticity:.2f}')

            self.action.args.fwhm[i] = FWHM_pix
            self.action.args.ellipticity[i] = ellipticity

        return self.action.args


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

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

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

        # Comple image info to store
        image_info = {'filename': self.action.args.kd.fitsfilename,
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
                      'FWHM_pix': np.mean(self.action.args.fwhm),
                      'ellipticity': np.mean(self.action.args.ellipticity),
                      'analyzed': True,
                      'SIDREversion': 'n/a',
                     }
        for key in image_info.keys():
            self.log.debug(f'  {key}: {image_info[key]}')

        import pymongo

        self.log.info('Connecting to mongo db at 192.168.1.101')
        try:
            client = pymongo.MongoClient('192.168.1.101', 27017)
            db = client.vysos
            images = db['images']
        except:
            self.log.error('Could not connect to mongo db')
            raise Exception('Failed to connect to mongo')
        else:
            # Remove old entries for this image file
            deletion = images.delete_many( {'filename': self.action.args.kd.fitsfilename} )
            self.log.info(f'  Deleted {deletion.deleted_count} previous entries for {self.action.args.kd.fitsfilename}')

        # Save new entry for this image file
        self.log.debug('Adding image info to mongo database')
        ## Save document
        try:
            inserted_id = images.insert_one(image_info).inserted_id
            self.log.info("  Inserted document id: {inserted_id}")
        except:
            e = sys.exc_info()[0]
            self.log.error('Failed to add new document')
            self.log.error(e)
        client.close()

        return self.action.args

