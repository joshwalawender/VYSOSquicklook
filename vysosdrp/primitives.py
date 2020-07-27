from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
plt.style.use('classic')

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import stats
from astropy.time import Time
import astropy.coordinates as c
from astropy.table import Table, Column
import ccdproc
import photutils
import sep
import ephem

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments


##-----------------------------------------------------------------------------
## Convenience Functions
def get_sunrise_sunset(start):
    obs = ephem.Observer()
    obs.lon = "-155:34:33.9"
    obs.lat = "+19:32:09.66"
    obs.elevation = 3400.0
    obs.temp = 10.0
    obs.pressure = 680.0
    obs.date = start.strftime('%Y/%m/%d 10:00:00')

    obs.horizon = '0.0'
    result = {'sunset': obs.previous_setting(ephem.Sun()).datetime(),
              'sunrise': obs.next_rising(ephem.Sun()).datetime(),
             }
    obs.horizon = '-6.0'
    result['evening_civil_twilight'] = obs.previous_setting(ephem.Sun(),
                                           use_center=True).datetime()
    result['morning_civil_twilight'] = obs.next_rising(ephem.Sun(),
                                           use_center=True).datetime()
    obs.horizon = '-12.0'
    result['evening_nautical_twilight'] = obs.previous_setting(ephem.Sun(),
                                              use_center=True).datetime()
    result['morning_nautical_twilight'] = obs.next_rising(ephem.Sun(),
                                              use_center=True).datetime()
    obs.horizon = '-18.0'
    result['evening_astronomical_twilight'] = obs.previous_setting(ephem.Sun(),
                                                  use_center=True).datetime()
    result['morning_astronomical_twilight'] = obs.next_rising(ephem.Sun(),
                                                  use_center=True).datetime()
    return result


def query_mongo(db, collection, query):
    if collection == 'weather':
        names=('date', 'temp', 'clouds', 'wind', 'gust', 'rain', 'safe')
        dtype=(datetime, np.float, np.float, np.float, np.float, np.int, np.bool)
    elif collection == 'V20status':
        names=('date', 'focuser_temperature', 'primary_temperature',
               'secondary_temperature', 'truss_temperature',
               'focuser_position', 'fan_speed',
               'alt', 'az', 'RA', 'DEC', 
              )
        dtype=(datetime, np.float, np.float, np.float, np.float, np.int, np.int,
               np.float, np.float, np.float, np.float)
    elif collection == 'images':
        names=('date', 'telescope', 'moon_separation', 'perr_arcmin',
               'airmass', 'FWHM_pix', 'ellipticity')
        dtype=(datetime, np.str, np.float, np.float, np.float, np.float, np.float)

    result = Table(names=names, dtype=dtype)
    for entry in db[collection].find(query):
        insert = {}
        for name in names:
            if name in entry.keys():
                insert[name] = entry[name]
        result.add_row(insert)
    return result


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
            self.log.info(f"File: {fitsfile}")
        else:
            self.log.info(f"Could not find file: {fitsfile}")
            return False

        # Check if this exists in the database already
        already_processed = [d for d in self.images.find( {'filename': fitsfile.name} )]
        if len(already_processed) == 0:
            self.action.args.skip = False
        else:
            self.log.info('File is already in the database, skipping further processing')
            self.action.args.skip = True
            return None

        # Read FITS file
        self.action.args.kd = fits_reader(fitsfile, datatype=VYSOS20)

        # Read some header info
        self.action.args.obstime = datetime.strptime(self.action.args.kd.get('DATE-OBS'), '%Y-%m-%dT%H:%M:%S')

        rawRA = self.action.args.kd.get('RA')
        rawDEC = self.action.args.kd.get('DEC')
        self.action.args.header_pointing = c.SkyCoord(rawRA, rawDEC, frame='fk5',
                                                      unit=(u.hourangle, u.deg))

        return self.action.args


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

        inst_cfg = self.context.config.instrument['VYSOS20']
        self.lat = c.Latitude(self.action.args.kd.get('SITELAT'), unit=u.degree)
        self.lon = c.Longitude(self.action.args.kd.get('SITELONG'), unit=u.degree)
        self.height = float(self.action.args.kd.get('ALT-OBS')) * u.meter
        self.loc = c.EarthLocation(self.lon, self.lat, self.height)
        tmperature = float(self.action.args.kd.get('AMBTEMP'))*u.Celsius
        press = inst_cfg.getfloat('pressure', 700)*u.mbar
        self.action.args.altazframe = c.AltAz(location=self.loc,
                          obstime=self.action.args.obstime,
                          temperature=tmperature,
                          pressure=press)
        self.action.args.moon = c.get_moon(Time(self.action.args.obstime),
                                           location=self.loc)

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
        some_pre_condition = not self.action.args.skip

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
                      'moon_alt': ((self.action.args.moon.transform_to(self.action.args.altazframe).alt).to(u.degree)).value,
                      'moon_separation': (self.action.args.moon.separation(self.action.args.header_pointing).to(u.degree)).value,
                      'FWHM_pix': np.mean(self.action.args.fwhm),
                      'ellipticity': np.mean(self.action.args.ellipticity),
                      'analyzed': True,
                      'SIDREversion': 'n/a',
                     }
        for key in image_info.keys():
            self.log.debug(f'  {key}: {image_info[key]}')

        # Remove old entries for this image file
        deletion = self.images.delete_many( {'filename': self.action.args.kd.fitsfilename} )
        self.log.info(f'  Deleted {deletion.deleted_count} previous entries for {self.action.args.kd.fitsfilename}')

        # Save new entry for this image file
        self.log.debug('Adding image info to mongo database')
        ## Save document
        try:
            inserted_id = self.images.insert_one(image_info).inserted_id
            self.log.info(f"  Inserted document id: {inserted_id}")
        except:
            e = sys.exc_info()[0]
            self.log.error('Failed to add new document')
            self.log.error(e)
        self.mongoclient.close()

        return self.action.args


class RegeneratePlot(BasePrimitive):
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
        some_pre_condition = not self.action.args.skip

        try:
            import pymongo
            self.log.debug('Connecting to mongo db at 192.168.1.101')
            self.mongoclient = pymongo.MongoClient('192.168.1.101', 27017)
            self.db = self.mongoclient['vysos']
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
        inst_cfg = self.context.config.instrument['VYSOS20']
        pixel_scale = inst_cfg.getfloat('pixel_scale', 1)
        weather_limits = {'Cloudiness (C)': [-30, -20],
                          'Wind (kph)': [20, 70],
                          'Rain': [2400, 2000],
                          }

        date_string = self.action.args.obstime.strftime('%Y%m%dUT')
        hours = HourLocator(byhour=range(24), interval=1)
        hours_fmt = DateFormatter('%H')
        mins = MinuteLocator(range(0,60,15))

        start = datetime.strptime(date_string, '%Y%m%dUT')
        end = start + timedelta(1)
        if end > datetime.utcnow():
            end = datetime.utcnow()

        images = query_mongo(self.db, 'images', {'date': {'$gt':start, '$lt':end}, 'telescope': 'V20' } )
        self.log.info(f"Found {len(images)} image entries")
        status = query_mongo(self.db, 'V20status', {'date': {'$gt':start, '$lt':end} } )
        self.log.info(f"Found {len(status)} status entries")
        weather = query_mongo(self.db, 'weather', {'date': {'$gt':start, '$lt':end} } )
        self.log.info(f"Found {len(weather)} weather entries")

        if len(images) == 0 and len(status) == 0:
            return self.action.args

        night_plot_file_name = f'{date_string}_{self.action.args.kd.instrument}.png'
        destination_path = Path('/var/www/nights/')
        night_plot_file = destination_path.joinpath(night_plot_file_name)
        self.log.debug(f'Generating plot file: {night_plot_file}')

        twilights = get_sunrise_sunset(start)
        plot_start = twilights['sunset'] - timedelta(0, 1.5*60*60)
        plot_end = twilights['sunrise'] + timedelta(0, 1800)

        time_ticks_values = np.arange(twilights['sunset'].hour,twilights['sunrise'].hour+1)
        plot_positions = [ ( [0.000, 0.755, 0.465, 0.245], [0.535, 0.760, 0.465, 0.240] ),
                           ( [0.000, 0.550, 0.465, 0.180], [0.535, 0.495, 0.465, 0.240] ),
                           ( [0.000, 0.490, 0.465, 0.050], [0.535, 0.245, 0.465, 0.240] ),
                           ( [0.000, 0.210, 0.465, 0.250], [0.535, 0.000, 0.465, 0.235] ),
                           ( [0.000, 0.000, 0.465, 0.200], None                         ) ]
        Figure = plt.figure(figsize=(13,9.5), dpi=100)

        ##------------------------------------------------------------------------
        ## Temperatures
        self.log.info('Adding temperature plot')
        t = plt.axes(plot_positions[0][0])
        plt.title(f"Temperatures for V20 on the Night of {date_string}")
        t.plot_date(weather['date'], weather['temp']*9/5+32, 'k-',
                         markersize=2, markeredgewidth=0, drawstyle="default",
                         label="Outside Temp")
        t.plot_date(status['date'], status['focuser_temperature']*9/5+32, 'y-',
                         markersize=2, markeredgewidth=0,
                         label="Focuser Temp")
        t.plot_date(status['date'], status['primary_temperature']*9/5+32, 'r-',
                         markersize=2, markeredgewidth=0,
                         label="Primary Temp")
        t.plot_date(status['date'], status['secondary_temperature']*9/5+32, 'g-',
                         markersize=2, markeredgewidth=0,
                         label="Secondary Temp")
        t.plot_date(status['date'], status['truss_temperature']*9/5+32, 'k-',
                         alpha=0.5,
                         markersize=2, markeredgewidth=0,
                         label="Truss Temp")

        plt.xlim(plot_start, plot_end)
        plt.ylim(28,87)
        t.xaxis.set_major_locator(hours)
        t.xaxis.set_major_formatter(hours_fmt)

        ## Overplot Twilights
        plt.axvspan(twilights['sunset'], twilights['evening_civil_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.1)
        plt.axvspan(twilights['evening_civil_twilight'], twilights['evening_nautical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.2)
        plt.axvspan(twilights['evening_nautical_twilight'], twilights['evening_astronomical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.3)
        plt.axvspan(twilights['evening_astronomical_twilight'], twilights['morning_astronomical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.5)
        plt.axvspan(twilights['morning_astronomical_twilight'], twilights['morning_nautical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.3)
        plt.axvspan(twilights['morning_nautical_twilight'], twilights['morning_civil_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.2)
        plt.axvspan(twilights['morning_civil_twilight'], twilights['sunrise'],
                    ymin=0, ymax=1, color='blue', alpha=0.1)

        plt.legend(loc='best', prop={'size':10})
        plt.ylabel("Temperature (F)")
        plt.grid(which='major', color='k')

        ##------------------------------------------------------------------------
        ## Temperature Differences (V20 Only)
        self.log.info('Adding temperature difference plot')
        d = plt.axes(plot_positions[1][0])

        from scipy import interpolate
        xw = [(x-weather['date'][0]).total_seconds() for x in weather['date']]
        outside = interpolate.interp1d(xw, weather['temp'],
                                       fill_value='extrapolate')
        xs = [(x-status['date'][0]).total_seconds() for x in status['date']]

        pdiff = status['primary_temperature'] - outside(xs)
        d.plot_date(status['date'], 9/5*pdiff, 'r-',
                         markersize=2, markeredgewidth=0,
                         label="Primary")
        sdiff = status['secondary_temperature'] - outside(xs)
        d.plot_date(status['date'], 9/5*sdiff, 'g-',
                         markersize=2, markeredgewidth=0,
                         label="Secondary")
        fdiff = status['focuser_temperature'] - outside(xs)
        d.plot_date(status['date'], 9/5*fdiff, 'y-',
                         markersize=2, markeredgewidth=0,
                         label="Focuser")
        tdiff = status['truss_temperature'] - outside(xs)
        d.plot_date(status['date'], 9/5*tdiff, 'k-', alpha=0.5,
                         markersize=2, markeredgewidth=0,
                         label="Truss")
        d.plot_date(status['date'], [0]*len(status), 'k-')
        plt.xlim(plot_start, plot_end)
        plt.ylim(-7,17)
        d.xaxis.set_major_locator(hours)
        d.xaxis.set_major_formatter(hours_fmt)
        d.xaxis.set_ticklabels([])
        plt.ylabel("Difference (F)")
        plt.grid(which='major', color='k')

        ##------------------------------------------------------------------------
        ## Fan State/Power (V20 Only)
        self.log.info('Adding fan state/power plot')
        f = plt.axes(plot_positions[2][0])
        f.plot_date(status['date'], status['fan_speed'], 'b-', \
                             label="Mirror Fans")
        plt.xlim(plot_start, plot_end)
        plt.ylim(-10,110)
        f.xaxis.set_major_locator(hours)
        f.xaxis.set_major_formatter(hours_fmt)
        f.xaxis.set_ticklabels([])
        plt.yticks(np.linspace(0,100,3,endpoint=True))
        plt.ylabel('Fan (%)')
        plt.grid(which='major', color='k')

        ##------------------------------------------------------------------------
        ## FWHM
        self.log.info('Adding FWHM plot')
        f = plt.axes(plot_positions[3][0])
        plt.title(f"Image Quality for V20 on the Night of {date_string}")

        fwhm = images['FWHM_pix']*u.pix * pixel_scale
        f.plot_date(images['date'], fwhm, 'ko',
                         markersize=3, markeredgewidth=0,
                         label="FWHM")
        plt.xlim(plot_start, plot_end)
        plt.ylim(0,10)
        f.xaxis.set_major_locator(hours)
        f.xaxis.set_major_formatter(hours_fmt)
        f.xaxis.set_ticklabels([])
        plt.ylabel(f"FWHM (arcsec)")
        plt.grid(which='major', color='k')

        ##------------------------------------------------------------------------
        ## ellipticity
        ##------------------------------------------------------------------------
        self.log.info('Adding ellipticity plot')
        e = plt.axes(plot_positions[4][0])
        e.plot_date(images['date'], images['ellipticity'], 'ko',
                         markersize=3, markeredgewidth=0,
                         label="ellipticity")
        plt.xlim(plot_start, plot_end)
        plt.ylim(0.95,1.75)
        e.xaxis.set_major_locator(hours)
        e.xaxis.set_major_formatter(hours_fmt)
        plt.ylabel(f"ellipticity")
        plt.grid(which='major', color='k')

        ##------------------------------------------------------------------------
        ## Cloudiness
        ##------------------------------------------------------------------------
        self.log.info('Adding cloudiness plot')
        c = plt.axes(plot_positions[0][1])
        plt.title(f"Cloudiness")
        wsafe = np.where(weather['clouds'] < weather_limits['Cloudiness (C)'][0])[0]
        wwarn = np.where(np.array(weather['clouds'] >= weather_limits['Cloudiness (C)'][0])\
                         & np.array(weather['clouds'] < weather_limits['Cloudiness (C)'][1]) )[0]
        wunsafe = np.where(weather['clouds'] >= weather_limits['Cloudiness (C)'][1])[0]
        if len(wsafe) > 0:
            c.plot_date(weather['date'][wsafe], weather['clouds'][wsafe], 'go',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")
        if len(wwarn) > 0:
            c.plot_date(weather['date'][wwarn], weather['clouds'][wwarn], 'yo',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")
        if len(wunsafe) > 0:
            c.plot_date(weather['date'][wunsafe], weather['clouds'][wunsafe], 'ro',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")

        plt.xlim(plot_start, plot_end)
        plt.ylim(-55,15)
        c.xaxis.set_major_locator(hours)
        c.xaxis.set_major_formatter(hours_fmt)

        ## Overplot Twilights
        plt.axvspan(twilights['sunset'], twilights['evening_civil_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.1)
        plt.axvspan(twilights['evening_civil_twilight'], twilights['evening_nautical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.2)
        plt.axvspan(twilights['evening_nautical_twilight'], twilights['evening_astronomical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.3)
        plt.axvspan(twilights['evening_astronomical_twilight'], twilights['morning_astronomical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.5)
        plt.axvspan(twilights['morning_astronomical_twilight'], twilights['morning_nautical_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.3)
        plt.axvspan(twilights['morning_nautical_twilight'], twilights['morning_civil_twilight'],
                    ymin=0, ymax=1, color='blue', alpha=0.2)
        plt.axvspan(twilights['morning_civil_twilight'], twilights['sunrise'],
                    ymin=0, ymax=1, color='blue', alpha=0.1)

        plt.ylabel("Cloudiness (C)")
        plt.grid(which='major', color='k')

        ## Overplot Moon Up Time
        obs = ephem.Observer()
        obs.lon = "-155:34:33.9"
        obs.lat = "+19:32:09.66"
        obs.elevation = 3400.0
        obs.temp = 10.0
        obs.pressure = 680.0
        obs.date = start.strftime('%Y/%m/%d 10:00:00')
        TheMoon = ephem.Moon()
        moon_alts = []
        moon_phases = []
        moon_time_list = []
        moon_time = plot_start
        while moon_time <= plot_end:
            obs.date = moon_time
            TheMoon.compute(obs)
            moon_time_list.append(moon_time)
            moon_alts.append(TheMoon.alt * 180. / ephem.pi)
            moon_phases.append(TheMoon.phase)
            moon_time += timedelta(0, 60*5)
        moon_phase = max(moon_phases)
        moon_fill = moon_phase/100.*0.4+0.05

        mc_axes = c.twinx()
        mc_axes.set_ylabel('Moon Alt (%.0f%% full)' % moon_phase, color='y')
        mc_axes.plot_date(moon_time_list, moon_alts, 'y-')
        mc_axes.xaxis.set_major_locator(hours)
        mc_axes.xaxis.set_major_formatter(hours_fmt)
        plt.ylim(0,100)
        plt.yticks([10,30,50,70,90], color='y')
        plt.xlim(plot_start, plot_end)
        plt.fill_between(moon_time_list, 0, moon_alts, where=np.array(moon_alts)>0,
                         color='yellow', alpha=moon_fill)        
        plt.ylabel('')

        ##------------------------------------------------------------------------
        ## Humidity, Wetness, Rain
        self.log.info('Adding rain plot')
        r = plt.axes(plot_positions[1][1])

        wsafe = np.where(weather['rain'] > weather_limits['Rain'][0])[0]
        wwarn = np.where(np.array(weather['rain'] <= weather_limits['Rain'][0])\
                         & np.array(weather['rain'] > weather_limits['Rain'][1]) )[0]
        wunsafe = np.where(weather['rain'] <= weather_limits['Rain'][1])[0]
        if len(wsafe) > 0:
            r.plot_date(weather['date'][wsafe], weather['rain'][wsafe], 'go',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")
        if len(wwarn) > 0:
            r.plot_date(weather['date'][wwarn], weather['rain'][wwarn], 'yo',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")
        if len(wunsafe) > 0:
            r.plot_date(weather['date'][wunsafe], weather['rain'][wunsafe], 'ro',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")

        plt.xlim(plot_start, plot_end)
        plt.ylim(-100,3000)
        r.xaxis.set_major_locator(hours)
        r.xaxis.set_major_formatter(hours_fmt)
        r.xaxis.set_ticklabels([])
        plt.ylabel("Rain")
        plt.grid(which='major', color='k')

        ##------------------------------------------------------------------------
        ## Wind Speed
        self.log.info('Adding wind speed plot')
        w = plt.axes(plot_positions[2][1])

        wsafe = np.where(weather['wind'] < weather_limits['Wind (kph)'][0])[0]
        wwarn = np.where(np.array(weather['wind'] >= weather_limits['Wind (kph)'][0])\
                         & np.array(weather['wind'] < weather_limits['Wind (kph)'][1]) )[0]
        wunsafe = np.where(weather['wind'] >= weather_limits['Wind (kph)'][1])[0]
        if len(wsafe) > 0:
            w.plot_date(weather['date'][wsafe], weather['wind'][wsafe], 'go',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")
        if len(wwarn) > 0:
            w.plot_date(weather['date'][wwarn], weather['wind'][wwarn], 'yo',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")
        if len(wunsafe) > 0:
            w.plot_date(weather['date'][wunsafe], weather['wind'][wunsafe], 'ro',
                             markersize=2, markeredgewidth=0,
                             drawstyle="default")
        windlim_data = list(weather['wind']*1.1)
        windlim_data.append(65) # minimum limit on plot is 65

        plt.xlim(plot_start, plot_end)
        plt.ylim(-2,max(windlim_data))
        w.xaxis.set_major_locator(hours)
        w.xaxis.set_major_formatter(hours_fmt)
        plt.ylabel("Wind (kph)")
        plt.grid(which='major', color='k')

        self.log.info(f'Saving figure: {night_plot_file}')
        plt.savefig(night_plot_file, dpi=100, bbox_inches='tight', pad_inches=0.10)
        self.log.info('Done.')

        return self.action.args









class Template(BasePrimitive):
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
        inst_cfg = self.context.config.instrument['VYSOS20']

        return self.action.args
