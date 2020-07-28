from pathlib import Path
from datetime import datetime, timedelta
import pymongo
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
plt.style.use('classic')

import numpy as np
from astropy import units as u
from astropy.table import Table, Column
import ephem


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


def make_nightly_plot(date_string=None, instrument='V20', pixel_scale=0.44,
                      log=None):

        if date_string is None:
            date_string = datetime.utcnow().strftime('%Y%m%dUT')

        weather_limits = {'Cloudiness (C)': [-30, -20],
                          'Wind (kph)': [20, 70],
                          'Rain': [2400, 2000],
                          }

        hours = HourLocator(byhour=range(24), interval=1)
        hours_fmt = DateFormatter('%H')
        mins = MinuteLocator(range(0,60,15))

        start = datetime.strptime(date_string, '%Y%m%dUT')
        end = start + timedelta(1)
        if end > datetime.utcnow():
            end = datetime.utcnow()

        if log: log.debug('Connecting to mongo db at 192.168.1.101')
        mongoclient = pymongo.MongoClient('192.168.1.101', 27017)
        db = mongoclient['vysos']

        images = query_mongo(db, 'images', {'date': {'$gt':start, '$lt':end}, 'telescope': 'V20' } )
        if log: log.info(f"Found {len(images)} image entries")
        status = query_mongo(db, 'V20status', {'date': {'$gt':start, '$lt':end} } )
        if log: log.info(f"Found {len(status)} status entries")
        weather = query_mongo(db, 'weather', {'date': {'$gt':start, '$lt':end} } )
        if log: log.info(f"Found {len(weather)} weather entries")

        if len(images) == 0 and len(status) == 0:
            return

        night_plot_file_name = f'{date_string}_{instrument}.png'
        destination_path = Path('/var/www/nights/')
        night_plot_file = destination_path.joinpath(night_plot_file_name)
        if log: log.debug(f'Generating plot file: {night_plot_file}')

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
        if log: log.info('Adding temperature plot')
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
        if log: log.info('Adding temperature difference plot')
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
        if log: log.info('Adding fan state/power plot')
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
        if log: log.info('Adding FWHM plot')
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
        if log: log.info('Adding ellipticity plot')
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
        if log: log.info('Adding cloudiness plot')
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
        if log: log.info('Adding rain plot')
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
        if log: log.info('Adding wind speed plot')
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

        if log: log.info(f'Saving figure: {night_plot_file}')
        plt.savefig(night_plot_file, dpi=100, bbox_inches='tight', pad_inches=0.10)
        if log: log.info('Done.')


