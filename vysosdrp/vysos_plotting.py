from pathlib import Path
from datetime import datetime, timedelta
import pymongo
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
plt.style.use('classic')

import numpy as np
from astropy import units as u
from astropy.table import Table, Column
from astropy.wcs.utils import proj_plane_pixel_scales
import ephem


##-----------------------------------------------------------------------------
## get_sunrise_sunset
##-----------------------------------------------------------------------------
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


##-----------------------------------------------------------------------------
## query_mongo
##-----------------------------------------------------------------------------
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


##-----------------------------------------------------------------------------
## generate_report
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
        pixel_scale = cfg['VYSOS20'].getfloat('pixel_scale', 1)

    fig = plt.figure(figsize=(2*sx, 1*sy), dpi=dpi)

    plotpos = [ [ [0.010, 0.010, 0.550, 0.965], [0.565, 0.775, 0.375, 0.200] ],
                [ None                        , [0.565, 0.540, 0.375, 0.200] ],
                [ None                        , [0.565, 0.265, 0.375, 0.240] ],
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
        avg_fwhm = np.median(objects['FWHM'])*pixel_scale
        fwhm_axes.set_title(f"FWHM = {avg_fwhm:.1f} arcsec")
        nstars, bins, p = fwhm_axes.hist(objects['FWHM']*pixel_scale,
                                         bins=np.arange(1,7,0.25),
                                         color='g', alpha=0.5)
        fwhm_axes.plot([avg_fwhm, avg_fwhm], [0,max(nstars)*1.2], 'r-', alpha=0.5)
        fwhm_axes.set_ylabel('N stars')
        fwhm_axes.set_ylim(0,max(nstars)*1.2)
    if associated is not None:
        nstars, bins, p = fwhm_axes.hist(associated['FWHM']*pixel_scale,
                                         bins=np.arange(1,7,0.25),
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
        fwhmmag_axes.set_ylabel(f"Flux (e-/s) [{max(objects['flux2']):.1f}]")
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
    jpeg_axes.set_title(titlestr)
    reportfilename = f'{fitsfile.split(".")[0]}.jpg'
    reportfile = Path('/var/www/plots/V20/') / reportfilename
    plt.savefig(reportfile, dpi=dpi)
    return reportfile


##-----------------------------------------------------------------------------
## make_nightly_plot
##-----------------------------------------------------------------------------
def make_nightly_plot(date_string=None, log=None, instrument='V20',
                      pixel_scale=0.44):
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
        plotpos = [ ( [0.000, 0.755, 0.465, 0.245], [0.535, 0.760, 0.465, 0.240] ),
                    ( [0.000, 0.550, 0.465, 0.180], [0.535, 0.495, 0.465, 0.240] ),
                    ( [0.000, 0.490, 0.465, 0.050], [0.535, 0.245, 0.465, 0.240] ),
                    ( [0.000, 0.210, 0.465, 0.250], [0.535, 0.000, 0.465, 0.235] ),
                    ( [0.000, 0.000, 0.465, 0.200], None                         ) ]
        Figure = plt.figure(figsize=(13,9.5), dpi=100)

        ##------------------------------------------------------------------------
        ## Temperatures
        if log: log.info('Adding temperature plot')
        t = plt.axes(plotpos[0][0])
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
        d = plt.axes(plotpos[1][0])

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
        f = plt.axes(plotpos[2][0])
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
        f = plt.axes(plotpos[3][0])
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
        e = plt.axes(plotpos[4][0])
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
        c = plt.axes(plotpos[0][1])
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
        r = plt.axes(plotpos[1][1])

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
        w = plt.axes(plotpos[2][1])

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


