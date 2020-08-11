from astropy import units as u
from astropy.table import Table
import astropy.coordinates as c
from astroquery.vizier import Vizier

def get_catalog(pointing, radius, catalog='UCAC4'):

    catalogs = {'UCAC4': 'I/322A', 'Gaia': 'I/345/gaia2'}
    if catalog not in catalogs.keys():
        print(f'{catalog} not in {catalogs.keys()}')
        raise NotImplementedError

    v = Vizier(columns=['_RAJ2000', '_DEJ2000', 'rmag'],
           column_filters={"rmag":">0"})
    v.ROW_LIMIT = 1e4

    stars = v.query_region(pointing, catalog=catalogs[catalog],
                           radius=c.Angle(radius, "deg"))[0]
    return Table(stars)


def get_moon_info(lat, lon, height, temperature, pressure, time):
    loc = None