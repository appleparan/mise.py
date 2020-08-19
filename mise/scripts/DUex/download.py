import cdsapi
import calendar

c = cdsapi.Client()

years = range(2018, 2020)
months = range(1, 13)

#years = [2017]
#months = range(11, 13)

for y in years:
    for m in months:
        days = [str(d).zfill(2) for d in range(1, calendar.monthrange(y, m)[1] + 1)]
        fname = 'DUex-' + str(y) + '-' + str(m).zfill(2)
        print(fname)
        req = {
                'format': 'tgz',
                'variable': 'dust_aerosol_optical_depth',
                'sensor_on_satellite': 'iasi_on_metopa',
                'algorithm': 'imars',
                'year': str(y),
                'month': str(m).zfill(2),
                'day': list(days),
                'orbit': 'ascending',
                'time_aggregation': 'daily_average',
                'version': 'v6.0',
        }
 
        try:
            c.retrieve(
                'satellite-aerosol-properties',
                req,
                fname + '.tar.gz')
        except cdsclient.exceptions.DataProviderFailed:
            # retry
            print("Retry...")
            c.retrieve(
                'satellite-aerosol-properties',
                req,
                fname + '.tar.gz')


