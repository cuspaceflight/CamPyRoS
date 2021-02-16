import iris
import scipy
import scipy.interpolate
import warnings
import os
import numpy as np
import requests
import numexpr as ne
import metpy.calc
from metpy.units import units

from .main import validate_lat_long, warning_on_one_line, closest, points


warnings.formatwarning = warning_on_one_line


class Wind:
    """Wind object

    Note
    ----
    Can give the wind vector for any lat long alt in the launch frame.
    Data collected from the NOAA's 0.25 degree 1 hour GFS forcast (https://nomads.ncep.noaa.gov/)

    Parameters
    ----------
    initial_lat : float
        Initial latitude /degrees
    initial_long : float
        Initial longitude /degrees
    data_loc : string, optional
        Route to folder where the data will be stored, defaults to data/wind/gfs
    variable : bool, optional
        Vary the wind or just use defaut for whole flight, defaults to True
    default : numpy array, optional
        Default wind vector [wind_x,wind_y,wind_z]/m/s, defauts to [0,0,0]
    run_date : string, optional
        Date for forcast data in format YYYYMMDD, defaults to current date
    forcast_time : string, optional
        Forcast run time, must be 00,06,12 or 18, defaults to 00
    forcast_plus_time : string, optional
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?), defaults to 000

    Attributes
    ----------
    centre_lat : float
        Initial latitude /degrees
    centre_long : float
        Initial longitude /degrees
    data_loc : string
        Route to folder where the data will be stored
    variable : bool
        Vary the wind or just use defaut for whole flight
    default : numpy array
        Default wind vector [wind_x,wind_y,wind_z]/m/s
    points : list
        List of available [latitude,longitude] points available
    date : string
        Date for forcast data in format YYYYMMDD
    forcast_time : string
        Forcast run time, must be 00,06,12 or 18
    run_time : string
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?)
    df : pandas DataFrame
        Dataframe holding wind data with columns lat, long, alt, wind x, wind y
    """

    def __init__(
        self,
        initial_long,
        initial_lat,
        variable=True,
        default=np.array([0, 0, 0]),
        data_loc="data/wind/gfs",
        run_date=date.today().strftime("%Y%m%d"),
        forcast_time="00",
        forcast_plus_time="000",
        fast=False,
    ):
        lat, long = validate_lat_long(initial_lat, initial_long)
        self.centre_lat = lat
        self.centre_long = long
        self.data_loc = data_loc  # must be in last week for now
        self.variable = variable
        self.default = default
        self.points = []
        self.fast = fast

        if variable == True:
            if lat < 2:
                warnings.warn(
                    "Wind data robustness has not yet been tested for the equator"
                )
            if abs(lat) > 87:
                warnings.warn(
                    "Wind data robustness has not yet been tested near the poles"
                )
            if forcast_time not in ["00", "06", "12", "18"]:
                warnings.warn(
                    "The forcast run selected is not valid, must be '00', '06', '12' or '18'. This will be fatal on file load"
                )
            valid_times = (
                ["00%s" % n for n in range(0, 10)]
                + ["0%s" % n for n in range(10, 100)]
                + ["%s" % n for n in range(100, 385)]
            )
            if forcast_plus_time not in valid_times:
                warnings.warn(
                    "The forcast time selected is not valid, must be three digit string time between 000 and 384. Thi siwll be fatal on file load"
                )
            self.date = run_date
            self.forcast_time = forcast_time
            self.run_time = forcast_plus_time
            self.df, self.points = self.load_data(
                closest(self.centre_lat, 0.25), closest(self.centre_long, 0.25)
            )

            if self.fast == True:
                self.winds = self.load_fast(lat, long)

    def load_fast(self, lat, long):
        """Returns an interpolation object of wind by altitude for the specified location
        This method is much faster than the normal method but if the rocket has significant downrange
        it becomes inaccurate

        Args:
            lat (float): latitude
            long (float): longitude

        Returns:
            scipy.interpolate.interpolate.interp1d: interpolation class of wind vector by altitude
        """
        mean = []
        lats = closest(lat, 0.25)
        longs = closest(long, 0.25)
        x = []
        y = []
        for n in [0, 1]:
            for m in [0, 1]:
                x.append(
                    scipy.interpolate.interp1d(
                        self.df.query("lat==%s" % lats[n]).query("long==%s" % longs[m])[
                            "alt"
                        ],
                        self.df.query("lat==%s" % lats[n]).query("long==%s" % longs[m])[
                            "w_x"
                        ],
                        fill_value="extrapolate",
                    )
                )
                y.append(
                    scipy.interpolate.interp1d(
                        self.df.query("lat==%s" % lats[n]).query("long==%s" % longs[m])[
                            "alt"
                        ],
                        self.df.query("lat==%s" % lats[n]).query("long==%s" % longs[m])[
                            "w_y"
                        ],
                        fill_value="extrapolate",
                    )
                )

        mean_x = []
        mean_y = []
        for alt in np.linspace(0, 45000, 1000):
            mean_x.append(np.mean([x[0](alt), x[1](alt), x[2](alt), x[3](alt)]))
            mean_y.append(np.mean([y[0](alt), y[1](alt), y[2](alt), y[3](alt)]))
        mean = np.array([-np.array(mean_y), np.array(mean_x), np.zeros(len(mean_x))])

        return scipy.interpolate.interp1d(
            np.linspace(0, 45000, 1000), mean, fill_value="extrapolate"
        )

    def load_data(self, lats, longs):
        """Loads wind data for particualr lat long to the objects df.

        Notes
        -----
        Checks if the file corespondin to the requested lat long at the time and date of the object is available.
        If not downloads. Then reads into the dataframe.
        The file has cubes for geopotential height, wind x and wind y by pressure at a square grid of lat longs.
        The wind x and y are itterated throgh the pressures for each lat long and the altitude found for each point
        by finding the geopotential height at the particular pressure which can be converted to altitude.
        Each point is then stored in the df separatly for ease of searching (because the library Iris is complelty inept for this).

        Parameters
        ----------
        lat : float:
            Requested latitude /degrees
        longi : float:
            Requested longitude /degrees
        Returns
        -------
        pandas DataFrame
            The new wind points
        points
            list of [lat,long] not available
        """
        lat_top = max(lats)
        lat_bottom = min(lats)
        long_left = min(longs)
        long_right = max(longs)

        lat_top, long_left = validate_lat_long(lat_top, long_left)
        lat_bottom, long_right = validate_lat_long(lat_bottom, long_right)

        if not os.path.isfile(
            "%s/%s_%s_%s_%s_%s.grb2"
            % (
                self.data_loc,
                lat_bottom,
                long_left,
                self.date,
                self.forcast_time,
                self.run_time,
            )
        ):
            # This does download 3 rows that aren't needed but I can't work out how to yeet them
            print("Downloading files")
            if long_left > long_right:
                long_left_request = long_left - 360
            else:
                long_left_request = long_left

            url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t{run}z.pgrb2.0p25.f{hour}&lev_0.4_mb=on&lev_1000_mb=on&lev_100_mb=on&lev_10_mb=on&lev_150_mb=on&lev_15_mb=on&lev_180-0_mb_above_ground=on&lev_1_mb=on&lev_200_mb=on&lev_20_mb=on&lev_250_mb=on&lev_255-0_mb_above_ground=on&lev_2_mb=on&lev_300_mb=on&lev_30-0_mb_above_ground=on&lev_30_mb=on&lev_350_mb=on&lev_3_mb=on&lev_400_mb=on&lev_40_mb=on&lev_450_mb=on&lev_500_mb=on&lev_50_mb=on&lev_550_mb=on&lev_5_mb=on&lev_600_mb=on&lev_650_mb=on&lev_700_mb=on&lev_70_mb=on&lev_750_mb=on&lev_7_mb=on&lev_800_mb=on&lev_850_mb=on&lev_900_mb=on&lev_925_mb=on&lev_950_mb=on&lev_975_mb=on&var_HGT=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon={leftlon}&rightlon={rightlon}&toplat={toplat}&bottomlat={bottomlat}&dir=%2Fgfs.{date}%2F{run}".format(
                leftlon=long_left_request,
                rightlon=long_right,
                toplat=lat_top,
                bottomlat=lat_bottom,
                date=self.date,
                run=self.forcast_time,
                hour=self.run_time,
            )
            r = requests.get(url, stream=True)
            with open(
                "%s/%s_%s_%s_%s_%s.grb2"
                % (
                    self.data_loc,
                    lat_bottom,
                    long_left,
                    self.date,
                    self.forcast_time,
                    self.run_time,
                ),
                "wb",
            ) as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        if (
            os.path.getsize(
                "%s/%s_%s_%s_%s_%s.grb2"
                % (
                    self.data_loc,
                    lat_bottom,
                    long_left,
                    self.date,
                    self.forcast_time,
                    self.run_time,
                )
            )
            < 1000
        ):
            raise RuntimeError(
                "The weather data you requested was not found, this is usually because it was for an invalid date/time. lat=%s,long=%s was requested"
                % (lat_bottom, long_left)
            )
        data = iris.load(
            "%s/%s_%s_%s_%s_%s.grb2"
            % (
                self.data_loc,
                lat_bottom,
                long_left,
                self.date,
                self.forcast_time,
                self.run_time,
            )
        )
        for index, row in enumerate(data):
            try:
                row.coord("pressure")
                if row.standard_name == "x_wind":
                    row_x_wind = index
                elif row.standard_name == "y_wind":
                    row_y_wind = index
                elif row.standard_name == "geopotential_height":
                    row_geo = index
            except:
                pass
        lats = list(data[row_geo].coord("latitude").points)
        longs = list(data[row_geo].coord("longitude").points)
        df = pd.DataFrame(columns=["lat", "long", "alt", "w_x", "w_y"])
        points = []
        for long in longs:
            for lat in lats:
                press_1 = (
                    data[row_x_wind]
                    .extract(iris.Constraint(latitude=lat, longitude=long))
                    .coord("pressure")
                    .points
                )
                press_2 = (
                    data[row_y_wind]
                    .extract(iris.Constraint(latitude=lat, longitude=long))
                    .coord("pressure")
                    .points
                )
                press_3 = (
                    data[row_geo]
                    .extract(iris.Constraint(latitude=lat, longitude=long))
                    .coord("pressure")
                    .points
                )
                press = []
                for pres in press_1:
                    if pres in press_2 and pres in press_3:
                        press.append(pres)
                for pres in press:
                    try:
                        if [lat, long] not in self.points or [lat, long] not in points:
                            w_x = (
                                data[row_x_wind]
                                .extract(
                                    iris.Constraint(
                                        latitude=lat, longitude=long, pressure=pres
                                    )
                                )
                                .data
                            )
                            w_y = (
                                data[row_y_wind]
                                .extract(
                                    iris.Constraint(
                                        latitude=lat, longitude=long, pressure=pres
                                    )
                                )
                                .data
                            )
                            alt = (
                                10
                                * metpy.calc.geopotential_to_height(
                                    data[row_geo]
                                    .extract(
                                        iris.Constraint(
                                            latitude=lat, longitude=long, pressure=pres
                                        )
                                    )
                                    .data
                                    * units.m ** 2
                                    / units.s ** 2
                                ).magnitude
                            )
                            row = {
                                "lat": lat,
                                "long": np.mod(long, 360),
                                "alt": alt,
                                "w_x": w_x,
                                "w_y": w_y,
                            }
                            df = df.append(row, ignore_index=True)
                    except KeyError:
                        warnings.warn(
                            "Wind datapoint lat=%s, long=%s, pres=%s was missed because of an unknown Iris error, this is non fatal as it will be interpolated from other values"
                            % (lat, long, pres)
                        )
                    except:
                        warnings.warn(
                            "Wind datapoint lat=%s, long=%s, pres=%s was missed because of an unknown Iris error, this may be a fatal result if there are many instances in one dataset"
                            % (lat, long, pres)
                        )
                # Add a lookup check here (i.e. query for lat long and check not none)
                points.append([lat, long])
        return df, points

    def get_wind(self, lat, long, alt):
        """Returns wind for a specific lat,long,alt

        Parameters
        ----------
        lat : float:
            Requested latitude /degrees
        longi : float:
            Requested longitude /degrees
        alt : float:
            Requested altitude /m
        Returns
        -------
        numpy array
            Wind speed vector [x,y,z]/m/s
        """
        lat, long = validate_lat_long(lat, long)
        if self.variable == True and self.fast == False and 0 < alt < 80000:
            lats = closest(lat, 0.25)
            longs = closest(long, 0.25)
            if not all(point in self.points for point in points(lats, longs)):
                new_df, new_points = self.load_data(lats, longs)
                self.df += new_df
                self.points += new_points
                # would self.df,self.points+=self.load_data(lats,longs) be valid?

            search_lats = self.df.lat.values
            # This search method was approx an order of magnitude faster in my testing

            m = self.df[ne.evaluate("search_lats==%s" % lats[0])]
            search_longs = m.long.values
            row = m[ne.evaluate("search_longs==%s" % longs[0])]
            a = scipy.interpolate.interp1d(
                row["alt"],
                np.array([-row["w_y"], row["w_x"], np.zeros(len(row["w_y"]))]),
                fill_value="extrapolate",
            )(alt)

            m = self.df[ne.evaluate("search_lats==%s" % lats[0])]
            search_longs = m.long.values
            row = m[ne.evaluate("search_longs==%s" % longs[1])]
            b = scipy.interpolate.interp1d(
                row["alt"],
                np.array([-row["w_y"], row["w_x"], np.zeros(len(row["w_y"]))]),
                fill_value="extrapolate",
            )(alt)
            y_0 = a + (long - longs[0]) * (b - a) / (longs[1] - longs[0])

            m = self.df[ne.evaluate("search_lats==%s" % lats[1])]
            search_longs = m.long.values
            row = m[ne.evaluate("search_longs==%s" % longs[0])]
            a = scipy.interpolate.interp1d(
                row["alt"],
                np.array([-row["w_y"], row["w_x"], np.zeros(len(row["w_y"]))]),
                fill_value="extrapolate",
            )(alt)

            m = self.df[ne.evaluate("search_lats==%s" % lats[1])]
            search_longs = m.long.values
            row = m[ne.evaluate("search_longs==%s" % longs[1])]
            b = scipy.interpolate.interp1d(
                row["alt"],
                np.array([-row["w_y"], row["w_x"], np.zeros(len(row["w_y"]))]),
                fill_value="extrapolate",
            )(alt)
            y_1 = a + (long - longs[0]) * (b - a) / (longs[1] - longs[0])

            return y_0 + (lat - lats[0]) * (y_1 - y_0) / (lats[1] - lats[0])
        elif self.variable == True and self.fast == True:
            return self.winds(alt)
        else:
            return self.default
