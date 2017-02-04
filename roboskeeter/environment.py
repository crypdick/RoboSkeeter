import numpy as np
import pandas as pd
from scipy.interpolate import Rbf, griddata
from scipy.spatial import cKDTree as kdt

from roboskeeter.io.i_o import get_directory
from roboskeeter.plotting.plot_environment import plot_windtunnel, plot_plume_gradient, draw_bool_plume


class Environment(object):
    def __init__(self, experiment):
        """
        Generate environmental objects

        Parameters
        ----------
        experiment
            (object)
        """
        self.condition = experiment.experiment_conditions['condition']
        self.bounded = experiment.experiment_conditions['bounded']
        self.heat_model = experiment.experiment_conditions['heat_model'].lower()

        if self.condition == 'Control' and self.heat_model != 'none':
            print "{} plume model selected for control condition, but there is no setting instead to no plume.".format(self.heat_model)
            print "TODO: make sure there isn't any Control temp recordings. If not, make Uniform plume"
            self.heat_model = 'none'
            print "TODO: make a uniform temp plume!!"

        self.windtunnel = WindTunnel(self.condition)
        try:
            self.heat = self._load_heat_model()
        except IOError:
            print """IOerror. You are probably missing the temperature data in /data/temperature. The data can be found at
            https://drive.google.com/file/d/0B1CyEg2BqCdjX21yZ0FSVWNEa1E/view?usp=sharing
            Note, the data is encrypted until we publish. If you're a collaborator, email Richard Decal for the password."""
        self.room_temperature = 19.0

    def _load_heat_model(self):
        if self.heat_model == "boolean":
            plume = BooleanPlumeModel(self)
        elif self.heat_model == "timeavg":
            plume = TimeAvgTempModel(self)
        elif self.heat_model == "none":
            plume = NoPlumeModel(self)
        elif self.heat_model == "unaveraged":
            plume = UnaveragedTempsModel(self)
        elif self.heat_model == "uniform-room-temp":
            plume = UniformRoomTemp(self)
        else:
            raise NotImplementedError("no such plume type {}".format(self.heat_model))

        return plume


class WindTunnel:
    def __init__(self, experimental_condition):
        """
        experimental_condition

        """
        self.walls = Walls()
        self.boundary = self.walls.boundary
        self.experimental_condition = experimental_condition

        self.heater_l = Heater("Left", self.experimental_condition)
        self.heater_r = Heater("Right", self.experimental_condition)

    def show(self):
        fig, ax = plot_windtunnel(self)
        return fig, ax


class Walls:
    def __init__(self):
        # these are real dims of our wind tunnel
        self.left = -0.127
        self.right = 0.127
        self.upwind = 1.0
        self.downwind = 0.0
        self.ceiling = 0.254
        self.floor = 0.
        self.boundary = [self.downwind, self.upwind, self.left, self.right, self.floor, self.ceiling]

    def check_in_bounds(self, position):
        xpos, ypos, zpos = position
        inside = True
        past_wall = []

        if xpos > self.upwind:  # beyond upwind(upwind) wall (end)
            inside = False
            past_wall.append('upwind')
        if xpos < self.downwind:  # too far behind
            inside = False
            past_wall.append('downwind')
        if ypos < self.left:  # too left
            inside = False
            past_wall.append('left')
        if ypos > self.right:  # too far right
            inside = False
            past_wall.append('right')
        if zpos > self.ceiling:  # too far above
            inside = False
            past_wall.append('ceiling')
        if zpos < self.floor:  # too far below
            inside = False
            past_wall.append('floor')

        return inside, past_wall


class Heater:
    def __init__(self, side, experimental_condition):
        """
        given side, generate heater

        Parameters
        ----------
        side
            {left, right, none, custom coords}
            Location of the heater
        experimental_condition
            determines whether the heater is on or off
        Returns
        -------
        None
        """
        self.side = side
        self.experimental_condition = experimental_condition

        if side == experimental_condition:
            self.is_on = True
            self.color = 'red'
        else:
            self.is_on = False
            self.color = 'black'

        self.zmin, self.zmax, self.diam, self.x_position, self.y_position = self._set_coordinates()

    def _set_coordinates(self):
        x_coord = 0.864
        zmin = 0.03800
        zmax = 0.11340
        diam = 0.00635

        if self.side in "leftLeftLEFT":
            y_coord = -0.0507
        elif self.side in "rightRightRIGHT":
            y_coord = 0.0507
        elif self.side in 'controlControlCONTROL':
            x_coord, y_coord = None, None
        else:
            raise Exception('invalid location type specified')

        return zmin, zmax, diam, x_coord, y_coord


class PlumeModel(object):
    def __init__(self, environment):
        """
        The plume base class
        Parameters
        ----------
        environment
            (object)

        Returns
        -------
        """
        # useful aliases
        self.environment = environment
        self.condition = environment.condition
        self.walls = environment.windtunnel.walls
        self.heat_model = environment.heat_model

        self.left = self.walls.left
        self.right = self.walls.right
        self.upwind = self.walls.upwind
        self.downwind = self.walls.downwind
        self.ceiling = self.walls.ceiling
        self.floor = self.walls.floor
        self.bounds = [self.downwind, self.upwind, self.left, self.right, self.floor, self.ceiling]


class NoPlumeModel(PlumeModel):
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)

    def check_in_plume_bounds(self, _):
        # always return false
        return False

    def get_nearest_gradient(self, _):
        """if trying to use gradient ascent decision policy with No Plume, return no gradient"""
        return np.array([0., 0., 0.])


class UniformRoomTemp(PlumeModel):
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)
        raise NotImplementedError('TODO make a uniform temperature plume for control simulations')

    def check_in_plume_bounds(self, _):
        # always return false
        return False

    def get_nearest_gradient(self, _):
        """if trying to use gradient ascent decision policy with No Plume, return no gradient"""
        return np.array([0., 0., 0.])


class BooleanPlumeModel(PlumeModel):
    """Are you in the plume Y/N"""
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)

        self.data = self._load_plume_data()

        self.resolution = self._calc_resolution()

    def check_in_plume_bounds(self, position):
        in_windtunnel_bounds, _ = self.walls.check_in_bounds(position)
        x, y, z = position

        x_distances_to_plume_planes = np.abs(self.data['x_position'] - x)

        if x_distances_to_plume_planes.min() > self.resolution:
            # if distance to nearest plume plane is greater than thresh, we are too far upwind or downwind from plume
            # to be inside the plume
            in_plume = False
        elif in_windtunnel_bounds is False:
            # print("WARNING: can't find plumes outside of windtunnel bounds")
            in_plume = False
        else:
            plume_plane = self._get_nearest_plume_plane(x)
            minor_axis = plume_plane.small_radius
            minor_ax_major_ax_ratio = 3
            major_axis = minor_axis * minor_ax_major_ax_ratio

            # check if position is within the elipsoid
            # implementation of http://math.stackexchange.com/a/76463/291217
            value = (((y - plume_plane.y_position) ** 2) / minor_axis ** 2) + \
                    (((z - plume_plane.z_position) ** 2) / major_axis ** 2)

            if value <= 1:
                in_plume = True
            else:
                in_plume = False

        return in_plume

    def show(self):
        fig, ax = plot_windtunnel(self.environment.windtunnel)
        ax.axis('off')
        draw_bool_plume(self, ax=ax)

    def _get_nearest_plume_plane(self, x_position):
        """given x position, find nearest plan"""
        closest_plume_index = (np.abs(self.data['x_position'] - x_position)).argmin()
        plume_plane = self.data.loc[closest_plume_index]

        return plume_plane

    def _load_plume_data(self):
        col_names = ['x_position', 'z_position', 'small_radius']

        if self.condition in 'controlControlCONTROL':
            raise Exception("This block shouldn't ever run.")
        elif self.condition in 'lLleftLeft':
            plume_dir = get_directory('BOOL_LEFT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
            df['y_position'] = self.environment.windtunnel.heater_l.y_position
        elif self.condition in 'rightRight':
            plume_dir = get_directory('BOOL_RIGHT_CSV')
            df = pd.read_csv(plume_dir, names=col_names)
            df['y_position'] = self.environment.windtunnel.heater_r.y_position
        else:
            raise Exception('problem with loading plume data {}'.format(self.condition))

        return df

    def _calc_resolution(self):
        """ use x data to calculate the resolution"""
        try:
            resolution = abs(self.data.x_position.diff()[1])
        except AttributeError:  # if no plume, can't take diff() of no data
            resolution = None

        return resolution

    def get_nearest_gradient(self, _):
        """if trying to use gradient ascent decision policy with Boolean, return no gradient"""
        return np.array([0., 0., 0.])


class TimeAvgTempModel(PlumeModel):
    """time-averaged temperature readings taken inside the windtunnel"""
    # TODO: test TimeAvgPlume
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)

        # number of x, y, z positions to interpolate the data. numbers chosen to reflect the spacing at which the
        # measurements were taken to avoid gradient values of 0 due to undersampling
        # resolution = (100j, 25j, 25j)  # stored as complex numbers for mgrid to work properly
        self.interpolation_resolution = .05  # in meters

        print "loading raw plume data"
        data_list = self._load_plume_data() # returns list. len(list) == 3 if precomputed.

        if len(data_list) == 3:
            print "loading precomputed padded and interpolated data"
            self.raw_data, self.padded_data, self.data = data_list
        elif len(data_list) == 1:
            self.raw_data = data_list[0]
            # print "filling area surrounding measured area with room temperature data"
            # self.padded_data = self._pad_plume_data()
            print "adding sheet of room temp data on outer windtunnel walls"
            self.padded_data = self._room_temp_wall_sheet()

            print "starting interpolation, resolution = {}".format(self.interpolation_resolution)
            self.grid_x, self.grid_y, self.grid_z = self._set_interpolation_coords(self.padded_data)
            print "WARNING: temporarily setting interpolation function by hand inside environment.py! ~line 317"
            interp_func = self._interpolate_data_griddata  # options: self._interpolate_data_RBF self._interpolate_data_griddata
            self.data, self.grid_temp = interp_func()
            # self.data, self.grid_x, self.grid_y, self.grid_z, self.grid_temp = self._interpolate_data_RBF()
            print "calculating gradient"
            self.gradient_x, self.gradient_y, self.gradient_z = self._calc_gradient()

        print "calculating kd-tree"
        self.tree = self._calc_kdtree()

        print """Timeaveraged plume stats:  TODO implement sanity checks
        interp function: {}
        raw data min temp: {}
        raw data max temp: {}
        interpolated min temp: {}
        interpolated max temp: {}
        """.format(interp_func, self.raw_data.avg_temp.min(), self.raw_data.avg_temp.max(),
                   self.data.avg_temp.min(), self.data.avg_temp.max())

        print """Warning: we don't know the plume bounds for the Timeavg plume, so the check_for_plume() method
                always returns False"""

    def check_in_plume_bounds(self, *_):
        """
        we don't know the plume bounds for the Timeavg plume, so the check_in_plume_bounds() method
                always returns False

        Returns
        -------
        in_plume
            always return False.
        """

        return False

    def get_nearest_prediction(self, position):
        """
        Given [x,y,z] return nearest temperature data
        Parameters
        ----------
        position
            [x,y,z]

        Returns
        -------
        temperature
        """

        _, index = self.tree.query(position)
        data = self.data.iloc[index]
        return data

    def get_nearest_gradient(self, position):
        data = self.get_nearest_prediction(position)
        return np.array([data['gradient_x'], data['gradient_y'], data['gradient_z']])

    def show_scatter_data(self, selection = 'raw', temp_thresh=0):
        print "selection={}".format(selection)
        data = self._select_data(selection)
        from roboskeeter.plotting.plot_environment import plot_windtunnel, plot_plume_recordings_scatter
        fig, ax = plot_windtunnel(self.environment.windtunnel)
        plot_plume_recordings_scatter(data, ax, temp_thresh)
        fig.show()

    def show(self):
        import roboskeeter.plotting.plot_environment_mayavi as pemavi
        from roboskeeter.plotting.plot_environment import plot_windtunnel, plot_plume_recordings_scatter
        fig, ax = plot_windtunnel(self.environment.windtunnel)
        plot_plume_recordings_scatter(self.data, ax)
        pemavi.plot_plume_recordings_volume(self.bounds, self.grid_x, self.grid_y, self.grid_z, self.grid_temp)
        fig.show()

    def show_gradient(self):
        import roboskeeter.plotting.plot_environment_mayavi as pemavi
        #pemavi.plot_plume_3d_quiver(self.gradient_x, self.gradient_y, self.gradient_z, self.bounds)

    def plot_gradient(self, thresh=0):
        from roboskeeter.plotting.plot_environment import plot_windtunnel, plot_plume_gradient
        fig, ax = plot_windtunnel(self.environment.windtunnel)
        plot_plume_gradient(self, ax, thresh)
        # fig.show()

    def calc_euclidean_distance_neighbords(self, selection='interpolated'):
        data = self._select_data(selection)
        kdtree = self._calc_kdtree(selection)

        coords = data[['x', 'y', 'z']]

        dist_neighbors = np.zeros(len(coords))
        for i in range(len(coords)):
            coord = coords.iloc[i]
            dists, _ = kdtree.query(coord, k=2, p=2)  # euclidean dist, select 2 nearest neighbords
            dist_neighbors[i] = dists[-1]  # select second entry

        return dist_neighbors.mean()

    def _load_plume_data(self):
        """

        Returns
        -------
        list of dataframes

        if precomputed files exist, will return [raw data, padded data, interpolated data]
        else, will return [raw data]
        """
        col_names = ['x', 'y', 'z', 'avg_temp']

        if self.condition in 'controlControlCONTROL':
            raise Exception("We shouldn't ever run this, unless we ever decide to precompute control temps")
        elif self.condition in 'lLleftLeft':
            plume_dir = get_directory('THERMOCOUPLE_TIMEAVG_LEFT_CSV')
            raw = pd.read_csv(plume_dir, names=col_names)
            raw = raw.dropna()

            # check for pre-computed padded files
            try:
                padded_f = get_directory('THERMOCOUPLE_TIMEAVG_LEFT_PADDED_CSV')
                padded_df = pd.read_csv(padded_f)

                interpolated_f = get_directory('THERMOCOUPLE_TIMEAVG_LEFT_INTERPOLATED_CSV')
                interpolated_df = pd.read_csv(interpolated_f)

                return [raw, padded_df, interpolated_df]
            except IOError:  # files doesn't exist
                print "did not find pre-computed padded temps"
                return [raw]

        elif self.condition in 'rightRight':
            plume_dir = get_directory('THERMOCOUPLE_TIMEAVG_RIGHT_CSV')
            raw = pd.read_csv(plume_dir, names=col_names)
            raw = raw.dropna()

            # check for pre-computed padded files
            try:
                padded_f = get_directory('THERMOCOUPLE_TIMEAVG_RIGHT_PADDED_CSV')
                padded_df = pd.read_csv(padded_f)

                interpolated_f = get_directory('THERMOCOUPLE_TIMEAVG_RIGHT_INTERPOLATED_CSV')
                interpolated_df = pd.read_csv(interpolated_f)

                return [raw, padded_df, interpolated_df]
            except IOError:  # files doesn't exist
                print "did not find pre-computed padded temps"
                return [raw]

        else:
            raise Exception('No such condition for loading plume data: {}'.format(self.condition))

    def _room_temp_wall_sheet(self):
        # generates a plane of room temp data points immediately outside of the wintunnel
        wall_thickness = 0.03  # make sure this value is >= resolution in the _make_uniform_data_grid()
        data_xmin = self.downwind - wall_thickness
        data_xmax = self.upwind + wall_thickness
        data_ymin = self.left - wall_thickness
        data_ymax = self.right + wall_thickness
        data_zmin = self.floor - wall_thickness
        data_zmax = self.ceiling + wall_thickness


        df_list = [self.raw_data]  # start with raw data

        # make a sheet of room temp data for each wall
        df_list.append(self._make_uniform_data_grid(data_xmin, self.downwind, self.left, self.right, self.floor, self.ceiling))
        df_list.append(self._make_uniform_data_grid(self.upwind, data_xmax, self.left, self.right, self.floor, self.ceiling))

        df_list.append(self._make_uniform_data_grid(self.downwind, self.upwind, data_ymin, self.left, self.floor, self.ceiling))
        df_list.append(self._make_uniform_data_grid(self.downwind, self.upwind, self.right, data_ymax, self.floor, self.ceiling))

        df_list.append(self._make_uniform_data_grid(self.downwind, self.upwind, self.left, self.right, data_zmin, self.floor))
        df_list.append(self._make_uniform_data_grid(self.downwind, self.upwind, self.left, self.right, self.ceiling, data_zmax))

        return pd.concat(df_list)

    def _pad_plume_data(self):
        """
        We are assuming that far away from the plume envelope the air will be room temperature. We are padding the
        recorded area with room temperature data points

        Appends the padded data to the raw data
        """

        padding_distance = 0.03  # start padding 3 cm away from recorded data

        data_xmin = self.raw_data.x.min() - padding_distance
        data_xmax = self.raw_data.x.max() + padding_distance
        data_ymin = self.raw_data.y.min() - padding_distance
        data_ymax = self.raw_data.y.max() + padding_distance
        data_zmin = self.raw_data.z.min() - padding_distance
        data_zmax = self.raw_data.z.max() + padding_distance


        df_list = [self.raw_data]  # append to raw data
        # make grids of room temp data to fill the volume surrounding the place we took measurements
        df_list.append(self._make_uniform_data_grid(self.downwind, data_xmin, self.left, self.right, self.floor, self.ceiling))
        df_list.append(self._make_uniform_data_grid(data_xmax, self.upwind, self.left, self.right, self.floor, self.ceiling))

        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, self.left, data_ymin, self.floor, self.ceiling))
        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, data_ymax, self.right, self.floor, self.ceiling))

        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, data_ymin, data_ymax, self.floor, data_zmin))
        df_list.append(self._make_uniform_data_grid(data_xmin, data_xmax, data_ymin, data_ymax, data_zmax, self.ceiling))

        return pd.concat(df_list)

    def _make_uniform_data_grid(self, xmin, xmax, ymin, ymax, zmin, zmax, temp=19., res=0.03):
        """given temp and resolution, fills a volume with a uniform temp grid"""

        # res is resolution in meters
        # left grid
        x = np.arange(xmin, xmax, res)
        y = np.arange(ymin, ymax, res)
        z = np.arange(zmin, zmax, res)
        # we save this grid b/c it helps us with the gradient func
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        df_dict = dict()
        df_dict['x'] = xx.ravel()
        df_dict['y'] = yy.ravel()
        df_dict['z'] = zz.ravel()
        df_dict['avg_temp'] = np.array([temp] * len(x) * len(y) * len(z))

        df = pd.DataFrame(data=df_dict)

        return df

    def _set_interpolation_coords(self, data):
        """generate the coords our interpolator will evaluate at"""
        # TODO: run this on a computer with lots of memory and save CSV so you don't run into memory errors (200, 60, 60)
        xi = np.arange(self.downwind, self.upwind, self.interpolation_resolution)
        yi = np.arange(self.left, self.right, self.interpolation_resolution)
        zi = np.arange(self.floor, self.ceiling, self.interpolation_resolution)

        grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')

        return grid_x, grid_y, grid_z

    def _interpolate_data_griddata(self):
        data = self.padded_data

        interpolated_temp_grid = griddata(data[['x', 'y', 'z']].values, data['avg_temp'].values, (self.grid_x, self.grid_y, self.grid_z), method='linear')

        # save to df
        df_dict = dict()
        df_dict['x'] = self.grid_x.ravel()
        df_dict['y'] = self.grid_y.ravel()
        df_dict['z'] = self.grid_z.ravel()
        df_dict['avg_temp'] = interpolated_temp_grid.ravel()
        interpolated_temps_df = pd.DataFrame(df_dict)

        return interpolated_temps_df, interpolated_temp_grid

    def _interpolate_data_RBF(self):
        """
        Replace data with a higher resolution interpolation
        Parameters
        ----------
        data
        resolution

        Returns
        -------
        interpolated_temps, (grid_x, grid_y, grid_z, grid_temps)
        """
        # TODO: review this function
        if self.condition in 'controlControlCONTROL':
            raise Exception("This code block shouldn't be running if Control is selected. \
                            use uniform temp plume class instead")

        data = self.padded_data

        # useful aliases
        x, y, z, temps = data.x.values, data.y.values, data.z.values, data.avg_temp.values

        # calculate average 3D euclidean distance b/w observations
        avg_distance = self.calc_euclidean_distance_neighbords(selection = 'raw')


        # init rbf interpolator
        """smoothing was determined by testing various numbers and looking at the minimum and maximum of the resulting plumes
        if I put values too far from this, the minimum and maximum temperature start to become extremely unnaturalistic.
        """
        smoothing = 2e-5  # TODO: we can disable smoothing by getting rid of duplicate positions
        self.rbfi = Rbf(x, y, z, temps, function='quintic', smooth=smoothing, epsilon=avg_distance)

        # interpolate at those locations
        grid_x_flat = self.grid_x.ravel()
        grid_y_flat = self.grid_y.ravel()
        grid_z_flat = self.grid_z.ravel()
        interp_temps = self.rbfi(grid_x_flat, grid_y_flat, grid_z_flat)

        # save to df
        df_dict = dict()
        df_dict['x'] = grid_x_flat
        df_dict['y'] = grid_y_flat
        df_dict['z'] = grid_z_flat
        df_dict['avg_temp'] = interp_temps
        interpolated_temps_df = pd.DataFrame(df_dict)

        # we save this grid b/c it helps us with the gradient func
        grid_temps = interp_temps.reshape(self.grid_x.shape)  # all the grid_* have the same shape

        return interpolated_temps_df, grid_temps

    def _calc_gradient(self):
        # impossible to do gradient with unevenly spaced  samples, see https://stackoverflow.com/questions/36781698/numpy-sample-distances-for-3d-gradient
        # so doing instead on regular grid
        # TODO: review this gradient function
        print "HIGH PRIORITY: audit the _calc_gradient function!"

        if self.condition in 'controlControlCONTROL':
            return None

        # grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')
        #grid_temps = interp_temps.reshape((len(xi), len(yi), len(zi)))

        # Solve for the spatial
        distances = [np.diff(self.data.x.unique())[0], np.diff(self.data.y.unique())[0], np.diff(self.data.z.unique())[0]]
        gradient_x, gradient_y, gradient_z = np.gradient(self.grid_temp, *distances)

        self.data['gradient_x'] = gradient_x.ravel()
        self.data['gradient_y'] = gradient_y.ravel()
        self.data['gradient_z'] = gradient_z.ravel()

        self.data.fillna(0, inplace=True)  # replace NaNs, infs before calculating norm
        self.data.replace([np.inf, -np.inf], 0, inplace=True)

        self.data['gradient_norm'] = np.linalg.norm(self.data[['gradient_x', 'gradient_y', 'gradient_z']], axis=1)

        return gradient_x, gradient_y, gradient_z

    def _calc_kdtree(self, selection = 'interpolated'):
        if self.condition in 'controlControlCONTROL':  # TODO: review this
            return None

        data = self._select_data(selection)

        zdata = zip(data.x, data.y, data.z)
        return kdt(zdata)

    def _select_data(self, selection):
        if selection == 'raw':
            data = self.raw_data
        elif selection == 'padded':
            data = self.padded_data
        elif selection == 'interpolated':
            data = self.data
        else:
            raise ValueError

        return data


class UnaveragedTempsModel:
    def __init__(self, environment):
        super(self.__class__, self).__init__(environment)
        raise NotImplementedError  # TODO: implement unaveraged plume