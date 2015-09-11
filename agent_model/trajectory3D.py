# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:15:28 2015

@author: richard

creates a trajectory object that has a bunch of sweet methods.
takes vectors from agent and adds it to ensemble

trajectory.ensemble
trajectory.append_ensemble(a trajectory)

trajectory.describe()
    agent info
    plots

trajectory.save
"""
import pandas as pd
import plotting_funcs3D
import numpy as np
import score


#def T_find_stats(t_targfinds):
#    """Target finding stats. Returns average time to find the target, and the
#    number of successes
#    
#    TODO:  find tfind stats for ensemble
#    """
#    t_targfinds = np.array(t_targfinds)
#    t_finds_NoNaNs = t_targfinds[~np.isnan(t_targfinds)]  # remove NaNs
#    if len(t_finds_NoNaNs) == 0:
#
#        return np.nan  # no target finds, no stats to run
#    else:
#        num_success = float(len(t_finds_NoNaNs))
#        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)
#
#        return Tfind_avg


class Trajectory():
    def __init__(self):
        self.ensemble = pd.DataFrame()

        
    def load_ensemble(self, data):
        if type(data) is dict:
            trajectory = pd.DataFrame(data)
            self.ensemble.append(trajectory)
        elif type(data) is list:
            self.ensemble = pd.concat(data)
        else:
            raise Exception

    def add_agent_info(self, agent_obj):
        self.agent_obj = agent_obj

    def crunch_analysis(self):
        calculate_curvature()
        calculate_heading()


    def plot_single_trajectory(self, trajectory_i=0):
        plot_kwargs = {'title':"Individual agent trajectory", 'titleappend':' (id = {})'.format(trajectory_i)}
        plotting_funcs3D.plot_single_trajectory(self.ensemble.loc[self.ensemble['trajectory_num']==trajectory_i], self.agent_obj, plot_kwargs)

            
            
    def plot_force_violin(self):
        plotting_funcs3D.force_violin(self.ensemble, self.agent_obj)
        
    
    def plot_posheatmap(self):
        plotting_funcs3D.heatmaps(self.ensemble, self.agent_obj)
        
    
    def plot_kinematic_hists(self, ensemble='none', titleappend=''):
        if type(ensemble) is str:
            ensemble = self.ensemble
            ensemble = ensemble.loc[(ensemble['position_x'] >0.25) & (ensemble['position_x'] <0.95)]
        plotting_funcs3D.stateHistograms(ensemble, self.agent_obj, titleappend=titleappend)
        
        
    def plot_door_velocity_compass(self, region='door', kind='avg_mag_per_bin'):
        
        if region == 'door':
            """plot the area """
            ensemble = self.ensemble.loc[((self.ensemble['position_x']>0.25) & (self.ensemble['position_x']<0.5)), ['velocity_x', 'velocity_y']]
        else:
            ensemble = self.ensemble
            
        plotting_funcs3D.velo_compass_histogram(ensemble, self.agent_obj, kind)
    

    def plot_kinematic_compass(self, kind='avg_mag_per_bin', data=None, flags='', title_append=''):
        if data is None: # TODO: wtf?
            data = self.ensemble
        # TODO: add heading angle plot
        for vector_name in self.agent_obj['forces']+self.agent_obj['kinematic_vals']:
            # from enemble, selec mag and theta
            df = eval("data[['"+vector_name+"_xy_mag', '"+vector_name+"_xy_theta']]")
            # rename col to play nice w plotting function
            df.rename(columns={vector_name+'_xy_mag': 'magnitude', vector_name+'_xy_theta': 'angle'}, inplace=True)
            
            title = "{flag} Avg. {name} vector magnitude in xy plane (n = {N}) {titeappend}".format(\
            N = self.agent_obj['total_trajectories'], name=vector_name,\
            flag=flags, titeappend=title_append)
            fname = "{flag} Avg mag distribution of {name}_xy compass _beta{beta}_bF{biasF_scale}_wf{wtf}_N{total_trajectories}".format(\
            beta=self.agent_obj['beta'], biasF_scale=self.agent_obj['randomF_strength'],
            wtf=self.agent_obj['wtF'], total_trajectories=self.agent_obj['total_trajectories'],
            name=vector_name, flag=flags)
  
            
            plotting_funcs3D.compass_histogram(vector_name, df, self.agent_obj, title=title, fname=fname)
#            magnitudes, thetas = getattr(self.ensemble, name+).values, getattr(V, name+'_xy_theta').values
#            plotting_funcs3D.compass_histogram(force, magnitudes, thetas, self.agent_obj)
       
    def plot_plume_triggered_compass(self, kind='avg_mag_per_bin'):
        behaviors = ['searching', 'entering', 'staying', 'Left_plume, exit left',
            'Left_plume, exit right', "Right_plume, exit left", "Right_plume, exit right"]
        for behavior in behaviors:
            ensemble = self.ensemble.loc[self.ensemble.plume_experience == behavior]
            if ensemble.empty is True:
                print "no data for ", behavior
                pass
            else:
                self.plot_kinematic_compass(data=ensemble, flags='Plume Triggered {}'.format(behavior))

    def plot_heading_compass(self):
        pass # TODO
        
        
    def plot_single_3Dtrajectory(self, trajectory_i=0):
        plot_kwargs = {'title':"Individual agent trajectory", 'titleappend':' (id = {})'.format(trajectory_i)}
        plotting_funcs3D.plot3D_trajectory(self.ensemble.loc[self.ensemble['trajectory_num']==trajectory_i], self.agent_obj, plot_kwargs)
      
      
        
        
    def plot_sliced_hists(self):
        """Plot histograms from 0.25 < x < 0.95, as well as that same space
        divided into 4 equal 0.15 m segments 
        """
        x_edges = np.linspace(.25,.95,5)
        
        print "full ensemble"
        full_ensemble = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[0]) \
            & (self.ensemble['position_x'] < x_edges[4])]
        self.plot_kinematic_hists(ensemble=full_ensemble, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[4]))
        
        print "downwind half ensemble"
        downwind_ensemble = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[0]) \
            & (self.ensemble['position_x'] < x_edges[2])]
        self.plot_kinematic_hists(ensemble=downwind_ensemble, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[2]))

        print "upwind half ensemble"
        upwind_ensemble = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[2]) \
            & (self.ensemble['position_x'] < x_edges[4])]
        self.plot_kinematic_hists(ensemble=upwind_ensemble, titleappend=' {} < x <= {}'.format(x_edges[2], x_edges[4]))    
#        
#        print "first quarter ensemble"
#        ensemble1 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[0]) \
#            & (self.ensemble['position_x'] <= x_edges[1])]
#        self.plot_kinematic_hists(ensemble=ensemble1, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[1]))
#        
#        print "second quarter ensemble"
#        ensemble2 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[1]) \
#            & (self.ensemble['position_x'] <= x_edges[2])]
#        self.plot_kinematic_hists(ensemble=ensemble2, titleappend=' {} < x <= {}'.format(x_edges[1], x_edges[2]))
#        
#        print "third quarter ensemble"
#        ensemble3 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[2]) \
#            & (self.ensemble['position_x'] <= x_edges[3])]
#        self.plot_kinematic_hists(ensemble=ensemble3, titleappend=' {} < x <= {}'.format(x_edges[2], x_edges[3]))
#
#        print "fourth quarter ensemble"
#        ensemble4 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[3]) \
#            & (self.ensemble['position_x'] <= x_edges[4])]
#        self.plot_kinematic_hists(ensemble=ensemble4, titleappend=' {} < x <= {}'.format(x_edges[3], x_edges[4]))
#        
        print "full- + up- + down-wind"
        plotting_funcs3D.stateHistograms(full_ensemble, self.agent_obj, titleappend = '', upw_ensemble = upwind_ensemble, downw_ensemble = downwind_ensemble)


    def plot_3D_kinematic_vecs(self, kinematic='acceleration'):
        plotting_funcs3D.plot_3D_kinematic_vecs(self.ensemble, kinematic)


    def plume_stats(self):
        """ 
        in plume == 1, out == 0. therefore sum/n is % in plume
        """
        in_out = self.ensemble.inPlume.values
        print "Total timesteps: {size}. Total time in plume: {vecsum}. Ratio:"\
            "{ratio}".format(size=in_out.size, vecsum=in_out.sum(), ratio=in_out.sum()/in_out.size)
            
            
    def describe(
            self,
            plot_kwargs={
                'trajectories':False,
                'heatmap':True,
                'states':True,
                'singletrajectories':False,
                'force_violin':True}):
        print self.ensemble.describe()
        if plot_kwargs['heatmap'] is True:
            self.plot_posheatmap()
        if plot_kwargs['singletrajectories'] is True:
            self.plot_single_trajectory()
        if plot_kwargs['states'] is True:
    #         plot histogram of pos, velo, accel distributions
            self.plot_kinematic_hists()
        if plot_kwargs['force_violin'] is True:
            self.plot_force_violin()
            
            
    def dump2csvs(self):
        for trajectory_i in self.ensemble.trajectory_num.unique():
            temp_traj = self.ensemble[self.ensemble['trajectory_num'] == trajectory_i]
            temp_array = temp_traj[['position_x', 'position_y']].values
            np.savetxt(str(trajectory_i) + ".csv", temp_array, delimiter=",")

    def visualize_forces(self, Npoints = 1000):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        # TODO: export plotting funcs to plotting funcs script
        # visualize direction selection
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        data = np.zeros((Npoints, 3))
        for point in range(Npoints):
            data[point] = math_sorcery.gen_symm_vecs(3)

        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        ax.scatter(x, y, z, c='b', marker='.', alpha=0.2)
        ax.scatter(0,0,0, c='r', marker='o', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Visualization of F_random direction selection")
        plt.savefig('F_rand_direction_selection.png')
        plt.show()

        ##############################################################################################
        #F_total
        ##############################################################################################

        fig = plt.figure(3)
        ax = fig.add_subplot(111, projection='3d')
        x = self.ensemble.totalF_x
        y = self.ensemble.totalF_y
        z = self.ensemble.totalF_z

        ax.scatter(x, y, z, c='b', marker='.', alpha=0.2)
        ax.scatter(0,0,0, c='r', marker='o', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Visualization of F_total")
        plt.savefig('F_total_cloud.png')
        plt.show()

        ##############################################################################################
        # F_stim
        ##############################################################################################


        fig = plt.figure(4)
        ax = fig.add_subplot(111, projection='3d')
        x = self.ensemble.stimF_x
        y = self.ensemble.stimF_y
        z = self.ensemble.stimF_z

        ax.scatter(x, y, z, c='b', marker='.', alpha=0.2)
        ax.scatter(0,0,0, c='r', marker='o', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Visualization of F_stim")
        plt.savefig('F_stim_cloud.png')
        plt.show()

        ##############################################################################################
        # F_upwind
        ##############################################################################################


        fig = plt.figure(5)
        ax = fig.add_subplot(111, projection='3d')
        x = self.ensemble.upwindF_x
        y = self.ensemble.upwindF_y
        z = self.ensemble.upwindF_z

        ax.scatter(x[:20], y[:20], z[:20], c='b', marker='.', alpha=0.2)
        ax.scatter(0,0,0, c='r', marker='o', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Visualization of F_upwind")
        plt.savefig('F_upwind_cloud.png')
        plt.show()

        ##############################################################################################
        # F_random
        ##############################################################################################

        fig = plt.figure(6)
        ax = fig.add_subplot(111, projection='3d')
        x = self.ensemble.biasF_x
        y = self.ensemble.biasF_y
        z = self.ensemble.biasF_z

        ax.scatter(x, y, z, c='b', marker='.', alpha=0.2)
        ax.scatter(0,0,0, c='r', marker='o', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Visualization of F_random")
        plt.savefig('F_random_cloud.png')
        plt.show()


        ##############################################################################################
        #F_wall repulsion
        ##############################################################################################

        fig = plt.figure(7)
        ax = fig.add_subplot(111, projection='3d')
        x = self.ensemble.wallRepulsiveF_x
        y = self.ensemble.wallRepulsiveF_y
        z = self.ensemble.wallRepulsiveF_z

        ax.scatter(x, y, z, c='b', marker='.', alpha=0.2)
        ax.scatter(0,0,0, c='r', marker='o', s=50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Visualization of F_wall repulsion")
        plt.savefig('F_wall repulsion_cloud.png')
        plt.show()
    
    def calc_score(self):
       return score.score(self.ensemble)