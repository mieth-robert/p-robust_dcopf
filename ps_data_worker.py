# Robert Mieth, 2023
# robert.mieth@rutgers.edu

import os
from collections import namedtuple

import pandas as pd
import numpy as np

PWLCost = namedtuple('PWLCost', ['slopes', 'intercepts', 'nums'])

def create_ps_data_from_rts_data(rtsdata):
    busdata = []
    busindexmap = {}
    gendata = []
    genindexmap = {}
    branchdata = []
    branchindexmap = {}
    slackbus = None

    area_load = {}
    for i,bus in enumerate(rtsdata.bus):
        if bus['area'] in area_load.keys():
            area_load[bus['area']] += bus['mw_load']
        else:
            area_load[bus['area']] = bus['mw_load']

    for i,bus in enumerate(rtsdata.bus):
        is_slack = False
        if bus['bus_type'] == 'Ref':
            is_slack = True
            slackbus = i
        newbus = {
            'id': bus['bus_id'],
            'type': bus['bus_type'],
            'area': bus['area'],
            'lat': bus['lat'],
            'lon': bus['lng'],
            'area_load_share': bus['mw_load']/area_load[bus['area']], 
            'is_slack': is_slack,
            'gens': [],
            'branches_out': [],
            'branches_in': [],
            }
        busindexmap[bus['bus_id']] = i
        busdata.append(newbus)
        
    for i,gen in enumerate(rtsdata.gen):
        newgen = {
            'id': gen['gen_uid'],
            'type': gen['unit_type'],
            'fuel': gen['fuel'],
            'bus': busindexmap[gen['bus_id']],
            'min_down_time_hr': gen['min_down_time_hr'],
            'min_up_time_hr': gen['min_up_time_hr'],
            'pmax_mw': gen['pmax_mw'],
            'pmax_pu': gen['pmax_mw']/rtsdata.basemva,
            'pmin_mw': gen['pmin_mw'],
            'pmin_pu': gen['pmin_mw']/rtsdata.basemva,
            'ramp_rate_mw_min': gen['ramp_rate_mw_per_min'],
            'ramp_rate_pu_min': gen['ramp_rate_mw_per_min']/rtsdata.basemva,
            }
        busdata[busindexmap[gen['bus_id']]]['gens'].append(i)
        genindexmap[gen['gen_uid']] = i
        gendata.append(newgen)
        
    for i,branch in enumerate(rtsdata.branch):
        newbranch = {
            'id': branch['uid'],
            'from_bus': busindexmap[branch['from_bus']],
            'to_bus': busindexmap[branch['to_bus']],
            'r_pu': branch['r'],
            'x_pu': branch['x'],
            'b_pu': branch['b'],
            'cap_mw': branch['cont_rating'],
            'cap_pu': branch['cont_rating']/rtsdata.basemva,
            'emergency_cap_mw': branch['ste_rating'],
            'emergency_cap_pu': branch['ste_rating']/rtsdata.basemva,
            }
        busdata[busindexmap[branch['from_bus']]]['branches_out'].append(i)
        busdata[busindexmap[branch['to_bus']]]['branches_in'].append(i)
        branchindexmap[branch['uid']] = i
        branchdata.append(newbranch)
    
    return PSData(busdata, busindexmap, gendata, genindexmap, branchdata, 
                 branchindexmap, rtsdata.basemva, slackbus)



def create_pwlcost_from_rts_data(rtsdata):
    # returns list of piece-wise linear cost functions for each generator
    # cost in $/MW
    gencost = []
    for g in rtsdata.gen:    
            segments = 1
            x1 = np.linspace(g['output_pct_0'], g['output_pct_1'], segments + 1)
            x2 = np.linspace(g['output_pct_1'], g['output_pct_2'], segments + 1)
            x3 = np.linspace(g['output_pct_2'], g['output_pct_3'], segments + 1)

            y1 = (x1 - x1.min()) * g['hr_incr_1'] +  g['hr_avg_0'] * x1.min()
            y2 = (x2 - x2.min()) * g['hr_incr_2'] +  y1.max()
            y3 = (x3 - x3.min()) * g['hr_incr_3'] +  y2.max()

            xs = list(np.concatenate([x1[:1], x1[-1:], x2[-1:], x3[-1:]]))
            ys = list(np.concatenate([y1[:1], y1[-1:], y2[-1:], y3[-1:]]))
            
            xs = [x * g['pmax_mw'] for x in xs]
            ys = [y * g['pmax_mw'] * g['fuel_price_dollar_per_mmbtu'] / 1000 for y in ys]
            # BTU/kWh * (1000kWH/MWh) * MWh * $/MMBTU  * (1MMBTU/100000BTU) = 1/1000

            slopes = []
            intercepts = []
            for s in [1,2,3]:
                dx = xs[s] - xs[s-1] 
                if dx==0:
                    slope = 0
                    intercept = 0
                else: 
                    slope = (ys[s] - ys[s-1]) / dx
                    intercept = ys[s-1] - slope*xs[s-1]
                slopes.append(slope)
                intercepts.append(intercept)
            
            gencost.append(PWLCost(slopes, intercepts, 3))
    return gencost


class RTSDataSet:

    def __init__(self, rtsdir, basemva=100):
        self.rtsdir = rtsdir
        self._read_system_data()
        self._read_timeseries()

        # additional fixed data
        self.basemva = basemva # 100, source: matpower file
        self.csp_h_max = 6 # source: Barrows et al.: "The IEEE Reliability Test System: A Proposed 2019 Update"
        self.csp_sm = 1.6 # source: Barrows et al.: "The IEEE Reliability Test System: A Proposed 2019 Update"
        self.csp_eff = 0.9 # simplified CSP efficiency, see https://www.osti.gov/biblio/1489331 for details

    def _read_system_data(self):
        # reads and sets relevant (static) system data
        print(">>> Reading system data")
        busdf = self._read_buses_from_csv(os.path.join( self.rtsdir, 'SourceData', 'bus.csv'))
        gendf = self._read_generators_from_csv(os.path.join( self.rtsdir, 'SourceData', 'gen.csv'))
        branchdf = self._read_branches_from_csv(os.path.join( self.rtsdir, 'SourceData', 'branch.csv')) 
        self.bus = busdf.to_dict('records')
        self.gen = gendf.to_dict('records')
        self.branch = branchdf.to_dict('records')
    
    def _read_timeseries(self):
        # reads and sets relevant time series data from RTS
        print(">>> Reading time series data")
        self.timeseries = {}
        series_to_read = {
            'load_da_regional':  os.path.join( self.rtsdir, 'timeseries_data_files', 'Load', 'DAY_AHEAD_regional_Load.csv'),
            'load_rt_regional':  os.path.join( self.rtsdir, 'timeseries_data_files', 'Load', 'REAL_TIME_regional_Load.csv'),
            'pv_da': os.path.join( self.rtsdir, 'timeseries_data_files', 'PV', 'DAY_AHEAD_pv.csv'),
            'pv_rt': os.path.join( self.rtsdir, 'timeseries_data_files', 'PV', 'REAL_TIME_pv.csv'),
            'wind_da': os.path.join( self.rtsdir, 'timeseries_data_files', 'WIND', 'DAY_AHEAD_wind.csv'),
            'hydro_da': os.path.join( self.rtsdir, 'timeseries_data_files', 'HYDRO', 'DAY_AHEAD_hydro.csv'),
            'csp_da': os.path.join( self.rtsdir, 'timeseries_data_files', 'CSP', 'DAY_AHEAD_Natural_Inflow.csv'),
            'rtpv_da': os.path.join( self.rtsdir, 'timeseries_data_files', 'RTPV', 'DAY_AHEAD_rtpv.csv'),
        }
        for key, file in series_to_read.items():
            data_df = self.read_rts_timeseries(file)
            self.timeseries[key] = data_df

    def _read_generators_from_csv(self, genfile):
        gendf = pd.read_csv(genfile)
        gendf.columns = [x.lower().replace(" ", "_").replace("$", "dollar").replace("/","_per_") for x in gendf.columns] 
        return gendf

    def _read_buses_from_csv(self, busfile):
        busdf = pd.read_csv(busfile)
        busdf.columns = [x.lower().replace(" ", "_") for x in busdf.columns] 
        return busdf

    def _read_branches_from_csv(self, branchfile): 
        branchdf = pd.read_csv(branchfile)
        branchdf.columns = [x.lower().replace(" ", "_") for x in branchdf.columns] 
        return branchdf
    
    def read_rts_timeseries(self, csv_file):
        # read rts timeseries from csv file
        df = pd.read_csv(csv_file)
        if df['Period'].max() == 24:
            df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day']]) +  pd.to_timedelta(df['Period']-1, unit='h')
        elif df['Period'].max() == 288:
            df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day']]) +  pd.to_timedelta(5*(df['Period']-1), unit='min')
        else:
            print(">>> Neither 5 min nor hourly data")
        return df
    

class PSData: 

    def __init__(self, busdata, busindexmap, gendata, genindexmap, branchdata, 
                 branchindexmap, basemva, slackbus):
        self.basemva = basemva
        self.busdata = busdata # list of dicts with bus data
        self.busindexmap = busindexmap # maps bus index to list index
        self.gendata = gendata # list of dicts with generator data
        self.genindexmap = genindexmap # maps generator index to list index
        self.branchdata = branchdata # list of dicts with branch data
        self.branchindexmap = branchindexmap # maps branch index to list index

        self.slackbus = slackbus 

        self.nbuses = len(busdata) # number of buses
        self.ngens = len(gendata) # number of generators
        self.nbranches = len(branchdata) # number of branches
        
        # self.gen2bus = [gen['bus'] for gen in self.gendata] # bus index of each generator (list index not 'name' index)
        # self.branchch2bus = [(branch['from_bus'], branch['to_bus']) for branch in self.branchdata] # list of tuples (from_bus, to_bus) as list index
        