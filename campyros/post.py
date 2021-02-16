'''
Useful functions for post-processing of the trajectory data, e.g. when imported from a .JSON file.

The trajectory data is usually stored in a minimalistic format, so only contains:
- time
- pos_i
- vel_i
- b2imat
- w_b

So things like the altitude, yaw, pitch, and roll, etc... need to be obtained from only this data.

'''
import numpy as np
from scipy.spatial.transform import Rotation

__copyright__ = """

    Copyright 2021 Jago Strong-Wright & Daniel Gibbons

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

def ypr_i(simulation_output):
    '''
    Get yaw, pitch and roll data (relative to inertial axes).
    
    Inputs
    ------
    simulation_output : pandas DataFrame
    
    Returns
    -------
    yaw : list
    pitch : list
    roll : list
    '''
    output_dict = simulation_output.to_dict(orient="list")
    time = output_dict["time"]
    
    yaw=[]
    pitch=[]
    roll=[]

    for i in range(len(time)):
        ypr = Rotation.from_matrix(output_dict["b2imat"][i]).as_euler("zyx")
        yaw.append(ypr[0])
        pitch.append(ypr[1])
        roll.append(ypr[2])
        
    return yaw, pitch, roll
