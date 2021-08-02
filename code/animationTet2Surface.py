# create character animation on surface from the prediction

import numpy as np
import os
import subprocess

# load txt file of float data
def loadData_Float(filename):
    data = np.loadtxt(filename)
    data = np.array(data)
    return data


def animationTet2Surface(mesh_path_root, eval_path_root, character_name, prefix):
    # step 1: combine all displacements into one text matrix file
    dis = []
    filename = os.path.join(eval_path_root, prefix + "u")
    u = loadData_Float(filename)
    dis.append(u)
    dis = np.array(dis)
    dis = np.transpose(dis)
    row = dis.shape[0]
    column = dis.shape[1]
    np.savetxt(os.path.join(eval_path_root, prefix + "TetDis_txt.dis"), dis)
    # print("Dis txt file saved:", os.path.join(eval_path_root, prefix + "TetDis_txt.dis"), row, column)

    # step 2: then use the textMatrix2Matrix utility to convert the displacement file to veg matrix file
    program='./vega_FEM/utilities/bin/textMatrix2Matrix'
    par= "TetDis_txt.dis TetDis.dis " + str(row) + " " + str(column) + " 1.0)"
    # print(program + " " + par)
    subprocess.call([program, os.path.join(eval_path_root, prefix + "TetDis_txt.dis"), os.path.join(eval_path_root, prefix + "TetDis.dis"), str(row), str(column), str(1.0)],stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # step 3: use the utility interpolateData to convert veg dis to obj dis file
    program='./vega_FEM/utilities/bin/interpolateData'
    par= character_name+".veg " + character_name + ".obj " + "TetDis.dis " + "SurfaceDis.u " + character_name + ".interp"
    # print(program + " " +par)
    subprocess.call([program, os.path.join(mesh_path_root,character_name+".veg"), os.path.join(mesh_path_root,character_name + ".obj"), os.path.join(eval_path_root, prefix + "TetDis.dis"), os.path.join(eval_path_root, prefix + "SurfaceDis.u"), "-i", os.path.join(mesh_path_root,character_name+".interp")],stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)