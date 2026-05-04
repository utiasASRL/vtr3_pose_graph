import matplotlib.pyplot as plt
import numpy as np
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
import pylgmath.so3.operations as so3op
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import numpy as np
import os
from matplotlib.patches import Patch
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path


# ========================================== Define Baselines ==========================================
LTR_dome = { # recaptured version
    "1": [0.039, 0.058, 0.062, 0.077, 0.029],
    "2": [0.041, 0.051, 0.061, 0.048, 0.05],
    "3": [0.059, 0.06, 0.069, 0.063, 0.068],
    "4": [0.016, 0.011, -0.025, 0.006, -0.015],
    "5": [0.078, 0.069, 0.064, 0.075, 0.056],
    "6": [-0.055, -0.037, -0.047, -0.036, -0.043],
    "7": [0.009, 0.016, 0.014, 0.002, 0.015],
    "8": [0.016, 0.025, 0.009, 0.014, 0.026],
    "9": [0.023, 0.015, 0.009, 0.008, 0.004],
    "10": [-0.027, -0.024, -0.037, -0.012, -0.021]
}
LTR_parking = {
    "1": [0.057, 0.06, 0.063, 0.032, 0.019],
    "2": [-0.013, -0.016, 0, -0.01, -0.003],
    "3": [0.079, 0.102, 0.076, 0.114, 0.083],
    "4": [-0.041, -0.044, -0.038, -0.032, -0.025],
    "5": [-0.079, -0.104, -0.079, -0.085, -0.083],
    "6": [0, 0.016, 0.019, 0.003, -0.006],
    "7": [-0.019, 0.01, 0.006, 0.019, 0.006],
    "8": [0, 0.003, 0.002, 0.006, 0.013],
    "9": [0.013, 0.016, 0.029, 0.019, 0.019],
    "10": [0.025, 0.025, 0.032, 0.044, 0.034]
}
LTR_bigpath = {
    "1": [0.013, 0.038, 0.025, 0.029, 0.035],
    "2": [-0.083, -0.146, -0.152, -0.108, -0.092],  
    "3": [0.013, 0.038, 0.025, 0.025, 0.022],
    "4": [0, -0.01, 0, 0, 0.016],
    "5": [0, -0.035, 0.013, 0.006, -0.022],
    "6": [0.013, 0, 0.019, 0.006, 0.01],
    "7": [0.029, 0.01, 0.019, 0.032, -0.029],
    "8": [0, 0.003, 0.019, 0.006, 0],
    "9": [0.029, 0.032, 0.022, 0.054, 0.063],
    "10": [0.006, -0.006, -0.013, -0.013, 0],
    "11": [0.016, 0, -0.006, -0.01, -0.013],
    "12": [0.041, 0.025, 0.029, 0.032, 0.029],
    "13": [0.124, 0.089, 0.121, 0.095, 0.098],
    "14": [0, 0.019, 0.006, 0.01, 0.003]
}
RTR_dome = {
    "1": [0.019, -0.006, -0.054, 0, -0.025],
    "2": [0.048, 0.035, 0.102, 0.051, -0.019],
    "3": [0.029, 0, 0, 0.035, 0.022],
    "4": [0.083, 0.102, 0.025, 0.07, 0.108],
    "5": [-0.077, -0.083, -0.095, -0.102, -0.098],
    "6": [-0.102, 0.052, -0.054, -0.048, -0.048],
    "7": [-0.067, -0.013, -0.051, -0.032, -0.054],
    "8": [-0.019, -0.029, -0.07, -0.01, -0.022],
    "9": [-0.019, 0, 0.006, 0.003, 0],
    "10": [-0.098, -0.133, -0.121, -0.076, -0.102]
}
RTR_parking = {
    "1": [0, 0.06, 0.021, 0.054, 0.041],
    "2": [-0.0325, -0.03, -0.043, -0.038, -0.028],
    "3": [0.08, 0.066, 0.085, 0.103, 0.073],
    "4": [-0.034, -0.023, 0, -0.0075, 0.03],
    "5": [-0.128, -0.19, -0.137, -0.2, -0.172],
    "6": [0.0075, 0, -0.015, -0.003, -0.037],
    "7": [0.054, 0.032, 0.027, 0.013, 0.035],
    "8": [-0.045, -0.075, -0.027, -0.042, -0.055],
    "9": [-0.016, 0.043, 0.017, 0.015, 0.06],
    "10": [0.042, 0.023, 0.021, 0.011, 0.032]
}
RTR_bigpath = {
    "1": [0.035, -0.007, -0.066, 0.061, -0.03],
    "2": [-0.03, -0.013, 0.038, -0.02, -0.023],
    "3": [-0.033, -0.079, -0.03, -0.111, -0.109],
    "4": [0.063, 0.068, 0.035, 0.022, 0.032],
    "5": [-0.022, -0.052, 0.055, 0.016, -0.071],
    "6": [0.067, 0.053, 0.119, 0.137, 0.153],
    "7": [-0.053, -0.037, -0.042, -0.068, -0.065],
    "8": [0.011, 0.055, 0.023, -0.014, -0.028],
    "9": [0.081, 0.105, 0.073, 0.005, 0.046],
    "10": [-0.046, -0.034, -0.02, -0.0075, 0.025],
    "11": [-0.012, -0.035, -0.022, -0.017, -0.005],
    "12": [0.027, 0.052, 0.06, 0.053, 0.02],
    "13": [0.035, -0.003, 0.039, 0.078, 0],
    "14": [-0.015, -0.06, -0.036, -0.195, -0.219]
}

# ========================================== Experimental Methods ==========================================
# ============ VirLTR ============
Pix4D_VirLTR_dome = {
    "1": [0.010475, 0.046475, 0.059475, 0.036475, 0.050475],
    "2": [-0.010125, 0.000875, 0.021875, 0.003875, 0.008875],
    "3": [0.012475, 0.014475, 0.010475, 0.015475, 0.007475],
    "4": [-0.076925, -0.081925, -0.098925, -0.085925, -0.086925],
    "5": [0.065275, 0.074275, 0.086275, 0.068275, 0.059275],
    "6": [-0.122125, -0.097125, -0.096125, -0.111125, -0.092125],
    "7": [0.039875, 0.038875, 0.039875, 0.055875, 0.049875],
    "8": [0.024675, 0.034675, 0.012675, 0.018675, 0.040675],
    "9": [-0.006525, -0.003525, -0.002525, 0.002475, -0.007525],
    "10": [0.027475, 0.033475, 0.019475, 0.015475, 0.007475]
}
Pix4D_VirLTR_parking = {
    "1": [0.073275, 0.041275, 0.024275, 0.012275, 0.029275],
    "2": [0.028475, 0.006475, 0.019475, 0.019475, 0.052475],
    "3": [-0.100525, -0.121475, -0.082525, -0.085525, -0.069525],
    "4": [-0.052725, -0.056725, -0.022725, -0.029725, -0.035725],
    "5": [-0.022925, -0.018925, -0.003925, 0.004075, -0.000925],
    "6": [0.114475, 0.097475, 0.095475, 0.078475, 0.107475],
    "7": [-0.034725, -0.013725, -0.065725, -0.015725, -0.024725],
    "8": [-0.037525, -0.049525, -0.051525, -0.061525, -0.032525],
    "9": [-0.024725, -0.017725, -0.023725, -0.022725, -0.021725],
    "10": [-0.039725, -0.034725, -0.047725, -0.035725, -0.030725]
}
Pix4D_VirLTR_bigpath = {
    "1": [0.018475, 0.043475, 0.040475, 0.042475, -0.015525],
    "2": [0.040275, 0.039275, 0.039275, 0.043275, 0.104275],
    "3": [-0.002925, 0.021075, -0.000925, -0.002925, 0.014075],
    "4": [-0.055525, -0.049525, -0.063525, 0.014475, -0.018525],
    "5": [-0.030925, -0.031925, -0.047925, -0.054925, -0.043925],
    "6": [0.033675, 0.000675, 0.021675, 0.009675, 0.011675],
    "7": [-0.022125, -0.059125, -0.026125, -0.050125, -0.049125],
    "8": [-0.039125, -0.054125, -0.049125, -0.057125, -0.039125],
    "9": [-0.149325, -0.082325, -0.084325, -0.090325, -0.103325],
    "10": [-0.026125, -0.029125, -0.029125, -0.027125, -0.025125],
    "11": [-0.032925, -0.017925, -0.022925, -0.014925, -0.016925],
    "12": [0.013475, -0.009525, -0.004525, 0.009475, 0.005475],
    "13": [-0.109925, -0.117925, -0.127925, -0.115925, -0.129925],
    "14": [-0.012725, -0.018725, -0.030725, -0.026725, -0.016725]
}
# ============ NeRF VirLTR ============
NeRF_VirLTR_dome = {
    "1": [-0.019725, -0.020725, -0.038725, -0.140725, -0.114725],
    "2": [-0.079925, -0.064925, -0.057925, -0.035925, -0.076925],
    "3": [0.031475, 0.000475, 0.035475, 0.009475, 0.031475],
    "4": [-0.012725, -0.010725, -0.014725, -0.024725, -0.028725],
    "5": [-0.108725, -0.093725, -0.072725, -0.087725, -0.092725],
    "6": [-0.015125, -0.022125, -0.052125, -0.054125, -0.024125],
    "7": [-0.061325, -0.054325, -0.048325, -0.040325, -0.049325],
    "8": [-0.001325, -0.049325, -0.033325, -0.011325, -0.022325],
    "9": [-0.050725, -0.049725, -0.061725, -0.054725, -0.070725],
    "10": [0.038075, 0.042075, 0.049075, 0.005075, 0.121075]
}
NeRF_VirLTR_parking = {
    "1": [0.063875, 0.080875, 0.074875, 0.069875, -0.054125],
    "2": [0.091275, 0.099275, 0.103275, 0.045275, 0.071275],
    "3": [-0.080925, -0.125925, -0.135925, -0.101925, -0.101925],
    "4": [-0.086925, -0.091925, -0.083925, -0.097925, -0.086925],
    "5": [-0.045525, -0.057525, -0.027525, -0.014525, -0.032525],
    "6": [-0.069925, -0.068925, -0.101925, -0.079925, -0.062925],
    "7": [-0.032925, -0.024925, -0.026925, -0.021925, -0.022925],
    "8": [0.042475, 0.012475, 0.031475, 0.026475, 0.049475],
    "9": [-0.053325, -0.055325, -0.039325, -0.054325, -0.105325],
    "10": [0.006675, -0.008325, -0.005325, 0.020675, -0.001325]
}
# ============ VirRTR ============
Pix4D_VirRTR_dome = {
    "1": [0.020475, 0.012475, 0.010475, 0.058475, 0.014475],
    "2": [-0.021125, -0.016125, 0.006875, -0.012125, -0.003125],
    "3": [0.099475, 0.045475, 0.062475, 0.103475, 0.072475],
    "4": [-0.015925, -0.048925, -0.083925, -0.069925, -0.166925],
    "5": [0.005275, 0.029275, 0.050275, 0.013275, 0.021275],
    "6": [-0.113125, -0.110125, -0.119125, -0.123125, -0.102125],
    "7": [0.156275, 0.154275, 0.144275, 0.209275, 0.110275],
    "8": [0.237875, 0.219875, 0.206875, 0.208875, 0.194875],
    "9": [0.125475, 0.115475, 0.191475, 0.025475, 0.112475],
    "10":[0.060475, -0.010525, -0.009525, 0.056475, 0.030475]
    
}
Pix4D_VirRTR_parking = {
    "1": [0.059875, 0.048875, 0.015875, 0.030875, 0.035875],
    "2": [-0.007525, 0.013475, 0.005475, -0.028525, 0.009475],
    "3": [-0.016525, 0.002475, -0.101525, -0.152525, -0.157525],
    "4": [-0.051725, 0.149275, -0.022725, 0.100275, 0.006275],
    "5": [-0.208925, 0.031075, -0.052925, -0.006925, -0.061925],
    "6": [-0.075525, -0.063525, -0.172525, -0.143525, -0.123525],
    "7": [0.100275, 0.103275, 0.227275, 0.135275, 0.007275],
    "8": [0.093475, 0.087475, 0.140475, 0.132475, 0.046475],
    "9": [-0.000725, 0.001275, 0.021275, 0.008275, -0.016725],
    "10":[0.100275, 0.097275, 0.106275, 0.092275, 0.093275]
}
# ============ NeRF VirRTR ============
NeRF_VirRTR_dome = {
    "1": [-0.069725, -0.021725, 0.007275, -0.052725, -0.040725],
    "2": [-0.046925, -0.045925, -0.050925, -0.021925, -0.058925],
    "3": [-0.340525, -0.405525, -0.439525, -0.391525, -0.463525],
    "4": [0.024275, 0.081275, -0.102725, -0.025725, -0.021725],
    "5": [0.181275, -0.009725, -0.010725, 0.050275, -0.044725],
    "6": [-0.045125, 0.008875, -0.034125, -0.095125, -0.107125],
    "7": [0.264675, 0.193675, 0.245675, 0.296675, 0.240675],
    "8": [0.264675, 0.381675, 0.305675, 0.271675, 0.237675],
    "9": [-0.083725, -0.081725, -0.032725, -0.102725, 0.200275],
    "10": [0.095075, 0.021075, -0.011925, 0.021075, 0.078075]
}
NeRF_VirRTR_parking = {
    "1": [0.015875, -0.031125, 0.058875, 0.108875, 0.045875],
    "2": [0.136275, 0.088275, 0.134275, 0.260275, 0.133275],
    "3": [0.138075, 0.204075, -0.019925, 0.180075, 0.140075],
    "4": [0.125075, 0.030075, 0.134075, 0.092075, 0.098075],
    "5": [-0.191525, -0.287525, -0.379525, -0.276525, -0.278525],
    "6": [-0.081925, -0.118925, -0.193925, -0.102925, -0.054925],
    "7": [-0.042925, -0.068925, 0.083075, 0.020075, 0.083075],
    "8": [0.035475, 0.066475, -0.003525, 0.063475, 0.045475],
    "9": [-0.025325, 0.043675, 0.002675, 0.003675, 0.023675],
    "10": [0.010675, 0.017675, 0.000675, 0.073675, 0.007675]
}

# ========================================== Define Comparisons ==========================================
EXPERIMENTS = {
    # lidar vs. virtual lidar with 3 methods
    "LT&R vs. VirLT&R (Pix4D) vs. VirLT&R (NeRF)": {
        "Structured-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/lidar/desi_LTR_Dome_Teach_from_her_laptop/graph", 
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/dome/graph",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/NeRF/dome/graph",

            method_1_runs=[9, 10, 11, 12, 13],  
            method_2_runs=[1, 3, 4, 5, 6],
            method_3_runs=[1, 3, 4, 5, 6],

            method_1_marker_errors=LTR_dome,
            method_2_marker_errors=Pix4D_VirLTR_dome,
            method_3_marker_errors=NeRF_VirLTR_dome,         

            method_1_marker_distances=[3, 32, 63, 106, 134, 174, 209, 240, 271, 302],
            method_2_marker_distances=[2, 30, 62, 105, 133, 177, 219, 241, 279, 312],
            method_3_marker_distances=[1, 33, 64, 102, 134, 175, 214, 242, 273, 308]
        ),
        "Urban-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/lidar/parking/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/parking/graph",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/NeRF/parking/graph",

            method_1_runs=[1, 2, 3, 4, 5],
            method_2_runs=[4, 7, 8, 9, 10],
            method_3_runs=[1, 2, 3, 5, 6],

            method_1_marker_errors=LTR_parking,
            method_2_marker_errors=Pix4D_VirLTR_parking,
            method_3_marker_errors=NeRF_VirLTR_parking,

            method_1_marker_distances=[1, 34, 86, 132, 180, 222, 274, 309, 344, 383],
            method_2_marker_distances=[2, 36, 85, 136, 184, 221, 270, 307, 343, 381],
            method_3_marker_distances=[3, 33, 87, 131, 181, 223, 271, 308, 342, 382]
        ),
    # },    
    # "LTR vs. VirLTR (Pix4D)": {
        "Semi-Structured-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/lidar/bigpath/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/bigpath/graph",

            method_1_runs=[1, 2, 4, 5, 6],
            method_2_runs=[1, 2, 3, 4, 5],

            method_1_marker_errors=LTR_bigpath,
            method_2_marker_errors=Pix4D_VirLTR_bigpath,

            method_1_marker_distances=[1, 21, 42, 70, 93, 114, 140, 169, 203, 229, 263, 278, 290, 312],
            method_2_marker_distances=[3, 23, 43, 69, 94, 116, 141, 172, 204, 233, 262, 277, 291, 311]
        ),
        "Rural-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/lidar/grassy/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/grassy/graph",

            method_1_runs=[2, 3, 4, 5, 6],
            method_2_runs=[1, 2, 3, 5, 6],

            method_1_marker_errors=None,
            method_2_marker_errors=None,

            method_1_marker_distances=None,
            method_2_marker_distances=None,
        ),
    },
    # radar vs. virtual radar with 3 methods
    "RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)": {
        "Structured-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/dome/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/Pix4D/dome/graph/",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/NeRF/dome/graph/",

            method_1_runs=[4, 5, 6, 7, 8],
            method_2_runs=[1, 2, 4, 5, 6],
            method_3_runs=[1, 2, 4, 5, 7], 

            method_1_marker_errors=RTR_dome,
            method_2_marker_errors=Pix4D_VirRTR_dome,
            method_3_marker_errors=NeRF_VirRTR_dome,
            
            method_1_marker_distances=[4, 32, 62, 105, 131, 174, 207, 239, 269, 300],
            method_2_marker_distances=[3, 31, 60, 106, 135, 176, 223, 248, 276, 311],
            method_3_marker_distances=[1, 33, 63, 108, 133, 172, 222, 243, 273, 303]  
        ),
        "Urban-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/parking/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/Pix4D/parking/graph",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/NeRF/parking/graph",

            method_1_runs=[2, 3, 4, 5, 6],
            method_2_runs=[1, 3, 4, 5, 7],
            method_3_runs=[3, 5, 6, 9, 10],

            method_1_marker_errors=RTR_parking,
            method_2_marker_errors=Pix4D_VirRTR_parking,
            method_3_marker_errors=NeRF_VirRTR_parking,

            method_1_marker_distances=[1, 35, 87, 132, 180, 224, 270, 307, 343, 385],
            method_2_marker_distances=[2, 36, 89, 134, 182, 223, 271, 308, 344, 381],
            method_3_marker_distances=[3, 34, 91, 133, 186, 222, 272, 306, 342, 382] 
        ),
        "Semi-Structured-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/bigpath/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/bigpath/graph",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/bigpath/graph",

            method_1_runs=[2, 3, 4, 5, 6],
            method_2_runs=[2, 3, 4, 5, 6],
            method_3_runs=[2, 3, 4, 5, 6], 

            method_1_marker_errors=RTR_bigpath,
            method_2_marker_errors=None,
            method_3_marker_errors=None,
            
            method_1_marker_distances=[4, 32, 62, 105, 131, 174, 207, 239, 269, 300],
            method_2_marker_distances=None,
            method_3_marker_distances=None  
        ),
        "Rural-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/grassy/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/grassy/graph",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/grassy/graph",

            method_1_runs=[3, 4, 5, 6, 7],
            method_2_runs=[3, 4, 5, 6, 7],
            method_3_runs=[3, 4, 5, 6, 7],

            method_1_marker_errors=None,
            method_2_marker_errors=None,
            method_3_marker_errors=None,
            
            method_1_marker_distances=None,
            method_2_marker_distances=None,
            method_3_marker_distances=None  
        ),
    },
    # for box plots
    "VirLT&R (Pix4D) vs. VirLT&R (NeRF) vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)": {
        "Structured-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/lidar/desi_LTR_Dome_Teach_from_her_laptop/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/dome/graph",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/NeRF/dome/graph",
            method_4="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/dome/graph",
            method_5="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/Pix4D/dome/graph",
            method_6="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/NeRF/dome/graph",

            method_1_runs=[9, 10, 11, 12, 13],
            method_2_runs=[1, 3, 4, 5, 6],
            method_3_runs=[1, 3, 4, 5, 6],
            method_4_runs=[4, 5, 6, 7, 8],
            method_5_runs=[1, 2, 4, 5, 6],
            method_6_runs=[1, 2, 4, 5, 7], 

            method_1_marker_errors=LTR_dome,
            method_2_marker_errors=Pix4D_VirLTR_dome,
            method_3_marker_errors=NeRF_VirLTR_dome,
            method_4_marker_errors=RTR_dome,
            method_5_marker_errors=Pix4D_VirRTR_dome,
            method_6_marker_errors=NeRF_VirRTR_dome,

            method_1_marker_distances=[3, 32, 63, 106, 134, 174, 209, 240, 271, 302],
            method_2_marker_distances=[2, 30, 62, 105, 133, 177, 219, 241, 279, 312],
            method_3_marker_distances=[1, 33, 64, 102, 134, 175, 214, 242, 273, 308],
            method_4_marker_distances=[4, 32, 62, 105, 131, 174, 207, 239, 269, 300],
            method_5_marker_distances=[3, 31, 60, 106, 135, 176, 223, 248, 276, 311],
            method_6_marker_distances=[1, 33, 63, 108, 133, 172, 222, 243, 273, 303]
        ),
        "Urban-C": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment2/lidar/parking/graph",
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/Pix4D/parking/graph",
            method_3="/home/desiree/ASRL/vtr3/temp/Experiment2/VirLTR/NeRF/parking/graph", 
            method_4="/home/desiree/ASRL/vtr3/temp/Experiment2/radar/kstrongest/parking/graph",
            method_5="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/Pix4D/parking/graph",
            method_6="/home/desiree/ASRL/vtr3/temp/Experiment2/VirRTR/NeRF/parking/graph",

            method_1_runs=[1, 2, 3, 4, 5],
            method_2_runs=[4, 7, 8, 9, 10],
            method_3_runs=[1, 2, 3, 5, 6],
            method_4_runs=[2, 3, 4, 5, 6],
            method_5_runs=[1, 3, 4, 5, 7],
            method_6_runs=[3, 5, 6, 9, 10],

            method_1_marker_errors=LTR_parking,
            method_2_marker_errors=Pix4D_VirLTR_parking,
            method_3_marker_errors=NeRF_VirLTR_parking,
            method_4_marker_errors=RTR_parking,
            method_5_marker_errors=Pix4D_VirRTR_parking,
            method_6_marker_errors=NeRF_VirRTR_parking,

            method_1_marker_distances=[1, 34, 86, 132, 180, 222, 274, 309, 344, 383],
            method_2_marker_distances=[2, 36, 85, 136, 184, 221, 270, 307, 343, 381],
            method_3_marker_distances=[3, 33, 87, 131, 181, 223, 271, 308, 342, 382],
            method_4_marker_distances=[1, 35, 87, 132, 180, 224, 270, 307, 343, 385],
            method_5_marker_distances=[2, 36, 89, 134, 182, 223, 271, 308, 344, 381],
            method_6_marker_distances=[3, 34, 91, 133, 186, 222, 272, 306, 342, 382] 
        ),
    },
}

# ========================================== Helper Functions ==========================================
def build_graph(graph_path):
    factory = Rosbag2GraphFactory(graph_path)
    graph = factory.buildGraph()
    g_utils.set_world_frame(graph, graph.root)
    return graph

def teach_path_matrix(graph):
    """Return the teach path as a matrix for distance computation."""
    return vtr_path.path_to_matrix(graph, PriviledgedIterator(graph.root))

def iter_repeats(graph, run_ids):
    """Yield (run_id, iterator_start_vertex) for each requested repeat."""
    for rid in run_ids:
        try:
            v_start = graph.get_vertex((rid, 0))
        except Exception:
            continue
        yield rid, v_start

def compute_repeat_pte(graph, teach_mat, v_start):
    """Compute path tracking error along the repeat and cumulative path length."""
    pose_vec = []
    times = []
    dists = []
    cum_len = []
    plen = 0.0

    for v, e in TemporalIterator(v_start):
        r = v.T_v_w.r_ba_ina()
        pose_vec.append(r)
        times.append(v.stamp / 1e9)
        dists.append(vtr_path.signed_distance_to_path(r, teach_mat))
        if e is not None:
            plen += np.linalg.norm(e.T.r_ba_ina())
        cum_len.append(plen)

    times = np.array(times)
    order = np.argsort(times)
    pose_vec = np.array(pose_vec)[order]
    dists = np.array(dists)[order]
    cum_len = np.array(cum_len)[order]
    rmse = np.sqrt(np.mean(np.square(dists))) if dists.size > 0 else float('nan')
    return pose_vec, cum_len, dists, rmse

def process_experiment(exp_name, path_name, cfg, color_map=None):
    """
    Process an experiment configuration that can include 2, 3, or 4 methods.
    Builds combined XY and PTE plots for all methods.
    """
    # Split experiment title to get individual method labels.
    method_labels = [s.strip() for s in exp_name.split(" vs. ")]

    # Determine available methods by scanning for keys "method_<num>"
    methods = []
    for key in cfg.keys():
        if key.startswith("method_") and key.count("_") == 1:
            try:
                idx = int(key.split("_")[1])
                methods.append(idx)
            except ValueError:
                continue
    methods = sorted(methods)

    default_xy_colors = ['red', 'blue', 'green', 'orange', 'purple', 'teal'] 
    default_pte_colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'moccasin', 'plum', 'paleturquoise']

    method_data = {}
    for m in methods:
        graph = build_graph(cfg[f"method_{m}"])
        teach_mat = teach_path_matrix(graph)
        runs = cfg.get(f"method_{m}_runs", [])
        repeats = []
        for rid, v_start in iter_repeats(graph, runs):
            pose_vec, cum_len, dists, _ = compute_repeat_pte(graph, teach_mat, v_start)
            if pose_vec.size > 0:
                repeats.append((rid, pose_vec, cum_len, dists))
        teach_xy = [(v.T_v_w.r_ba_ina()[0], v.T_v_w.r_ba_ina()[1])
                    for v, _ in PriviledgedIterator(graph.root)]
        method_data[m] = {"graph": graph,
                          "teach_mat": teach_mat,
                          "repeats": repeats,
                          "teach_xy": teach_xy,
                          "runs": runs,
                          "marker_errors": cfg.get(f"method_{m}_marker_errors", None),
                          "marker_distances": cfg.get(f"method_{m}_marker_distances", None)}
    
    # Combined XY Plot
    fig_xy, ax_xy = plt.subplots()
    ax_xy.set_title(f"{exp_name} – {path_name} – Combined XY Plot")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.axis('equal')
    for m in methods:
        color = color_map[m] if color_map and m in color_map else default_xy_colors[(m - 1) % len(default_xy_colors)]
        # Use label from method_labels if available, otherwise fall back to "Method N"
        label_name = method_labels[m - 1] if m - 1 < len(method_labels) else f"Method {m}"
        teach_x, teach_y = zip(*method_data[m]["teach_xy"])
        ax_xy.plot(teach_x, teach_y, linestyle=':', color=color, label=f"{label_name} Teach")
        for rid, pose_vec, _, _ in method_data[m]["repeats"]:
            ax_xy.plot(pose_vec[:, 0], pose_vec[:, 1], '.', color=color, label=f"{label_name} Repeat {rid}")
    ax_xy.legend(loc='upper left')

    # Combined PTE Plot
    fig_pte, ax_pte = plt.subplots()
    ax_pte.set_title(f"{exp_name} – {path_name} Path Tracking Error")
    ax_pte.set_xlabel("Path Length (m)")
    ax_pte.set_ylabel("PTE (m)")
    ax_pte.grid(True)
    ax_pte.axhline(0, linestyle='--', linewidth=1.0, color='gray')

    for m in methods:
        color = default_pte_colors[(m - 1) % len(default_pte_colors)]
        # Use label from method_labels if available, otherwise fall back to "Method N"
        label_name = method_labels[m - 1] if m - 1 < len(method_labels) else f"Method {m}"
        repeats   = method_data[m]["repeats"]              # list of (rid, pose_vec, cum_len, dists)
        markers   = method_data[m]["marker_errors"]        # dict of hardcoded marker errors (for scatter)
        distances = method_data[m]["marker_distances"]     # list of marker distances along path

        if not repeats:
            continue

        # ---------- Average curve over common domain (for plotting) ----------
        min_end = min(r[2][-1] for r in repeats if r[2].size > 0)
        common_x = np.linspace(0, min_end, 500)
        interpolated_dists = []
        for _, _, cum_len_run, dists_run in repeats:
            interp_d = np.interp(common_x, cum_len_run, dists_run)
            interpolated_dists.append(interp_d)
        average_dists = np.mean(interpolated_dists, axis=0)

        rmse_avg = np.sqrt(np.mean(np.square(average_dists))) if average_dists.size else float('nan')
        max_avg  = np.max(np.abs(average_dists))           if average_dists.size else float('nan')

        # ---------- Overall metrics from ALL per-run samples (pooled, not averaged) ----------
        pooled = np.concatenate([d for _, _, _, d in repeats if isinstance(d, np.ndarray) and d.size > 0]) \
                 if any((d.size > 0) for _, _, _, d in repeats) else np.array([])
        rmse_all = np.sqrt(np.mean(pooled**2)) if pooled.size > 0 else float('nan')
        max_all  = np.max(np.abs(pooled))      if pooled.size > 0 else float('nan')

        # ---------- Posegraph PTE sampled at marker locations (pooled across runs) ----------
        rmse_at_marks  = float('nan')
        max_at_marks   = float('nan')
        if distances and len(distances) > 0:
            mark_arrays = []
            for _, _, cum_len_run, dists_run in repeats:
                if cum_len_run.size == 0 or dists_run.size == 0:
                    continue
                valid_md = np.asarray([md for md in distances if md <= cum_len_run[-1]], dtype=float)
                if valid_md.size == 0:
                    continue
                samples = np.interp(valid_md, cum_len_run, dists_run)
                mark_arrays.append(samples)
            if mark_arrays:
                mark_samples_flat = np.concatenate(mark_arrays)
                if mark_samples_flat.size > 0:
                    rmse_at_marks = np.sqrt(np.mean(mark_samples_flat**2))
                    max_at_marks  = np.max(np.abs(mark_samples_flat))

        # ---------- Plot average curve with a legend that shows ALL the metrics ----------
        ax_pte.plot(
            common_x, average_dists, linewidth=1.5, color=color,
            label=(
                f"{label_name} Avg PTE Curve (Overall RMSE={rmse_all:.3f} m, Overall Max={max_all:.3f} m)"
            )
        )
        print(f"{label_name} PTE Avg "
                f"(AvgCurve RMSE={rmse_avg:.3f} m, Max={max_avg:.3f} m | "
                f"Overall RMSE={rmse_all:.3f} m, Max={max_all:.3f} m | "
                f"PTE@Marks RMSE={rmse_at_marks:.3f} m, Max={max_at_marks:.3f} m)")
        
        # ---------- Scatter of all hardcoded marker measurements (if provided) ----------
        if markers and distances:
            flat_vals = [v for mlist in markers.values() for v in mlist]
            rmse_hard = np.sqrt(np.mean(np.square(flat_vals))) if flat_vals else float('nan')
            max_hard  = max((abs(v) for v in flat_vals), default=float('nan'))

            marker_color = default_xy_colors[(m - 1) % len(default_xy_colors)]
            x_vals, y_vals = [], []
            sorted_keys = sorted(markers.keys(), key=lambda k: int(k))
            for i, key in enumerate(sorted_keys):
                errs = markers[key]
                if i < len(distances):
                    x_vals.extend([distances[i]] * len(errs))
                    y_vals.extend(errs)

            ax_pte.scatter(
                x_vals, y_vals, s=80, marker='x', linewidths=1.2, alpha=0.9,
                color=marker_color, zorder=10,
                label=(f"{label_name} Marker Measurements "
                       f"(RMSE={rmse_hard:.3f} m, Max={max_hard:.3f} m)")
            )
    ax_pte.legend(loc='upper left')

    # Save figures to a folder named "plots" within this directory.
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    safe_exp = exp_name.replace(" ", "_")
    safe_path = path_name.replace(" ", "_")
    xy_filename = os.path.join(save_dir, f"{safe_exp}_{safe_path}_xy.png")
    pte_filename = os.path.join(save_dir, f"{safe_exp}_{safe_path}_pte.png")
    fig_xy.savefig(xy_filename, dpi=300, bbox_inches='tight')
    fig_pte.savefig(pte_filename, dpi=300, bbox_inches='tight')
    plt.show()
    return fig_xy, fig_pte

def summary_marker_box_plots(experiments):
    """
    For each experiment, create a single box plot that aggregates the marker error distributions
    for all its paths. For each path (grouped on the x-axis), each method's errors are shown side‐by‐side.
    Two-method experiments (with "baseline_markers" and "marker_errors") and multi-method experiments
    (with keys like "method_?_marker_errors") are supported.
    RMSE and maximum error for each method (across all paths) are included in the legend.
    The plot is saved in the "plots" folder.
    """

    colors = ['red', 'blue', 'green', 'orange']

    for exp_name, paths in experiments.items():
        # Collect marker data per path. Each path will yield a dictionary:
        #   { method_label: [list of errors] }
        per_path_method_data = []
        path_names = []
        for path_name, cfg in paths.items():
            markers_data = {}
            # Check for two-method case using legacy keys: baseline_markers and marker_errors.
            if (cfg.get("baseline_markers") is not None and 
                cfg.get("marker_errors") is not None):
                baseline_vals = []
                for vals in cfg["baseline_markers"].values():
                    baseline_vals.extend(vals)
                experimental_vals = []
                for vals in cfg["marker_errors"].values():
                    experimental_vals.extend(vals)
                markers_data["Baseline"] = baseline_vals
                markers_data["Experimental"] = experimental_vals
            else:
                # Multi-method case: look for keys like "method_?_marker_errors"
                method_keys = [key for key in cfg.keys() 
                               if key.startswith("method_") and "marker_errors" in key and cfg.get(key) is not None]
                if method_keys:
                    # Parse method labels from the experiment title; if not enough, fall back to default names.
                    parsed_labels = [s.strip() for s in exp_name.split(" vs. ")]
                    sorted_keys = sorted(method_keys, key=lambda key: int(key.split("_")[1]))
                    for i, key in enumerate(sorted_keys):
                        label = parsed_labels[i] if i < len(parsed_labels) else f"Method {i+1}"
                        vals = []
                        for mvals in cfg[key].values():
                            vals.extend(mvals)
                        markers_data[label] = vals
            # Only add this path if it has 2+ non-empty method datasets
            if len(markers_data) >= 2:
                per_path_method_data.append(markers_data)
                path_names.append(path_name)

        if not per_path_method_data:
            continue

        # Determine the set of method labels across all paths (sorted for consistent order)
        parsed_labels     = [s.strip() for s in exp_name.split(" vs. ")]
        all_method_labels = [
            lbl for lbl in parsed_labels
            if any(lbl in d for d in per_path_method_data)
        ]

        num_methods = len(all_method_labels)
        method_color = {label: colors[i % len(colors)]
                        for i, label in enumerate(all_method_labels)}

        # Define box group configuration
        group_width = 0.15  # horizontal spacing between methods within each group

        # Create a figure for the experiment
        fig, ax = plt.subplots()
        ax.set_title(f"{exp_name} – Marker Errors Summary")
        ax.set_ylabel("Marker Error (m)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, linestyle='--', linewidth=2, color='darkgrey')

        # For each method, collect data across paths along with x positions.
        method_to_positions = {label: [] for label in all_method_labels}
        method_to_data = {label: [] for label in all_method_labels}
        x_ticks = []
        for i, pdata in enumerate(per_path_method_data):
            group_center = i + 1
            x_ticks.append(group_center)
            # For each method, compute an offset position within the group
            for j, m_label in enumerate(all_method_labels):
                offset = (j - (num_methods - 1) / 2) * group_width
                method_to_positions[m_label].append(group_center + offset)
                # Use an empty list if the method is not present in this path
                method_to_data[m_label].append(pdata.get(m_label, []))

        # Plot a box for each method in each path group.
        for m_label in all_method_labels:
            bp = ax.boxplot(method_to_data[m_label],
                            positions=method_to_positions[m_label],
                            widths=group_width * 0.8,
                            patch_artist=True,
                            manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(method_color[m_label])

        # Set x-axis ticks to group centers with path names.
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(path_names, rotation=45, ha='right')

        # Build legend entries with one patch per method.
        legend_handles = [Patch(facecolor=method_color[m_label], edgecolor='black', label=m_label)
                  for m_label in all_method_labels]
              
        ax.legend(handles=legend_handles, loc='best')

        plt.tight_layout()

        # Save the combined summary plot
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        safe_exp = exp_name.replace(" ", "_")
        combined_filename = os.path.join(save_dir, f"{safe_exp}_combined_marker_summary.png")
        fig.savefig(combined_filename, dpi=300, bbox_inches='tight')
        plt.show()

def summary_cross_platform_route_box_plot(experiments):
    """
    Create a grouped box plot for:
      LTR vs. VirLTR (Pix4D) vs. VirLTR (NeRF) vs. RTR vs. VirRTR (Pix4D) vs. VirRTR (NeRF)
    over Urban, Structured, and Semi-Structured routes.
    """
    route_order = ["Urban-C", "Structured-C", "Semi-Structured-C"]
    route_labels = ["Urban-C", "Structured-C", "Semi-Structured-C"]

    method_specs = [
        ("LT&R", "LT&R vs. VirLT&R (Pix4D) vs. VirLT&R (NeRF)", "method_1_marker_errors"),
        ("VirLT&R (Pix4D)", "LT&R vs. VirLT&R (Pix4D) vs. VirLT&R (NeRF)", "method_2_marker_errors"),
        ("VirLT&R (NeRF)", "LT&R vs. VirLT&R (Pix4D) vs. VirLT&R (NeRF)", "method_3_marker_errors"),
        ("RT&R", "RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)", "method_1_marker_errors"),
        ("VirRT&R (Pix4D)", "RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)", "method_2_marker_errors"),
        ("VirRT&R (NeRF)", "RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)", "method_3_marker_errors"),
    ]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'teal']

    def flatten_marker_dict(marker_dict):
        vals = []
        for seq in marker_dict.values():
            vals.extend(seq)
        return vals

    data = {
        route: {method: [] for method, _, _ in method_specs}
        for route in route_order
    }

    for route in route_order:
        for method, exp_name, marker_key in method_specs:
            cfg = experiments.get(exp_name, {}).get(route, {})
            marker_dict = cfg.get(marker_key, None)
            if marker_dict:
                data[route][method] = flatten_marker_dict(marker_dict)

    fig, ax = plt.subplots(figsize=(16, 7))
    group_gap = 0.95
    box_width = 0.12
    x_centers = []

    # Larger text settings for this figure
    title_fs = 22
    label_fs = 20
    tick_fs = 18
    legend_fs = 16

    flierprops = dict(
        marker='o',
        markerfacecolor='none',
        markeredgecolor='black',
        markersize=4,
        linestyle='none'
    )

    for i, route in enumerate(route_order):
        center = (i + 1) * group_gap
        x_centers.append(center)

        for j, (method, _, _) in enumerate(method_specs):
            pos = center + (j - (len(method_specs) - 1) / 2) * box_width
            vals = data[route][method]

            if vals:
                bp = ax.boxplot(
                    [vals],
                    positions=[pos],
                    widths=box_width * 0.8,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=True,
                    flierprops=flierprops,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(colors[j])
                    patch.set_edgecolor("black")
                for element in ("whiskers", "caps", "medians"):
                    for artist in bp[element]:
                        artist.set_color("black")
            else:
                ax.text(
                    pos, 0.02, "",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=8, color="gray", rotation=90
                )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(route_labels, fontsize=tick_fs)
    ax.set_ylabel("Marker Error (m)", fontsize=label_fs)
    ax.set_title(
        "LT&R vs. VirLT&R (Pix4D) vs. VirLT&R (NeRF) vs. RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)",
        fontsize=title_fs
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, linestyle="--", linewidth=1.5, color="darkgrey")
    ax.tick_params(axis='y', labelsize=tick_fs)

    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", label=method_specs[i][0])
        for i in range(len(method_specs))
    ]
    ax.legend(handles=legend_handles, loc="best", ncol=2, fontsize=legend_fs)

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(
        save_dir,
        "LT&R_VirLT&R_RT&R_VirRT&R_Urban_Structured_SemiStructured_marker_summary.png"
    )
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def summary_lidar_pte_plots(experiments):
    """
    Create a single figure with 4 subplots showing PTE curves for LiDAR across all routes.
    Each subplot shows the methods available for that route.
    """
    routes = ["Structured-C", "Urban-C", "Semi-Structured-C", "Rural-C"]
    exp_name = "LT&R vs. VirLT&R (Pix4D) vs. VirLT&R (NeRF)"

    fig, axs = plt.subplots(4, 1, figsize=(18, 10))
    axs = np.atleast_1d(axs).flatten()

    default_pte_colors = ['lightcoral', 'lightskyblue', 'lightgreen']
    default_xy_colors = ['red', 'blue', 'green']

    for idx, route in enumerate(routes):
        ax_pte = axs[idx]
        cfg = experiments.get(exp_name, {}).get(route, {})

        if not cfg:
            ax_pte.axis("off")
            continue

        method_nums = sorted(
            int(k.split("_")[1])
            for k in cfg.keys()
            if k.startswith("method_") and k.count("_") == 1 and f"method_{k.split('_')[1]}" in cfg
        )

        for m in method_nums:
            method_key = f"method_{m}"
            if method_key not in cfg:
                continue

            graph = build_graph(cfg[method_key])
            teach_mat = teach_path_matrix(graph)
            runs = cfg.get(f"method_{m}_runs", [])
            repeats = []
            for rid, v_start in iter_repeats(graph, runs):
                pose_vec, cum_len, dists, _ = compute_repeat_pte(graph, teach_mat, v_start)
                if pose_vec.size > 0:
                    repeats.append((rid, pose_vec, cum_len, dists))

            if not repeats:
                continue

            label_name = ["LT&R", "VirLT&R (Pix4D)", "VirLT&R (NeRF)"][m - 1] if m <= 3 else f"Method {m}"
            color = default_pte_colors[(m - 1) % len(default_pte_colors)]

            min_end = min(r[2][-1] for r in repeats if r[2].size > 0)
            common_x = np.linspace(0, min_end, 500)
            interpolated_dists = []
            for _, _, cum_len_run, dists_run in repeats:
                interp_d = np.interp(common_x, cum_len_run, dists_run)
                interpolated_dists.append(interp_d)
            average_dists = np.mean(interpolated_dists, axis=0)

            pooled = np.concatenate([d for _, _, _, d in repeats if isinstance(d, np.ndarray) and d.size > 0]) \
                     if any((d.size > 0) for _, _, _, d in repeats) else np.array([])
            rmse_all = np.sqrt(np.mean(pooled**2)) if pooled.size > 0 else float('nan')
            max_all = np.max(np.abs(pooled)) if pooled.size > 0 else float('nan')

            ax_pte.plot(
                common_x, average_dists, linewidth=1.5, color=color,
                label=f"{label_name} (RMSE={rmse_all:.3f} m, Max={max_all:.3f} m)"
            )

            markers = cfg.get(f"method_{m}_marker_errors", None)
            distances = cfg.get(f"method_{m}_marker_distances", None)
            if markers and distances:
                marker_color = default_xy_colors[(m - 1) % len(default_xy_colors)]
                x_vals, y_vals = [], []
                sorted_keys = sorted(markers.keys(), key=lambda k: int(k))
                for i, key in enumerate(sorted_keys):
                    errs = markers[key]
                    if i < len(distances):
                        x_vals.extend([distances[i]] * len(errs))
                        y_vals.extend(errs)

                ax_pte.scatter(
                    x_vals, y_vals, s=80, marker='x', linewidths=1.2, alpha=0.9,
                    color=marker_color, zorder=10
                )

        ax_pte.set_title(route, fontsize=14)
        ax_pte.grid(True, linestyle='--', alpha=0.5)
        ax_pte.axhline(0, linestyle='--', linewidth=1.0, color='gray')
        ax_pte.legend(loc='lower right', fontsize=10)

    fig.suptitle("LiDAR PTE Summary: LT&R vs. VirLT&R (Pix4D) vs. VirLT&R (NeRF)", fontsize=16)
    fig.supxlabel("Path Length (m)", fontsize=12)
    fig.supylabel("PTE (m)", fontsize=12)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "LiDAR_PTE_Summary.png")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def summary_radar_pte_plots(experiments):
    """
    Create a single figure with 2 subplots showing PTE curves for Radar across selected routes.
    Each subplot shows: RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)
    """
    routes = ["Structured-C", "Urban-C"]
    route_labels = ["Structured-C", "Urban-C"]
    exp_name = "RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)"
    
    fig, axs = plt.subplots(2, 1, figsize=(14, 5))
    axs = np.atleast_1d(axs)
    
    default_pte_colors = ['lightcoral', 'lightskyblue', 'lightgreen']
    default_xy_colors = ['red', 'blue', 'green']
    
    for idx, route in enumerate(routes):
        ax_pte = axs[idx]
        cfg = experiments.get(exp_name, {}).get(route, {})
        
        if not cfg:
            continue
        
        method_labels = ["RT&R", "VirRT&R (Pix4D)", "VirRT&R (NeRF)"]
        
        for m in [1, 2, 3]:
            graph = build_graph(cfg[f"method_{m}"])
            teach_mat = teach_path_matrix(graph)
            runs = cfg.get(f"method_{m}_runs", [])
            repeats = []
            for rid, v_start in iter_repeats(graph, runs):
                pose_vec, cum_len, dists, _ = compute_repeat_pte(graph, teach_mat, v_start)
                if pose_vec.size > 0:
                    repeats.append((rid, pose_vec, cum_len, dists))
            
            if not repeats:
                continue
            
            color = default_pte_colors[m - 1]
            
            # Average curve over common domain
            min_end = min(r[2][-1] for r in repeats if r[2].size > 0)
            common_x = np.linspace(0, min_end, 500)
            interpolated_dists = []
            for _, _, cum_len_run, dists_run in repeats:
                interp_d = np.interp(common_x, cum_len_run, dists_run)
                interpolated_dists.append(interp_d)
            average_dists = np.mean(interpolated_dists, axis=0)
            
            pooled = np.concatenate([d for _, _, _, d in repeats if isinstance(d, np.ndarray) and d.size > 0]) \
                     if any((d.size > 0) for _, _, _, d in repeats) else np.array([])
            rmse_all = np.sqrt(np.mean(pooled**2)) if pooled.size > 0 else float('nan')
            max_all  = np.max(np.abs(pooled))      if pooled.size > 0 else float('nan')
            
            # Plot average curve
            ax_pte.plot(
                common_x, average_dists, linewidth=1.5, color=color,
                label=(f"{method_labels[m-1]} (RMSE={rmse_all:.3f} m, Max={max_all:.3f} m)")
            )
            
            # Scatter markers
            markers = cfg.get(f"method_{m}_marker_errors", None)
            distances = cfg.get(f"method_{m}_marker_distances", None)
            
            if markers and distances:
                marker_color = default_xy_colors[m - 1]
                x_vals, y_vals = [], []
                sorted_keys = sorted(markers.keys(), key=lambda k: int(k))
                for i, key in enumerate(sorted_keys):
                    errs = markers[key]
                    if i < len(distances):
                        x_vals.extend([distances[i]] * len(errs))
                        y_vals.extend(errs)
                
                ax_pte.scatter(
                    x_vals, y_vals, s=80, marker='x', linewidths=1.2, alpha=0.9,
                    color=marker_color, zorder=10
                )
        
        ax_pte.set_title(f"{route}", fontsize=14)
        ax_pte.grid(True, linestyle='--', alpha=0.5)
        ax_pte.axhline(0, linestyle='--', linewidth=1.0, color='gray')
        ax_pte.legend(loc='lower right', fontsize=10)
    
    fig.suptitle("Radar PTE Summary: RT&R vs. VirRT&R (Pix4D) vs. VirRT&R (NeRF)", fontsize=16)
    fig.supxlabel("Path Length (m)", fontsize=12)
    fig.supylabel("PTE (m)", fontsize=12)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "Radar_PTE_Summary.png")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

# ========================================== Main Execution ==========================================
if __name__ == "__main__":
    # Call the metrics printing function
    #compute_and_print_metrics()

    # # Generate individual comparison plots for each experiment and path.
    # for exp_name, paths in EXPERIMENTS.items():
    #     for path_name, cfg in paths.items():
    #         process_experiment(exp_name, path_name, cfg)

    # # # Call the summary marker box plots function using the EXPERIMENTS dictionary.
    # #summary_marker_box_plots(EXPERIMENTS)
    summary_cross_platform_route_box_plot(EXPERIMENTS)

    # # Call the summary PTE functions
    # summary_lidar_pte_plots(EXPERIMENTS)
    # summary_radar_pte_plots(EXPERIMENTS)