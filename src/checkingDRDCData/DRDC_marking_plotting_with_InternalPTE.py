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
# LTR_office = {# 1 is 13/15, 2 is 17,18, 3 is 20, 4 is 21
#     "1":  [0.000000,  0.000000,  0.000000,  0.000000],
#     "2":  [-0.009525, -0.053975, -0.012700, -0.044450],
#     "3":  [-0.022098,  0.000000,  0.012700,  0.009525],
#     "4":  [0.031750,  0.076200,  0.082550,  0.044450],
#     "5":  [-0.028575, -0.012700, -0.015875, -0.015875],
#     "6":  [-0.038100, -0.022225, -0.022225, -0.019050],
#     "7":  [0.012700, -0.009525, -0.009525,  0.015875],
#     "8":  [-0.158750, -0.206375, -0.196850, -0.228600],
#     "9":  [0.031750,  0.022225,  0.038100, -0.146050],
#     "10": [0.060325,  0.088900,  0.079375,  0.079375],
#     "11": [0.028575,  0.028575,  0.028575,  0.050800],
#     "12": [-0.028575, -0.022225,  0.009525,  0.000000],
#     "13": [0.031750,  0.025400,  0.041275,  0.015875],
# }

LTR_office = {# 1 is 13/15, 2 is 17,18, 3 is 20, 4 is 21 THIS IS 3 AND 4
    "1":  [0.000000,  0.000000],
    "2":  [-0.012700, -0.044450],
    "3":  [0.012700,  0.009525],
    "4":  [0.082550,  0.044450],
    "5":  [-0.015875, -0.015875],
    "6":  [-0.022225, -0.019050],
    "7":  [-0.009525,  0.015875],
    "8":  [-0.196850, -0.228600],
    "9":  [0.038100, -0.146050],
    "10": [0.079375,  0.079375],
    "11": [0.028575,  0.050800],
    "12": [0.009525,  0.000000],
    "13": [0.041275,  0.015875],
}

# LTR_rural_all = { # some missing or moved markers 
#     "1": [0.00000,      0.00000, 0.00000, 0.00000],
#     "2": [0.10160,      0.08890, -0.07938, -0.05080],
#     "3": [0.02858,      0.12383, 0.16193, 0.11430],
#     "4": [-0.01270,     -0.04128, 0.09843, -0.00318],
#     "5": [float('nan'), -0.05080, 0.05715, -0.09208],
#     "6": [0.00318,      0.03493,  0.06350, 0.06985],
#     "7": [0.09208,      0.03810,  0.10478, 0.10478],
#     "8": [0.20955,      0.19685,  0.24130, 0.26353],
#     "9": [0.21908,      0.20320,  0.23495, 0.15875],
#     }

LTR_rural_for_clicked = { # 1 is 4, 2 is 9/10, 3 is 14, 4 is 13 THIS IS LTR RURAL 3(14) and 4(13) 
    "1": [0.00000, 0.00000],
    "2": [-0.07938, -0.05080],
    "3": [0.16193, 0.11430],
    "4": [0.09843, -0.00318],
    "5": [0.05715, -0.09208],
    "6": [0.06350, 0.06985],
    "7": [0.10478, 0.10478],
    "8": [0.24130, 0.26353],
    "9": [0.23495, 0.15875],
    }

LTR_rural_for_driven = { # 1 is 4, 2 is 9/10, 3 is 14, 4 is 13 THIS IS LTR RURAL 4(13) 
    "1": [0.00000],
    "2": [-0.05080],
    "3": [0.11430],
    "4": [-0.00318],
    "5": [-0.09208],
    "6": [0.06985],
    "7": [0.10478],
    "8": [0.26353],
    "9": [0.15875],
    }

LTR_urban = { # excluding marker 4 (renumbered) # 1 is 1, 2 is 2, 3 is 3, 4 is 4
    "1":  [0.000000,  0.000000,  0.000000,  0.000000],
    "2":  [-0.025400, -0.015875,  0.000000, -0.003175],
    "3":  [0.003175,  0.006350,  0.003175,  0.025400],
    "4":  [-0.012700, -0.066675,  0.047625,  0.044450],
    "5":  [-0.006350, -0.025400,  0.006350,  0.012700],
    "6":  [-0.019050, -0.028575, -0.012700, -0.028575],
    "7":  [0.050800,  0.031750,  0.063500, -0.022225],
    "8":  [0.012700,  0.015875,  0.019050,  0.022225],
    "9": [-0.041275, -0.050800, -0.028575, -0.057150],
    "10": [-0.009525, -0.012700, -0.015875, -0.015875],
    "11": [0.000000, -0.009525, -0.003175, -0.019050],
}

# ========================================== Experimental Methods ==========================================
# Pix4D_VirLTR_office = { # 1 is 3, 2, is 4, 3 is 5, 4 is 7
#     "1":  [0.0,      0.0,       0.0,         0.0],
#     "2":  [0.1524,   0.14605,   0.12065,     0.17145],
#     "3":  [0.022225, 0.01905,   0.03175,     0.0127],
#     "4":  [0.06985,  0.08255,   0.06985,     0.10795],
#     "5":  [0.1651,   0.149225,  0.15875,     0.15875],
#     "6":  [-0.09525, -0.0889,   -0.092075,  -0.09525],
#     "7":  [-0.098425, -0.130175, -0.117475, -0.1397],
#     "8":  [-0.104775, -0.0889,  -0.060325,  -0.05715],
#     "9":  [0.0635,   0.041275,   0.0381,    0.03175],
#     "10": [-0.03175, -0.047625, -0.028575,  -0.04445],
#     "11": [-0.0762, -0.034925,  -0.066675,  -0.0635],
#     "12": [-0.06985, -0.03175,  -0.0381,    -0.0381],
#     "13": [0.1905,   0.1905,     0.2159,    0.193675],
# }

Pix4D_VirLTR_office = { # 1 is 3, 2, is 4, 3 is 5, 4 is 7 THIS IS 3 AND 4 
    "1":  [0.0,         0.0],
    "2":  [0.12065,     0.17145],
    "3":  [0.03175,     0.0127],
    "4":  [0.06985,     0.10795],
    "5":  [0.15875,     0.15875],
    "6":  [-0.092075,  -0.09525],
    "7":  [-0.117475, -0.1397],
    "8":  [-0.060325,  -0.05715],
    "9":  [0.0381,    0.03175],
    "10": [-0.028575,  -0.04445],
    "11": [-0.066675,  -0.0635],
    "12": [-0.0381,    -0.0381],
    "13": [0.2159,    0.193675],
}

# Pix4D_VirLTR_rural_all = {  # clicked1, clicked2, driven
#     "1":  [-0.0254,   -0.0254,   -0.0254],
#     "2":  [-0.0508,   -0.00635,  -0.1143],
#     "3":  [0.04445,    0.1143,    0.5969],
#     "4":  [-0.0254,   -0.0508,    0.18415],
#     "5":  [0.22225,   -0.08255,   0.7747],
#     "6":  [-0.01905,  -0.022225,  0.003175],
#     "7":  [-0.3683,   -0.4953,    0.4318],
#     "8":  [-0.3175,   -0.4699,    0.62865],
#     "9":  [-0.1778,   -0.1905,    1.1303],
# }

Pix4D_VirLTR_rural_clicked = {  # 1 is 1, 2 is 2
    "1":  [-0.0254,   -0.0254],
    "2":  [-0.0508,   -0.00635],
    "3":  [0.04445,    0.1143],
    "4":  [-0.0254,   -0.0508],
    "5":  [0.22225,   -0.08255],
    "6":  [-0.01905,  -0.02222575],
    "7":  [-0.3683,   -0.4953],
    "8":  [-0.3175,   -0.4699],
    "9":  [-0.1778,   -0.1905],
}

Pix4D_VirLTR_rural_driven = {  # 1 is 1
    "1":  [-0.0254],
    "2":  [-0.1143],
    "3":  [0.5969],
    "4":  [0.18415],
    "5":  [0.7747],
    "6":  [0.003175],
    "7":  [0.4318],
    "8":  [0.62865],
    "9":  [1.1303], #****
}

Pix4D_VirLTR_urban = { # 1 is 1, 2 is 4, 3 is 6, 4 is 7
    "1":  [0.0508,     0.0508,     0.0508,     0.0508],
    "2":  [0.06985,    0.06985,    0.05715,    0.06985],
    "3":  [0.111125,   0.1016,     0.1016,     0.12065],
    "4":  [-0.0889,    -0.1524,    0.117475,   -0.2413],
    "5":  [-0.00635,   -0.0254,    -0.003175,  -0.015875],
    "6":  [-0.08255,   -0.0762,    -0.079375,  -0.0762],
    "7":  [-0.142875,  -0.1524,    -0.161925,  -0.149225],
    "8":  [-0.127,     -0.123825,  -0.13335,   -0.136525],
    "9": [-0.136525,  -0.130175,  -0.136525,  -0.1397],
    "10": [-0.1397,    -0.130175,  -0.130175,  -0.127],
    "11": [0.1016,     0.15875,    0.17145,    0.174625],
}

# ========================================== Define Comparisons ==========================================
EXPERIMENTS = {
    # lidar vs. virtual lidar for all routes using runs from pose graph and markings
    "LTR vs. VirLTR (Pix4D)": {
        # "DRDC Office Loop": dict(
        #     method_1="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/LTR/ltr_office2/graph", 
        #     method_2="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/VirLTR/virtr_office/graph",

        #     method_1_runs=[13, 15, 20, 21],  
        #     method_2_runs=[3, 4, 5, 7],

        #     method_1_marker_errors=LTR_office,
        #     method_2_marker_errors=Pix4D_VirLTR_office,

        #     method_1_marker_distances=[3, 32, 63, 106, 134, 174, 209, 240, 271, 302, 330, 350, 370],
        #     method_2_marker_distances=[2, 30, 62, 105, 133, 177, 219, 241, 279, 312, 330, 350, 370],
        # ),
        "DRDC Office Loop": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/LTR/ltr_office2/graph", 
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/VirLTR/virtr_office_clicked/graph",

            method_1_runs=[20, 21],  
            method_2_runs=[5, 7], # 5 started 3 or so vertices too early and ends 3 or so too early, 

            method_1_run_trim=[2, 2],
            method_2_run_trim=[5, 5],

            method_1_marker_errors=LTR_office,
            method_2_marker_errors=Pix4D_VirLTR_office,

            method_1_marker_distances=[3, 45, 134, 209, 286, 361, 439, 517, 602, 698, 790, 875, 954], # new updated
            method_2_marker_distances=[2, 46, 135, 210, 287, 362, 441, 520, 599, 696, 793, 886, 970],
        ),
        "DRDC Rural Loop - Clicked ": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/LTR/ltr_rural/graph", 
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/VirLTR/virtr_rural_clicked/graph",

            method_1_runs=[14, 13],
            method_2_runs=[1, 2],

            method_1_run_trim=[0, 0],
            method_2_run_trim=[4, 5],

            method_1_marker_errors=LTR_rural_for_clicked,
            method_2_marker_errors=Pix4D_VirLTR_rural_clicked,

            method_1_marker_distances=[2, 54, 120, 197, 301, 398, 555, 651, 752], # new updated
            method_2_marker_distances=[2, 54, 120, 197, 301, 398, 555, 651, 752],
        ),
        "DRDC Rural Loop - Driven": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/LTR/ltr_rural/graph", 
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/VirLTR/virtr_rural_driven/graph",

            method_1_runs=[14],
            method_2_runs=[1],

            method_1_run_trim=[0],
            method_2_run_trim=[5],

            method_1_marker_errors=LTR_rural_for_driven,
            method_2_marker_errors=Pix4D_VirLTR_rural_driven,

            method_1_marker_distances=[2, 54, 120, 197, 301, 398, 555, 651, 752], # new updated same as above
            method_2_marker_distances=[2, 54, 120, 197, 301, 398, 555, 651, 752],
        ),
        "DRDC Urban Loop": dict(
            method_1="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/LTR/ltr_urban/graph", 
            method_2="/home/desiree/ASRL/vtr3/temp/Experiment3/final_DRDC_Posegraphs_Without_Data_Folders/VirLTR/new_virtr_urban_clicked/graph",

            method_1_runs=[1, 2, 3, 5],
            method_2_runs=[1, 4, 6, 7],

            method_1_run_trim=[0, 0, 0, 0],
            method_2_run_trim=[3, 3, 3, 3],

            method_1_marker_errors=LTR_urban,
            method_2_marker_errors=Pix4D_VirLTR_urban,

            method_1_marker_distances=[1, 107, 194, 300, 411, 505, 646, 775, 833, 898, 976], # new udated
            method_2_marker_distances=[1, 107, 194, 300, 411, 505, 646, 775, 833, 898, 976],
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

def get_run_trim_count(cfg, method_num, run_id):
    """
    Determine how many vertices to trim for a given method and run.
    Supports these config formats for key 'method_<num>_run_trim':
      - dict: { run_id: trim_count, ... } (keys can be ints or strings)
      - list: [trim_for_first_run, trim_for_second_run, ...] aligned with method_<num>_runs
      - int: single value applied to all runs
    Returns int (>=0).
    """
    key = f"method_{method_num}_run_trim"
    val = cfg.get(key, None)
    if val is None:
        return 0
    # dict case
    if isinstance(val, dict):
        if run_id in val:
            return int(val[run_id])
        # maybe string key
        if str(run_id) in val:
            return int(val[str(run_id)])
        return 0
    # list case: find index from runs list if available
    if isinstance(val, (list, tuple)):
        runs = cfg.get(f"method_{method_num}_runs", [])
        try:
            idx = runs.index(run_id)
            return int(val[idx])
        except (ValueError, IndexError):
            return 0
    # scalar int
    try:
        return int(val)
    except Exception:
        return 0

def compute_repeat_pte(graph, teach_mat, v_start, m, skip_vertices):
    """Compute path tracking error along the repeat and cumulative path length.
    The first `skip_vertices` vertices (and the edges leading to them) are excluded
    from the returned pose sequence and cumulative distance. This ensures trimming
    actually affects the plotted curves and computed metrics.
    """
    pose_vec = []
    times = []
    cum_len = []
    plen = 0.0

    iter_count = 0
    included_started = False
    repeat_z = []

    # Single pass: skip initial vertices, then collect poses/times and build cumulative length
    for v, e in TemporalIterator(v_start):
        if iter_count < skip_vertices:
            iter_count += 1
            continue

        r = v.T_v_w.r_ba_ina().copy()
        # accumulate Z for bias calculation (only for included vertices)
        repeat_z.append(r[2])

        # For the first included vertex, do NOT add the incoming edge length (that would count skipped motion).
        if e is not None and included_started:
            plen += np.linalg.norm(e.T.r_ba_ina())

        pose_vec.append(r)
        times.append(v.stamp / 1e9)
        cum_len.append(plen)
        included_started = True

    # If no included vertices, return empty arrays / nan rmse
    if len(pose_vec) == 0:
        return np.array([]), np.array([]), np.array([]), float('nan')

    pose_vec = np.array(pose_vec)
    times = np.array(times)
    cum_len = np.array(cum_len)

    # Z-bias adjustment (align repeat's mean Z with teach path mean, except for method 1 where bias=0)
    if m == 1 or len(repeat_z) == 0:
        z_bias = 0.0
    else:
        z_bias = np.mean(teach_mat[:, 2]) - np.mean(repeat_z)

    # Apply z_bias and compute distances to teach path
    dists = []
    for r in pose_vec:
        r_adj = r.copy()
        r_adj[2] += z_bias
        dists.append(vtr_path.signed_distance_to_path(r_adj, teach_mat))
    dists = np.array(dists)

    # Sort by time to ensure monotonic cum_len and dists
    order = np.argsort(times)
    pose_vec = pose_vec[order]
    dists = dists[order]
    cum_len = cum_len[order]

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

    default_xy_colors = ['red', 'blue', 'green', 'orange']
    default_pte_colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'moccasin']

    method_data = {}
    for m in methods:
        graph = build_graph(cfg[f"method_{m}"])
        teach_mat = teach_path_matrix(graph)
        runs = cfg.get(f"method_{m}_runs", [])
        repeats = []
        for rid, v_start in iter_repeats(graph, runs):
            trim = get_run_trim_count(cfg, m, rid)
            pose_vec, cum_len, dists, _ = compute_repeat_pte(graph, teach_mat, v_start, m, skip_vertices=trim)
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
        teach_x, teach_y = zip(*method_data[m]["teach_xy"])
        ax_xy.plot(teach_x, teach_y, linestyle=':', color=color, label=f"{method_labels[m-1]} Teach")
        for rid, pose_vec, _, _ in method_data[m]["repeats"]:
            ax_xy.plot(pose_vec[:, 0], pose_vec[:, 1], '.', color=color, label=f"{method_labels[m-1]} Repeat {rid}")
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
        max_avg = np.max(np.abs(average_dists)) if average_dists.size else float('nan')

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
                f"{method_labels[m-1]} Avg PTE Curve (Overall RMSE={rmse_all:.3f} m, Overall Max={max_all:.3f} m)"
            )
        )
        print(f"{method_labels[m-1]} PTE Avg "
                f"(AvgCurve RMSE={rmse_avg:.3f} m, Max={max_avg:.3f} m | "
                f"Overall RMSE={rmse_all:.3f} m, Max={max_all:.3f} m | "
                f"PTE@Marks RMSE={rmse_at_marks:.3f} m, Max={max_at_marks:.3f} m)")
        
        # ---------- Scatter of all hardcoded marker measurements (if provided) ----------
        if markers and distances:
            flat_vals = [v for mlist in markers.values() for v in mlist]
            rmse_hard = np.sqrt(np.mean(np.square(flat_vals))) if flat_vals else float('nan')
            max_hard  = max((abs(v) for v in flat_vals), default=float('nan'))

            marker_color = default_xy_colors[(m - 1) % len(default_xy_colors)]
            x_vals = []
            y_vals = []
            sorted_keys = sorted(markers.keys(), key=lambda k: int(k))
            for i, key in enumerate(sorted_keys):
                errs = markers[key]
                if i < len(distances):
                    x_vals.extend([distances[i]] * len(errs))
                    y_vals.extend(errs)

            ax_pte.scatter(
                x_vals, y_vals, s=80, marker='x', linewidths=1.2, alpha=0.9,
                color=marker_color, zorder=10,
                label=(f"{method_labels[m-1]} Marker Measurements "
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
    for all its paths. For each path (grouped on the x-axis), each method's errors are shown side‐by-side.
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

def compute_and_print_metrics():
    # Iterate over all experiments and their paths
    for exp_name, paths in EXPERIMENTS.items():
        print("\n======================================================")
        print(f"Experiment: {exp_name}")
        print("======================================================")
        for path_name, cfg in paths.items():
            print(f"\n--- Path: {path_name} ---")
            # Find all methods (keys like "method_X")
            method_nums = []
            for key in cfg.keys():
                if key.startswith("method_") and key.count("_") == 1:
                    try:
                        method_nums.append(int(key.split("_")[1]))
                    except ValueError:
                        continue
            method_nums = sorted(method_nums)
            for m in method_nums:
                # Build posegraph and teach path information
                graph = build_graph(cfg[f"method_{m}"])
                teach_mat = teach_path_matrix(graph)
                runs = cfg.get(f"method_{m}_runs", [])
                all_pose_errors = []
                marker_interp_errors = []
                # Process each run in the posegraph
                for rid, v_start in iter_repeats(graph, runs):
                    trim = get_run_trim_count(cfg, m, rid)
                    _, cum_len, dists, _ = compute_repeat_pte(graph, teach_mat, v_start, m, skip_vertices=trim)
                    if dists.size == 0:
                        continue
                    all_pose_errors.extend(dists.tolist())
                    marker_dists = cfg.get(f"method_{m}_marker_distances", None)
                    if marker_dists is not None:
                        # Only consider markers that fall within the cumulative path length
                        valid_markers = [md for md in marker_dists if md <= cum_len[-1]]
                        if valid_markers:
                            interp_errs = np.interp(valid_markers, cum_len, dists)
                            marker_interp_errors.extend(interp_errs.tolist())
                # Compute statistics for the posegraph data (all runs)
                if all_pose_errors:
                    all_arr = np.array(all_pose_errors)
                    rmse_pose = np.sqrt(np.mean(np.square(all_arr))) 
                    max_pose = max(abs(all_arr)) #    max_error = max(abs(v) for v in dist)

                else:
                    rmse_pose = float('nan')
                    max_pose = float('nan')
                # Compute statistics for posegraph data at marker distances
                if marker_interp_errors:
                    marker_arr = np.array(marker_interp_errors)
                    rmse_marker_pose = np.sqrt(np.mean(np.square(marker_arr)))
                    max_marker_pose = max(abs(marker_arr))
                    mean_marker_pose = np.mean(np.abs(marker_arr))
                else:
                    rmse_marker_pose = max_marker_pose = mean_marker_pose = float('nan')
                # Compute statistics from hardcoded marker errors (if provided)
                hardcoded = cfg.get(f"method_{m}_marker_errors", None)
                if hardcoded:
                    flat_vals = []
                    for k in hardcoded:
                        flat_vals.extend(hardcoded[k])
                    flat_arr = np.array(flat_vals) if flat_vals else np.array([])
                    if flat_arr.size > 0:
                        rmse_hard = np.sqrt(np.mean(np.square(flat_arr)))
                        max_hard = max(abs(flat_arr))
                    else:
                        rmse_hard = max_hard = float('nan')
                else:
                    rmse_hard = max_hard = float('nan')
                print(f"\nMethod {m}:")
                print(f"  Hardcoded marks    -> RMSE: {rmse_hard:.3f} m, Max Error: {max_hard:.3f} m")
                print(f"  Posegraph (all)    -> RMSE: {rmse_pose:.3f} m, Max Error: {max_pose:.3f} m")
                print(f"  Posegraph (markers)-> RMSE: {rmse_marker_pose:.3f} m, Max Error: {max_marker_pose:.3f} m, Mean Error: {mean_marker_pose:.3f} m")

if __name__ == "__main__":
    # Call the metrics printing function
    #compute_and_print_metrics()

    # Generate individual comparison plots for each experiment and path.
    for exp_name, paths in EXPERIMENTS.items():
        for path_name, cfg in paths.items():
            process_experiment(exp_name, path_name, cfg)

    # Call the summary marker box plots function using the EXPERIMENTS dictionary.
    summary_marker_box_plots(EXPERIMENTS) 