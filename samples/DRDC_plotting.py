import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import pdb

"""
TODO:
 x merge partitioned trials
 x plot merged trials
 x plot discounted region
 x discount actual data (mask)
 x box plots
"""

pkl_dir = "/home/desiree/ASRL/vtr3/vtr3_posegraph_tools/vtr3_pose_graph/samples/pkls"

# raw trials
r_office_LTR = []
r_office_VirTR = []
r_urban_LTR = []
r_urban_VirTR = []
r_rural_LTR = []
r_rural_VirTR = []
r_rural_VirTR2 = []

# finalized (merged) data
office_LTR = []
office_VirTR = []
urban_LTR = []
urban_VirTR = []
rural_LTR = []
rural_VirTR = []
rural_VirTR2 = []

for fname in os.listdir(pkl_dir):
    if fname.endswith(".pkl"):
        path = os.path.join(pkl_dir, fname)
        with open(path, "rb") as f:
            data = pkl.load(f)

        if fname.startswith("Office_LTR"):
            r_office_LTR.append(data)
        if fname.startswith("Office_VirTR"):
            r_office_VirTR.append(data)
        if fname.startswith("Urban_LTR"):
            r_urban_LTR.append(data)
        if fname.startswith("Urban_VirTR"):
            r_urban_VirTR.append(data)
        if fname.startswith("Rural_LTR"):
            r_rural_LTR.append(data)
        if fname.startswith("Rural_VirTR (clicked)"):
            r_rural_VirTR.append(data)
        if fname.startswith("Rural_VirTR (driven)"):
            r_rural_VirTR2.append(data)

r_office_LTR = sorted(r_office_LTR, key=lambda x: x[0])
urban_LTR = sorted(r_urban_LTR, key=lambda x: x[0])
r_rural_LTR = sorted(r_rural_LTR, key=lambda x: x[0])
office_VirTR = sorted(r_office_VirTR, key=lambda x: x[0])
urban_VirTR = sorted(r_urban_VirTR, key=lambda x: x[0])
rural_VirTR = sorted(r_rural_VirTR, key=lambda x: x[0])
rural_VirTR2 = sorted(r_rural_VirTR2, key=lambda x: x[0])

def merge_trials(t1, t2):
    shifted_dists = t2[2] + t1[2][-1]
    merged_trial = (t1[0], np.vstack((t1[1],t2[1])), np.hstack((t1[2],shifted_dists)),np.hstack((t1[3],t2[3])))
    return merged_trial   

# fix office LT&R 1,2 ->  [13,15],[12,18], 20, 21
office_LTR.append(merge_trials(r_office_LTR[0],r_office_LTR[1]))
office_LTR.append(merge_trials(r_office_LTR[2],r_office_LTR[3]))
office_LTR.append(r_office_LTR[4])
office_LTR.append(r_office_LTR[5])

rural_LTR.append(r_rural_LTR[0])
# rural_LTR.append(merge_trials(r_rural_LTR[1],r_rural_LTR[2]))
rural_LTR.append(r_rural_LTR[3])
rural_LTR.append(r_rural_LTR[4])

rural_VirTR.append(r_rural_VirTR2[0])

# discounted regions
office_discount = (515,600, 'office')
urban_discount = (990,1030, 'urban')
rural_discount = (297,355, 'rural')

# markers
dists_officeltr = np.array([0,45,134,210,286,362,439,698,790,877,960]) # removed 7 @ 517, 8 @ 600
markers_officeltr = np.array((
        [0,0,0,0],
        [-0.009525, -0.053975, -0.012700, -0.044450],
        [-0.022098, 0.000000, 0.012700, 0.009525],
        [0.031750, 0.076200, 0.082550, 0.044450],
        [-0.028575, -0.012700, -0.015875, -0.015875],
        [-0.038100, -0.022225, -0.022225, -0.019050],
        [0.012700, -0.009525, -0.009525, 0.015875],
        [0.060325, 0.088900, 0.079375, 0.079375],
        [0.028575, 0.028575, 0.028575, 0.050800],
        [-0.028575, -0.022225, 0.009525, 0.000000],
        [0.031750, 0.025400, 0.041275, 0.015875]
        ))

dists_officevirtr = np.array(([0,45,142,217,293,368,448,702,799,892,976]))
markers_officevirtr = np.array((
    [0,0,0,0],
    [0.1524, 0.14605, 0.12065, 0.17145],
    [0.022225, 0.01905, 0.03175, 0.0127],
    [0.06985, 0.08255, 0.06985, 0.10795],
    [0.1651, 0.149225, 0.15875, 0.15875],
    [-0.09525, -0.0889, -0.092075, -0.09525],
    [-0.098425, -0.130175, -0.117475, -0.1397],
    [-0.03175, -0.047625, -0.028575, -0.04445],
    [-0.0762, -0.034925, -0.066675, -0.0635],
    [-0.06985, -0.03175, -0.0381, -0.0381],
    [0.1905, 0.1905, 0.2159, 0.193675],
))


dists_urbanltr = np.array(([0,107,195,410,505,646,775,833,897,976])) #,1028]))
markers_urbanltr = np.array((
    [0,0,0,0],
    [-0.025400, -0.015875, 0.000000, -0.003175],
    [0.003175, 0.006350, 0.003175, 0.025400],
    [-0.012700, -0.066675, 0.047625, 0.044450],
    [-0.006350, -0.025400, 0.006350, 0.012700],
    [-0.019050, -0.028575, -0.012700, -0.028575],
    [0.050800, 0.031750, 0.063500, -0.022225],
    [0.012700, 0.015875, 0.019050, 0.022225],
    [-0.041275, -0.050800, -0.028575, -0.057150],
    [-0.009525, -0.012700, -0.015875, -0.015875],
    # [0.000000, -0.009525, -0.003175, -0.019050]
))

dists_urbanvirtr = np.array(([0,107,194,406,500,642,769,829,896,978])) #,1022]))
markers_urbanvirtr = np.array((
    [0.0508, 0.0508, 0.0508, 0.0508],
    [0.06985, 0.06985, 0.05715, 0.06985],
    [0.111125, 0.1016, 0.1016, 0.12065],
    [-0.0889, -0.1524, 0.117475, -0.2413],
    [-0.00635, -0.0254, -0.003175, -0.015875],
    [-0.08255, -0.0762, -0.079375, -0.0762],
    [-0.142875, -0.1524, -0.161925, -0.149225],
    [-0.127, -0.123825, -0.13335, -0.136525],
    [-0.136525, -0.130175, -0.136525, -0.1397],
    [-0.1397, -0.130175, -0.130175, -0.127],
    # [0.1016, 0.15875, 0.17145, 0.174625]
))


dists_ruralltr = np.array(([0,54, 119, 197, 397, 553, 650, 748]))
markers_ruralltr = np.array((
    [0,0,0,0],
    [0.101600, 0.088900, -0.079375, -0.050800],
    [0.028575, 0.123825, 0.161925, 0.114300],
    [-0.012700, -0.041275, 0.098425, -0.003175],
    [0.003175, 0.034925, 0.063500, 0.069850],
    [0.092075, 0.038100, 0.104775, 0.104775],
    [0.209550, 0.196850, 0.241300, 0.263525],
    [0.219075, 0.203200, 0.234950, 0.158750]
))

dists_ruralvirtr = np.array(([0,53, 118, 195, 375, 530, 627, 726]))
markers_ruralvirtr = np.array((
    [-0.0254, -0.0254],
    [-0.0508, -0.00635],
    [0.04445, 0.1143],
    [-0.0254, -0.0508],
    [-0.01905, -0.022225],
    [-0.3683, -0.4953],
    [-0.3175, -0.4699],
    [-0.1778, -0.1905],
))

office_markers = {
    'ltr_dist' : dists_officeltr,
    'ltr_markers' : markers_officeltr, 
    'virtr_dist' : dists_officevirtr,
    'virtr_markers' : markers_officevirtr
    }

urban_markers = {
    'ltr_dist' : dists_urbanltr,
    'ltr_markers' : markers_urbanltr, 
    'virtr_dist' : dists_urbanvirtr,
    'virtr_markers' : markers_urbanvirtr
    }

rural_markers = {
    'ltr_dist' : dists_ruralltr,
    'ltr_markers' : markers_ruralltr, 
    'virtr_dist' : dists_ruralvirtr,
    'virtr_markers' : markers_ruralvirtr
    }

def get_stats(repeats, discounted, marker_distances=None):
    """Mimic process_experiment | repeats is ex. office_LTR"""
    min_end = min(r[2][-1] for r in repeats if r[2].size > 0)
    common_x = np.linspace(0, min_end, 500)
    valid_mask = (common_x < discounted[0]) | (common_x > discounted[1])

    interpolated_dists = []
    valid_interpolated_dists = []
    for _, _, cum_len_run, dists_run in repeats:
        interp_d = np.interp(common_x, cum_len_run, dists_run)
        # valid_interp_d = interp_d
        # valid_interp_d[valid_mask==False] = np.nan
        interpolated_dists.append(interp_d)
        # valid_interpolated_dists.append(valid_interp_d)
        
    average_dists = np.mean(interpolated_dists, axis=0)
    valid_average_dists = np.mean(interpolated_dists, axis=0)

    rmse_avg = np.sqrt(np.mean(np.square(average_dists))) if average_dists.size else float('nan')
    max_avg  = np.max(np.abs(average_dists))           if average_dists.size else float('nan')
    # ---------- Overall metrics from ALL per-run samples (pooled, not averaged) ----------
    # pooled = np.concatenate([d for _, _, _, d in repeats if isinstance(d, np.ndarray) and d.size > 0]) \
    #             if any((d.size > 0) for _, _, _, d in repeats) else np.array([])
    discounted_arrays = []
    arrays = []
    for _, _, c, d in repeats:
        if isinstance(d, np.ndarray) and d.size > 0:
            # mask by discount region
            keep_mask = ~((c > discounted[0]) & (c < discounted[1]))
            discounted_arrays.append(d[keep_mask])
            arrays.append(d)

    if len(arrays) > 0:
        d_pooled = np.concatenate(discounted_arrays)
        pooled = np.concatenate(arrays)
    else:
        pooled = np.array([])
        d_pooled = np.array([])

    rmse_all = np.sqrt(np.mean(pooled**2)) if pooled.size > 0 else float('nan')
    max_all  = np.max(np.abs(pooled))      if pooled.size > 0 else float('nan')
    d_rmse_all = np.sqrt(np.mean(d_pooled**2)) if d_pooled.size > 0 else float('nan')
    d_max_all  = np.max(np.abs(d_pooled))      if d_pooled.size > 0 else float('nan')

    # ---------- PTE sampled at marker locations (pooled across runs) ----------
    rmse_at_marks = float('nan')
    max_at_marks  = float('nan')
    if marker_distances is not None and marker_distances.size > 0:
        mark_arrays = []
        for _, _, cum_len_run, dists_run in repeats:
            if cum_len_run.size == 0 or dists_run.size == 0:
                continue
            valid_md = marker_distances[
                (marker_distances <= cum_len_run[-1]) &
                ~((marker_distances > discounted[0]) & (marker_distances < discounted[1]))
            ]            
            if valid_md.size == 0:
                continue
            samples = np.interp(valid_md, cum_len_run, dists_run)
            mark_arrays.append(samples)
        if mark_arrays:
            mark_samples_flat = np.concatenate(mark_arrays)
            if mark_samples_flat.size > 0:
                rmse_at_marks = np.sqrt(np.mean(mark_samples_flat**2))
                max_at_marks  = np.max(np.abs(mark_samples_flat))

    stats = {
        'common_x' : common_x,
        'average_dists' : average_dists,
        'rmse_all' : rmse_all,
        'max_all' : max_all,
        'discounted_rmse_all' : d_rmse_all,
        'discounted_max_all' : d_max_all,
        'rmse_at_marks'       : rmse_at_marks,
        'max_at_marks'        : max_at_marks,
    }

    return stats

def get_marker_stats(distances, markers):
    flat_vals = np.ravel(markers)
    rmse_hard = np.sqrt(np.mean(np.square(flat_vals)))
    max_hard  = max((abs(v) for v in flat_vals)) # Q: per trials or over smoothed?

    return rmse_hard, max_hard 

def summary_marker_box_plot(office_markers, urban_markers, rural_markers):
    """
    Plot box plots for marker errors across all environments.
    Converts numpy array marker data to the format expected by the plotting logic.
    """
    from matplotlib.patches import Patch
    
    # Convert numpy arrays to dict format for compatibility
    def markers_array_to_dict(markers_array):
        """Convert 2D numpy array to dict with string keys."""
        return {str(i+1): list(row) for i, row in enumerate(markers_array)}
    
    # Structure data like the original EXPERIMENTS dict
    experiments_data = {
        "LT&R vs. VirLT&R (Pix4D)": {
            "Urban-R": {
                'method_1_marker_errors': markers_array_to_dict(office_markers['ltr_markers']),
                'method_2_marker_errors': markers_array_to_dict(office_markers['virtr_markers']),
            },
            "Structured-R": {
                'method_1_marker_errors': markers_array_to_dict(urban_markers['ltr_markers']),
                'method_2_marker_errors': markers_array_to_dict(urban_markers['virtr_markers']),
            },
            "Sparse-R": {
                'method_1_marker_errors': markers_array_to_dict(rural_markers['ltr_markers']),
                'method_2_marker_errors': markers_array_to_dict(rural_markers['virtr_markers']),
            },
        }
    }
    
    colors = ['r', 'b', 'green', 'orange']
    
    for exp_name, paths in experiments_data.items():
        per_path_method_data = []
        path_names = []
        
        for path_name, cfg in paths.items():
            markers_data = {}
            
            # Multi-method case: look for keys like "method_?_marker_errors"
            method_keys = [key for key in cfg.keys() 
                          if key.startswith("method_") and "marker_errors" in key]
            
            if method_keys:
                parsed_labels = ["LT&R", "VirLT&R (Pix4D)"]  # Explicit labels
                sorted_keys = sorted(method_keys, key=lambda key: int(key.split("_")[1]))
                
                for i, key in enumerate(sorted_keys):
                    label = parsed_labels[i] if i < len(parsed_labels) else f"Method {i+1}"
                    vals = []
                    for mvals in cfg[key].values():
                        vals.extend(mvals)
                    markers_data[label] = vals
            
            if len(markers_data) >= 2:
                per_path_method_data.append(markers_data)
                path_names.append(path_name)
        
        if not per_path_method_data:
            continue
        
        # Method labels
        all_method_labels = ["LT&R", "VirLT&R (Pix4D)"]
        num_methods = len(all_method_labels)
        method_color = {label: colors[i % len(colors)]
                       for i, label in enumerate(all_method_labels)}
        
        # Define box group configuration (tighter spacing)
        group_width = 0.12
        group_gap = 0.4
        
        # Larger text settings for this figure
        title_fs = 22
        label_fs = 22
        tick_fs = 20
        legend_fs = 18

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xticklabels(all_method_labels, fontsize=tick_fs)
        ax.set_ylabel("Marker Error (m)", fontsize=label_fs)
        #ax.set_title(f"{exp_name} – Marker Errors Summary", fontsize=title_fs)
        ax.set_ylabel("Marker Error (m)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, linestyle="--", linewidth=1.5, color="darkgrey")
        ax.tick_params(axis='y', labelsize=tick_fs)
        
        # Collect positions and data for each method
        method_to_positions = {label: [] for label in all_method_labels}
        method_to_data = {label: [] for label in all_method_labels}
        x_ticks = []
        
        for i, pdata in enumerate(per_path_method_data):
            group_center = (i + 1) * group_gap
            x_ticks.append(group_center)

            for j, m_label in enumerate(all_method_labels):
                offset = (j - (num_methods - 1) / 2) * group_width
                method_to_positions[m_label].append(group_center + offset)
                method_to_data[m_label].append(pdata.get(m_label, []))
        
        # Plot boxes for each method
        for m_label in all_method_labels:
            bp = ax.boxplot(
                method_to_data[m_label],
                positions=method_to_positions[m_label],
                widths=group_width * 0.8,
                patch_artist=True,
                manage_ticks=False
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(method_color[m_label])
                patch.set_edgecolor("black")
            for element in ("whiskers", "caps", "medians"):
                for artist in bp[element]:
                    artist.set_color("black")

        # Set x-axis ticks
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(path_names, rotation=0, ha='center')
        ax.tick_params(axis='x', labelsize=tick_fs)

        # Build legend
        legend_handles = [
            Patch(facecolor=method_color[m_label], edgecolor='black', label=m_label)
            for m_label in all_method_labels
        ]
        ax.legend(handles=legend_handles, loc='best', fontsize=legend_fs)

        plt.tight_layout()
        # Optional if labels are still cramped:
        # plt.subplots_adjust(bottom=0.18)
        plt.savefig('box_plot', dpi=300, bbox_inches='tight')
        plt.show()


def plot_experiment(ltr, virtr, markers, discounted):
    # Larger text settings for this figure
    title_fs = 20
    label_fs = 20
    tick_fs = 18
    legend_fs = 12
    fig_pte, ax_pte = plt.subplots()
    ax_pte.set_xlabel("Path Length (m)", fontsize=label_fs)
    ax_pte.set_ylabel("PTE (m)", fontsize=label_fs)
    ax_pte.grid(True)
    ax_pte.axhline(0, linestyle='--', linewidth=1.0, color='gray')
    ax_pte.tick_params(axis='y', labelsize=tick_fs)
    ax_pte.tick_params(axis='x', labelsize=tick_fs)



    if discounted[2] == 'office' or discounted[2] == 'rural':
        ax_pte.axvspan(discounted[0], discounted[1], color='grey', alpha=0.5, label="Operator Error")
    if discounted[2] == 'urban':
        ax_pte.axvspan(discounted[0], discounted[1], color='burlywood', alpha=0.5, label="Platform-Specific Miscalibration")

    ltr_stats = get_stats(ltr, discounted, marker_distances=markers['ltr_dist'])
    virtr_stats = get_stats(virtr, discounted, marker_distances=markers['virtr_dist'])

    ltr_marker_rmse, ltr_marker_max = get_marker_stats(markers['ltr_dist'], markers['ltr_markers'])
    virtr_marker_rmse, virtr_marker_max = get_marker_stats(markers['virtr_dist'], markers['virtr_markers'])

    # get internally estimated distance at markers
    # idx = np.searchsorted(ltr_stats['common_x'], markers['ltr_dist'])
    # pte_marks = ltr_stats['average_dists'][idx-1]
    print(f"LT&R   PTE@Marks RMSE={ltr_stats['rmse_at_marks']:.3f} m, Max={ltr_stats['max_at_marks']:.3f} m)")
    # idx = np.searchsorted(virtr_stats['common_x'], markers['virtr_dist'])
    # pte_marks = virtr_stats['average_dists'][idx-1]
    print(f"VirT&R PTE@Marks RMSE={virtr_stats['rmse_at_marks']:.3f} m, Max={virtr_stats['max_at_marks']:.3f} m)")

    # LT&R
    ax_pte.plot(ltr_stats['common_x'], ltr_stats['average_dists'], linewidth=1.5, color='lightcoral',
        label=(f"LT&R Discounted Avg PTE Curve (RMSE={ltr_stats['discounted_rmse_all']:.3f} m, Max={ltr_stats['discounted_max_all']:.3f} m)"))
    ax_pte.plot(ltr_stats['common_x'], ltr_stats['average_dists'], linewidth=1.5, color='lightcoral',
        label=(f"LT&R Avg PTE Curve (RMSE={ltr_stats['rmse_all']:.3f} m, Max={ltr_stats['max_all']:.3f} m)"))
    ax_pte.scatter(markers['ltr_dist'], markers['ltr_markers'][:,0], s=80, marker='x', linewidths=1.2, alpha=0.9, c='r', zorder=10,
                    label=(f"LT&R Marker Measurements "
                    f"(RMSE={ltr_marker_rmse:.3f} m, Max={ltr_marker_max:.3f} m)"
                    ))
    for i in range(1,markers['ltr_markers'].shape[1]):
        ax_pte.scatter(markers['ltr_dist'], markers['ltr_markers'][:,i], s=80, marker='x', linewidths=1.2, alpha=0.9, c='r', zorder=10)
    

    # VirT&R
    ax_pte.plot(virtr_stats['common_x'], virtr_stats['average_dists'], linewidth=1.5, color='lightskyblue',
        label=(f"VirT&R Discounted Avg PTE Curve (RMSE={virtr_stats['discounted_rmse_all']:.3f} m, Max={virtr_stats['discounted_max_all']:.3f} m)"))
    # ax_pte.plot(virtr_stats['common_x'], virtr_stats['average_dists'], linewidth=1.5, color='lightskyblue',
    #     label=(f"Avg PTE Curve (RMSE={virtr_stats['rmse_all']:.3f} m, Max={virtr_stats['max_all']:.3f} m)"))
    ax_pte.scatter(markers['virtr_dist'], markers['virtr_markers'][:,0], s=80, marker='x', linewidths=1.2, alpha=0.9, c='b', zorder=10,
                label=(f"VirT&R Marker Measurements "
                f"(RMSE={virtr_marker_rmse:.3f} m, Max={virtr_marker_max:.3f} m)"
                ))
    for i in range(1,markers['virtr_markers'].shape[1]):
        ax_pte.scatter(markers['virtr_dist'], markers['virtr_markers'][:,i], s=80, marker='x', linewidths=1.2, alpha=0.9, c='b', zorder=10)
    

    ax_pte.legend(fontsize=legend_fs)
    plt.show()
    fig_pte.savefig(discounted[2], dpi=300, bbox_inches='tight')


def summary_pte_plots(office_ltr, office_virtr, office_markers, office_discount,
                      urban_ltr, urban_virtr, urban_markers, urban_discount,
                      rural_ltr, rural_virtr, rural_markers, rural_discount):
    """
    Create a single figure with 3 subplots showing PTE curves across all environments.
    Each subplot shows LT&R vs. VirT&R for one environment.
    """
    environments = [
        ("Urban-R", office_ltr, office_virtr, office_markers, office_discount),
        ("Structured-R", urban_ltr, urban_VirTR, urban_markers, urban_discount),
        ("Sparse-R", rural_ltr, rural_VirTR, rural_markers, rural_discount),
    ]
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 12))
    axs = np.atleast_1d(axs).flatten()
    
    title_fs = 20
    label_fs = 18
    tick_fs = 16
    legend_fs = 10
    
    for idx, (env_name, ltr, virtr, markers, discounted) in enumerate(environments):
        ax_pte = axs[idx]
        
        # Add shaded regions
        if discounted[2] == 'office' or discounted[2] == 'rural':
            ax_pte.axvspan(discounted[0], discounted[1], color='grey', alpha=0.5, label="Operator Error")
        if discounted[2] == 'urban':
            ax_pte.axvspan(discounted[0], discounted[1], color='burlywood', alpha=0.5, label="Platform-Specific Miscalibration")
        
        ltr_stats = get_stats(ltr, discounted, marker_distances=markers['ltr_dist'])
        virtr_stats = get_stats(virtr, discounted, marker_distances=markers['virtr_dist'])
        
        ltr_marker_rmse, ltr_marker_max = get_marker_stats(markers['ltr_dist'], markers['ltr_markers'])
        virtr_marker_rmse, virtr_marker_max = get_marker_stats(markers['virtr_dist'], markers['virtr_markers'])
        
        # LT&R curve
        ax_pte.plot(ltr_stats['common_x'], ltr_stats['average_dists'], linewidth=1.5, color='lightcoral',
            label=(f"LT&R"))
                    #label=(f"LT&R (RMSE={ltr_stats['discounted_rmse_all']:.3f} m, Max={ltr_stats['discounted_max_all']:.3f} m)"))

        # LT&R markers
        ax_pte.scatter(markers['ltr_dist'], markers['ltr_markers'][:,0], s=80, marker='x', linewidths=1.2, alpha=0.9, c='r', zorder=10,
                       label=(f"LT&R Markers"))
                               #label=(f"LT&R Markers (RMSE={ltr_marker_rmse:.3f} m, Max={ltr_marker_max:.3f} m)"))

        for i in range(1, markers['ltr_markers'].shape[1]):
            ax_pte.scatter(markers['ltr_dist'], markers['ltr_markers'][:,i], s=80, marker='x', linewidths=1.2, alpha=0.9, c='r', zorder=10)
        
        # VirT&R curve
        ax_pte.plot(virtr_stats['common_x'], virtr_stats['average_dists'], linewidth=1.5, color='lightskyblue',
            label=(f"VirLT&R"))
                    #label=(f"VirT&R (RMSE={virtr_stats['discounted_rmse_all']:.3f} m, Max={virtr_stats['discounted_max_all']:.3f} m)"))

        # VirT&R markers
        ax_pte.scatter(markers['virtr_dist'], markers['virtr_markers'][:,0], s=80, marker='x', linewidths=1.2, alpha=0.9, c='b', zorder=10,
                       label=(f"VirLT&R Markers"))
                               #label=(f"VirT&R Markers (RMSE={virtr_marker_rmse:.3f} m, Max={virtr_marker_max:.3f} m)"))
        for i in range(1, markers['virtr_markers'].shape[1]):
            ax_pte.scatter(markers['virtr_dist'], markers['virtr_markers'][:,i], s=80, marker='x', linewidths=1.2, alpha=0.9, c='b', zorder=10)
        
        ax_pte.set_title(env_name, fontsize=label_fs)
        ax_pte.grid(True, linestyle='--', alpha=0.5)
        ax_pte.axhline(0, linestyle='--', linewidth=1.0, color='gray')
        ax_pte.tick_params(axis='y', labelsize=tick_fs)
        ax_pte.tick_params(axis='x', labelsize=tick_fs)
        ax_pte.legend(loc='best', fontsize=legend_fs)

    fig.supxlabel("Path Length (m)", fontsize=label_fs, y=0.02)
    ylab = fig.supylabel("PTE (m)", fontsize=label_fs)
    ylab.set_x(0.02)

    # Maximize usable subplot area
    fig.subplots_adjust(left=0.075, right=0.995, top=0.95, bottom=0.08, hspace=0.18)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, "PTE_Summary.png")
    fig.savefig(filename, dpi=300)
    plt.show()


# plot_experiment(office_LTR, office_VirTR, office_markers, office_discount)
# plot_experiment(urban_LTR, urban_VirTR, urban_markers, urban_discount)
# plot_experiment(rural_LTR, rural_VirTR, rural_markers, rural_discount)

# summary_marker_box_plot(office_markers, urban_markers, rural_markers)
summary_pte_plots(
    office_LTR, office_VirTR, office_markers, office_discount,
    urban_LTR, urban_VirTR, urban_markers, urban_discount,
    rural_LTR, rural_VirTR, rural_markers, rural_discount
)