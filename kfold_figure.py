import numpy as np
import matplotlib.pyplot as plt
from nested_sampling import call_model, fixed_observations_duration, ns

n = 14
k = 5

offset = 4

labels = ['Fold 0 (Ingress)', 'Fold 1 (Pre-Mid)', 'Fold 2 (Midtransit)', 'Fold 3 (Post-Mid)', 'Fold 4 (Egress)']
colors = ['red', 'orange', 'green', 'blue', 'purple']
line_styles = ['-', '--', '-.', ':', '-']
point_styles = ['o', '^', 's', 'D', 'v']

kfold_params = [
    {'exoplanet_orbit_eccentricity': 0.008672, 'exoplanet_orbit_inclination': 37.765317,
     'exoplanet_longitude_of_ascending_node': 90.0, 'exoplanet_argument_of_periapsis': 12.327428,
     'exoplanet_radius': 58534339.43, 'ring_eccentricity': 0.801089, 'ring_semi_major_axis': 3478119296.88,
     'ring_width': 87864478.66, 'ring_obliquity': 0.705084, 'ring_azimuthal_angle': 126.518057,
     'ring_argument_of_periapsis': 51.954711},
    {'exoplanet_orbit_eccentricity': 0.000045, 'exoplanet_orbit_inclination': 72.089298,
     'exoplanet_longitude_of_ascending_node': 90.0, 'exoplanet_argument_of_periapsis': 14.683721,
     'exoplanet_radius': 58374821.51, 'ring_eccentricity': 0.543706, 'ring_semi_major_axis': 3474018323.96,
     'ring_width': 72991412.84, 'ring_obliquity': 1.549794, 'ring_azimuthal_angle': 117.856305,
     'ring_argument_of_periapsis': 73.633800},
    {'exoplanet_orbit_eccentricity': 0.001660, 'exoplanet_orbit_inclination': 77.281416,
     'exoplanet_longitude_of_ascending_node': 90.0, 'exoplanet_argument_of_periapsis': 4.057686,
     'exoplanet_radius': 46571179.29, 'ring_eccentricity': 0.435136, 'ring_semi_major_axis': 2451450186.30,
     'ring_width': 263816768.90, 'ring_obliquity': 1.584390, 'ring_azimuthal_angle': 150.373378,
     'ring_argument_of_periapsis': 134.489960},
    {'exoplanet_orbit_eccentricity': 0.002363, 'exoplanet_orbit_inclination': 86.650391,
     'exoplanet_longitude_of_ascending_node': 90.0, 'exoplanet_argument_of_periapsis': 9.076583,
     'exoplanet_radius': 55917121.21, 'ring_eccentricity': 0.621450, 'ring_semi_major_axis': 3389281374.44,
     'ring_width': 478925032.23, 'ring_obliquity': 4.623564, 'ring_azimuthal_angle': 56.818595,
     'ring_argument_of_periapsis': 132.314381},
    {'exoplanet_orbit_eccentricity': 0.000450, 'exoplanet_orbit_inclination': 87.841136,
     'exoplanet_longitude_of_ascending_node': 90.0, 'exoplanet_argument_of_periapsis': 5.766493,
     'exoplanet_radius': 48510939.00, 'ring_eccentricity': 0.793939, 'ring_semi_major_axis': 1878159559.34,
     'ring_width': 184417118.77, 'ring_obliquity': 3.907295, 'ring_azimuthal_angle': 111.840848,
     'ring_argument_of_periapsis': 161.295060}
]

fig = plt.figure(figsize=(14, 20))
gs = fig.add_gridspec(k + 1, 1, height_ratios=[3] + [1] * k, hspace=0.3)

ax_main = fig.add_subplot(gs[0])
residual_axes = [fig.add_subplot(gs[i + 1], sharex=ax_main) for i in range(k)]

segment_boundaries = np.linspace(0, 1, k + 1)
all_observations = []

for i in range(k):
    best_fit_model_output, transit_duration, _ = call_model(list(kfold_params[i].values()))
    model_lc = np.array(best_fit_model_output)
    current_train_fold = offset + i
    all_obs_indices = [0, 1, 2, 3, 9, 10, 11, 12, 13]
    gray_indices = [current_train_fold]
    color_indices = [idx for idx in all_obs_indices if idx != current_train_fold]
    gray_data = ns.kfold_split(n, gray_indices)
    color_data = ns.kfold_split(n, color_indices)

    match i:
        case 1:
            d = 0.496
        case 3:
            d = 0.493
        case _:
            d = 0.5

    all_obs = ns.kfold_split(n, all_obs_indices)
    all_p = (all_obs[:, 0] - 0.5) * 0.5 * fixed_observations_duration / transit_duration + d
    x_min, x_max = np.min(all_p), np.max(all_p)

    c_phases = (color_data[:, 0] - 0.5) * 0.5 * fixed_observations_duration / transit_duration + d
    c_norm_p = (c_phases - x_min) / (x_max - x_min)
    c_model_interp = np.interp(c_phases, model_lc[:, 0], model_lc[:, 1])
    c_res = color_data[:, 1] - c_model_interp

    g_phases = (gray_data[:, 0] - 0.5) * 0.5 * fixed_observations_duration / transit_duration + d
    g_norm_p = (g_phases - x_min) / (x_max - x_min)
    g_model_interp = np.interp(g_phases, model_lc[:, 0], model_lc[:, 1])
    g_res = gray_data[:, 1] - g_model_interp

    mask_mod = (model_lc[:, 0] >= x_min) & (model_lc[:, 0] <= x_max)
    norm_mod_p = (model_lc[mask_mod, 0] - x_min) / (x_max - x_min)
    mod_m = model_lc[mask_mod, 1]

    all_observations.append((c_norm_p, color_data[:, 1], g_norm_p, gray_data[:, 1]))

    if colors[i] == 'red' and line_styles[i] == '-':
        width = 4
    else:
        width = 2

    ax_main.plot(norm_mod_p, mod_m, color=colors[i], linestyle=line_styles[i], linewidth=width, zorder=10,
                 label=f'Fold {i}: {labels[i]}')

    ax_res = residual_axes[i]

    segment_c_data_for_plot = []
    segment_c_res_for_plot = []
    other_c_data_for_plot = []
    other_c_res_for_plot = []

    for seg_idx in range(k):
        segment_mask = (c_norm_p >= segment_boundaries[seg_idx]) & (c_norm_p <= segment_boundaries[seg_idx + 1])
        segment_c_norm_p = c_norm_p[segment_mask]
        segment_c_res = c_res[segment_mask]

        if len(segment_c_norm_p) > 0:
            if seg_idx == i:
                segment_c_data_for_plot.append((segment_c_norm_p, segment_c_res))
            else:
                other_c_data_for_plot.append((segment_c_norm_p, segment_c_res))

    for seg_norm_p, seg_res in other_c_data_for_plot:
        ax_res.plot(seg_norm_p, seg_res, color='gray', marker='o', markersize=3, linestyle='', alpha=0.5)

    for seg_norm_p, seg_res in segment_c_data_for_plot:
        ax_res.plot(seg_norm_p, seg_res, color=colors[i], marker=point_styles[i], markersize=4, linestyle='', alpha=0.7)

    ax_res.axhline(0, color='black', linestyle='-', linewidth=1)
    ax_res.set_ylabel(f'Fold {current_train_fold - offset}\nResiduals')
    ax_res.grid(True, alpha=0.2)

for i in range(k):
    c_norm_p, c_data, g_norm_p, g_data = all_observations[i]

    for seg_idx in range(k):
        segment_mask = (c_norm_p >= segment_boundaries[seg_idx]) & (c_norm_p <= segment_boundaries[seg_idx + 1])
        segment_c_norm_p = c_norm_p[segment_mask]
        segment_c_data = c_data[segment_mask]

        if len(segment_c_norm_p) > 0 and seg_idx == i:
            ax_main.plot(segment_c_norm_p, segment_c_data, color=colors[i],
                         marker=point_styles[i], markersize=4, linestyle='', alpha=0.5, zorder=5)

for i in range(1, k):
    ax_main.axvline(x=segment_boundaries[i], color='gray', linestyle=':', alpha=0.3, linewidth=1, zorder=1)
    for ax_res in residual_axes:
        ax_res.axvline(x=segment_boundaries[i], color='gray', linestyle=':', alpha=0.2, linewidth=0.5)

for i in range(k):
    segment_center = (segment_boundaries[i] + segment_boundaries[i + 1]) / 2
    ax_main.text(segment_center, ax_main.get_ylim()[1] - 0.02 * (ax_main.get_ylim()[1] - ax_main.get_ylim()[0]),
                 labels[i], ha='center', va='top', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[i], alpha=0.2))

ax_main.set_title("Best-Fit Light Curves for K-Fold Cross-Validation", fontsize=14, fontweight='bold')
ax_main.set_ylabel('Magnitude Change')
ax_main.invert_yaxis()
ax_main.set_xlim(0, 1)

from matplotlib.lines import Line2D

legend_elements = []
for i in range(k):
    legend_elements.append(Line2D([0], [0], color=colors[i], linestyle=line_styles[i], linewidth=2,
                                  label=f'Fold {i} Model'))
for i in range(k):
    legend_elements.append(Line2D([0], [0], color=colors[i], marker=point_styles[i], markersize=6, linestyle='',
                                  label=f'Fold {i} Data and Residuals', markeredgecolor='black'))

ax_main.legend(handles=legend_elements, loc='upper center', fontsize='small')

residual_axes[-1].set_xlabel('Normalized Phase')
plt.tight_layout()
plt.savefig('kfold_figure.png')