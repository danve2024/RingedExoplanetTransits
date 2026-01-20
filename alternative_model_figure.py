from nested_sampling import call_model, ns, fixed_observations_duration
import numpy as np
import matplotlib.pyplot as plt
from ringless_nested_sampling import call as call_ringless
from oblate_planet_nested_sampling import call as call_oblate
from star_spots_nested_sampling import call as call_spot
from noise_nested_sampling import call as call_noise

def plot_residuals(axes, residuals, title='Residuals', color: str = 'red', pointer: str = 'o'):
    axes.plot(observed_phases, residuals, pointer,
                  color=color, markersize=3, alpha=0.5)
    axes.axhline(0, color='black', linestyle='-', linewidth=1)
    axes.set_ylabel(title)
    axes.grid(True)

main = call_model([0.002683, 84.290292, 90.000000, 0.807971, 45485031.127797, 0.416792, 2701646296.410850, 369052443.984327, 0.972302, 56.787311, 42.513863])
ringless = call_ringless([0.000070, 54.384045, 90.000000, 2.322188, 59650150.793948])
oblate = call_oblate([0.001, 28.401, 90.000, 3.056, 108331855.913, 0.702, 15.048])
spot = call_spot([0.003, 16.631, 90.000, 2.370, 58712009.773, 0.001, 77.530, 0.028, 1.543])
noise = call_noise([0.000, 58.880, 90.000, 0.984, 58715608.912, 0.002, 0.002])

main_fit = np.array(main[0])
main_phases = (main_fit[:, 0] - 0.5) * 2 * main[1] / fixed_observations_duration + 0.5

ringless_fit = np.array(ringless[0])
ringless_phases = (ringless_fit[:, 0] - 0.5) * 2 * ringless[1] / fixed_observations_duration + 0.5

oblate_fit = np.array(oblate[0])
oblate_phases = (oblate_fit[:, 0] - 0.5) * 2 * oblate[1] / fixed_observations_duration + 0.5

spot_fit = np.array(spot[0])
spot_phases = (spot_fit[:, 0] - 0.5) * 2 * spot[1] / fixed_observations_duration + 0.5

noise_fit = np.array(noise[0])
noise_phases = (noise_fit[:, 0] - 0.5) * 2 * noise[1] / fixed_observations_duration + 0.5

observed_phases = ns.observations[:, 0]
observed_magnitudes = ns.observations[:, 1]

# Main model
main_intersection = (main_phases >= 0) & (main_phases <= 1)

main_interp = np.interp(
    observed_phases,
    main_phases[main_intersection],
    main_fit[:, 1][main_intersection]
)
main_residuals = observed_magnitudes - main_interp

# Ringless model
ringless_intersection = (ringless_phases >= 0) & (ringless_phases <= 1)

ringless_interp = np.interp(
    observed_phases,
    ringless_phases[ringless_intersection],
    ringless_fit[:, 1][ringless_intersection]
)
ringless_residuals = observed_magnitudes - ringless_interp

# Oblate model
oblate_intersection = (oblate_phases >= 0) & (oblate_phases <= 1)

oblate_interp = np.interp(
    observed_phases,
    oblate_phases[oblate_intersection],
    oblate_fit[:, 1][oblate_intersection]
)
oblate_residuals = observed_magnitudes - oblate_interp

# Spot model
spot_intersection = (spot_phases >= 0) & (spot_phases <= 1)

spot_interp = np.interp(
    observed_phases,
    spot_phases[spot_intersection],
    spot_fit[:, 1][spot_intersection]
)
spot_residuals = observed_magnitudes - spot_interp

# Noise model
noise_intersection = (noise_phases >= 0) & (noise_phases <= 1)

noise_interp = np.interp(
    observed_phases,
    noise_phases[noise_intersection],
    noise_fit[:, 1][noise_intersection]
)
noise_residuals = observed_magnitudes - noise_interp

# Plotting
fig, (ax_main, res_main, res_ringless, res_oblate, res_noise, res_spot) = plt.subplots(6, 1, figsize=(14, 20),
                                        height_ratios=[3, 1, 1, 1, 1, 1],
                                        sharex=True)

ax_main.plot(observed_phases, observed_magnitudes, 'o', color='gray', markersize=3, alpha=0.5, label='Observed Data')
ax_main.plot(main_phases[main_intersection], main_fit[:, 1][main_intersection],
         '-', color='red', linewidth=4, label='Ringed Model')
ax_main.plot(ringless_phases[ringless_intersection], ringless_fit[:, 1][ringless_intersection],
         '--', color='orange', linewidth=2, label='Ringless Model')
ax_main.plot(oblate_phases[oblate_intersection], oblate_fit[:, 1][oblate_intersection],
         '-.', color='green', linewidth=2, label='Oblate Planet Model')
ax_main.plot(noise_phases[noise_intersection], noise_fit[:, 1][noise_intersection],
         ':', color='blue', linewidth=2, label='Observational Noise Model')
ax_main.plot(spot_phases[spot_intersection], spot_fit[:, 1][spot_intersection],
         '-', color='purple', linewidth=2, label='Starspots/Faculae Model')

ax_main.set_ylabel('Magnitude Change')
ax_main.legend()
ax_main.set_title('Best-Fit Model vs. Observed Data')
ax_main.grid(True)
ax_main.invert_yaxis()

# Residuals
plot_residuals(res_main, main_residuals, "Ringed Model Residuals")
plot_residuals(res_ringless, ringless_residuals, "Ringless Model Residuals", 'orange', '^')
plot_residuals(res_oblate, oblate_residuals, "Oblate Planet Model Residuals", 'green', 's')
plot_residuals(res_noise, noise_residuals, "Obs. Noise Model Residuals", 'blue', 'D')
plot_residuals(res_spot, spot_residuals, "Starpots/Faculae Residuals", 'purple', 'v')
res_spot.set_xlabel('Phase')

plt.tight_layout()
plt.savefig('alternative_model_figure.png', dpi=300)
plt.close()