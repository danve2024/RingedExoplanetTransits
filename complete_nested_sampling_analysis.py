import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import corner
import sys
from typing import Union, List, Dict, Tuple
from dynesty import utils as dyfunc
from formulas import roche_sma_max, roche_density, volume
from main import calculate_data, defaults
from models import quadratic_star_model
from units import *
from space import CustomStarModel
import os
import json
import random
from collections import OrderedDict

fixed_pixel_size = defaults['pixel_size']
fixed_star_radius = 1.28 * sun_radius  # Grouffal et al., 2022
fixed_star_log_g = 4.3  # from TICv8
fixed_star_coefficients = [0.0678, 0.188]  # Grant & Wakeford, 2024
fixed_observations_duration = 80 * hrs
fixed_star_object = CustomStarModel(quadratic_star_model, fixed_star_radius,
                                    fixed_star_log_g, fixed_star_coefficients,
                                    fixed_pixel_size)
fixed_exoplanet_mass = 12 * earth_mass  # Santerne et al., 2019
fixed_exoplanet_period = 542.08 * days  # Santerne et al., 2019
fixed_exoplanet_sma = 1.377 * au
fixed_specific_absorption_coefficient = defaults['specific_absorption_coefficient']

min_radius =  2000 * km
max_radius = 9.2 * 6_400 * km
max_semi_major_axis_max = roche_sma_max(max_radius, fixed_exoplanet_mass/volume(min_radius), 0, roche_density(fixed_exoplanet_mass, max_radius, 0.9))
max_width_max = max_semi_major_axis_max - min_radius

specific_parameter_boundaries = {
    'exoplanet_orbit_eccentricity': (0, 0.4),
    'exoplanet_orbit_inclination': (0, 90),
    'exoplanet_longitude_of_ascending_node': (90 - 5e-8, 90 + 5e-8),
    'exoplanet_argument_of_periapsis': (0, 180),
    'exoplanet_radius': (min_radius, max_radius),
    'eccentricity': (0, 0.9),
    'semi_major_axis': (min_radius, max_semi_major_axis_max), # dynamic boundaries
    'width': (0, max_width_max), # dynamic boundaries
    'obliquity': (0, 90),
    'azimuthal_angle': (0, 180),
    'argument_of_periapsis': (0, 180)
}

class NestedSamplingAnalysis:
    def __init__(self, results_file: str = "nested_sampling_results.npz"):
        print("Loading nested sampling results...")
        self.results_file = results_file
        self.load_results()
        self.load_observations()

        # stdev=(b-a)/sqrt(12) for uniform distribution
        param_bounds = list(specific_parameter_boundaries.values())
        self.prior_stdev = np.array([(high - low) / np.sqrt(12) for low, high in param_bounds])
        
    def load_results(self):
        data = np.load(self.results_file)
        self.samples = data['samples']
        self.logwt = data['logwt']
        self.logz = data['logz']
        self.logl = data['logl']
        print(data.files)
        self.logzerr = data['logzerr'] if 'logzerr' in data.files else None

        finite_logz_idx = np.where(np.isfinite(self.logz))[0]
        self._logz_last_finite = self.logz[finite_logz_idx[-1]] if finite_logz_idx.size > 0 else np.nan
        if self.logzerr is not None:
            finite_logzerr_idx = np.where(np.isfinite(self.logzerr))[0]
            self._logzerr_last_finite = self.logzerr[finite_logzerr_idx[-1]] if finite_logzerr_idx.size > 0 else np.nan
        else:
            self._logzerr_last_finite = np.nan

        self.labels = [
            'Orbit Eccentricity',
            'Orbit Inclination, °',
            'Ex. Long. Asc. Node, °',
            'Ex. Arg. of Periapsis, °',
            'Exoplanet Radius, km',
            'Ring Eccentricity',
            'Ring Semi-Major Axis, km',
            'Ring Width, km',
            'Ring Obliquity, °',
            'Ring Azimuthal Angle, °',
            'Ring Arg. of Periapsis, °'
        ]
        
        print(f"Loaded {len(self.samples)} samples with {len(self.labels)} parameters")
        print(f"Log-evidence: {self._logz_last_finite:.2f}")
        
    def load_observations(self):
        """Load observational data."""
        files = {
            "C18 short cadence": ("observations/C18_short_cadence.csv", "red"),
            "C18 long cadence": ("observations/C18_long_cadence.csv", "green"),
            "C5": ("observations/C5.csv", "blue"),
        }
        
        all_times = []
        all_mag_changes = []

        for label, (filename, color) in files.items():
            try:
                df = pd.read_csv(filename, header=None, names=['time', 'flux'])
                mag_change = -2.5 * np.log10(df['flux'])
                all_times.extend(df['time'].tolist())
                all_mag_changes.extend(mag_change.tolist())
            except FileNotFoundError:
                print(f"Warning: Could not find the file {filename}. Skipping.")
                continue
            except Exception as e:
                print(f"An error occurred while reading {filename}: {e}")
                continue
        
        if not all_times:
            print("No data was loaded. Exiting.")
            sys.exit(1)
        
        all_times = np.array(all_times)
        all_mag_changes = np.array(all_mag_changes)
        observations_duration = np.max(all_times) - np.min(all_times)

        phases = (all_times / observations_duration) + 0.5

        sorted_indices = np.argsort(phases)
        self.observations = np.vstack((phases[sorted_indices], all_mag_changes[sorted_indices])).T

    def kfold_split(self, n_splits: int, removed: List[int], _dir: str = 'kfold_observations.jpg',
                    title: str = 'K-Fold Observational Light Curve',
                    x_label: str = 'Phase',
                    y_label: str = 'Magnitude Change',
                    invert_x: bool = False,
                    invert_y: bool = True):
        print(f"Loading and processing observational data with {n_splits}-fold splitting...")
        print(f"Removing chunks: {removed}")

        all_times = []
        all_mag_changes = []

        plt.figure(figsize=(12, 7))

        files = {
            "C18 short cadence": ("observations/C18_short_cadence.csv", "red"),
            "C18 long cadence": ("observations/C18_long_cadence.csv", "green"),
            "C5": ("observations/C5.csv", "blue"),
        }

        # Load data from each file
        for label, (filename, color) in files.items():
            try:
                df = pd.read_csv(filename, header=None, names=['time', 'flux'])
                mag_change = -2.5 * np.log10(df['flux'])

                all_times.extend(df['time'].tolist())
                all_mag_changes.extend(mag_change.tolist())

            except FileNotFoundError:
                print(f"Warning: Could not find the file {filename}. Skipping.")
                continue
            except Exception as e:
                print(f"An error occurred while reading {filename}: {e}")
                continue

        if not all_times:
            print("No data was loaded. Exiting.")
            sys.exit(1)

        all_times = np.array(all_times)
        all_mag_changes = np.array(all_mag_changes)
        observations_duration = np.max(all_times) - np.min(all_times)

        phases = (all_times / observations_duration) + 0.5

        sorted_indices = np.argsort(phases)
        sorted_phases = phases[sorted_indices]
        sorted_mag_changes = all_mag_changes[sorted_indices]

        chunk_size = len(sorted_phases) // n_splits
        chunks_phases = []
        chunks_mag_changes = []

        for i in range(n_splits):
            start_idx = i * chunk_size
            if i == n_splits - 1:  # last chunk gets remaining data
                end_idx = len(sorted_phases)
            else:
                end_idx = (i + 1) * chunk_size

            chunks_phases.append(sorted_phases[start_idx:end_idx])
            chunks_mag_changes.append(sorted_mag_changes[start_idx:end_idx])

        remaining_phases = []
        remaining_mag_changes = []
        for i, (chunk_phases, chunk_mag_changes) in enumerate(zip(chunks_phases, chunks_mag_changes)):
            if i not in removed:
                remaining_phases.extend(chunk_phases)
                remaining_mag_changes.extend(chunk_mag_changes)

        if not remaining_phases:
            print("All chunks were removed. No data remaining.")
            sys.exit(1)

        remaining_phases = np.array(remaining_phases)
        remaining_mag_changes = np.array(remaining_mag_changes)

        plt.plot(remaining_phases, remaining_mag_changes, 'o',
                 markersize=3, color='blue', alpha=0.7, label='Training Data')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{title} (Chunks {list(set(range(n_splits)) - set(removed))} retained)")
        plt.grid(True)
        plt.legend()
        if invert_x:
            plt.gca().invert_xaxis()
        if invert_y:
            plt.gca().invert_yaxis()
        plt.savefig(_dir)
        plt.close()
        print(f"K-fold observational data plot saved to {_dir}")

        sorted_remaining_indices = np.argsort(remaining_phases)
        self.observations = np.vstack((remaining_phases[sorted_remaining_indices],
                                       remaining_mag_changes[sorted_remaining_indices])).T
        
    def get_posterior_samples(self, nsamples=1000):
        valid_mask = np.isfinite(self.logwt) & (self.logwt > -1e100) # remove infinite or extremely negative results
        
        if not np.any(valid_mask):
            print("Warning: No valid samples found. Using best-fit sample only.")
            # Return copies of the best-fit sample
            best_fit_idx = np.argmax(self.logl)
            best_fit_sample = self.samples[best_fit_idx]
            return np.array([best_fit_sample for _ in range(min(nsamples, 100))])
        
        # Use only valid samples
        valid_samples = self.samples[valid_mask]
        valid_logwt = self.logwt[valid_mask]

        # Return maximum weight for normalization
        valid_logz = self.logz[np.isfinite(self.logz)]
        if len(valid_logz) > 0:
            logz_final = valid_logz[-1]
        else:
            logz_final = np.max(valid_logwt)
        
        weights = np.exp(valid_logwt - logz_final)
        
        # Check if weights are valid
        if np.all(weights == 0) or np.any(~np.isfinite(weights)):
            print("Warning: Invalid weights detected. Using uniform weights for valid samples.")
            weights = np.ones_like(weights)

        weights = weights / np.sum(weights)
        
        try:
            samples = dyfunc.resample_equal(
                samples=valid_samples,
                weights=weights
            )
            return samples[:nsamples]
        except Exception as e:
            print(f"Warning: Resampling failed ({e}). Using best-fit sample only.")
            best_fit_idx = np.argmax(self.logl[valid_mask])
            best_fit_sample = valid_samples[best_fit_idx]
            return np.array([best_fit_sample for _ in range(min(nsamples, 100))])
    
    def _estimate_logzerr(self, min_tail: int = 10, tail_fraction: float = 0.2) -> float:
        finite_logz = self.logz[np.isfinite(self.logz)]
        n = finite_logz.size
        if n < 2:
            return np.nan
        k = max(min_tail, int(np.ceil(n * tail_fraction)))
        k = min(k, n)
        tail = finite_logz[-k:]
        if tail.size < 2:
            return np.nan
        return float(np.std(tail, ddof=1))
    
    def summary(self, q1=16, q2=50, q3=84):
        samples = self.get_posterior_samples()
        ans = ''
        for i in range(len(self.labels)):
            percentiles = np.percentile(samples[:, i], [q1, q2, q3])
            posterior_stdev = (percentiles[2] - percentiles[0]) / 1.349
            if posterior_stdev < 1e-10:
                print(f"Warning: Parameter {self.labels[i]} has zero uncertainty. Using prior-based estimate.")
                param_bounds = list(specific_parameter_boundaries.values())
                prior_width = param_bounds[i][1] - param_bounds[i][0]
                posterior_stdev = prior_width * 0.1
                center = percentiles[1]
                percentiles[0] = center - posterior_stdev
                percentiles[2] = center + posterior_stdev
            shrinkage = 1 - (posterior_stdev / self.prior_stdev[i])
            ans += f"{self.labels[i]:<28}: {percentiles[1]:.3f} +{percentiles[2] - percentiles[1]:.3f} / -{percentiles[1] - percentiles[0]:.3f} (shrinkage: {shrinkage:.4f})\n"

        logz_final = self._logz_last_finite if np.isfinite(self._logz_last_finite) else (self.logz[-1] if np.isfinite(self.logz[-1]) else np.nan)
        if getattr(self, 'logzerr', None) is not None and np.isfinite(self._logzerr_last_finite):
            ans += f"\nLog-evidence: {logz_final:.2f} +/- {self._logzerr_last_finite:.2f}\n"
        else:
            dlogz_est = self._estimate_logzerr()
            if np.isfinite(dlogz_est) and dlogz_est > 0:
                ans += f"\nLog-evidence: {logz_final:.2f} +/- {dlogz_est:.2f} (est.)\n"
            else:
                ans += f"\nLog-evidence: {logz_final:.2f}\n"
        
        return ans

    def corner_plot(self, filename: str = 'corner_plot.png', n_prior_samples: int = 1000):
        print("\nGenerating corner plot...")

        try:
            post_samples = self.get_posterior_samples()

            if post_samples is None or len(post_samples) == 0:
                print("No posterior samples available for corner plot.")
                return

            ndim = post_samples.shape[1]
            param_bounds = list(specific_parameter_boundaries.values())
            km_indices = [4, 6, 7] # parameters to divide by 1000 for converting from m to km for display
            
            plot_samples = post_samples.copy()
            for idx in km_indices:
                if idx < plot_samples.shape[1]:
                    plot_samples[:, idx] = plot_samples[:, idx] / 1000.0

            ranges = []
            for i in range(ndim):
                min_val, max_val = np.min(plot_samples[:, i]), np.max(plot_samples[:, i])
                padding = 0.1 * (max_val - min_val) if max_val != min_val else 0.1
                ranges.append((min_val - padding, max_val + padding))

            fig = corner.corner(
                plot_samples,
                labels=self.labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt=".3f",
                title_kwargs={"fontsize": 10},
                label_kwargs={"fontsize": 12},
                labelpad=0.25,
                plot_datapoints=True,
                plot_density=True,
                plot_contours=True,
                fill_contours=True,
                levels=1.0 - np.exp(-0.5 * np.array([1.0, 2.0]) ** 2),
                alpha=0.5,
                bins=30,
                smooth=1.0,
                smooth1d=1.0,
                range=ranges
            )

            # Add uniform prior lines to the diagonal panels
            axes = np.array(fig.axes).reshape((ndim, ndim))
            for i in range(ndim):
                ax = axes[i, i]
                low, high = param_bounds[i]

                if i in km_indices:
                    low_plot, high_plot = low / 1000.0, high / 1000.0
                else:
                    low_plot, high_plot = low, high

                prior_height = 1.0 / (high_plot - low_plot)
                ax.plot([low_plot, high_plot], [prior_height, prior_height],
                        color='blue', linestyle='-', linewidth=2.5,
                        label='Uniform Prior', alpha=0.7)

            fig.suptitle('Corner Plot for Posterior Distributions vs. Uniform Priors', y=1.02, fontsize=16)

            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Corner plot with prior overlays saved to {os.path.abspath(filename)}")

        except Exception as e:
            print(f"Error generating corner plot: {str(e)}")
            raise
    
    def sample_prior(self, nsamples=1000):
        param_bounds = list(specific_parameter_boundaries.values())
        u_samples = np.random.uniform(0, 1, size=(nsamples, len(param_bounds)))
        return np.array([self.prior_transform(u, param_bounds) for u in u_samples])
    
    def prior_transform(self, u, param_bounds):
        params = np.zeros_like(u)
        for i, (u_i, (low, high)) in enumerate(zip(u, param_bounds)):
            params[i] = low + (high - low) * u_i
        return params
    
    def best_fit_vs_observations(self, filename: str = 'best_fit_model_vs_observed_data.png'):
        print("\nGenerating best-fit model vs. observed data plot...")
        best_fit_params = self.samples[np.argmax(self.logl)]
        
        best_fit_model_output, transit_duration, _ = calculate_data(
            fixed_exoplanet_sma,
            best_fit_params[0],  # exoplanet_orbit_eccentricity
            best_fit_params[1],  # exoplanet_orbit_inclination
            best_fit_params[2],  # exoplanet_longitude_of_ascending_node
            best_fit_params[3],  # exoplanet_argument_of_periapsis
            best_fit_params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            roche_density(fixed_exoplanet_mass, best_fit_params[6], best_fit_params[5]),  # density
            best_fit_params[5],  # eccentricity
            best_fit_params[6],  # sma
            best_fit_params[7],  # width
            np.nan,
            best_fit_params[8],  # obliquity
            best_fit_params[9],  # azimuthal_angle
            best_fit_params[10],  # argument_of_periapsis
            fixed_specific_absorption_coefficient,
            fixed_star_object,
            fixed_pixel_size,
            custom_units=False
        )
        
        best_fit_model_lightcurve = np.array(best_fit_model_output)
        
        # Observations
        observed_phases = (self.observations[:, 0] - 0.5) * 0.5 * fixed_observations_duration / transit_duration + 0.5
        observed_magnitudes = self.observations[:, 1]
        min_phase = np.maximum(np.min(observed_phases), np.min(best_fit_model_lightcurve[:, 0]))
        max_phase = np.minimum(np.max(observed_phases), np.max(best_fit_model_lightcurve[:, 0]))
        intersection = (observed_phases >= min_phase) & (observed_phases <= max_phase)
        obs_phase_intersect = observed_phases[intersection]
        obs_mag_intersect = observed_magnitudes[intersection]
        
        # Residuals
        model_interp = np.interp(
            obs_phase_intersect,
            best_fit_model_lightcurve[:, 0],
            best_fit_model_lightcurve[:, 1]
        )
        residuals = obs_mag_intersect - model_interp
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [2, 1]},
                                       sharex=True)
        
        # Top panel: data vs. model
        ax1.plot(obs_phase_intersect, obs_mag_intersect, 'o',
                 color='blue', markersize=3, alpha=0.5, label='Observed Data')
        ax1.plot(best_fit_model_lightcurve[:, 0], best_fit_model_lightcurve[:, 1],
                 '-', color='black', linewidth=2, label='Best-Fit Model')
        ax1.set_ylabel('Magnitude Change')
        ax1.legend()
        ax1.set_title('Best-Fit Model vs. Observed Data')
        ax1.grid(True)
        ax1.invert_yaxis()
        
        # Bottom panel: residuals
        ax2.plot(obs_phase_intersect, residuals, 'o',
                 color='red', markersize=3, alpha=0.5)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Residuals')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Best-fit model vs. observed data plot saved to {filename}")
    
    def ppc(self, n_samples=100, filename='ppc.png', deviation=0.02):
        print("\nGenerating posterior predictive check...")
        
        best_fit_params = self.samples[np.argmax(self.logl)]
        
        fig, ax = plt.subplots()
        ax.set_ylabel('Magnitude Change')
        ax.set_title('Posterior Predictive Check')
        plt.grid(True)
        ax.invert_yaxis()

        best_fit_model_output, transit_duration, _ = calculate_data(
            fixed_exoplanet_sma,
            best_fit_params[0],  # exoplanet_orbit_eccentricity
            best_fit_params[1],  # exoplanet_orbit_inclination
            best_fit_params[2],  # exoplanet_longitude_of_ascending_node
            best_fit_params[3],  # exoplanet_argument_of_periapsis
            best_fit_params[4],  # exoplanet_radius
            fixed_exoplanet_mass,
            roche_density(fixed_exoplanet_mass, best_fit_params[6], best_fit_params[5]),  # density
            best_fit_params[5],  # eccentricity
            best_fit_params[6],  # sma
            best_fit_params[7],  # width
            np.nan,
            best_fit_params[8],  # obliquity
            best_fit_params[9],  # azimuthal_angle
            best_fit_params[10],  # argument_of_periapsis
            fixed_specific_absorption_coefficient,
            fixed_star_object,
            fixed_pixel_size,
            custom_units=False
        )
        
        best_fit_model_lightcurve = np.array(best_fit_model_output)
        predictives = []
        
        for _ in range(n_samples):
            sample = best_fit_params.copy()
            for i in range(len(sample)):
                sample[i] = random.uniform(sample[i] * (1 - deviation), sample[i] * (1 + deviation))
            best_fit_model_output, transit_duration, _ = calculate_data(
                fixed_exoplanet_sma,
                sample[0],  # exoplanet_orbit_eccentricity
                sample[1],  # exoplanet_orbit_inclination
                90,  # exoplanet_longitude_of_ascending_node
                sample[3],  # exoplanet_argument_of_periapsis
                sample[4],  # exoplanet_radius
                fixed_exoplanet_mass,
                roche_density(fixed_exoplanet_mass, best_fit_params[6], best_fit_params[5]),  # density
                sample[5],  # eccentricity
                sample[6],  # sma
                sample[7],  # width
                np.nan,
                sample[8],  # obliquity
                sample[9],  # azimuthal_angle
                sample[10],  # argument_of_periapsis
                fixed_specific_absorption_coefficient,
                fixed_star_object,
                fixed_pixel_size,
                custom_units=False
            )
            
            model_lightcurve = np.array(best_fit_model_output)
            predictives.append(model_lightcurve[:, 1])
            
            plt.plot(model_lightcurve[:, 0], model_lightcurve[:, 1],
                     '-', color='blue', alpha=0.1, label='Posterior Predictive')
        
        plt.plot(best_fit_model_lightcurve[:, 0], np.mean(np.vstack(predictives), axis=0), '--', color='orange', label="Posterior Predictive Mean")
        plt.plot(best_fit_model_lightcurve[:, 0], best_fit_model_lightcurve[:, 1],
                 '-', color='black', linewidth=2, label='Best-Fit Model')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(filename)
        plt.close()
        print(f"Posterior predictive check plot saved to {filename}")
    
    def trace_plots(self, filename='trace_plots.png'):
        print("\nGenerating trace plots...")
        
        samples = self.get_posterior_samples(500)
        
        fig, axes = plt.subplots(len(self.labels), 1, figsize=(10, 2*len(self.labels)))
        if len(self.labels) == 1:
            axes = [axes]
        
        for i, (ax, label) in enumerate(zip(axes, self.labels)):
            ax.plot(samples[:, i], 'b-', alpha=0.7)
            ax.set_ylabel(label)
            ax.grid(True)
        
        ax.set_xlabel('Sample Number')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Trace plots saved to {filename}")
    
    def parameter_distributions(self, filename='parameter_distributions.png'):
        print("\nGenerating parameter distribution plots...")
        
        samples = self.get_posterior_samples()
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, label) in enumerate(zip(axes[:len(self.labels)], self.labels)):
            ax.hist(samples[:, i], bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.grid(True)
            
            # Add statistics
            mean = np.mean(samples[:, i])
            std = np.std(samples[:, i])
            median = np.median(samples[:, i])
            
            ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
            ax.axvline(median, color='green', linestyle='--', label=f'Median: {median:.3f}')
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(self.labels), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Parameter distributions saved to {filename}")
    
    def correlation_heatmap(self, filename='correlation_heatmap.png'):
        """Generate a correlation heatmap of parameters."""
        print("\nGenerating correlation heatmap...")
        
        samples = self.get_posterior_samples()
        corr_matrix = np.corrcoef(samples.T)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=self.labels,
                   yticklabels=self.labels,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   vmin=-1, 
                   vmax=1)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Correlation heatmap saved to {filename}")
    
    def analyze(self, folder: str = ''):
        """Run all analyses and save results."""
        print("Running complete nested sampling analysis...")

        sns.set_theme(style="whitegrid")

        self.corner_plot(folder + 'corner_plot.png')
        self.best_fit_vs_observations(folder + 'best_fit_model_vs_observed_data.png')
        self.ppc(filename=folder + 'ppc.png')
        self.trace_plots(folder + 'trace_plots.png')
        self.parameter_distributions(folder + 'parameter_distributions.png')
        self.correlation_heatmap(folder + 'correlation_heatmap.png')
        stats = self.summary()
        print("\nParameter estimates (50th percentile) with 1-sigma uncertainties:")
        print(stats)
        
        with open(folder + 'nested_sampling_analysis_summary.txt', 'w') as f:
            f.write("Complete Nested Sampling Analysis Results\n")
            f.write("==========================================\n\n")
            f.write("Parameter estimates (50th percentile) with 1-sigma uncertainties:\n")
            f.write(stats)
            
            # Best-fit parameters
            best_fit_params = self.samples[np.argmax(self.logl)]
            f.write("\n\nBest-fit Parameters:\n")
            for i, label in enumerate(self.labels):
                f.write(f"{label}: {best_fit_params[i]:.6f}\n")
            
            f.write(f"\nMaximum Log-Likelihood: {np.max(self.logl):.2f}\n")
            logz_final = self._logz_last_finite if np.isfinite(self._logz_last_finite) else (self.logz[-1] if np.isfinite(self.logz[-1]) else np.nan)
            if getattr(self, 'logzerr', None) is not None and np.isfinite(self._logzerr_last_finite):
                f.write(f"Log-Evidence: {logz_final:.2f} +/- {self._logzerr_last_finite:.2f}\n")
            else:
                dlogz_est = self._estimate_logzerr()
                if np.isfinite(dlogz_est) and dlogz_est > 0:
                    f.write(f"Log-Evidence: {logz_final:.2f} +/- {dlogz_est:.2f} (est.)\n")
                else:
                    f.write(f"Log-Evidence: {logz_final:.2f}\n")
        
        print("Analysis complete! All plots and summary saved.")
        return stats

if __name__ == "__main__":
    nsa = NestedSamplingAnalysis('kfolds/4/nested_sampling_result.npz')
    nsa.analyze('kfolds/4/')
