# Exoplanetary Ring Systems: Identification and Parameter Estimation from Transit Photometry Data

This project provides a code framework for modeling and analyzing transits of ringed exoplanets, with a focus on HIP 41378f. The codebase includes physical models, parameter estimation methods (MCMC and nested sampling), visualization tools and other analysis utilities.

## Main File

**`nested_sampling.py`** - This is the main file used for performing nested sampling analysis of the ring model on the HIP 41378f dataset. It implements dynamic nested sampling using the `dynesty` library to estimate posterior distributions of the parameters of an exoplanet and its ring system. The file includes:

- `NestedSampler` class for running nested sampling analysis
- Fixed parameters specific to HIP 41378f (star properties, exoplanet mass, orbital period, etc.)
- Parameter boundaries optimized for HIP 41378f analysis (for the primary model with rings)
- Methods for saving results and generating analysis plots

### Main Ringed Planet Model (`nested_sampling.py`)

**Fixed Parameters:**
- Pixel size: 10,000 km
- Star radius: 1.28 Solar radius (Grouffal et al., 2022)
- Star log(g): 4.3 (from TICv8)
- Exoplanet mass: 12 Earth mass (Berardo et al., 2019)
- Exoplanet orbital period: 542 days (Santerne et al., 2019)
- Star limb darkening coefficients: [0.0678, 0.188] (Grant & Wakeford, 2024)
- Ring specific absorption coefficient: 2.3e-3 m^2/g (Utry et al., 2014)
- Observation duration: 80 hours

**Parameter Boundaries (11 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.9)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 58,880 km) [Must be lower than in Santerne et al., 2019]
- Ring eccentricity: (0, 0.4)
- Ring semi-major axis: (dynamic, based on Roche limit)
- Ring width: (0, dynamic maximum based on Roche limit)
- Ring obliquity: (0°, 90°)
- Ring azimuthal angle: (0°, 180°)
- Ring argument of periapsis: (0°, 180°)

**Nested Sampling Configuration:**
- `nlive`: 5000 live points
- `ndim`: 11 dimensions
- Sampler: Dynamic nested sampling (`dynesty.DynamicNestedSampler`)
- Log-likelihood cache: `loglikes/loglike.json`

### Noise Model (`noise_nested_sampling.py`)

**Parameter Boundaries (7 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 128,000 km)
- Noise scale: (0, auto-calculated from data)
- Noise magnitude: (0, auto-calculated from data)

**Nested Sampling Configuration:**
- `nlive`: 5000 live points
- `ndim`: 7 dimensions
- Sampler: Dynamic nested sampling (`dynesty.DynamicNestedSampler`)
- Log-likelihood cache: `loglikes/noise.json`
- Auto-calculated noise boundaries from observation data

### Oblate Planet Model (`oblate_planet_nested_sampling.py`)

**Parameter Boundaries (7 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (46,170 km, 128,000 km)
- Planet oblateness: (0, 0.9)
- Projection rotation: (0°, 180°)

**Nested Sampling Configuration:**
- `nlive`: 5000 live points
- `ndim`: 7 dimensions
- Sampler: Dynamic nested sampling (`dynesty.DynamicNestedSampler`)
- Log-likelihood cache: `loglikes/oblate_planet.json`

### Ringless Model (`ringless_nested_sampling.py`)

**Parameter Boundaries (5 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (46,170 km, 128,000 km)

**Nested Sampling Configuration:**
- `nlive`: 5000 live points
- `ndim`: 5 dimensions
- Sampler: Dynamic nested sampling (`dynesty.DynamicNestedSampler`)
- Log-likelihood cache: `loglikes/ringless.json`

### Star Spots Model (`star_spots_nested_sampling.py`)

**Parameter Boundaries (9 dimensions):**
- Exoplanet orbit eccentricity: (0, 0.4)
- Exoplanet orbit inclination: (0°, 90°)
- Exoplanet longitude of ascending node: (90° - 5e-8, 90° + 5e-8)
- Exoplanet argument of periapsis: (0°, 180°)
- Exoplanet radius: (2,000 km, 128,880 km)
- Star angular velocity: (-2.5/hr, 2.5/hr)
- Spot longitude: (0°, 360°)
- Spot radius: (0, 1) [in star radii]
- Spot brightness: (0, 2) [relative to star]

**Nested Sampling Configuration:**
- `nlive`: 5000 live points
- `ndim`: 9 dimensions
- Sampler: Dynamic nested sampling (`dynesty.DynamicNestedSampler`)
- Log-likelihood cache: `loglikes/star_spots.json`

**Common Observation Files (for all models):**
- C18 short cadence: `observations/C18_short_cadence.csv`
- C18 long cadence: `observations/C18_long_cadence.csv`
- C5: `observations/C5.csv`

## Figure-Producing Files and Outputs

### Nested Sampling Analysis Figures

**`nested_sampling.py`** - Main nested sampling analysis generates:
- `corner_plot.png` - Corner plot showing parameter posterior distributions with prior overlays
- `best_fit_model_vs_observed_data.png` - Comparison of best-fit model with observational data
- `observational_data.png` - Plots of observational data from all datasets
- `kfold_observational_data.png` - K-fold cross-validation observational data plots
- `posterior_predictive_check.png` - Posterior predictive checks
- `trace_plots.png` - trace plots for parameter convergence
- `parameter_distributions.png` - Parameter distribution histograms
- `correlation_heatmap.png` - Parameter correlation matrix heatmap

**`complete_nested_sampling_analysis.py`** - Extended analysis generates:
- `corner_plot.png` - Enhanced corner plots with prior overlays
- `best_fit_model_vs_observed_data.png` - Detailed model vs data comparison
- `kfold_observational_data.png` - K-fold validation plots
- `posterior_predictive_check.png` - Posterior predictive analysis
- `trace_plots.png` - Parameter trace analysis
- `parameter_distributions.png` - Parameter distribution analysis
- `correlation_heatmap.png` - Correlation analysis

**MCMC Analysis (`data_fitting.py`)** generates:
- `trace_plots.png` - MCMC chain trace plots
- `posterior_histograms.png` - Parameter posterior histograms
- `best_fit_model_vs_observed_data.png` - Model comparison with data
- `corner_plot.png` - Corner plots with priors
- `corner_plot_reduced.png` - Reduced corner plots
- `correlation_matrix.png` - Parameter correlation matrices

### Alternative Model Comparison Figures

**`alternative_model_figure.py`** generates:
- `alternative_model_figure.png` - Comparison of different alternative models (noise, oblate planet, ringless, star spots)

**`kfold_figure.py`** generates:
- `kfold_figure.png` - K-fold cross-validation results visualization

### Animation and Visualization Files

**`animation_generation.py`** produces:
- **Static parameter effect images** (saved to `effect/` directories):
  - `effect/argument_of_periapsis/` - Argument of periapsis variation frames
  - `effect/azimuthal_angle/` - Azimuthal angle variation frames  
  - `effect/band/` - Photometric band effect frames
  - `effect/eccentricity/` - Eccentricity variation frames
- **Animated GIFs** (saved to various directories):
  - `argument_of_periapsis.gif` - Argument of periapsis animation
  - `azimuthal_angle.gif` - Azimuthal angle animation
  - `band.gif` - Photometric band animation
  - `eccentricity.gif` - Eccentricity animation
  - `oblate_planet_light_curve_oblateness.gif` - Oblate planet oblateness effect
  - `oblate_planet_light_curve_rotation.gif` - Oblate planet rotation effect

**`space.py`** demonstration script generates:
- `stellar_spots.gif` - Star spots rotation animation

**`star_spots_demos/`** directory contains additional animations:
- `spot_size.gif` - Star spot size variations
- `spot_radius.gif` - Star spot radius variations
- `spot_count.gif` - Multiple star spots demonstration
- `spot_contrast.gif` - Star spot contrast variations
- `spot_brightness.gif` - Star spot brightness variations
- `longitude.gif` - Star spot longitude variations
- `exoplanet_orbit_eccentricity.gif` - Exoplanet orbital eccentricity effects
- `angular_velocity.gif` - Star angular velocity variations

**`parameter_demos/`** directory contains static demonstration images:
- `width/` - Ring width parameter variations (7 frames)
- `semi-major_axis/` - Ring semi-major axis variations (7 frames)
- `rotation_angle/` - Rotation angle variations (6 frames)

### GUI and Interactive Visualization

**`visualization.py`** provides:
- Interactive parameter selection interface
- Real-time transit model visualization
- Animation display windows with save capabilities
- Exportable transit animation frames (`.jpg` format)
- Convertible animations to GIF format

## File Descriptions

### Core Physics and Models

- **`space.py`** - Defines the main physical objects: `Star`, `Exoplanet`, `Orbit`, `Rings`, `StarSpot`, and `CustomStarModel`. Handles star limb darkening models, exoplanet ring systems, and orbital mechanics. Includes a demonstration script for star spots animation.

- **`models.py`** - Contains functions for generating physical models:
  - `planet()` - Creates 2D projections of planets with oblateness and rotation
  - `disk()` - Creates circular disk models
  - `transit()` - Calculates transit light curves based on orbital mechanics
  - `transit_animation()` - Generates animation frames for transit visualization
  - `star_spot()` - Models stellar spots with rotation
  - `quadratic_star_model()` and `square_root_star_model()` - Limb darkening models

- **`formulas.py`** - Mathematical formulas and physical calculations:
  - Roche limit calculations (minimum/maximum ring semi-major axis)
  - Hill sphere calculations (not used in the main data-fitting process)
  - Orbital mechanics (mean anomaly, eccentric anomaly, true anomaly, radius vector)
  - Newton-Raphson method for solving Kepler's equation
  - Star mass calculations
  - Unit conversions (to_pixels)
  - Light curve normalization

- **`units.py`** - Contains constants for physical unit conversion.

- **`measure.py`** - Implements the `Measure` class for handling physical quantities with units, including unit conversions and arithmetic operations.

### Main Application

- **`main.py`** - Main application entry point. Defines default parameter ranges and the `calculate_data()` function that orchestrates the simulation by creating rings, orbits, exoplanets, and computing transit light curves. Can be run as a GUI application for interactive parameter exploration (although it is currently not functional).

### Parameter Estimation

- **`data_fitting.py`** - Implements MCMC (Markov Chain Monte Carlo) parameter estimation using `emcee`:
  - `MCMC` class extending `emcee.EnsembleSampler`
  - Log-likelihood and log-prior calculations
  - Chain analysis and visualization (corner plots, trace plots, posterior histograms)
  - Correlation heatmaps

- **`nested_sampling.py`** - **Main file** for nested sampling analysis (see above for details)

- **`complete_nested_sampling_analysis.py`** - Extended analysis tools for nested sampling results, including additional visualization and statistical analysis methods.

### Data Handling

- **`observations.py`** - `Observations` class for loading and processing observation data from CSV files. Handles time and magnitude shifts, normalization, and data manipulation for comparison with models.


### Visualization

- **`visualization.py`** - PyQt6-based GUI components:
  - `Selection` - Parameter selection interface
  - `Model` - Main visualization window for transit models
  - `FramesWindow` - Animation display window
  - `AnimatedGraph` - Animated graph visualization
  - `AnimationWindow` - General animation window

- **`animation_generation.py`** - Functions for generating animations and visualizations of parameter effects, including light curve animations and ring parameter demonstrations.

### Analysis and Results

- **`kfold.py`** - K-fold cross-validation implementation for model validation.

- **`alternative_models.py`** - Alternative model implementations for comparison studies.

### Data and Configuration

- **`limb_darkening/`** - Directory containing limb darkening coefficient data:
  - `quadratic.json` - Quadratic limb darkening coefficients
  - `square-root.json` - Square-root limb darkening coefficients

- **`observations/`** - Directory containing observation data files in CSV format

- **`loglikes/`** - Directory for nested sampling log-likelihood cache and results. Now it does not contain the cache files, as they take up too much space. Instead, they will be automatically generated by the nested sampling algorithm.

- **`parameter_demos/`** - Directory containing parameter demonstration images:
  - `width/` - Ring width parameter variations
  - `semi-major_axis/` - Ring semi-major axis variations
  - `rotation_angle/` - Rotation angle variations

- **`star_spots_demos/`** - Directory containing star spots demonstration animations:
  - `stellar_spots.gif` - Star spots rotation
  - `spot_size.gif` - Star spot size variations
  - `spot_radius.gif` - Star spot radius variations
  - `spot_count.gif` - Multiple star spots demonstration
  - `spot_contrast.gif` - Star spot contrast variations
  - `spot_brightness.gif` - Star spot brightness variations
  - `longitude.gif` - Star spot longitude variations
  - `exoplanet_orbit_eccentricity.gif` - Exoplanet orbital eccentricity effects
  - `angular_velocity.gif` - Star angular velocity variations

- **`improved_gifs/`** - Directory containing enhanced animation outputs

- **`effect/`** - Directory for parameter effect frames generated by animations:
  - `argument_of_periapsis/` - Argument of periapsis variation frames
  - `azimuthal_angle/` - Azimuthal angle variation frames
  - `band/` - Photometric band effect frames
  - `eccentricity/` - Eccentricity variation frames

- **`synthetic_light_curves/`** - Directory containing synthetic light curve demonstrations

- **`animations/`** - Directory for generated animation files

- **`effect_gifs/`** - Directory for parameter effect GIF animations

- **`figures/`** - Directory for analysis and comparison figures

- **`nested_sampling/`** - Directory for nested sampling results and analysis

- **`alternative_model_runs/`** - Directory containing alternative model analysis results:
  - `noise/` - Noise model results
  - `oblate_planet/` - Oblate planet model results
  - `ringless/` - Ringless model results
  - `star_spots/` - Star spots model results

- **`kfolds/`** - Directory containing k-fold cross-validation results (0-4)

- **`MCMC_run1/`, `MCMC_run2/`, `MCMC_best_run/`** - Directories containing MCMC analysis results

- **`failed_nested_samlipng_run/`** - Directory containing failed nested sampling run results for debugging

## Usage

### Running Nested Sampling Analysis

#### Main Ringed Planet Model (HIP 41378f)
```python
python nested_sampling.py
```

#### Alternative Models
```python
# Noise model
python noise_nested_sampling.py

# Oblate planet model  
python oblate_planet_nested_sampling.py

# Ringless model
python ringless_nested_sampling.py

# Star spots model
python star_spots_nested_sampling.py
```

Each nested sampling run will:
1. Initialize the nested sampler with model-specific parameters
2. Run dynamic nested sampling analysis
3. Save results to `{model}_nested_sampling_results.npz`
4. Generate analysis plots and summary statistics
5. Cache log-likelihood calculations to `loglikes/{model}.json`

### Running MCMC Analysis

```python
python data_fitting.py
```

### Running Interactive GUI

```python
python main.py
```

### Generating Animations and Visualizations

```python
# Generate parameter effect animations
python animation_generation.py

# Generate alternative model comparison figure
python alternative_model_figure.py

# Generate k-fold cross-validation figure
python kfold_figure.py

# Run star spots demonstration (generates stellar_spots.gif)
python space.py
```

### Complete Analysis Pipeline

For a comprehensive analysis of all models:
```bash
# Run all nested sampling variations
python nested_sampling.py
python noise_nested_sampling.py
python oblate_planet_nested_sampling.py
python ringless_nested_sampling.py
python star_spots_nested_sampling.py

# Generate comparison figures
python alternative_model_figure.py
python kfold_figure.py

# Run extended analysis
python complete_nested_sampling_analysis.py
```

## Output Files

### Nested Sampling Results
- **`nested_sampling_results.npz`** - Main model nested sampling results (samples, weights, log-likelihoods)
- **`noise_nested_sampling_results.npz`** - Noise model results
- **`oblate_planet_nested_sampling_results.npz`** - Oblate planet model results  
- **`ringless_nested_sampling_results.npz`** - Ringless model results
- **`star_spots_nested_sampling_results.npz`** - Star spots model results

### Analysis Summaries
- **`nested_sampling_analysis_summary.txt`** - Main model analysis summary
- **Alternative model summaries** - Generated in respective model directories

### Visualization Files
- **`corner_plot.png`** - Parameter posterior distributions with priors
- **`best_fit_model_vs_observed_data.png`** - Model comparison with observations
- **`parameter_distributions.png`** - Parameter distribution histograms
- **`trace_plots.png`** - Parameter convergence trace plots
- **`correlation_heatmap.png`** - Parameter correlation matrices
- **`posterior_predictive_check.png`** - Posterior predictive analysis
- **`observational_data.png`** - Observational data plots
- **`kfold_observational_data.png`** - K-fold validation plots

### Comparison and Analysis Figures
- **`alternative_model_figure.png`** - Alternative model comparison
- **`kfold_figure.png`** - K-fold cross-validation results

### Animation Files
- **Root directory animations**: Generated by `animation_generation.py`
  - `argument_of_periapsis.gif` - Argument of periapsis variation
  - `azimuthal_angle.gif` - Azimuthal angle variation
  - `band.gif` - Photometric band effects
  - `eccentricity.gif` - Eccentricity variation
  - `oblate_planet_light_curve_oblateness.gif` - Oblateness effects
  - `oblate_planet_light_curve_rotation.gif` - Rotation effects

- **`star_spots_demos/` animations**:
  - `stellar_spots.gif` - Star spots rotation
  - `spot_size.gif` - Star spot size variations
  - `spot_radius.gif` - Star spot radius variations
  - `spot_count.gif` - Multiple star spots demonstration
  - `spot_contrast.gif` - Star spot contrast variations
  - `spot_brightness.gif` - Star spot brightness variations
  - `longitude.gif` - Star spot longitude variation
  - `exoplanet_orbit_eccentricity.gif` - Exoplanet orbital eccentricity effects
  - `angular_velocity.gif` - Star angular velocity variation

- **`improved_gifs/` enhanced animations**:
  - `exoplanet_argument_of_periapsis.gif` - Enhanced argument of periapsis animation
  - `optical_depth.gif` - Optical depth variation animation

### Static Effect Frames
- **`effect/` directories** containing parameter variation frames:
  - `argument_of_periapsis/` - Argument of periapsis effect frames
  - `azimuthal_angle/` - Azimuthal angle effect frames
  - `band/` - Photometric band effect frames
  - `eccentricity/` - Eccentricity effect frames

- **`parameter_demos/` static demonstration images**:
  - `width/width0.jpg` through `width6.jpg` - Ring width variations
  - `semi-major_axis/semi-major_axis0.jpg` through `semi-major_axis6.jpg` - Semi-major axis variations
  - `rotation_angle/rotation_angle0.jpg` through `rotation_angle5.jpg` - Rotation angle variations

- **`synthetic_light_curves/` demonstration frames**:
  - Various subdirectories with synthetic light curve examples

### Analysis Results Directories
- **`alternative_model_runs/`** - Results from alternative model analyses:
  - `noise/` - Noise model results and figures
  - `oblate_planet/` - Oblate planet model results and figures
  - `ringless/` - Ringless model results and figures
  - `star_spots/` - Star spots model results and figures

- **`kfolds/`** - K-fold cross-validation results:
  - `0/`, `1/`, `2/`, `3/`, `4/` - Individual fold results with figures and analysis

- **`MCMC_run1/`, `MCMC_run2/`, `MCMC_best_run/`** - MCMC analysis results

- **`nested_sampling_run/`** - Main nested sampling analysis results

- **`failed_nested_samlipng_run/`** - Debugging information from failed runs

- **`figures/`** - Analysis and comparison figures
- **`animations/`** - Generated animation files

### Log-Likelihood Cache
- **`loglikes/`** directory containing cached calculations:
  - `loglike.json` - Main model cache
  - `noise.json` - Noise model cache
  - `oblate_planet.json` - Oblate planet cache
  - `ringless.json` - Ringless model cache
  - `star_spots.json` - Star spots model cache
  - `kfold0.json` through `kfold4.json` - K-fold validation caches

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```




