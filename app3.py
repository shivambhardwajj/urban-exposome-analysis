import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from collections import Counter
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import measure, morphology, segmentation
from sklearn.cluster import KMeans
import math

# Try to import YOLO, handle gracefully if not available
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("⚠️ YOLO not available. Please install: pip install ultralytics")

# Set page config
st.set_page_config(page_title="Comprehensive Urban Exposome Analysis - Utrecht", layout="wide")


# Cache the model loading with better error handling
@st.cache_resource
def load_yolo_model():
    """Load YOLO model with caching and proper error handling"""
    if not YOLO_AVAILABLE:
        return None

    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.info("Please ensure you have internet connection for initial model download")
        return None


# Initialize model only if YOLO is available
model = load_yolo_model() if YOLO_AVAILABLE else None


# =============================================================================
# 🌿 ENHANCED GREENERY & VEGETATION ANALYSIS
# =============================================================================
def comprehensive_vegetation_analysis(image):
    """
    Multi-method vegetation analysis including coverage, distribution, and type
    """
    try:
        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("Invalid image format")

        # Multiple green detection methods
        # Method 1: Traditional HSV green detection
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Method 2: NDVI-inspired calculation (for vegetation health)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        # Add small epsilon to avoid division by zero
        ndvi_like = np.where((g + r) > 0, (g - r) / (g + r + 1e-8), 0)
        vegetation_mask = (ndvi_like > 0.1) & (g > 100)

        # Method 3: Tree canopy detection (darker greens, larger connected components)
        lower_tree = np.array([40, 50, 30])
        upper_tree = np.array([80, 255, 150])
        tree_mask = cv2.inRange(hsv, lower_tree, upper_tree)

        # Connected component analysis for tree detection
        tree_components = measure.label(tree_mask > 0)
        tree_areas = [region.area for region in measure.regionprops(tree_components)]
        large_trees = sum(1 for area in tree_areas if area > 1000)  # Trees > 1000 pixels

        # Grass/ground vegetation (lighter greens)
        lower_grass = np.array([40, 30, 60])
        upper_grass = np.array([80, 180, 255])
        grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)

        # Combine all vegetation
        combined_vegetation = cv2.bitwise_or(green_mask, vegetation_mask.astype(np.uint8) * 255)
        combined_vegetation = cv2.bitwise_or(combined_vegetation, tree_mask)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        combined_vegetation = cv2.morphologyEx(combined_vegetation, cv2.MORPH_CLOSE, kernel)

        # Calculate metrics
        total_pixels = image.shape[0] * image.shape[1]
        vegetation_coverage = np.sum(combined_vegetation > 0) / total_pixels * 100
        tree_coverage = np.sum(tree_mask > 0) / total_pixels * 100
        grass_coverage = np.sum(grass_mask > 0) / total_pixels * 100

        # Vegetation distribution analysis
        vegetation_binary = combined_vegetation > 0
        vegetation_fragments = measure.label(vegetation_binary)
        fragment_props = measure.regionprops(vegetation_fragments)

        vegetation_connectivity = len(fragment_props)  # Number of separate vegetation patches
        avg_patch_size = np.mean([prop.area for prop in fragment_props]) if fragment_props else 0

        return {
            'total_vegetation_percent': float(vegetation_coverage),
            'tree_canopy_percent': float(tree_coverage),
            'grass_vegetation_percent': float(grass_coverage),
            'vegetation_connectivity_index': int(vegetation_connectivity),
            'average_patch_size': float(avg_patch_size),
            'large_trees_count': int(large_trees),
            'vegetation_health_score': float(np.mean(ndvi_like[ndvi_like > 0])) if np.any(ndvi_like > 0) else 0.0
        }
    except Exception as e:
        st.warning(f"Error in vegetation analysis: {e}")
        return {
            'total_vegetation_percent': 0.0,
            'tree_canopy_percent': 0.0,
            'grass_vegetation_percent': 0.0,
            'vegetation_connectivity_index': 0,
            'average_patch_size': 0.0,
            'large_trees_count': 0,
            'vegetation_health_score': 0.0
        }


# =============================================================================
# 🏗️ BUILT ENVIRONMENT ANALYSIS
# =============================================================================
def analyze_built_environment(image, detections):
    """
    Comprehensive built environment feature extraction
    """
    try:
        # Convert to grayscale properly
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        height, width = gray.shape

        # Building density estimation using edge detection and vertical line analysis
        edges = cv2.Canny(gray, 50, 150)

        # Detect vertical lines (buildings)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        building_density = np.sum(vertical_lines > 0) / (height * width) * 100

        # Road width estimation using horizontal edge detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)

        # Estimate road width by finding the largest horizontal continuous space
        road_mask = gray > 80  # Assume road is lighter colored
        horizontal_projection = np.sum(road_mask, axis=0)
        road_width_estimate = float(np.max(horizontal_projection)) / height * 100 if len(
            horizontal_projection) > 0 else 0.0

        # Building height estimation using shadow analysis
        shadows = gray < 60  # Dark areas likely to be shadows
        shadow_coverage = np.sum(shadows) / (height * width) * 100

        # Estimate building height from shadow length (rough approximation)
        if shadow_coverage > 5:  # Only if significant shadows
            shadow_regions = measure.label(shadows)
            shadow_props = measure.regionprops(shadow_regions)
            avg_shadow_length = np.mean([prop.major_axis_length for prop in shadow_props]) if shadow_props else 0.0
            estimated_building_height = avg_shadow_length * 0.7  # Rough conversion factor
        else:
            estimated_building_height = 0.0

        # Urban density indicators
        building_objects = detections.get('building', 0) + detections.get('house', 0)

        return {
            'building_density_score': float(building_density),
            'estimated_road_width_percent': float(road_width_estimate),
            'shadow_coverage_percent': float(shadow_coverage),
            'estimated_building_height_pixels': float(estimated_building_height),
            'visible_buildings_count': int(building_objects),
            'urban_density_score': float((building_density + shadow_coverage) / 2)
        }
    except Exception as e:
        st.warning(f"Error in built environment analysis: {e}")
        return {
            'building_density_score': 0.0,
            'estimated_road_width_percent': 0.0,
            'shadow_coverage_percent': 0.0,
            'estimated_building_height_pixels': 0.0,
            'visible_buildings_count': 0,
            'urban_density_score': 0.0
        }


# =============================================================================
# 🚲 TRANSPORTATION INFRASTRUCTURE ANALYSIS
# =============================================================================
def analyze_transportation_infrastructure(image, detections):
    """
    Detailed transportation and mobility infrastructure analysis
    """
    try:
        # Vehicle analysis
        cars = detections.get('car', 0)
        trucks = detections.get('truck', 0)
        buses = detections.get('bus', 0)
        motorcycles = detections.get('motorcycle', 0)
        bicycles = detections.get('bicycle', 0)

        total_vehicles = cars + trucks + buses + motorcycles

        # Traffic density estimation
        image_area = image.shape[0] * image.shape[1]
        vehicle_density = total_vehicles / (image_area / 100000)  # Vehicles per 100k pixels

        # Active transportation indicators
        active_transport_ratio = bicycles / max(total_vehicles, 1)  # Bikes vs motorized vehicles

        # Parking analysis (cars stationary vs moving - rough estimation)
        parking_density = cars * 0.7  # Assume 70% of visible cars are parked

        # Infrastructure detection
        traffic_lights = detections.get('traffic light', 0)
        stop_signs = detections.get('stop sign', 0)

        # Road marking analysis using line detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Detect white lines (road markings, bike lanes)
        white_mask = gray > 200
        try:
            white_lines = cv2.HoughLinesP(white_mask.astype(np.uint8) * 255,
                                          1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
            road_marking_density = len(white_lines) if white_lines is not None else 0
        except Exception:
            road_marking_density = 0

        # Bike lane detection (combination of white lines and positioning)
        bike_infrastructure_score = (road_marking_density * 0.1 + bicycles * 2) / 2

        # Determine traffic intensity level
        if vehicle_density > 2:
            traffic_intensity_level = 'High'
        elif vehicle_density > 0.5:
            traffic_intensity_level = 'Medium'
        else:
            traffic_intensity_level = 'Low'

        return {
            'total_vehicles': int(total_vehicles),
            'car_count': int(cars),
            'bicycle_count': int(bicycles),
            'bus_count': int(buses),
            'motorcycle_count': int(motorcycles),
            'vehicle_density_score': float(vehicle_density),
            'active_transport_ratio': float(active_transport_ratio),
            'parking_density_estimate': float(parking_density),
            'traffic_control_devices': int(traffic_lights + stop_signs),
            'road_marking_density': int(road_marking_density),
            'bike_infrastructure_score': float(bike_infrastructure_score),
            'traffic_intensity_level': traffic_intensity_level
        }
    except Exception as e:
        st.warning(f"Error in transportation analysis: {e}")
        return {
            'total_vehicles': 0,
            'car_count': 0,
            'bicycle_count': 0,
            'bus_count': 0,
            'motorcycle_count': 0,
            'vehicle_density_score': 0.0,
            'active_transport_ratio': 0.0,
            'parking_density_estimate': 0.0,
            'traffic_control_devices': 0,
            'road_marking_density': 0,
            'bike_infrastructure_score': 0.0,
            'traffic_intensity_level': 'Low'
        }


# =============================================================================
# 🌍 ENVIRONMENTAL QUALITY INDICATORS
# =============================================================================
def assess_environmental_quality(image, detections, built_env, transport):
    """
    Comprehensive environmental quality assessment for epidemiological applications
    """
    try:
        # Air quality proxies
        vehicle_count = transport['total_vehicles']
        traffic_density = transport['vehicle_density_score']

        # Get vegetation score
        vegetation_score = comprehensive_vegetation_analysis(image)['total_vegetation_percent']

        # Air pollution proxy score (higher vehicles + lower vegetation = worse air quality)
        air_quality_proxy = max(0.0, 100.0 - (traffic_density * 10 + max(0.0, 20.0 - vegetation_score)))

        # Noise exposure estimation
        noise_sources = vehicle_count * 2 + detections.get('motorcycle', 0) * 3  # Motorcycles louder
        noise_barriers = vegetation_score * 0.5 + built_env.get('building_density_score', 0) * 0.3
        noise_exposure_level = max(0.0, min(100.0, noise_sources * 5 - noise_barriers))

        # Walkability assessment
        people_count = detections.get('person', 0)
        sidewalk_space = 100.0 - built_env.get('estimated_road_width_percent', 50.0)  # Inverse of road dominance
        walkability_score = (people_count * 10 + sidewalk_space + vegetation_score * 0.5) / 3

        # Social infrastructure indicators
        benches = detections.get('bench', 0)
        social_spaces = benches + detections.get('dining table', 0)  # Outdoor dining
        social_infrastructure_score = min(100.0, social_spaces * 20 + people_count * 5)

        # Heat island effect proxy
        built_surface_ratio = built_env.get('building_density_score', 0) + built_env.get('shadow_coverage_percent', 0)
        heat_island_risk = max(0.0, built_surface_ratio - vegetation_score)

        # Overall environmental health score
        env_health_score = (air_quality_proxy + (100.0 - noise_exposure_level) + walkability_score) / 3

        # Categorize risks
        pollution_risk_category = 'High' if air_quality_proxy < 40 else 'Medium' if air_quality_proxy < 70 else 'Low'
        noise_risk_category = 'High' if noise_exposure_level > 70 else 'Medium' if noise_exposure_level > 40 else 'Low'

        return {
            'air_quality_proxy_score': float(air_quality_proxy),
            'noise_exposure_level': float(noise_exposure_level),
            'walkability_score': float(walkability_score),
            'social_infrastructure_score': float(social_infrastructure_score),
            'heat_island_risk_score': float(heat_island_risk),
            'overall_environmental_health_score': float(env_health_score),
            'pollution_risk_category': pollution_risk_category,
            'noise_risk_category': noise_risk_category
        }
    except Exception as e:
        st.warning(f"Error in environmental quality assessment: {e}")
        return {
            'air_quality_proxy_score': 50.0,
            'noise_exposure_level': 50.0,
            'walkability_score': 50.0,
            'social_infrastructure_score': 50.0,
            'heat_island_risk_score': 50.0,
            'overall_environmental_health_score': 50.0,
            'pollution_risk_category': 'Medium',
            'noise_risk_category': 'Medium'
        }


# =============================================================================
# 🧠 EPIDEMIOLOGICAL APPLICATIONS CALCULATOR
# =============================================================================
def calculate_health_outcomes(vegetation, built_env, transport, env_quality):
    """
    Calculate epidemiological indicators based on environmental features
    """
    try:
        # Physical Activity & Obesity Prevention Indicators
        bike_infrastructure = transport['bike_infrastructure_score']
        walkability = env_quality['walkability_score']
        green_space_access = vegetation['total_vegetation_percent']

        physical_activity_promotion_score = (bike_infrastructure + walkability + green_space_access * 0.8) / 3
        obesity_prevention_potential = min(100.0, physical_activity_promotion_score * 1.2)

        # Respiratory Health Indicators
        air_quality = env_quality['air_quality_proxy_score']
        vehicle_exposure = transport['vehicle_density_score']
        vegetation_buffer = vegetation['total_vegetation_percent']

        respiratory_health_risk = max(0.0, 100.0 - air_quality - vegetation_buffer * 0.5)
        asthma_risk_level = 'High' if respiratory_health_risk > 60 else 'Medium' if respiratory_health_risk > 30 else 'Low'

        # Mental Health & Well-being Indicators
        nature_access = vegetation['total_vegetation_percent'] + vegetation['tree_canopy_percent']
        social_spaces = env_quality['social_infrastructure_score']
        noise_stress = env_quality['noise_exposure_level']

        mental_wellbeing_score = (nature_access + social_spaces - noise_stress * 0.8) / 2
        stress_reduction_potential = max(0.0, min(100.0, mental_wellbeing_score))

        # Cardiovascular Health Indicators
        cardio_protective_factors = (air_quality + walkability + nature_access * 0.6) / 3
        cardio_risk_factors = (vehicle_exposure * 10 + noise_stress) / 2
        cardiovascular_health_score = max(0.0, cardio_protective_factors - cardio_risk_factors * 0.5)

        # Overall health supportive environment
        overall_health_supportive = (obesity_prevention_potential + (100.0 - respiratory_health_risk) +
                                     stress_reduction_potential + cardiovascular_health_score) / 4

        return {
            'physical_activity_promotion_score': float(physical_activity_promotion_score),
            'obesity_prevention_potential': float(obesity_prevention_potential),
            'respiratory_health_risk': float(respiratory_health_risk),
            'asthma_risk_level': asthma_risk_level,
            'mental_wellbeing_score': float(mental_wellbeing_score),
            'stress_reduction_potential': float(stress_reduction_potential),
            'cardiovascular_health_score': float(cardiovascular_health_score),
            'overall_health_supportive_environment': float(overall_health_supportive)
        }
    except Exception as e:
        st.warning(f"Error in health outcomes calculation: {e}")
        return {
            'physical_activity_promotion_score': 50.0,
            'obesity_prevention_potential': 50.0,
            'respiratory_health_risk': 50.0,
            'asthma_risk_level': 'Medium',
            'mental_wellbeing_score': 50.0,
            'stress_reduction_potential': 50.0,
            'cardiovascular_health_score': 50.0,
            'overall_health_supportive_environment': 50.0
        }


# =============================================================================
# 🚗 ENHANCED YOLO OBJECT DETECTION
# =============================================================================
def comprehensive_object_detection(image_path):
    """Enhanced object detection with health-relevant categorization"""
    if model is None:
        st.warning("YOLO model not available - using simplified analysis")
        return Counter()

    try:
        results = model(image_path, verbose=False)
        names = model.names

        if len(results[0].boxes) > 0:
            detections = results[0].boxes.cls.cpu().numpy()
            detection_names = [names[int(cls)] for cls in detections]
            detection_counts = Counter(detection_names)

            # Add health-relevant categorizations
            health_relevant_objects = {
                'vehicles_total': (detection_counts.get('car', 0) + detection_counts.get('truck', 0) +
                                   detection_counts.get('bus', 0) + detection_counts.get('motorcycle', 0)),
                'active_transport': detection_counts.get('bicycle', 0),
                'pedestrians': detection_counts.get('person', 0),
                'green_infrastructure': (detection_counts.get('potted plant', 0) +
                                         detection_counts.get('vase', 0)),  # Proxy for urban greenery
                'social_infrastructure': (detection_counts.get('bench', 0) +
                                          detection_counts.get('dining table', 0)),
                'safety_infrastructure': (detection_counts.get('traffic light', 0) +
                                          detection_counts.get('stop sign', 0))
            }

            detection_counts.update(health_relevant_objects)
            return detection_counts
        else:
            return Counter()
    except Exception as e:
        st.warning(f"Detection failed for image: {e}")
        return Counter()


# =============================================================================
# 🖼️ COMPREHENSIVE IMAGE PROCESSING
# =============================================================================
def comprehensive_image_analysis(uploaded_file):
    """
    Master function for comprehensive image analysis
    """
    try:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Save to temporary file for YOLO (if available)
        detections = Counter()  # Default empty detections
        temp_path = None

        if model is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    image.save(temp_file.name, quality=95)
                    temp_path = temp_file.name

                # Run YOLO detection
                detections = comprehensive_object_detection(temp_path)

                # Clean up temp file with retry mechanism for Windows
                if temp_path and os.path.exists(temp_path):
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            os.unlink(temp_path)
                            break
                        except PermissionError:
                            if attempt < max_retries - 1:
                                import time
                                time.sleep(0.1)  # Wait 100ms before retry
                            else:
                                st.warning(f"Could not delete temporary file: {temp_path}")
            except Exception as e:
                st.warning(f"YOLO processing error: {e}")
                # Clean up on error
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

        # Comprehensive analysis pipeline
        vegetation = comprehensive_vegetation_analysis(image_np)
        built_env = analyze_built_environment(image_np, detections)
        transport = analyze_transportation_infrastructure(image_np, detections)
        env_quality = assess_environmental_quality(image_np, detections, built_env, transport)
        health_outcomes = calculate_health_outcomes(vegetation, built_env, transport, env_quality)

        # Combine all results
        comprehensive_results = {
            'filename': uploaded_file.name,
            **vegetation,
            **built_env,
            **transport,
            **env_quality,
            **health_outcomes,
            'raw_detections': dict(detections)
        }

        return comprehensive_results, image_np

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return None, None


# =============================================================================
# 📊 ADVANCED VISUALIZATION FUNCTIONS
# =============================================================================
def create_environmental_dashboard(df):
    """Create comprehensive environmental analysis dashboard"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Utrecht Street View Environmental Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Vegetation Coverage Distribution
        if 'total_vegetation_percent' in df.columns:
            axes[0, 0].hist(df['total_vegetation_percent'], bins=15, color='forestgreen', alpha=0.7,
                            edgecolor='darkgreen')
            axes[0, 0].set_title('Vegetation Coverage Distribution')
            axes[0, 0].set_xlabel('Vegetation Coverage (%)')
            axes[0, 0].set_ylabel('Frequency')
            mean_veg = df['total_vegetation_percent'].mean()
            axes[0, 0].axvline(mean_veg, color='red', linestyle='--', label=f'Mean: {mean_veg:.1f}%')
            axes[0, 0].legend()

        # 2. Transportation Analysis
        transport_data = [
            df['car_count'].mean() if 'car_count' in df.columns else 0,
            df['bicycle_count'].mean() if 'bicycle_count' in df.columns else 0,
            df['bus_count'].mean() if 'bus_count' in df.columns else 0
        ]
        transport_labels = ['Cars', 'Bicycles', 'Buses']
        axes[0, 1].bar(transport_labels, transport_data, color=['red', 'blue', 'orange'])
        axes[0, 1].set_title('Average Transportation Mode Presence')
        axes[0, 1].set_ylabel('Average Count per Image')

        # 3. Environmental Health Scores
        health_columns = ['air_quality_proxy_score', 'walkability_score', 'overall_environmental_health_score']
        available_columns = [col for col in health_columns if col in df.columns]
        if available_columns:
            health_means = [df[score].mean() for score in available_columns]
            health_labels = ['Air Quality', 'Walkability', 'Overall Health'][:len(available_columns)]
            axes[0, 2].bar(health_labels, health_means,
                           color=['lightblue', 'lightgreen', 'gold'][:len(available_columns)])
            axes[0, 2].set_title('Environmental Health Indicators')
            axes[0, 2].set_ylabel('Score (0-100)')
            axes[0, 2].tick_params(axis='x', rotation=45)
        else:
            axes[0, 2].text(0.5, 0.5, 'No health data available', ha='center', va='center',
                            transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Environmental Health Indicators')

        # 4. Risk Assessment Matrix
        if 'pollution_risk_category' in df.columns and 'noise_risk_category' in df.columns:
            risk_categories = ['Low', 'Medium', 'High']
            pollution_risk = df['pollution_risk_category'].value_counts()
            noise_risk = df['noise_risk_category'].value_counts()

            x = np.arange(len(risk_categories))
            width = 0.35

            axes[1, 0].bar(x - width / 2, [pollution_risk.get(cat, 0) for cat in risk_categories],
                           width, label='Pollution Risk', color='red', alpha=0.7)
            axes[1, 0].bar(x + width / 2, [noise_risk.get(cat, 0) for cat in risk_categories],
                           width, label='Noise Risk', color='orange', alpha=0.7)
            axes[1, 0].set_title('Environmental Risk Distribution')
            axes[1, 0].set_xlabel('Risk Level')
            axes[1, 0].set_ylabel('Number of Locations')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(risk_categories)
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No risk data available', ha='center', va='center',
                            transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Environmental Risk Distribution')

        # 5. Health Outcome Potential
        health_outcome_columns = ['obesity_prevention_potential', 'stress_reduction_potential',
                                  'cardiovascular_health_score']
        available_outcome_columns = [col for col in health_outcome_columns if col in df.columns]
        if available_outcome_columns:
            outcome_means = [df[outcome].mean() for outcome in available_outcome_columns]
            outcome_labels = ['Obesity Prevention', 'Stress Reduction', 'Cardiovascular'][
                             :len(available_outcome_columns)]
            colors = ['purple', 'teal', 'crimson'][:len(available_outcome_columns)]
            axes[1, 1].bar(outcome_labels, outcome_means, color=colors)
            axes[1, 1].set_title('Health Outcome Support Potential')
            axes[1, 1].set_ylabel('Potential Score (0-100)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No health outcome data available', ha='center', va='center',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Health Outcome Support Potential')

        # 6. Correlation Heatmap of Key Variables
        correlation_vars = ['total_vegetation_percent', 'vehicle_density_score', 'walkability_score',
                            'air_quality_proxy_score', 'overall_environmental_health_score']
        available_corr_vars = [var for var in correlation_vars if var in df.columns]

        if len(available_corr_vars) >= 2:  # Need at least 2 variables for correlation
            try:
                correlation_matrix = df[available_corr_vars].corr()

                im = axes[1, 2].imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
                axes[1, 2].set_title('Environmental Factor Correlations')

                # Set ticks and labels
                axes[1, 2].set_xticks(range(len(available_corr_vars)))
                axes[1, 2].set_yticks(range(len(available_corr_vars)))

                # Create shorter labels for display
                short_labels = []
                for var in available_corr_vars:
                    if 'vegetation' in var:
                        short_labels.append('Vegetation')
                    elif 'vehicle' in var:
                        short_labels.append('Vehicle Density')
                    elif 'walkability' in var:
                        short_labels.append('Walkability')
                    elif 'air_quality' in var:
                        short_labels.append('Air Quality')
                    elif 'health_score' in var:
                        short_labels.append('Health Score')
                    else:
                        short_labels.append(var[:10])

                axes[1, 2].set_xticklabels(short_labels, rotation=45, ha='right')
                axes[1, 2].set_yticklabels(short_labels)

                # Add correlation values to heatmap
                for i in range(len(available_corr_vars)):
                    for j in range(len(available_corr_vars)):
                        text_color = 'white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black'
                        axes[1, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                        ha='center', va='center', color=text_color, fontsize=8)

                # Add colorbar
                plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

            except Exception as e:
                axes[1, 2].text(0.5, 0.5, f'Correlation error:\n{str(e)[:50]}', ha='center', va='center',
                                transform=axes[1, 2].transAxes, fontsize=8)
                axes[1, 2].set_title('Environmental Factor Correlations')
        else:
            axes[1, 2].text(0.5, 0.5, 'Need at least 2 variables\nfor correlation analysis',
                            ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Environmental Factor Correlations')

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating dashboard: {e}")
        # Return simple placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Dashboard Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        return fig


def create_health_impact_radar(df_row):
    """Create radar chart for individual location health impact"""
    try:
        categories = ['Physical Activity\nPromotion', 'Respiratory\nHealth Support', 'Mental\nWellbeing',
                      'Cardiovascular\nHealth', 'Air Quality', 'Walkability']

        values = [
            df_row.get('physical_activity_promotion_score', 50),
            100 - df_row.get('respiratory_health_risk', 50),  # Invert risk to support
            df_row.get('mental_wellbeing_score', 50),
            df_row.get('cardiovascular_health_score', 50),
            df_row.get('air_quality_proxy_score', 50),
            df_row.get('walkability_score', 50)
        ]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        ax.set_title(f'Health Impact Profile: {df_row.get("filename", "Unknown")}', pad=20, fontsize=14,
                     fontweight='bold')

        return fig
    except Exception as e:
        st.error(f"Error creating radar chart: {e}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Radar Chart Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        return fig


# =============================================================================
# 🌍 STREAMLIT APPLICATION
# =============================================================================
def main():
    st.title("🌆 Comprehensive Urban Exposome Analysis by Shivam Bhardwaj")
    st.markdown("##### Advanced Environmental Epidemiology Tool using Street View Imagery")
    st.markdown(
        "**Objective:** Extract comprehensive environmental health determinants from street-level imagery to support population health research and urban planning")

    # Show YOLO status
    if not YOLO_AVAILABLE:
        st.warning(
            "⚠️ YOLO object detection is not available. Analysis will continue with computer vision methods only.")
    elif model is None:
        st.warning("⚠️ YOLO model failed to load. Analysis will continue with computer vision methods only.")
    else:
        st.success("✅ YOLO object detection is ready!")

    # Enhanced info section
    with st.expander("🔬 Comprehensive Analysis Framework"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🌿 Environmental Exposures:**
            - Vegetation coverage & distribution analysis
            - Tree canopy density assessment
            - Green space connectivity metrics
            - Air quality proxy indicators
            - Noise exposure estimation

            **🏗️ Built Environment:**
            - Building density & height estimation
            - Road width & infrastructure assessment
            - Urban density scoring
            - Shadow analysis for microclimate
            """)

        with col2:
            st.markdown("""
            **🚲 Transportation Analysis:**
            - Multi-modal transport detection
            - Active transportation infrastructure
            - Traffic intensity assessment
            - Parking density estimation

            **🏥 Health Outcome Modeling:**
            - Physical activity promotion potential
            - Respiratory health risk assessment
            - Mental wellbeing support indicators
            - Cardiovascular health environment scoring
            """)

    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Upload Street View Images for Comprehensive Analysis",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple high-quality street view images for detailed environmental health analysis"
    )

    if uploaded_files:
        st.success(f"✅ Processing {len(uploaded_files)} images with comprehensive environmental health analysis...")

        results = []
        processed_images = {}

        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process all images
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name} - Extracting features...")

            comprehensive_results, image_np = comprehensive_image_analysis(uploaded_file)

            if comprehensive_results is not None:
                processed_images[uploaded_file.name] = image_np

                # Add simplified key metrics for easy access
                comprehensive_results.update({
                    'Cars': comprehensive_results.get('car_count', 0),
                    'Bicycles': comprehensive_results.get('bicycle_count', 0),
                    'People': comprehensive_results.get('raw_detections', {}).get('person', 0),
                    'Total Vehicles': comprehensive_results.get('total_vehicles', 0),
                    'Greenery %': round(comprehensive_results.get('total_vegetation_percent', 0), 2),
                    'Infrastructure Items': (comprehensive_results.get('raw_detections', {}).get('bench', 0) +
                                             comprehensive_results.get('raw_detections', {}).get('fire hydrant', 0) +
                                             comprehensive_results.get('raw_detections', {}).get('stop sign', 0) +
                                             comprehensive_results.get('raw_detections', {}).get('traffic light', 0)),
                    'Air Quality Score': round(comprehensive_results.get('air_quality_proxy_score', 0), 1),
                    'Health Score': round(comprehensive_results.get('overall_environmental_health_score', 0), 1)
                })

                results.append(comprehensive_results)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.text("✅ Comprehensive analysis complete!")

        if results:
            df = pd.DataFrame(results)

            # =============================================================================
            # 📊 COMPREHENSIVE DASHBOARD
            # =============================================================================
            st.subheader("🎯 Environmental Health Dashboard")

            # Key Performance Indicators
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                if 'total_vegetation_percent' in df.columns:
                    avg_vegetation = df['total_vegetation_percent'].mean()
                    st.metric("🌿 Avg Vegetation", f"{avg_vegetation:.1f}%",
                              delta=f"{avg_vegetation - 25:.1f}%" if avg_vegetation != 25 else None)
                else:
                    st.metric("🌿 Avg Vegetation", "N/A")

            with col2:
                if 'air_quality_proxy_score' in df.columns:
                    avg_air_quality = df['air_quality_proxy_score'].mean()
                    st.metric("🌬️ Air Quality Score", f"{avg_air_quality:.1f}/100",
                              delta=f"{avg_air_quality - 70:.1f}" if avg_air_quality != 70 else None)
                else:
                    st.metric("🌬️ Air Quality Score", "N/A")

            with col3:
                if 'walkability_score' in df.columns:
                    avg_walkability = df['walkability_score'].mean()
                    st.metric("🚶 Walkability", f"{avg_walkability:.1f}/100",
                              delta=f"{avg_walkability - 60:.1f}" if avg_walkability != 60 else None)
                else:
                    st.metric("🚶 Walkability", "N/A")

            with col4:
                if 'overall_environmental_health_score' in df.columns:
                    avg_health_score = df['overall_environmental_health_score'].mean()
                    st.metric("🏥 Health Score", f"{avg_health_score:.1f}/100",
                              delta=f"{avg_health_score - 65:.1f}" if avg_health_score != 65 else None)
                else:
                    st.metric("🏥 Health Score", "N/A")

            with col5:
                if 'bicycle_count' in df.columns:
                    bike_friendly = len(df[df['bicycle_count'] > 0]) / len(df) * 100 if len(df) > 0 else 0
                    st.metric("🚲 Bike Friendly", f"{bike_friendly:.1f}%")
                else:
                    st.metric("🚲 Bike Friendly", "N/A")

            # Comprehensive visualization
            try:
                dashboard_fig = create_environmental_dashboard(df)
                st.pyplot(dashboard_fig)
            except Exception as e:
                st.error(f"Error creating dashboard: {e}")

            # =============================================================================
            # 🏥 EPIDEMIOLOGICAL INSIGHTS
            # =============================================================================
            st.subheader("🏥 Epidemiological Applications & Health Impact Assessment")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["📍 Individual Locations", "📈 Comparative Insights", "🧬 Health Profiles", "📤 Export Results"])

            with tab1:
                st.markdown("Explore detailed health-related environmental indicators for each uploaded image.")

                if len(df) > 0:
                    selected_filename = st.selectbox("Select Image", df['filename'].unique())
                    selected_row = df[df['filename'] == selected_filename].iloc[0]

                    if selected_filename in processed_images:
                        st.image(processed_images[selected_filename], caption=f"📸 {selected_filename}",
                                 use_column_width=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("🌿 Vegetation Coverage", f"{selected_row.get('total_vegetation_percent', 0):.1f}%")
                        st.metric("🚲 Active Transport Ratio", f"{selected_row.get('active_transport_ratio', 0):.2f}")
                        st.metric("🏥 Health Environment Score",
                                  f"{selected_row.get('overall_environmental_health_score', 0):.1f}/100")
                        st.metric("🧘 Stress Reduction Potential",
                                  f"{selected_row.get('stress_reduction_potential', 0):.1f}/100")

                    with col2:
                        st.metric("💨 Air Quality Proxy", f"{selected_row.get('air_quality_proxy_score', 0):.1f}/100")
                        st.metric("🔊 Noise Exposure", f"{selected_row.get('noise_exposure_level', 0):.1f}/100")
                        st.metric("💚 Mental Wellbeing Score",
                                  f"{selected_row.get('mental_wellbeing_score', 0):.1f}/100")
                        st.metric("❤️ Cardiovascular Health",
                                  f"{selected_row.get('cardiovascular_health_score', 0):.1f}/100")

                    try:
                        radar_fig = create_health_impact_radar(selected_row)
                        st.pyplot(radar_fig)
                    except Exception as e:
                        st.error(f"Error creating radar chart: {e}")
                else:
                    st.warning("No data available for individual analysis.")

            with tab2:
                st.markdown("Aggregate analysis across all images.")
                if len(df) > 0:
                    # Display key summary table with main metrics
                    st.markdown("### 📊 Key Metrics Summary")
                    summary_columns = ['filename', 'Cars', 'Bicycles', 'People', 'Total Vehicles',
                                       'Greenery %', 'Infrastructure Items', 'Air Quality Score', 'Health Score']
                    available_summary_cols = [col for col in summary_columns if col in df.columns]

                    if available_summary_cols:
                        st.dataframe(df[available_summary_cols].round(2))

                    # Show aggregate statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if 'Cars' in df.columns:
                            st.metric("🚗 Total Cars Detected", int(df['Cars'].sum()))
                            st.metric("📊 Avg Cars per Image", f"{df['Cars'].mean():.1f}")

                    with col2:
                        if 'Bicycles' in df.columns:
                            st.metric("🚲 Total Bicycles Detected", int(df['Bicycles'].sum()))
                            st.metric("📊 Avg Bicycles per Image", f"{df['Bicycles'].mean():.1f}")

                    with col3:
                        if 'People' in df.columns:
                            st.metric("👥 Total People Detected", int(df['People'].sum()))
                            st.metric("📊 Avg People per Image", f"{df['People'].mean():.1f}")

                    with col4:
                        if 'Greenery %' in df.columns:
                            st.metric("🌿 Avg Greenery Coverage", f"{df['Greenery %'].mean():.1f}%")
                            st.metric("🏞️ Max Greenery Found", f"{df['Greenery %'].max():.1f}%")

                    # Detailed analysis table
                    st.markdown("### 📋 Complete Environmental Analysis")
                    display_columns = ['filename']
                    optional_columns = ['total_vegetation_percent', 'air_quality_proxy_score',
                                        'walkability_score', 'vehicle_density_score',
                                        'obesity_prevention_potential', 'respiratory_health_risk',
                                        'mental_wellbeing_score', 'cardiovascular_health_score']

                    for col in optional_columns:
                        if col in df.columns:
                            display_columns.append(col)

                    st.dataframe(df[display_columns])

                    # Correlation analysis
                    if len(display_columns) > 2:  # Need at least some numeric columns
                        st.markdown("### 📊 Correlation Matrix of Health-Related Factors")
                        correlation_vars = [col for col in optional_columns if col in df.columns]

                        if len(correlation_vars) >= 2:
                            try:
                                corr_matrix = df[correlation_vars].corr()
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                                ax.set_title("Environmental Health Factor Correlations")
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error creating correlation matrix: {e}")
                        else:
                            st.info("Not enough numeric variables for correlation analysis.")
                else:
                    st.warning("No data available for comparative analysis.")

            with tab3:
                st.markdown("### 🧬 Location Health Profiles Comparison")
                if len(df) > 0:
                    profiles_to_compare = st.multiselect("Select Locations to Compare",
                                                         df['filename'].tolist(),
                                                         default=df['filename'].tolist()[:min(3, len(df))])

                    if profiles_to_compare:
                        try:
                            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
                            categories = ['Physical Activity\nPromotion', 'Respiratory\nHealth Support',
                                          'Mental\nWellbeing', 'Cardiovascular\nHealth',
                                          'Air Quality', 'Walkability']
                            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                            angles += angles[:1]

                            colors = plt.cm.Set3(np.linspace(0, 1, len(profiles_to_compare)))

                            for idx, profile in enumerate(profiles_to_compare):
                                row = df[df['filename'] == profile].iloc[0]
                                values = [
                                    row.get('physical_activity_promotion_score', 50),
                                    100 - row.get('respiratory_health_risk', 50),
                                    row.get('mental_wellbeing_score', 50),
                                    row.get('cardiovascular_health_score', 50),
                                    row.get('air_quality_proxy_score', 50),
                                    row.get('walkability_score', 50)
                                ]
                                values += values[:1]
                                ax.plot(angles, values, label=profile, color=colors[idx], linewidth=2)
                                ax.fill(angles, values, alpha=0.1, color=colors[idx])

                            ax.set_xticks(angles[:-1])
                            ax.set_xticklabels(categories)
                            ax.set_ylim(0, 100)
                            ax.set_title("Health Impact Radar Comparison", size=14, pad=20)
                            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error creating comparison radar: {e}")
                    else:
                        st.info("Please select at least one location to compare.")
                else:
                    st.warning("No data available for health profile comparison.")

            with tab4:
                st.markdown("### 📤 Export Analysis Results")
                if len(df) > 0:
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV Summary",
                        data=csv_data,
                        file_name='utrecht_environmental_analysis.csv',
                        mime='text/csv'
                    )

                    st.markdown("### 📋 Data Summary")
                    st.write(f"**Total Images Processed:** {len(df)}")
                    st.write(f"**Analysis Columns:** {len(df.columns)}")

                    if 'total_vegetation_percent' in df.columns:
                        st.write(f"**Average Vegetation Coverage:** {df['total_vegetation_percent'].mean():.1f}%")
                    if 'overall_environmental_health_score' in df.columns:
                        st.write(f"**Average Health Score:** {df['overall_environmental_health_score'].mean():.1f}/100")

                    st.markdown(
                        "You can now use this data for further statistical modeling or epidemiological research.")
                else:
                    st.warning("No data available for export.")

        else:
            st.error("No images were successfully processed. Please check your image files and try again.")

    st.markdown("---")
    st.markdown(
        "🔚 **End of Analysis.** Thank you")


if __name__ == "__main__":
    main()