import streamlit as st

st.set_page_config(page_title="Urban Greening Planner - README", layout="wide")

st.title("About the Urban Greening Planner")

st.header("Goal")
st.markdown("""
The **Urban Greening Planner** is an interactive application designed to help urban planners and environmental enthusiasts visualize and predict the microclimatic impact of urban greening scenarios. By allowing users to modify land use on a map, the tool forecasts changes in Land Surface Temperature (LST) and Normalized Difference Vegetation Index (NDVI), providing insights into the environmental effects of their plans.
""")

st.info(" This is not a production-ready tool. It is intended for demonstration and educational purposes only",icon="💡")

st.header("Data Used")
st.markdown("""
This application leverages satellite imagery and land cover data from Google Earth Engine (GEE):
*   **Sentinel-2:** Provides high-resolution multispectral imagery (including RGB and near-infrared bands for NDVI calculation).
*   **Landsat 8:** Used for deriving Land Surface Temperature (LST).
*   **Dynamic World:** A 10m resolution land cover classification dataset, providing detailed information on various land use types (e.g., trees, built-up areas, water).
*   **SimpleMaps World Cities Population Data:** Utilized to automatically assign a population to the selected location, which is then fed into the prediction model as a contextual feature.
""")

st.header("Live Data Fetching and Quality Considerations")
st.markdown("""
A critical aspect of this tool is its reliance on **live data fetching** from Google Earth Engine. When you select a location and a time period, the application queries GEE in real-time to retrieve the latest available satellite imagery.

**Important considerations regarding data quality and results:**
*   **Satellite Availability:** The quality and completeness of the fetched data are directly dependent on the availability of cloud-free satellite imagery for your chosen location and time. GEE's vast archives are generally excellent, but specific dates or highly cloudy regions might yield less complete data.
*   **Cloud Cover:** While some internal processing (like pixel masking and compositing) is applied to mitigate cloud effects, persistent cloud cover over a region can impact the quality of the raw data fetched.
*   **Minimal Pre-processing:** Beyond fetching, resizing, and basic normalization for the model, no extensive pre-processing or gap-filling is performed on the raw satellite inputs by the application itself. The prediction model is then applied to this fetched data.
*   **Model's Role:** The model's predictions are based on the patterns it learned from a large dataset. Therefore, the accuracy of the forecast is a combination of the quality of the live input data and the predictive power of the underlying model.
""")

st.header("How to Use")
st.markdown("""
1.  **Configure Location & Time:** Use the sidebar to set the latitude, longitude, and desired time periods for current (T1) and future (T2) scenarios.
2.  **Fetch Satellite Data:** Click the "Fetch Satellite Data" button to retrieve current satellite imagery for the selected location. The nearest city population will be automatically determined.
3.  **Design Future Scenario:** The canvas will display the current land use. Use the brush palette and drawing tools to simulate changes (e.g., adding green spaces, expanding built-up areas).
4.  **Forecast Impact:** Select a trained model and click "Run Prediction" to see the forecasted NDVI and LST for your designed future scenario, along with the predicted temperature change.
""")
