# app.py

from shapely.geometry import Point
import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import folium
from math import radians, sin, cos, sqrt, atan2
import requests
import gdown
import os
from streamlit_folium import st_folium
from matplotlib import cm, colors as mcolors

st.title("🌊 Flood-Safe Route Planner (Demo)")

# Google Drive file URL (replace with your own)
drive_id = "1gr1Z1EyY2h-R0o4ZVEy4dxCMpHFpvjRQ"
local_path = "chennai.graphml"

# ------------------------------
# 1️⃣ Load the map from Drive (cached)
# ------------------------------
@st.cache_data(show_spinner=True)
def load_graph_from_drive():
    if not os.path.exists(local_path):
        st.info("📥 Downloading map data from Google Drive (only once)...")
        gdown.download(id=drive_id, output=local_path, quiet=False)
    G = ox.load_graphml(local_path)
    return G

# Load the graph
G = load_graph_from_drive()

# Convert to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# ------------------------------
# 2️⃣ Generate synthetic flood data
# ------------------------------
np.random.seed(42)
flood_vals = np.random.rand(len(nodes))
nodes["flood_value"] = flood_vals

# Normalization for color scaling
min_val, max_val = float(np.min(flood_vals)), float(np.max(flood_vals))
range_val = max_val - min_val if max_val != min_val else 1.0

# ------------------------------
# 3️⃣ Interactive Map (Select Start/End)
# ------------------------------
st.subheader("🗺️ Pick Start and End Locations on the Map")

center_lat, center_lon = nodes.geometry.y.mean(), nodes.geometry.x.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

# Draw flood risk color markers
for _, row in nodes.iterrows():
    flood_norm = (row["flood_value"] - min_val) / range_val
    color = cm.Reds(flood_norm)
    hexcolor = mcolors.rgb2hex(color[:3])
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color=hexcolor,
        fill=True,
        fill_opacity=0.5,
    ).add_to(m)

# Let the user click on map
coords = st_folium(m, height=500, width=700, returned_objects=["last_clicked"])

# ------------------------------
# 4️⃣ Handle map clicks
# ------------------------------
start_lat, start_lon, end_lat, end_lon = None, None, None, None

if coords.get("last_clicked"):
    if "all_clicks" not in st.session_state:
        st.session_state["all_clicks"] = []
    if (
        len(st.session_state["all_clicks"]) == 0
        or coords["last_clicked"] != st.session_state["all_clicks"][-1]
    ):
        st.session_state["all_clicks"].append(coords["last_clicked"])
        if len(st.session_state["all_clicks"]) > 2:
            st.session_state["all_clicks"] = st.session_state["all_clicks"][-2:]

    if len(st.session_state["all_clicks"]) >= 1:
        start_lat = st.session_state["all_clicks"][0]["lat"]
        start_lon = st.session_state["all_clicks"][0]["lng"]
    if len(st.session_state["all_clicks"]) >= 2:
        end_lat = st.session_state["all_clicks"][1]["lat"]
        end_lon = st.session_state["all_clicks"][1]["lng"]

# ------------------------------
# 5️⃣ Route Calculation
# ------------------------------
if start_lat and end_lat:
    st.success(f"Start: ({start_lat:.5f}, {start_lon:.5f}) → End: ({end_lat:.5f}, {end_lon:.5f})")

    def nearest_node(G, lat, lon):
    """Find the nearest node manually (no rtree or pygeos needed)."""
        min_dist = float("inf")
        nearest = None
        for node, data in G.nodes(data=True):
            node_lat = data.get('y')
            node_lon = data.get('x')
            if node_lat is None or node_lon is None:
                continue
            # Haversine distance (in km)
            R = 6371
            dlat = radians(lat - node_lat)
            dlon = radians(lon - node_lon)
            a = sin(dlat / 2)**2 + cos(radians(lat)) * cos(radians(node_lat)) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            dist = R * c
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    u = nearest_node(G, start_lat, start_lon)
    v = nearest_node(G, end_lat, end_lon)

    def edge_weight(u, v, d):
        f1 = nodes.at[u, "flood_value"]
        f2 = nodes.at[v, "flood_value"]
        factor = (f1 + f2) / 2
        return d["length"] * (1 + 10 * factor)

    try:
        safest_path = nx.shortest_path(G, source=u, target=v, weight=edge_weight)
        path_coords = [(nodes.at[n, "geometry"].y, nodes.at[n, "geometry"].x) for n in safest_path]

        # Display route on map
        m2 = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles="OpenStreetMap")
        folium.Marker([start_lat, start_lon], icon=folium.Icon(color="green"), tooltip="Start").add_to(m2)
        folium.Marker([end_lat, end_lon], icon=folium.Icon(color="red"), tooltip="End").add_to(m2)
        folium.PolyLine(path_coords, color="blue", weight=6, tooltip="Safest Route").add_to(m2)
        for pt in path_coords:
            folium.CircleMarker(pt, radius=2, color="blue", fill=True).add_to(m2)
        st_folium(m2, height=500, width=700)

    except Exception as e:
        st.error(f"⚠️ Path could not be found: {e}")
else:
    st.info("Click once for start point and once for end point on the map.")
