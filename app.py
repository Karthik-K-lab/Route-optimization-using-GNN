# app.py

from shapely.geometry import Point
import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import folium
from math import radians, sin, cos, sqrt, atan2
import gdown
import os
from streamlit_folium import st_folium
from matplotlib import cm, colors as mcolors

# ------------------------------
# 🌍 App Title
# ------------------------------
st.set_page_config(page_title="Flood-Safe Route Planner", layout="wide")
st.title("🌊 Flood-Safe Route Planner (Optimized Demo)")

# ------------------------------
# 📁 Google Drive graph file
# ------------------------------
drive_id = "18EcOIP4ReNoOgh-JKah42vsqhG4k1cXr"
local_path = "tbm.graphml"

# ------------------------------
# 1️⃣ Load Map Data (Cached)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_graph_from_drive():
    """Load the pre-downloaded OSM graph from Google Drive (cached once)."""
    if not os.path.exists(local_path):
        st.info("📥 Downloading map data from Google Drive (first time only)...")
        gdown.download(id=drive_id, output=local_path, quiet=False)
    return ox.load_graphml(local_path)

def add_edge_lengths(G):
    """Ensure every edge has a 'length' attribute."""
    for u, v, key, data in G.edges(keys=True, data=True):
        if "length" not in data or data["length"] is None:
            lat1, lon1 = G.nodes[u]["y"], G.nodes[u]["x"]
            lat2, lon2 = G.nodes[v]["y"], G.nodes[v]["x"]
            R = 6371000
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            data["length"] = R * c
    return G

G = load_graph_from_drive()
G = add_edge_lengths(G)

# Convert graph to GeoDataFrames (cached to avoid reprocessing)
@st.cache_resource
def get_graph_gdfs(G):
    return ox.graph_to_gdfs(G)

nodes, edges = get_graph_gdfs(G)

# ------------------------------
# 2️⃣ Flood Risk Data (Synthetic, Cached)
# ------------------------------
@st.cache_data
def generate_flood_values(num_nodes, seed=42):
    np.random.seed(seed)
    return np.random.rand(num_nodes)

def compute_flood_data(nodes):
    flood_vals = generate_flood_values(len(nodes))
    nodes = nodes.copy()
    nodes["flood_value"] = flood_vals
    min_val = float(np.min(flood_vals))
    max_val = float(np.max(flood_vals))
    range_val = max_val - min_val if max_val != min_val else 1.0
    return nodes, min_val, max_val, range_val

nodes, min_val, max_val, range_val = compute_flood_data(nodes)

# ------------------------------
# 3️⃣ Map Setup (Faster tiles)
# ------------------------------
st.subheader("🗺️ Pick Start and End Locations on the Map")

center_lat, center_lon = nodes.geometry.y.mean(), nodes.geometry.x.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

# Draw flood risk markers (simplified to speed up rendering)
sample_nodes = nodes.sample(min(2000, len(nodes)))  # plot fewer points for speed
for _, row in sample_nodes.iterrows():
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

coords = st_folium(m, height=500, width=800, returned_objects=["last_clicked"])

# ------------------------------
# 4️⃣ Handle Map Clicks
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
# 5️⃣ Route Calculation (Optimized)
# ------------------------------
def nearest_node_fast(G, lat, lon):
    """Vectorized nearest node search (fast NumPy method)."""
    x = np.array([d["x"] for n, d in G.nodes(data=True)])
    y = np.array([d["y"] for n, d in G.nodes(data=True)])
    dlat = np.radians(y - lat)
    dlon = np.radians(x - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(y)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = 6371 * c
    nearest_idx = np.argmin(dist)
    return list(G.nodes())[nearest_idx]

if start_lat and end_lat:
    st.success(f"Start: ({start_lat:.5f}, {start_lon:.5f}) → End: ({end_lat:.5f}, {end_lon:.5f})")

    u = nearest_node_fast(G, start_lat, start_lon)
    v = nearest_node_fast(G, end_lat, end_lon)

    def edge_weight(u, v, d):
        f1 = nodes.at[u, "flood_value"]
        f2 = nodes.at[v, "flood_value"]
        factor = (f1 + f2) / 2
        length = d.get("length", 1.0)
        return length * (1 + 10 * factor)

    try:
        safest_path = nx.shortest_path(G, source=u, target=v, weight=edge_weight)
        path_coords = [(nodes.at[n, "geometry"].y, nodes.at[n, "geometry"].x) for n in safest_path]

        m2 = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles="CartoDB positron")
        folium.Marker([start_lat, start_lon], icon=folium.Icon(color="green"), tooltip="Start").add_to(m2)
        folium.Marker([end_lat, end_lon], icon=folium.Icon(color="red"), tooltip="End").add_to(m2)
        folium.PolyLine(path_coords, color="blue", weight=6, tooltip="Safest Route").add_to(m2)
        st_folium(m2, height=500, width=800)

    except Exception as e:
        st.error(f"⚠️ Path could not be found: {e}")
else:
    st.info("🟢 Click once for start point and once for end point on the map.")
