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
# Page config & title
# ------------------------------
st.set_page_config(page_title="Flood-Safe Route Planner", layout="wide")
st.title("🌊 Flood-Safe Route Planner")

# ------------------------------
# Google Drive graph file (replace drive_id)
# ------------------------------
drive_id = "1gr1Z1EyY2h-R0o4ZVEy4dxCMpHFpvjRQ"  # <-- replace with your file id
local_path = "chennai.graphml"

# ------------------------------
# 1) Load graph from Drive (cached safely: no unhashable args)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_graph_from_drive(drive_id_local: str = drive_id, local_path_local: str = local_path):
    """Download (once) and load GraphML from Google Drive using gdown."""
    if not os.path.exists(local_path_local):
        st.info("📥 Downloading map data...")
        # gdown handles Google Drive confirmation tokens for large files
        gdown.download(id=drive_id_local, output=local_path_local, quiet=False)
    G_loaded = ox.load_graphml(local_path_local)
    return G_loaded

G = load_graph_from_drive()

# ------------------------------
# 2) Ensure 'length' on edges (do not cache; G is unhashable)
# ------------------------------
def add_edge_lengths(G_local):
    """Ensure every edge has a 'length' attribute (meters)."""
    for u, v, key, data in G_local.edges(keys=True, data=True):
        if "length" not in data or data["length"] is None:
            # node coordinates must exist
            lat1, lon1 = G_local.nodes[u].get("y"), G_local.nodes[u].get("x")
            lat2, lon2 = G_local.nodes[v].get("y"), G_local.nodes[v].get("x")
            # if any coordinate missing, skip
            if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
                data["length"] = 1.0
                continue
            R = 6371000  # meters
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            data["length"] = R * c
    return G_local

G = add_edge_lengths(G)

# ------------------------------
# 3) Graph -> GeoDataFrames (no caching of functions that take G)
# ------------------------------
nodes, edges = ox.graph_to_gdfs(G)  # nodes: GeoDataFrame

# ------------------------------
# 4) Flood values: cache only numeric generator (safe)
# ------------------------------
@st.cache_data
def generate_flood_values(num_nodes: int, seed: int = 42):
    np.random.seed(seed)
    return np.random.rand(num_nodes)

def attach_flood_values(nodes_gdf):
    flood_vals = generate_flood_values(len(nodes_gdf))
    nodes_copy = nodes_gdf.copy()
    nodes_copy["flood_value"] = flood_vals
    min_val = float(np.min(flood_vals))
    max_val = float(np.max(flood_vals))
    range_val = max_val - min_val if max_val != min_val else 1.0
    return nodes_copy, min_val, max_val, range_val

nodes, min_val, max_val, range_val = attach_flood_values(nodes)

# ------------------------------
# 5) Precompute simple arrays for fast nearest-node (no hashing)
# ------------------------------
# Build node_id list and coordinate arrays only once per run
node_ids = list(nodes.index)
node_y = nodes.geometry.y.to_numpy()  # lat
node_x = nodes.geometry.x.to_numpy()  # lon

def nearest_node_fast(lat, lon):
    """Vectorized approximate nearest node (Haversine, returns node id)."""
    # Compute haversine distances vectorized (fast in numpy)
    dlat = np.radians(node_y - lat)
    dlon = np.radians(node_x - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(node_y)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = 6371 * c  # kilometers
    idx = np.argmin(dist)
    return node_ids[int(idx)]

# ------------------------------
# 6) Build initial Folium map (sampled markers for speed)
# ------------------------------
st.subheader("🗺️ Pick Start and End Locations on the Map")

center_lat, center_lon = float(node_y.mean()), float(node_x.mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

# Sample nodes to plot (limit for speed in browser)
max_plot = 3000
if len(nodes) <= max_plot:
    plot_nodes = nodes
else:
    plot_nodes = nodes.sample(max_plot, random_state=42)

for _, row in plot_nodes.iterrows():
    flood_norm = (row["flood_value"] - min_val) / range_val
    color = cm.Reds(flood_norm)
    hexcolor = mcolors.rgb2hex(color[:3])
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color=hexcolor,
        fill=True,
        fill_opacity=0.5,
        parse_html=False,
    ).add_to(m)

# Render the map and capture clicks
coords = st_folium(m, height=560, width=1000, returned_objects=["last_clicked"])

# ------------------------------
# 7) Handle clicks (session_state keeps last two)
# ------------------------------
start_lat = start_lon = end_lat = end_lon = None

if coords.get("last_clicked"):
    if "all_clicks" not in st.session_state:
        st.session_state["all_clicks"] = []
    last = coords["last_clicked"]
    if not st.session_state["all_clicks"] or last != st.session_state["all_clicks"][-1]:
        st.session_state["all_clicks"].append(last)
        if len(st.session_state["all_clicks"]) > 2:
            st.session_state["all_clicks"] = st.session_state["all_clicks"][-2:]

    if len(st.session_state["all_clicks"]) >= 1:
        start_lat = st.session_state["all_clicks"][0]["lat"]
        start_lon = st.session_state["all_clicks"][0]["lng"]
    if len(st.session_state["all_clicks"]) >= 2:
        end_lat = st.session_state["all_clicks"][1]["lat"]
        end_lon = st.session_state["all_clicks"][1]["lng"]

# ------------------------------
# 8) If both points selected => compute and draw path
# ------------------------------
if start_lat and end_lat:
    st.success(f"Start: ({start_lat:.5f}, {start_lon:.5f}) → End: ({end_lat:.5f}, {end_lon:.5f})")

    u = nearest_node_fast(start_lat, start_lon)
    v = nearest_node_fast(end_lat, end_lon)

    def edge_weight(u_local, v_local, data_local):
        # safe flood lookup (fallback to 0 if node missing)
        try:
            f1 = nodes.at[u_local, "flood_value"]
        except Exception:
            f1 = 0.0
        try:
            f2 = nodes.at[v_local, "flood_value"]
        except Exception:
            f2 = 0.0
        factor = (f1 + f2) / 2
        length = data_local.get("length", 1.0)
        return length * (1 + 10 * factor)

    with st.spinner("🧭 Calculating safest path..."):
        try:
            safest_path = nx.shortest_path(G, source=u, target=v, weight=edge_weight)
            path_coords = [(nodes.at[n, "geometry"].y, nodes.at[n, "geometry"].x) for n in safest_path]

            # Draw result map (only path + start/end markers)
            m2 = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles="CartoDB positron")
            folium.Marker([start_lat, start_lon], icon=folium.Icon(color="green"), tooltip="Start").add_to(m2)
            folium.Marker([end_lat, end_lon], icon=folium.Icon(color="red"), tooltip="End").add_to(m2)
            folium.PolyLine(path_coords, color="blue", weight=6, tooltip="Safest Route").add_to(m2)
            for pt in path_coords:
                folium.CircleMarker(pt, radius=2, color="blue", fill=True).add_to(m2)
            st_folium(m2, height=560, width=1000)

        except Exception as e:
            st.error(f"⚠️ Path could not be found: {e}")
else:
    st.info("Click once for start and once for end on the map (two clicks).")
