# app.py
import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import folium
from math import radians, sin, cos, sqrt, atan2
import gdown, os
from streamlit_folium import st_folium
from matplotlib import cm, colors as mcolors
from scipy.spatial import cKDTree

st.set_page_config(page_title="Flood-Safe Route Planner", layout="wide")
st.title("🌊 Flood-Safe Route Planner ( Demo)")

# Google Drive GraphML file
drive_id = "18EcOIP4ReNoOgh-JKah42vsqhG4k1cXr"
local_path = "tbm.graphml"

# -----------------------------------------------------
# 1️⃣ Cached Graph Loader
# -----------------------------------------------------
@st.cache_resource(show_spinner="Loading map graph, please wait...")
def load_graph():
    if not os.path.exists(local_path):
        st.info("📥 Downloading map data from Google Drive (only once)...")
        gdown.download(id=drive_id, output=local_path, quiet=False)
    G = ox.load_graphml(local_path)

    # Ensure 'length' exists for all edges
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'length' not in data or data['length'] is None:
            lat1, lon1 = G.nodes[u]['y'], G.nodes[u]['x']
            lat2, lon2 = G.nodes[v]['y'], G.nodes[v]['x']
            R = 6371000
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            data['length'] = R * c
    return G

G = load_graph()
nodes, edges = ox.graph_to_gdfs(G)

# -----------------------------------------------------
# 2️⃣ Precompute Flood Risk (cached)
# -----------------------------------------------------
@st.cache_data
def compute_flood_data(nodes):
    np.random.seed(42)
    flood_vals = np.random.rand(len(nodes))
    nodes["flood_value"] = flood_vals
    min_val, max_val = float(flood_vals.min()), float(flood_vals.max())
    range_val = max_val - min_val if max_val != min_val else 1.0
    return nodes, min_val, max_val, range_val

nodes, min_val, max_val, range_val = compute_flood_data(nodes)

# -----------------------------------------------------
# 3️⃣ KDTree for Fast Nearest Node Search
# -----------------------------------------------------
@st.cache_resource
def build_kdtree(nodes):
    coords = np.array(list(zip(nodes.geometry.y, nodes.geometry.x)))
    tree = cKDTree(coords)
    return tree, nodes.index.to_list()

kdtree, node_ids = build_kdtree(nodes)

def nearest_node(lat, lon):
    dist, idx = kdtree.query((lat, lon))
    return node_ids[idx]

# -----------------------------------------------------
# 4️⃣ Map Visualization
# -----------------------------------------------------
st.subheader("🗺️ Pick Start and End Locations on the Map")

center_lat, center_lon = nodes.geometry.y.mean(), nodes.geometry.x.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

# Show fewer markers for speed
sampled_nodes = nodes.sample(min(3000, len(nodes)), random_state=42)
for _, row in sampled_nodes.iterrows():
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

coords = st_folium(m, height=500, width=900, returned_objects=["last_clicked"])

# -----------------------------------------------------
# 5️⃣ Handle clicks & path calculation
# -----------------------------------------------------
start_lat, start_lon, end_lat, end_lon = None, None, None, None
if coords.get("last_clicked"):
    if "clicks" not in st.session_state:
        st.session_state["clicks"] = []
    click = coords["last_clicked"]
    if not st.session_state["clicks"] or click != st.session_state["clicks"][-1]:
        st.session_state["clicks"].append(click)
        if len(st.session_state["clicks"]) > 2:
            st.session_state["clicks"] = st.session_state["clicks"][-2:]

    if len(st.session_state["clicks"]) >= 1:
        start_lat = st.session_state["clicks"][0]["lat"]
        start_lon = st.session_state["clicks"][0]["lng"]
    if len(st.session_state["clicks"]) >= 2:
        end_lat = st.session_state["clicks"][1]["lat"]
        end_lon = st.session_state["clicks"][1]["lng"]

if start_lat and end_lat:
    st.success(f"Start: ({start_lat:.5f}, {start_lon:.5f}) → End: ({end_lat:.5f}, {end_lon:.5f})")

    u = nearest_node(start_lat, start_lon)
    v = nearest_node(end_lat, end_lon)

    def edge_weight(u, v, d):
        f1 = nodes.at[u, "flood_value"]
        f2 = nodes.at[v, "flood_value"]
        factor = (f1 + f2) / 2
        length = d.get("length", 1.0)
        return length * (1 + 10 * factor)

    with st.spinner("🧮 Calculating safest route..."):
        try:
            safest_path = nx.shortest_path(G, source=u, target=v, weight=edge_weight)
            path_coords = [(nodes.at[n, "geometry"].y, nodes.at[n, "geometry"].x) for n in safest_path]

            m2 = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles="CartoDB positron")
            folium.Marker([start_lat, start_lon], icon=folium.Icon(color="green"), tooltip="Start").add_to(m2)
            folium.Marker([end_lat, end_lon], icon=folium.Icon(color="red"), tooltip="End").add_to(m2)
            folium.PolyLine(path_coords, color="blue", weight=6, tooltip="Safest Route").add_to(m2)
            st_folium(m2, height=500, width=900)
        except Exception as e:
            st.error(f"⚠️ Path could not be found: {e}")
else:
    st.info("Click once for start and once for end on the map.")
