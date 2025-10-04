# app.py

import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import MarkerDrag
from streamlit_folium import st_folium
from shapely.geometry import Point
from matplotlib import cm, colors as mcolors

st.title("Flood-Safe Route Planner (Demo)")

# 1. Area of interest input
place = st.text_input("Enter your area or city (e.g., Chennai, India):", "Chennai, India")

@st.cache_resource(show_spinner=False)
def load_graph(place):
    G = ox.graph_from_place(place, network_type='drive')
    G = ox.project_graph(G)
    nodes, edges = ox.graph_to_gdfs(G)
    return G, nodes, edges

if place:
    G, nodes, edges = load_graph(place)

    # 2. Generate synthetic flood values for demonstration
    np.random.seed(42)
    flood_vals = np.random.rand(len(nodes))
    nodes['flood_value'] = flood_vals

    # 3. Normalize for coloring
    min_val = float(np.min(flood_vals))
    max_val = float(np.max(flood_vals))
    range_val = max_val - min_val if max_val != min_val else 1.0

    # 4. Pick start/end by clicking map
    st.subheader('Pick start and end locations on the map (click once for start, twice for end):')

    center_lat, center_lon = nodes.geometry.y.mean(), nodes.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    # Draw flooding by color
    for _, row in nodes.iterrows():
        flood_norm = (row['flood_value'] - min_val) / range_val
        color = cm.Reds(flood_norm)
        hexcolor = mcolors.rgb2hex(color[:3])
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color=hexcolor,
            fill=True,
            fill_opacity=0.5,
        ).add_to(m)

    # Interactive marker click
    coords = st_folium(m, height=500, width=700, returned_objects=['last_clicked'])

    start_lat, start_lon = None, None
    end_lat, end_lon = None, None

    # Keep two most recent clicks for start/end
    if coords.get('last_clicked'):
        if 'all_clicks' not in st.session_state:
            st.session_state['all_clicks'] = []
        if len(st.session_state['all_clicks']) == 0 or (coords['last_clicked'] != st.session_state['all_clicks'][-1]):
            st.session_state['all_clicks'].append(coords['last_clicked'])
            # Cap at 2 clicks (start, end only)
            if len(st.session_state['all_clicks']) > 2:
                st.session_state['all_clicks'] = st.session_state['all_clicks'][-2:]

        if len(st.session_state['all_clicks']) >= 1:
            start_lat, start_lon = st.session_state['all_clicks'][0]['lat'], st.session_state['all_clicks'][0]['lng']
        if len(st.session_state['all_clicks']) >= 2:
            end_lat, end_lon = st.session_state['all_clicks'][1]['lat'], st.session_state['all_clicks'][1]['lng']

    if start_lat and end_lat:
        st.success(f"Start: ({start_lat:.5f}, {start_lon:.5f}), End: ({end_lat:.5f}, {end_lon:.5f})")

        # 5. Route finding logic
        def nearest_node(graph_nodes, lat, lon):
            point = Point(lon, lat)
            distances = graph_nodes.geometry.distance(point)
            return distances.idxmin()

        u = nearest_node(nodes, start_lat, start_lon)
        v = nearest_node(nodes, end_lat, end_lon)

        def edge_weight(u, v, d):
            f1 = nodes.at[u, 'flood_value']
            f2 = nodes.at[v, 'flood_value']
            factor = (f1 + f2)/2
            return d['length'] * (1 + 10*factor)

        try:
            safest_path = nx.shortest_path(G, source=u, target=v, weight=edge_weight)
            path_coords = [(nodes.at[n, 'geometry'].y, nodes.at[n, 'geometry'].x) for n in safest_path]

            # Map route with polyline and markers
            m2 = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles='OpenStreetMap')
            folium.Marker([start_lat, start_lon], icon=folium.Icon(color='green')).add_to(m2)
            folium.Marker([end_lat, end_lon], icon=folium.Icon(color='red')).add_to(m2)
            folium.PolyLine(path_coords, color='blue', weight=6, tooltip="Safest Route").add_to(m2)
            for pt in path_coords:
                folium.CircleMarker(pt, radius=2, color='blue', fill=True).add_to(m2)
            st_folium(m2, height=500, width=700)
        except Exception as e:
            st.error(f"Path could not be found: {str(e)}")
    else:
        st.info("Please click start and end points on the map above.")
else:
    st.warning("Enter a valid city or area to begin.")
