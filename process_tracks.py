import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plotpolygon
from shapely.geometry import Point, Polygon
import sys
from shapely.geometry import Point, LineString
from array import array

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv('101 Cypress Tracks/101 Cypress Tracks/' + input_file)
print(df)

def convert_coordinates(df, target_width, target_height):
    scale_x = target_width / 640
    scale_y = target_height / 336

    scaled_df = df.copy()  # Create a copy of the original DataFrame

    scaled_df['x_center'] = scaled_df['x_center'] * scale_x
    scaled_df['y_center'] = scaled_df['y_center'] * scale_y

    return scaled_df

df['x_center'] = (df['xmin'] + df['xmax']) / 2
df['y_center'] = (df['ymin'] + df['ymax']) / 2

df = convert_coordinates(df, 3840, 2160)
print(df)
unique_count = df['new_track_id'].nunique()
print("Number of unique entries:", unique_count)

# Assuming 'polygon_points' is a list of (x, y) points that form the polygon
polygon_points = [(9, 905), (2217, 391), (2443, 477), (7, 1257)]

# Create a figure and axis with the specified grid size
fig, ax = plt.subplots(figsize=(19.2, 10.8))

# Create a Polygon object from the polygon points
polygon = plotpolygon(polygon_points, closed=True, edgecolor='r', facecolor='none')

# Add the polygon to the axis
ax.add_patch(polygon)

# Set the limits and aspect ratio of the axis to match the grid
ax.set_xlim(0, 3840)
ax.set_ylim(0, 2160)
ax.set_aspect('equal')
plt.gca().invert_yaxis() 
# Display the grid and polygon
plt.grid(True)
# plt.show()




# Assuming 'df' is your DataFrame with 'x' and 'y' columns
# Assuming 'polygon_points' is a list of (x, y) points that form the polygon

# Create a Polygon object from the polygon points
polygon = Polygon(polygon_points)

# Create an empty DataFrame to store selected points
selected_points = pd.DataFrame(columns=['x_center', 'y_center'])

# Iterate over all rows of the DataFrame
for index, row in df.iterrows():
    # print(index)
    point = Point(row['x_center'], row['y_center'])
    if polygon.contains(point):
        selected_points = selected_points.append(row)

# Print the selected points
print(selected_points)


df_selected = selected_points

fig, ax = plt.subplots(figsize=(19.2, 10.8))

# Load the background image
background_image = plt.imread('bg3.jpg')

# Display the background image
ax.imshow(background_image, extent=[0, 3840, 0, 2160])

# Plot the 'x_center' and 'y_center' values
ax.plot(df_selected['x_center'], 2160 - df_selected['y_center'], 'bo', markersize=3)

# Set the limits and aspect ratio of the axis to match the grid
ax.set_xlim(0, 3840)
ax.set_ylim(0, 2160)
ax.set_aspect('equal')

ax.grid(True)

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Background Image')

# Show the plot
# plt.show()

lane1_polygon_points = [(580, 970), (1362, 737), (1478, 781), (743, 1016)]
lane2_polygon_points = [(397, 931), (1200, 713), (1362, 737), (580, 970)]
lane3_polygon_points = [(257, 887), (1090, 679), (1200, 713), (397, 931)]


lane1_polygon = Polygon(lane1_polygon_points)
lane2_polygon = Polygon(lane2_polygon_points)
lane3_polygon = Polygon(lane3_polygon_points)



# Example polygons and labels
polygons = [lane1_polygon, lane2_polygon, lane3_polygon]
polygon_labels = ["lane1", "lane2", "lane3"]

# Create an empty list to store the labels
labels = []

# Group the DataFrame by 'track'
grouped_df = df_selected.groupby('new_track_id')

# Iterate over each group
for track, group in grouped_df:
    print(track)
    # Get the first and last coordinates of the group
    first_point = Point(group.iloc[0]['x_center'], group.iloc[0]['y_center'])
    last_point = Point(group.iloc[-1]['x_center'], group.iloc[-1]['y_center'])
    
    # Create a LineString from the first and last points
    line = LineString([first_point, last_point])

    found_polygon = False
    label = None

    # Check if the LineString intersects any polygon
    for i, polygon in enumerate(polygons):
        if line.intersects(polygon):
            label = polygon_labels[i]
            found_polygon = True
            break

    # Assign the label to the whole group
    if found_polygon:
        group_labels = [label] * len(group)
        labels.extend(group_labels)
    else:
        labels.extend(['No Intersection'] * len(group))

# Assign the labels as a new column in the DataFrame
df_selected['polygon_label'] = labels
df_clusters = df_selected

bg_image = plt.imread('bg3.jpg')
# Create a list of unique labels
unique_labels = df_clusters['polygon_label'].unique()

# Create a color map for the labels
# label_color_map = {label: f'C{i}' for i, label in enumerate(unique_labels)}
label_color_map = {'lane1':'red', 'lane2':'blue', 'lane3':'green'}


# Create a figure and axis with the specified grid size
fig, ax = plt.subplots(figsize=(19.2, 10.8))

# Display the background image
ax.imshow(bg_image, extent=[0, 3840, 0, 2160])

# Plot the points with colors based on the labels
for label, color in label_color_map.items():
    if label is not 'No Intersection':
        data = df_clusters[df_clusters['polygon_label'] == label]
        ax.scatter(data['x_center'], 2160 - data['y_center'], color=color, label=label, marker='.')

# Set the limits and aspect ratio of the axis to match the grid
ax.set_xlim(0, 3840)
ax.set_ylim(0, 2160)
ax.set_aspect('equal')

# Set the origin at the top left corner
# ax.invert_yaxis()

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Colored Points on Background Image')

# Show the legend
plt.legend()

# Show the plot
# plt.show()


dist1_polygon_points = [(736, 903), (1203, 768), (1341, 827), (887, 969)]
dist2_polygon_points = [(578, 865), (1041, 738), (1220, 781), (771, 913)]
dist3_polygon_points = [(458, 819), (928, 704), (1068, 752), (600, 877)]

dist1_polygon = Polygon(dist1_polygon_points)
dist2_polygon = Polygon(dist2_polygon_points)
dist3_polygon = Polygon(dist3_polygon_points)

dist_dict = {'lane1':dist1_polygon, 'lane2': dist2_polygon, 'lane3':dist3_polygon, 'No Polygon': None}
lane_dict = {'lane1':lane1_polygon, 'lane2': lane2_polygon, 'lane3':lane3_polygon}

df_clusters['pos_point'] = [Point(row['x_center'], row['y_center']) for _, row in df_clusters.iterrows()]



result_df = pd.DataFrame(columns=['track_id', 'intersect_point1', 'intersect_point2', ])

# Group the DataFrame by track ID
grouped = df_clusters.groupby('new_track_id')
print(grouped.ngroups)
# Iterate over each group
for track_id, group in grouped:
    group_size_val = len(group)
#     print(group_size_val)
    # Get the x and y coordinates of the group as a list of tuples
    if group_size_val < 10:
        continue
        
    # Get the x and y coordinates of the group as a list of tuples
    points = list(zip(group['x_center'], group['y_center']))
    if len(points) == 1:
        continue
    # Create a LineString object from the points
    line = LineString(points)
    lane_str = group['polygon_label'].iloc[0]
    if lane_str == "No Intersection":
        continue
    polygon = dist_dict[lane_str]
    
    if polygon is not None and line.intersects(polygon):
        # Get the intersection points between the line and the polygon
        intersection = line.intersection(polygon)
        # Check if the intersection is a LineString (i.e., multiple intersection points)
        if isinstance(intersection, LineString):
            intersect_point1 = Point(intersection.coords[0])
            intersect_point2 = Point(intersection.coords[-1])
        else:
#             intersect_point1 = Point(intersection.coords)
#             intersect_point2 = None
            size = len(intersection.geoms)
            if size == 1:
                continue
            intersect_point1 = Point(intersection.geoms[0].coords[0])
            intersect_point2 = Point(intersection.geoms[size - 1].coords[1])
        
        if intersect_point1 == group.iloc[0]['pos_point'] or intersect_point2 == group.iloc[group_size_val - 1]['pos_point']:
            continue
        
        distances1 = group['pos_point'].apply(lambda row_point: Point(row_point).distance(intersect_point1))
        distances2 = group['pos_point'].apply(lambda row_point: Point(row_point).distance(intersect_point2))
        closest_index1 = distances1.idxmin()
        closest_index2 = distances2.idxmin()
        
        p1 = Point(group.loc[closest_index1]['x_center'], group.loc[closest_index1]['y_center'])
        p2 = Point(group.loc[closest_index2]['x_center'], group.loc[closest_index2]['y_center'])
        
        frame1 = group.loc[closest_index1]['frame_num']
        frame2 = group.loc[closest_index2]['frame_num']
        
        time1 = group.loc[closest_index1]['timestamp']
        time2 = group.loc[closest_index2]['timestamp']
        
        if abs(frame2 - frame1) < 4:
            continue
        # Add the results to the result DataFrame
        result_df = result_df.append({'track_id': track_id, 'intersect_point1': intersect_point1, 'intersect_point2': intersect_point2, 'track_point1': p1, 'track_point2': p2, 'frame_num1':frame1,'frame_num2': frame2, 'lane':lane_str, 'time1':time1, 'time2':time2}, ignore_index=True)
result_df

df_times = result_df

df_times['frame_diff'] = df_times['frame_num2'] - df_times['frame_num1']
df_times

df_times['elapsed_time'] = df_times['frame_diff'] / 30
df_times

gap_between_dotted_lines = 9.14

df_times['speed'] = 9.14 / df_times['elapsed_time']
df_times

df_times['speed'] = 2.236 * df_times['speed'] 
df_times

df_times.to_excel('output/'+output_file , index=False, engine='openpyxl')