import numpy as np
import cv2
from objects import Objecs3D
import matplotlib.pyplot as plt
import math
import tkinter as tk
import time

# Define 3D cube vertices (centered at z = 200)
cube = Objecs3D(0, 0, 0, "cube", (1, 1, 1))
cube_3D = cube._nodes

# Projection parameters
focal_length = 1 # Controls perspective depth
vanishing_point = np.array([500, 500])  # Screen center


def plane_equation_from_points(points):
    """
    Calculate the plane equation ax + by + cz + d = 0 given at least 3 coplanar points.
    :param points: numpy array of shape (N, 3), where N >= 3.
    :return: (a, b, c, d) coefficients of the plane equation.
    """
    p1, p2, p3 = points[0], points[1], points[2]
    # Create two vectors from p1 to p2 and p1 to p3
    v1 = p2 - p1
    v2 = p3 - p1
    # Compute the normal of the plane using cross product
    normal = np.cross(v1, v2)
    a, b, c = normal
    # Calculate d using the plane equation with point p1
    d = -np.dot(normal, p1)
    return a, b, c, d

def distance_to_plane(plane, point):
    """
    Calculate the distance from a point to a plane.
    :param plane: numpy array of shape (4,) representing the plane coefficients (a, b, c, d).
    :param point: numpy array of shape (3,) representing the point coordinates (x, y, z).
    :return: float, the distance from the point to the plane.
    """
    a, b, c, d = plane
    x, y, z = point
    return np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)

def project_point_onto_plane(point, plane):
    """
    Perpendicularly projects a 3D point onto a plane.

    :param point: numpy array of shape (3,) representing the point coordinates (x, y, z).
    :param plane: array-like of shape (4,) representing the plane coefficients (a, b, c, d)
                  where the plane equation is: a*x + b*y + c*z + d = 0.
    :return: numpy array of shape (3,) representing the projected point on the plane.
    """
    a, b, c, d = plane
    # Create the normal vector of the plane.
    normal = np.array([a, b, c])

    # Compute the distance from the point to the plane along the normal direction.
    distance = (np.dot(normal, point) + d) / np.dot(normal, normal)

    # Subtract the component in the direction of the normal.
    projected_point = point - distance * normal
    return projected_point

def vector_between_points(point1, point2):
    return point2 - point1

def project_cube_into_plane(cube_3D, focal_length, plane_points):
    plane = plane_equation_from_points(plane_points)

    transformed_points = []

    for point in cube_3D:
        distance = distance_to_plane(plane, point)
        # print(f"Distance to plane: {distance:.2f}")

        projected_point = project_point_onto_plane(point, plane)

        plane_middle = np.mean(plane_points, axis=0)

        vector = vector_between_points(projected_point, point)

        scaled_plane_middle = plane_middle + vector

        vector2 = vector_between_points(scaled_plane_middle, point)

        transformed_point = plane_middle + (vector2*(focal_length/distance))

        transformed_points.append(transformed_point)

    transformed = np.array(transformed_points)
    # print("Transformed:", transformed)
    return transformed

def display_projected_points_on_plane_tk_anim(plane_points_start, plane_points_end, cube_3D, focal_length, edges=None):
    """
    Animates a smooth transition of the plane from plane_points_start to plane_points_end
    and continuously displays the projected cube along with an FPS counter.
    The window is set to full-screen mode, and the animation lasts about 5 seconds.
    """
    root = tk.Tk()
    root.title("Projected Points on Plane (Animated)")

    # Make the window full screen.
    root.attributes("-fullscreen", True)

    margin = 50
    scale = 400

    # Estimate initial canvas size based on plane_points_start
    plane_temp = plane_points_start
    origin = plane_temp[0]
    vx_temp = plane_temp[1] - origin
    vy_temp = plane_temp[3] - origin
    vxu_temp = vx_temp / np.linalg.norm(vx_temp)
    vyu_temp = vy_temp / np.linalg.norm(vy_temp)

    def to_local_temp(p):
        d = p - origin
        return np.array([np.dot(d, vxu_temp), np.dot(d, vyu_temp)])
    local_plane_temp = np.array([to_local_temp(p) for p in plane_temp])
    x_min, y_min = local_plane_temp.min(axis=0)
    x_max, y_max = local_plane_temp.max(axis=0)
    width = int((x_max - x_min) * scale + 2 * margin)
    height = int((y_max - y_min) * scale + 2 * margin)

    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack(fill="both", expand=True)

    # For FPS calculation.
    last_time = time.perf_counter()
    fps_text = canvas.create_text(10, 10, anchor="nw", text="FPS: --", fill="black", font=("Helvetica", 12))

    t = 0.0  # interpolation factor
    # Adjust dt so that the animation lasts about 5 seconds (assuming ~60 FPS).
    # ~300 frames in 5 sec => dt ~ 1/300.
    dt = 1 / 300

    def update_display():
        nonlocal t, last_time

        # Interpolate plane corners.
        plane_points = (1 - t) * plane_points_start + t * plane_points_end

        # Recalculate projection.
        projected_pts = project_cube_into_plane(cube_3D, focal_length, plane_points)

        # Build local axes from current plane points.
        origin = plane_points[0]
        vx = plane_points[1] - origin
        vy = plane_points[3] - origin
        vx_unit = vx / np.linalg.norm(vx)
        vy_unit = vy / np.linalg.norm(vy)

        def to_local_coords(p):
            diff = p - origin
            return np.array([np.dot(diff, vx_unit), np.dot(diff, vy_unit)])

        local_plane = np.array([to_local_coords(p) for p in plane_points])
        lmin = local_plane.min(axis=0)
        lmax = local_plane.max(axis=0)
        x_range, y_range = lmax[0] - lmin[0], lmax[1] - lmin[1]

        # Resize canvas if necessary.
        w_new = int(x_range * scale + 2 * margin)
        h_new = int(y_range * scale + 2 * margin)
        canvas.config(width=w_new, height=h_new)

        def to_canvas_coords(pt):
            x_c = (pt[0] - lmin[0]) * scale + margin
            y_c = h_new - ((pt[1] - lmin[1]) * scale + margin)
            return (x_c, y_c)

        # Clear old shapes except FPS text.
        canvas.delete("all")
        # Redraw FPS text so it stays on top.
        fps_text_local = canvas.create_text(10, 10, anchor="nw",
                                           text="FPS: --", fill="black",
                                           font=("Helvetica", 12), tags="fps")

        # Draw plane boundary as a rectangle.
        plane_box = np.array([
            [lmin[0], lmin[1]],
            [lmax[0], lmin[1]],
            [lmax[0], lmax[1]],
            [lmin[0], lmax[1]]
        ])
        plane_coords = []
        for corner in plane_box:
            px, py = to_canvas_coords(corner)
            plane_coords.extend((px, py))
        canvas.create_polygon(plane_coords, outline="black", fill="", width=2)

        # Draw projected points.
        local_pts = np.array([to_local_coords(p) for p in projected_pts])
        r = 4
        for p in local_pts:
            cx, cy = to_canvas_coords(p)
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="red")

        # Draw edges if any.
        if edges is not None:
            for e in edges:
                p1 = local_pts[e[0]]
                p2 = local_pts[e[1]]
                x1, y1 = to_canvas_coords(p1)
                x2, y2 = to_canvas_coords(p2)
                canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

        # Compute and display FPS.
        now = time.perf_counter()
        fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
        last_time = now
        canvas.itemconfig(fps_text_local, text=f"FPS: {fps:.1f}")

        # Increase interpolation factor and clamp.
        t_new = t + dt
        t = t_new if t_new <= 1.0 else 1.0

        if t < 1.0:
            root.after(16, update_display)

    update_display()
    root.mainloop()

def interactive_plane_points_tk(plane_points, cube_3D, focal_length, edges=None):
    """
    Displays the projected cube onto plane_points, allowing you to move the plane
    with W/S/A/D keys to translate, arrow keys to rotate around its center,
    and space/b keys to raise/lower the plane's Z coordinate. FPS is displayed.
    """
    root = tk.Tk()
    root.title("Interactive Plane Points (Movement & Rotation)")
    root.attributes("-fullscreen", False)  # Change to True if full screen needed

    margin = 50
    scale = 400
    move_step = 0.1  # Shift increment for translations
    rotate_step = np.radians(5)  # Rotation in degrees converted to radians

    last_time = [time.perf_counter()]  # For FPS

    canvas = tk.Canvas(root, bg="white")
    canvas.pack(fill="both", expand=True)

    def rotate_around_center(points, angle, axis):
        center = points.mean(axis=0)
        shifted = points - center

        c = math.cos(angle)
        s = math.sin(angle)

        if axis == 'x':
            # Rotate around X axis
            R = np.array([
                [1,  0,  0],
                [0,  c, -s],
                [0,  s,  c]
            ], dtype=float)
        elif axis == 'y':
            # Rotate around Y axis
            R = np.array([
                [ c, 0,  s],
                [ 0, 1,  0],
                [-s, 0,  c]
            ], dtype=float)
        elif axis == 'z':
            # Rotate around Z axis
            R = np.array([
                [ c, -s, 0],
                [ s,  c, 0],
                [ 0,  0, 1]
            ], dtype=float)
        else:
            R = np.eye(3)

        rotated = shifted @ R.T
        return rotated + center

    def update_display():
        canvas.delete("all")

        projected_pts = project_cube_into_plane(cube_3D, focal_length, plane_points)

        origin = plane_points[0]
        vx = plane_points[1] - origin
        vy = plane_points[3] - origin
        vx_unit = vx / np.linalg.norm(vx)
        vy_unit = vy / np.linalg.norm(vy)

        def to_local_coords(p):
            d = p - origin
            return np.array([np.dot(d, vx_unit), np.dot(d, vy_unit)])

        local_plane = np.array([to_local_coords(p) for p in plane_points])
        lmin = local_plane.min(axis=0)
        lmax = local_plane.max(axis=0)
        x_range, y_range = lmax[0] - lmin[0], lmax[1] - lmin[1]

        w_new = int(x_range * scale + 2 * margin)
        h_new = int(y_range * scale + 2 * margin)
        canvas.config(width=w_new, height=h_new)

        def to_canvas_coords(pt):
            x_c = (pt[0] - lmin[0]) * scale + margin
            y_c = h_new - ((pt[1] - lmin[1]) * scale + margin)
            return (x_c, y_c)

        plane_box = np.array([
            [lmin[0], lmin[1]],
            [lmax[0], lmin[1]],
            [lmax[0], lmax[1]],
            [lmin[0], lmax[1]],
        ])
        plane_coords = []
        for corner in plane_box:
            px, py = to_canvas_coords(corner)
            plane_coords.extend((px, py))
        canvas.create_polygon(plane_coords, outline="black", fill="", width=2)

        local_pts = np.array([to_local_coords(p) for p in projected_pts])
        r = 4
        for p in local_pts:
            cx, cy = to_canvas_coords(p)
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="red")

        if edges is not None:
            for e in edges:
                p1 = local_pts[e[0]]
                p2 = local_pts[e[1]]
                x1, y1 = to_canvas_coords(p1)
                x2, y2 = to_canvas_coords(p2)
                canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

        now = time.perf_counter()
        fps = 1.0 / (now - last_time[0]) if (now - last_time[0]) > 0 else 0
        last_time[0] = now
        canvas.create_text(10, 10, anchor="nw",
                           text=f"FPS: {fps:.1f}", fill="black",
                           font=("Helvetica", 12), tags="fps")

    def on_key_press(event):
        key = event.keysym.lower()
        if key == 'w':
            plane_points[:, 1] += move_step
        elif key == 's':
            plane_points[:, 1] -= move_step
        elif key == 'a':
            plane_points[:, 0] -= move_step
        elif key == 'd':
            plane_points[:, 0] += move_step
        elif key == 'space':
            plane_points[:, 2] += move_step
        elif key == 'b':
            plane_points[:, 2] -= move_step
        elif key == 'left':
            new_points = rotate_around_center(plane_points, rotate_step, 'z')
            plane_points[:] = new_points
        elif key == 'right':
            new_points = rotate_around_center(plane_points, -rotate_step, 'z')
            plane_points[:] = new_points
        elif key == 'up':
            new_points = rotate_around_center(plane_points, rotate_step, 'x')
            plane_points[:] = new_points
        elif key == 'down':
            new_points = rotate_around_center(plane_points, -rotate_step, 'x')
            plane_points[:] = new_points

        update_display()

    root.bind("<Key>", on_key_press)
    update_display()
    root.mainloop()

plane_points = np.array([
    [-2, -1, -1],
    [2, -1, -1],
    [2, -1, 1],
    [-2, -1, 1]], dtype=float)

plane_points2 = np.array([
    [-2, -1, -2],
    [2, -1, -2],
    [2, -2, 0],
    [-2, -2, 0]])

projected_points = project_cube_into_plane(cube_3D, focal_length, plane_points)


# # Adjust points relative to the vanishing point
# adjusted_points = projected_points

# # Plot projected points (X vs Z)
# plt.scatter(adjusted_points[:, 0], adjusted_points[:, 2])

# # Draw edges based on the edge indexes using X (index 0) and Z (index 2)
# for edge in edges:
#     pt1 = adjusted_points[edge[0]]
#     pt2 = adjusted_points[edge[1]]
#     plt.plot([pt1[0], pt2[0]], [pt1[2], pt2[2]], 'b-', linewidth=2)

# plt.title("Projected Cube Points with Edges (X vs Z)")
# plt.xlabel("X")
# plt.ylabel("Z")
# plt.grid(True)
# plt.show()

# Example usage:
# Convert the plane points to 2D (using X and Y or any chosen two axes).
# Here we assume the plane is defined in 3D, but we only use the first two coordinates.

# display_projected_points_on_plane_tk_anim(plane_points, plane_points2, cube_3D, focal_length, cube._edges_idx)

interactive_plane_points_tk(plane_points, cube_3D, focal_length, cube._edges_idx)

# plane = np.array([
#         [-2, -1, -1],
#         [2, -1, -1],
#         [2, -1, 1],
#         [-2, -1, 1]])
# # Calculate and print the plane equation
# a, b, c, d = plane_equation_from_points(plane)
# print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")




# def project_cube(cube_3D, focal_length, vanishing_point):
#     perspective_matrix = np.array([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 1, 1/focal_length],
#         [0, 0, 0, 0]  # Changed from 0 to 1 here.
#     ], dtype=np.float32)

#     all_points = np.concatenate([cube_3D, np.ones((cube_3D.shape[0], 1), dtype=np.float32)], axis=1)
#     print("All points:", all_points)
#     transformed_points = []
#     for point in all_points:
#         transformed_point = point @ perspective_matrix
#         transformed_point = transformed_point / transformed_point[3]
#         transformed_points.append(transformed_point[:2])
#     transformed = np.array(transformed_points)
#     print("Transformed:", transformed)
#     return transformed


# projected_points = project_cube(cube_3D, focal_length, vanishing_point)

# edges = cube._edges_idx

# # Adjust points relative to the vanishing point
# adjusted_points = projected_points

# # Plot projected points
# plt.scatter(adjusted_points[:, 0], adjusted_points[:, 1])

# # Draw edges based on the edge indexes
# for edge in edges:
#     pt1 = adjusted_points[edge[0]]
#     pt2 = adjusted_points[edge[1]]
#     plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=2)

# plt.title("Projected Cube Points with Edges")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.show()

# # Apply perspective projection function
# def project_point(point, focal_length, vanishing_point):
#     x, y, z = point
#     x_proj = (focal_length * x / z) + vanishing_point[0]
#     y_proj = (focal_length * y / z) + vanishing_point[1]
#     return int(x_proj), int(y_proj)

# # Project all cube vertices
# cube_2D = np.array([project_point(p, focal_length, vanishing_point) for p in cube_3D])

# edges = cube._edges_idx

# # Create an image
# img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

# # Draw cube edges
# for edge in edges:
#     pt1, pt2 = cube_2D[edge[0]], cube_2D[edge[1]]
#     cv2.line(img, tuple(pt1), tuple(pt2), (0, 0, 255), 2)

# # Show the result
# cv2.imshow("One-Point Perspective Cube", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
