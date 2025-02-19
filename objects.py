import math
import numpy as np
import time
import tkinter as tk

def rotate_around_center(points, angle, axis):
    center = points.mean(axis=0)
    shifted = points - center
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=float)
    elif axis == 'y':
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=float)
    elif axis == 'z':
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=float)
    else:
        R = np.eye(3)
    rotated = shifted @ R.T
    return rotated + center

class ObjectsToRender:
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

class Camera:
    def __init__(self, x, y, z, focal_length=500, proportions=(2, 1), scale=1):
        self.x = x
        self.y = y
        self.z = z
        self.focal_length = float(focal_length)
        self._monitor_plane_points = np.array([
            [x - ((proportions[0] / 2) * scale), y, z + ((proportions[1] / 2) * scale)],
            [x + ((proportions[0] / 2) * scale), y, z + ((proportions[1] / 2) * scale)],
            [x + ((proportions[0] / 2) * scale), y, z - ((proportions[1] / 2) * scale)],
            [x - ((proportions[0] / 2) * scale), y, z - ((proportions[1] / 2) * scale)]
        ])
        self._plane = np.array([0, 0, 0, 0], dtype=float)

    def display(self, objToRender):
        root = tk.Tk()
        root.title("Interactive Plane Points (Movement & Rotation)")
        root.attributes("-fullscreen", False)
        margin = 50
        scale = 400
        move_step = 0.1
        rotate_step = np.radians(2)
        last_time = [time.perf_counter()]
        canvas = tk.Canvas(root, bg="white")
        canvas.pack(fill="both", expand=True)

        def update_display():
            canvas.delete("all")
            origin = self._monitor_plane_points[0]
            vx = self._monitor_plane_points[1] - origin
            vy = self._monitor_plane_points[3] - origin
            vxu = vx / np.linalg.norm(vx)
            vyu = vy / np.linalg.norm(vy)

            def to_local_coords(p):
                d = p - origin
                return np.array([np.dot(d, vxu), np.dot(d, vyu)], dtype=float)

            local_plane = np.array([to_local_coords(pt) for pt in self._monitor_plane_points], dtype=float)
            lmin = local_plane.min(axis=0)
            lmax = local_plane.max(axis=0)
            x_range = lmax[0] - lmin[0]
            y_range = lmax[1] - lmin[1]
            w_new = int(x_range * scale + 2 * margin)
            h_new = int(y_range * scale + 2 * margin)
            canvas.config(width=w_new, height=h_new)

            def to_canvas_coords(pt):
                x_c = (pt[0] - lmin[0]) * scale + margin
                y_c = (pt[1] - lmin[1]) * scale + margin
                return (x_c, y_c)

            plane_box = np.array([
                [lmin[0], lmin[1]],
                [lmax[0], lmin[1]],
                [lmax[0], lmax[1]],
                [lmin[0], lmax[1]]
            ])
            plane_coords = []
            for corner in plane_box:
                px, py = to_canvas_coords(corner)
                plane_coords.extend([px, py])
            canvas.create_polygon(plane_coords, outline="black", fill="", width=2)

            for obj in objToRender.objects:
                projected_pts = self.project_object_into_plane_perspective(obj._nodes)
                local_pts = []
                for p in projected_pts:
                    if p is None:
                        local_pts.append(None)
                    else:
                        local_pts.append(to_local_coords(p))
                for edge in obj._edges_idx:
                    i1, i2 = edge
                    pt1, pt2 = local_pts[i1], local_pts[i2]
                    if pt1 is None or pt2 is None:
                        continue
                    x1, y1 = to_canvas_coords(pt1)
                    x2, y2 = to_canvas_coords(pt2)
                    canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

            now = time.perf_counter()
            fps = 1.0 / (now - last_time[0]) if (now - last_time[0]) > 0 else 0
            last_time[0] = now
            canvas.create_text(10, 10, anchor="nw", text=f"FPS: {fps:.1f}", fill="black", font=("Helvetica", 12), tags="fps")

        def on_key_press(event):
            key = event.keysym.lower()
            if key == 'w':
                self._monitor_plane_points[:, 1] += move_step
            elif key == 's':
                self._monitor_plane_points[:, 1] -= move_step
            elif key == 'a':
                self._monitor_plane_points[:, 0] -= move_step
            elif key == 'd':
                self._monitor_plane_points[:, 0] += move_step
            elif key == 'space':
                self._monitor_plane_points[:, 2] += move_step
            elif key == 'b':
                self._monitor_plane_points[:, 2] -= move_step
            elif key == 'left':
                self._monitor_plane_points[...] = rotate_around_center(self._monitor_plane_points, rotate_step, 'z')
            elif key == 'right':
                self._monitor_plane_points[...] = rotate_around_center(self._monitor_plane_points, -rotate_step, 'z')
            elif key == 'up':
                self._monitor_plane_points[...] = rotate_around_center(self._monitor_plane_points, rotate_step, 'x')
            elif key == 'down':
                self._monitor_plane_points[...] = rotate_around_center(self._monitor_plane_points, -rotate_step, 'x')
            update_display()

        root.bind("<Key>", on_key_press)
        update_display()
        root.mainloop()

    def project_object_into_plane_perspective(self, object_nodes):
        self.plane_equation_from_points_update()
        plane_middle = np.mean(self._monitor_plane_points, axis=0)
        transformed_points = []
        for point in object_nodes:
            if (self._plane[0] * point[0] + self._plane[1] * point[1] + self._plane[2] * point[2] + self._plane[3]) < 0:
                transformed_points.append(None)
                continue
            dist = self.distance_to_plane(point)
            projected_point = self.project_point_onto_plane(point)
            vec = self.vector_between_points(projected_point, point)
            scaled_pm = plane_middle + vec
            vec2 = self.vector_between_points(scaled_pm, point)
            if dist != 0:
                transformed_point = plane_middle + (vec2 * (self.focal_length / dist))
            else:
                transformed_point = plane_middle
            transformed_points.append(transformed_point)
        return np.array(transformed_points, dtype=object)

    def plane_equation_from_points_update(self):
        p1, p2, p3 = self._monitor_plane_points[0], self._monitor_plane_points[1], self._monitor_plane_points[2]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        a, b, c = normal
        d = -np.dot(normal, p1)
        self._plane = np.array([a, b, c, d], dtype=float)

    def distance_to_plane(self, point):
        a, b, c, d = self._plane
        x, y, z = point
        return abs(a * x + b * y + c * z + d) / math.sqrt(a * a + b * b + c * c)

    def project_point_onto_plane(self, point):
        a, b, c, d = self._plane
        normal = np.array([a, b, c], dtype=float)
        dist = (np.dot(normal, point) + d) / np.dot(normal, normal)
        return point - dist * normal

    def vector_between_points(self, p1, p2):
        return p2 - p1

    def __str__(self):
        return f"Camera @ ({self.x}, {self.y}, {self.z}), focal_length={self.focal_length}"

class Objecs3D:
    def __init__(self, x, y, z, type="none", dimensions=(1, 1, 1)):
        self.x = x
        self.y = y
        self.z = z
        self.dimensions = dimensions
        self.rotation = (0, 0, 0)
        if type == "cube":
            dx = dimensions[0] / 2
            dy = dimensions[1] / 2
            dz = dimensions[2] / 2
            self._nodes = np.array([
                [x - dx, y - dy, z - dz],
                [x + dx, y - dy, z - dz],
                [x + dx, y + dy, z - dz],
                [x - dx, y + dy, z - dz],
                [x - dx, y - dy, z + dz],
                [x + dx, y - dy, z + dz],
                [x + dx, y + dy, z + dz],
                [x - dx, y + dy, z + dz],
            ])
            self._edges_idx = np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ])
            self._faces_idx = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [0, 3, 7, 4],
                [1, 2, 6, 5],
            ])
            num_nodes = self._nodes.shape[0]
            self._edges = np.zeros((self._edges_idx.shape[0], 2, num_nodes))
            for e, (i, j) in enumerate(self._edges_idx):
                self._edges[e, 0, i] = 1
                self._edges[e, 1, j] = 1
            self._faces = np.zeros((self._faces_idx.shape[0], 4, num_nodes))
            for f, (i, j, k, l) in enumerate(self._faces_idx):
                self._faces[f, 0, i] = 1
                self._faces[f, 1, j] = 1
                self._faces[f, 2, k] = 1
                self._faces[f, 3, l] = 1

    @property
    def edges_positions(self):
        return self._edges @ self._nodes

    @property
    def faces_positions(self):
        return self._faces @ self._nodes

    def __str__(self):
        return f"pos: ({self.x}, {self.y}, {self.z}) dim: {self.dimensions} rot: {self.rotation}"