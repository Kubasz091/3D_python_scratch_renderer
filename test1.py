import numpy as np
import cv2
from objects import Objecs3D
import matplotlib.pyplot as plt

# Define 3D cube vertices (centered at z = 200)
cube = Objecs3D(1, -1, 4, "cube", (1, 1, 1))
cube_3D = cube._nodes

# Projection parameters
focal_length = 10  # Controls perspective depth
vanishing_point = np.array([500, 500])  # Screen center


def project_cube(cube_3D, focal_length, vanishing_point):
    perspective_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1/focal_length],
        [0, 0, 0, 0]  # Changed from 0 to 1 here.
    ], dtype=np.float32)

    all_points = np.concatenate([cube_3D, np.ones((cube_3D.shape[0], 1), dtype=np.float32)], axis=1)
    print("All points:", all_points)
    transformed_points = []
    for point in all_points:
        transformed_point = point @ perspective_matrix
        transformed_point = transformed_point / transformed_point[3]
        transformed_points.append(transformed_point[:2])
    transformed = np.array(transformed_points)
    print("Transformed:", transformed)
    return transformed


projected_points = project_cube(cube_3D, focal_length, vanishing_point)

edges = cube._edges_idx

# Adjust points relative to the vanishing point
adjusted_points = projected_points

# Plot projected points
plt.scatter(adjusted_points[:, 0], adjusted_points[:, 1])

# Draw edges based on the edge indexes
for edge in edges:
    pt1 = adjusted_points[edge[0]]
    pt2 = adjusted_points[edge[1]]
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=2)

plt.title("Projected Cube Points with Edges")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

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
