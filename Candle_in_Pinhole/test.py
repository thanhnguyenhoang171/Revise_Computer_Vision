import cv2
import numpy as np

# Tạo một ma trận biến đổi 3x3 để xoay khối
rotation_matrix = np.array(
    [
        [1, 0, 0],
        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
    ]
)

# Tạo các điểm của khối 3D
cube = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

# Biến đổi các điểm của khối theo ma trận biến đổi
cube_transformed = np.dot(cube, rotation_matrix.T)

# Chuyển các điểm từ không gian 3D thành 2D bằng phép chiếu phối cảnh (perspective projection)
focal_length = 1000
cube_2d = cube_transformed[:, :2] * focal_length / cube_transformed[:, 2, None]

# Vẽ các cạnh của khối 2D
edges = [
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
]

# Tạo ảnh trắng
image = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Vẽ các cạnh của khối
for edge in edges:
    start = tuple(cube_2d[edge[0]].astype(int))
    end = tuple(cube_2d[edge[1]].astype(int))
    cv2.line(image, start, end, (0, 0, 0), 2)

# Hiển thị khối 2D
cv2.imshow("Cube", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
