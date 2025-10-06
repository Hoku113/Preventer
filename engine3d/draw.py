import math

import cv2
import numpy as np

# 定数定義
SCALE_DX = 450
SCALE_DY = 337
AXIS_LENGTH = 200

# グローバル変数
should_rotate = False
previous_position = []

class Plotter3d:
    def __init__(self, canvas_size, origin=(0.5, 0.5), scale=1):
        self.origin = np.array(
            [origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta = 3.1415/4
        self.phi = -3.1415/6
        axes = [
                np.array([[-AXIS_LENGTH/2, -AXIS_LENGTH/2, 0],
                        [AXIS_LENGTH/2, -AXIS_LENGTH/2, 0]], dtype=np.float32),
                np.array([[-AXIS_LENGTH/2, -AXIS_LENGTH/2, 0],
                        [-AXIS_LENGTH/2, AXIS_LENGTH/2, 0]], dtype=np.float32),
                np.array([[-AXIS_LENGTH/2, -AXIS_LENGTH/2, 0],
                        [-AXIS_LENGTH/2, -AXIS_LENGTH/2, AXIS_LENGTH]], dtype=np.float32)
            ]
        step = 20
        for step_id in range(AXIS_LENGTH // step + 1):  # add grid
            axes.append(np.array([[-AXIS_LENGTH / 2, -AXIS_LENGTH / 2 + step_id * step, 0],
                                  [AXIS_LENGTH / 2, -AXIS_LENGTH / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-AXIS_LENGTH / 2 + step_id * step, -AXIS_LENGTH / 2, 0],
                                  [-AXIS_LENGTH / 2 + step_id * step, AXIS_LENGTH / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, img, vertices, edges):
        img.fill(0)
        R = self._get_rotation()
        self._draw_axes(img, R)
        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d * self.scale + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]),
                     (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape(
            (-1, 2))[edges] * self.scale + self.origin
        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(
                edge_vertices[1]), (255, 255, 255), 1, cv2.LINE_AA)

    def _get_rotation(self):
        sin, cos = math.sin, math.cos
        return np.array([
            [cos(self.theta),  sin(self.theta) * sin(self.phi)],
            [-sin(self.theta),  cos(self.theta) * sin(self.phi)],
            [0,                       -cos(self.phi)]
        ], dtype=np.float32)  # transposed

    def mouse_callback(self,event, x, y, flags, param):
        global should_rotate, previous_position
        if event == cv2.EVENT_LBUTTONDOWN:
            previous_position = [x, y]
            should_rotate = True
        if event == cv2.EVENT_MOUSEMOVE and should_rotate:
            self.theta += (x - previous_position[0]) / SCALE_DX * 6.2831  # 360 deg
            self.phi -= (y - previous_position[1]) / SCALE_DY * 6.2831 * 2  # 360 deg
            self.phi = max(min(3.1415 / 2, self.phi), -3.1415 / 2)
            previous_position = [x, y]
        if event == cv2.EVENT_LBUTTONUP:
            should_rotate = False


