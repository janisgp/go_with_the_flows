import numpy as np
import open3d

from lib.visualization.utils import get_rotation_matrix


COLORS = [
    np.array([0.9, 0, 0]),
    np.array([0, 0.9, 0]),
    np.array([0, 0, 0.9]),
    np.array([0.9, 0, 0.9]),
    np.array([0, 0.9, 0.9]),
    np.array([0.9, 0.9, 0]),
    np.array([0.25, 0.25, 0.9]),
    np.array([0.25, 0.9, 0.25]),
    np.array([0.9, 0.25, 0.25])
]


def rotate_pc(pc, angle_axis0, angle_axis1, angle_axis2):
    mat1 = get_rotation_matrix(axis=0, angle=angle_axis0)
    mat2 = get_rotation_matrix(axis=1, angle=angle_axis1)
    mat3 = get_rotation_matrix(axis=2, angle=angle_axis2)

    pc.rotate(R=mat1.dot(mat2).dot(mat3), center=np.zeros((3, 1)))
    return pc


def numpy2ply(points, labels=None, heatmap: bool=False):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(np.transpose(points))
    if labels is not None:
        colors = []
        if heatmap:
            labels_normed = (labels/labels.max())**(1/2)
            for l in labels_normed:
                colors.append([l, 0, 1.0-l])
        else:
            for l in labels:
                colors.append(COLORS[int(l - 1)])
        colors = np.array(colors)
        point_cloud.colors = open3d.utility.Vector3dVector(colors)
    else:
        point_cloud.colors = open3d.utility.Vector3dVector(np.tile(np.expand_dims(COLORS[0], 0),
                                                                   [points.shape[-1], 1]))
    return point_cloud


def capture_ply_image(pc, lbl):
    samples_point_cloud = numpy2ply(pc, lbl)
    samples_point_cloud = rotate_pc(samples_point_cloud, 25, 135, 0)
    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(samples_point_cloud)
    def move_forward(vis):
        image = vis.capture_screen_float_buffer(True)
        image = np.asarray(image) * 255
        np.save('tmp.npy', image)
        vis.register_animation_callback(None)
        vis.destroy_window()
        return False
    vis.register_animation_callback(move_forward)
    vis.run()
    return np.load('tmp.npy')