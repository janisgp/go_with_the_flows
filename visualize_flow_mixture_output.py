import os
import argparse

import numpy as np
import h5py as h5
import torch


def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('experiment_path', type=str, help='Path to experiment which contains .npy-files.')
    parser.add_argument('nr_samples', type=int, help='Number figures.')
    parser.add_argument('h5_file', type=str, help='generated h5 file containing gt, samples and labels.')
    return parser


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


xml_head = \
    """
    <scene version="0.5.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="1600"/>
                <integer name="height" value="1200"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>

    """

xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.015"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
                <scale value="0.7"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001,
                  1.0)  # limit the value to a certain range, value outside the range will be set to the boundries
    norm = np.sqrt(np.sum(vec ** 2))  # get the distance of every point
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def single_point_cloud_render(pcl, label, cnt, experiment_path):
    pcl = np.transpose(pcl[[2, 0, 1], :], (1, 0))
    pcl = pcl * 2
    # pcl = standardize_bbox(pcl, 2048)
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125
    xml_segments = [xml_head]
    for j in range(pcl.shape[0]):
        if label[j] % 8 == 1:
            color = [1, 0.8, 0]  # 黄褐色
        elif label[j] % 8 == 2:
            color = [0, 0.6, 0.2]  # 绿色
        elif label[j] % 8 == 3:
            color = [0.2, 0.4, 0.8]  # 蓝色
        elif label[j] % 8 == 4:
            color = [0.8, 0.2, 0.6]  # 紫色
        elif label[j] % 8 == 5:
            color = [0.44, 0.43, 0.11]
        elif label[j] % 8 == 6:
            color = [1, 0.2, 0.47]
        elif label[j] % 8 == 7:
            color = [0, 0.4, 0.6]
        elif label[j] % 8 == 0:
            color = [0.64, 0.76, 0.57]
        xml_segments.append(xml_ball_segment.format(pcl[j, 0], pcl[j, 1], pcl[j, 2], *color))
        # for the whole shape
        # color = colormap(pcl[j, 0] + 0.5, pcl[j, 1] + 0.5, pcl[j, 2] + 0.5 - 0.0125)
        # xml_segments.append(xml_ball_segment.format(pcl[j, 0], pcl[j, 1], pcl[j, 2], *color))

    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    rendered_pcl_path = os.path.join(experiment_path, 'rendered_pcl')
    if not os.path.exists(rendered_pcl_path):
        os.mkdir(rendered_pcl_path)
    with open(os.path.join(rendered_pcl_path, '%d.xml' % cnt), 'w') as f:
        f.write(xml_content)


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()

    '''
    smp = np.load(os.path.join(args.experiment_path, 'all_samples.npy'))
    labels = np.load(os.path.join(args.experiment_path, 'all_labels.npy'))
    '''
    f = h5.File(os.path.join(args.experiment_path, args.h5_file), "a")
    gt_clouds = torch.from_numpy(f['gt_clouds'][:])
    smp = torch.from_numpy(f['sampled_clouds'][:])
    labels = torch.from_numpy(f['sampled_labels'][:])
    for i in range(0, args.nr_samples):
        smp_cur = smp[i, :, :]
        labels_cur = labels[i]
        single_point_cloud_render(smp_cur, labels_cur, i, args.experiment_path)