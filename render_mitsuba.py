from PIL import Image
import h5py as h5
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import subprocess
import argparse

def standardize_bbox(pcl):    
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)    
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    
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
            <integer name="width" value="800"/>
            <integer name="height" value="600"/>
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
            <scale x="10" y="10" z="10"/>
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

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def mitsuba(pcl, path, colors=None):
    xml_segments = [xml_head]    
    pcl = standardize_bbox(pcl, pcl.shape[0])
    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    h = np.min(pcl[:,2])

    for i in range(pcl.shape[0]):
        if colors is None:
            color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5)            
        else:
            color = colors[i]
        if h < -0.25:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2]-h-0.6875, *color))
        else:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open(path, 'w') as f:
        f.write(xml_content)


def from_exr_to_png(hdr):    
    # Simply clamp values to a 0-1 range for tone-mapping
    ldr = np.clip(hdr, 0, 1)
    # Color space conversion
    ldr = ldr**(1/2.2)
    # 0-255 remapping for bit-depth conversion
    ldr = 255.0 * ldr
    return ldr    
    
def ex_to_rgb(exa):
    rgb = [float(int(exa[i:i+2], 16) / 255.0) for i in (0, 2, 4)]
    return rgb

maps_color = [[1, 0.8, 0], [0, 0.6, 0.2], [0.2, 0.4, 0.8], [0.8, 0.2, 0.6]]
color_gt = [0.8, 0.2, 0.6]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_h5", type=str, help="Path to h5 file data.", required=True)
    parser.add_argument("--path_png", type=str, help="path to data", required=True)
    parser.add_argument("--name_png", type=str, help="path to data", required=True)
    parser.add_argument("--path_mitsuba", type=str, help="path to mitsuba render bin.", required=True)
    parser.add_argument("--indices", type=int, nargs='+', help="Index for the shape to render.", required=True)
    args = parser.parse_args()
    
    path_out_render = args.path_png
    path_file_h5 = args.path_h5
    path_mitsuba = args.path_mitsuba
    name_png = args.name_png
    
    #set path for output rendered images
    path_out_render = Path(path_out_render)
    path_out_render.mkdir(exist_ok=True)
    
    h5_file = h5.File(path_file_h5, "a")
    indices = args.indices
    
    # get generated gt clouds and sampled clouds
    pcds_gt = h5_file['gt_clouds'][:].transpose(0, 2, 1)    
    pcds_pred = h5_file['sampled_clouds'][:].transpose(0, 2, 1)
    labels = h5_file['sampled_labels'][:]

    for index in tqdm(indices):
        pcd_gt = pcds_gt[index]
        pcd_pred = pcds_pred[index]        
        colors_gt = np.asarray(color_gt * pcd_gt.shape[0]).reshape(-1, 3)        
        pcd_labels = labels[index] - 1
        colors_pred = np.asarray([maps_color[l] for l in pcd_labels])

        dict_pcds = {name_png: [pcd_pred, colors_pred], "gt": [pcd_gt, colors_gt]}

        for key, value in dict_pcds.items():            
            pcd, colors = value[0], value[1]            
            name_file = f"{index}_{key}"
            path_xml = path_out_render / f"{name_file}.xml" #path for xml file
            
            mitsuba(pcd, path_xml, colors)  # use mitsuba to generate xml

            # call mitsuba    
            path_exr = path_out_render / f"{name_file}.exr"
            subprocess.call([f"{path_mitsuba}/mitsuba", path_xml, "-o", path_exr], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            path_xml.unlink()
            
            # transfer exr to png file and save
            path_exr = path_out_render / f"{name_file}.exr"
            hdr = cv2.imread(str(path_exr), flags=cv2.IMREAD_UNCHANGED) 
            ldr = from_exr_to_png(hdr)
            path_png = path_out_render / f"{name_file}.png"            
            cv2.imwrite(str(path_png), ldr)
            path_exr.unlink()
