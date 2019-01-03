
import argparse, copy, math, os.path, re, shutil, sys, time

import numpy as np
from PIL import Image
from stl import mesh
import os
import cv2
from colorama import Fore, Back, Style
ENABLE_PROFILING = False
if not ENABLE_PROFILING:
    from multiprocessing import Pool

class VideoCombiner(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir

        if not os.path.exists(self.img_dir):
            print(Fore.RED + '=> Error: ' + '{} not exist.'.format(self.img_dir))
            exit(0)

        self._get_video_shape()

    def _get_video_shape(self):
        self.all_images = [os.path.join(self.img_dir, i) for i in os.listdir(self.img_dir)]
        sample_img = np.random.choice(self.all_images)
        if os.path.exists(sample_img):
            img = cv2.imread(sample_img)
            self.video_shape = img.shape
        else:
            print(Fore.RED + '=> Error: ' + '{} not found or open failed, try again.'.format(sample_img))
            exit(0)

    def combine(self, target_file='combined.mp4'):
        size = (self.video_shape[1], self.video_shape[0])
        print('=> target video frame size: ', size)
        print('=> all {} frames to solve.'.format(len(self.all_images)))
        video_writer = cv2.VideoWriter(target_file, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
        i = 0
        print('=> Solving, be patient.')
        for img in self.all_images:
            img = cv2.imread(img, cv2.COLOR_BGR2RGB)
            i += 1
            # print('=> Solving: ', i)
            video_writer.write(img)
        video_writer.release()
        print('Done!')




def main():

    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='slices an .stl file into .png images')
    parser.add_argument('-i', '--input-file', required=True,
        help='Input .stl file using mm units')
    parser.add_argument('-o', '--output-dir', default=None,
        help='If unspecified, the name of the input will be used as a new \
        directory')
    parser.add_argument('-x', '--output-width', default=200,
        help='Width in pixels of the output .pngs.this argument is optional.')
    parser.add_argument('-y', '--output-height', default=500,
        help='Height in pixels of the output .pngs. this argument is optional.')
    parser.add_argument('-s', '--scale', default='10',
        help='Pixels per mm in the .stl file.this argument is optional.')

    args = parser.parse_args()

    if (not re.match(".+?\.stl", args.input_file)):
        print("Please specify an input .stl file.")
        return 0

    if (not os.path.isfile(args.input_file)):
        print(args.input_file + " does not exist!")
        return 0

    if (not re.match("^([0-9]+|[0-9]+\.[0-9]*)$", args.scale)):
        print(args.scale + " is not a valid float value for scale!")
        return 0

    if (args.output_dir is None):
        args.output_dir = args.input_file[:-4]

    if (os.path.isdir(args.output_dir)):
        shutil.rmtree(args.output_dir)
        time.sleep(1.0)

    os.mkdir(args.output_dir)


    try:
        scene = mesh.Mesh.from_file(args.input_file)
    except:
        print("Unable to read the .stl file!")
        return 0


    image_width = int(args.output_width)
    image_height = int(args.output_height)


    scene = mirror_scene(scene,(1,-1,1))

    # Scale the vertices to the actual pixels
    scale = float(args.scale)


    x_min = min(scene.points[:,0:9:3].flatten())
    x_max = max(scene.points[:,0:9:3].flatten())
    y_min = min(scene.points[:,1:9:3].flatten())
    y_max = max(scene.points[:,1:9:3].flatten())
    z_min = min(scene.points[:,2:9:3].flatten())
    z_max = max(scene.points[:,2:9:3].flatten())
    best_min_xy = (x_min, y_min)
    best_xy = (image_width/2 - scale*(x_max - x_min)/2,image_height/2 -scale*(y_max - y_min)/2)
    best_angle = 0

    print("Moving object to best position...")

    best_scene = transform_scene(scene, 1, best_angle, (0,0,0))
    best_scene = transform_scene(best_scene, scale, 0, (-best_min_xy[0]*scale + best_xy[0], -best_min_xy[1]*scale + best_xy[1], -z_min*scale))

    print("Generating layers...")
    num_layers = int(round((z_max - z_min) * scale))

    # change scene to [polygon][vertex][dimension]
    optimized_scene = best_scene.points.reshape((-1,3,3))

    # sort the scene on z-axis per polygon
    optimized_scene = optimized_scene[np.arange(len(optimized_scene))[:,np.newaxis], np.argsort(optimized_scene[:, :, 2])]

    pool_args = []
    for layer_z in range(num_layers):
        pool_args.append((optimized_scene, layer_z, image_width, image_height, args.output_dir))

    if ENABLE_PROFILING:

        for the_args in pool_args:
            slice_layer(the_args)

    else:
        # Divide and conquer!
        with Pool() as p:
            p.map(slice_layer, pool_args)

    end_time = time.time()
    combiner = VideoCombiner(args.output_dir)
    combiner.combine()
    print("Total time taken: {:.2f} s".format(end_time - start_time))


def slice_layer(args):
    optimized_scene, layer_z, image_width, image_height, output_dir = args
    layer_image_data = get_slice_image_data(optimized_scene, layer_z + 0.5, image_width, image_height)
    save_image_data(layer_image_data, '{!s}/layer{:04d}.png'.format(output_dir, layer_z))


def save_image_data(image_data, filename):

    R_data = (image_data * 255)

    Image.fromarray(R_data).convert('RGBA').save(filename)


def mirror_scene(scene, mirror):
    transform = np.array([[mirror[0],  0,           0],
                          [0,          mirror[1],   0],
                          [0,          0,           mirror[2]]])

    num_facets = len(scene.points)
    num_points = num_facets * 3

    tmp_scene = copy.deepcopy(scene)

    tmp_scene.points = np.dot(tmp_scene.points.reshape((num_points, 3)), transform.transpose()).reshape(num_facets, 9)
    tmp_scene.update_normals()

    return tmp_scene

def transform_scene(scene, scale, angle, translate):
    transform = np.array([[scale * math.cos(angle),  scale * math.sin(angle),  0,     translate[0]],
                          [scale * -math.sin(angle), scale * math.cos(angle),  0,     translate[1]],
                          [0,                        0,                        scale, translate[2]]])

    num_facets = len(scene.points)
    num_points = num_facets * 3

    tmp_scene = copy.deepcopy(scene)

    tmp_scene.points = np.dot(np.concatenate((tmp_scene.points.reshape((num_points, 3)),np.ones((num_points,1))),axis=1), transform.transpose()).reshape(num_facets, 9)
    tmp_scene.update_normals()

    return tmp_scene


def get_slice_image_data(scene, z_height, width, height):
    image = np.zeros((height, width), dtype='uint8')

    new_facet_points = scene

    # filter the matrix
    new_facet_points = new_facet_points[new_facet_points[:,0,2] <= z_height]
    new_facet_points = new_facet_points[new_facet_points[:,2,2] > z_height]

    # split the matrix based on whether the base is above or
    # below the z-height plane
    base_above = new_facet_points[new_facet_points[:,1,2] <= z_height]
    base_below = new_facet_points[new_facet_points[:,1,2] > z_height]

    scale_below = (z_height - base_below[:,0,2]).reshape(-1,1) / (base_below[:,1:3,2] - base_below[:,0,2].reshape(-1,1))

    scale_above = (z_height - base_above[:,0:2,2]) / (base_above[:,2,2].reshape(-1,1) - base_above[:,0:2,2])

    # should be a [polygon][vertex][x,y]
    above_coords = (base_above[:,2,0:2].reshape(-1,1,2) - base_above[:,0:2,0:2]) * scale_above.reshape(-1,2,1) + base_above[:,0:2,0:2]
    below_coords = (base_below[:,1:3,0:2] - base_below[:,0,0:2].reshape(-1,1,2)) * scale_below.reshape(-1,2,1) + base_below[:,0,0:2].reshape(-1,1,2)

    coords = np.concatenate((above_coords, below_coords), axis=0)

    # Filter out points where the y values don't have a difference
    coords = coords[coords[:,0,1].astype(int) != coords[:,1,1].astype(int)]

    # sort the coords so smallest y is listed first in the vertex dimension
    coords = coords[np.arange(len(coords))[:,np.newaxis], np.argsort(coords[:,:,1])].reshape(-1,4)

    #   The line between the coords pairs marks the transition points (outside
    #   -> inside or inside -> outside). Find all the points on that line and
    #   mark them

    x_trans = []

    total_row_list = np.arange(1, height + 1, dtype='float')

    for x0,y0,x1,y1 in coords:
        row_list = total_row_list[int(y0):int(y1)]
        col_list = (row_list - y0) * ((x1 - x0) / (y1 - y0)) + x0
        x_trans.append((col_list,row_list))

    transition_points = np.round(np.hstack(x_trans)).T.astype(int)

    # sort the coords by first by ascending y then by ascending x.
    # Remove redundant y dimension, since these coordinates will have y-pairs
    transition_points = transition_points[np.lexsort((transition_points[:,0], transition_points[:,1]))].reshape(-1,4)[:,0:3]

    char_one = np.uint8(1)
    for x0,y0,x1 in transition_points:
        image[y0,x0:x1] = char_one

    return image


if __name__ == '__main__':
    sys.exit(main())

