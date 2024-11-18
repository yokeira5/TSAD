import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import os, cv2
import copy, random
from tqdm import tqdm
# Load the original image
from noise import pnoise2, snoise2

stripe_widths = [10, 25]
gap_widths = [10, 40]


def create_variable_shape_mask(dimensions, shape):
    mask = Image.new('L', dimensions, 0)
    draw = ImageDraw.Draw(mask)

    if shape == 'rectangle':
        width = np.random.randint(80, 251)
        height = np.random.randint(20, 60)
        position = (np.random.randint(0, dimensions[0] - width), np.random.randint(0, dimensions[1] - height))
        draw.rectangle([position[0], position[1], position[0] + width, position[1] + height], fill=255)
        shape_size = (width, height)
    elif shape == 'ellipse':
        radius_x = np.random.randint(30, 51)
        radius_y = np.random.randint(20, 70)
        position = (
        np.random.randint(0, dimensions[0] - radius_x * 2), np.random.randint(0, dimensions[1] - radius_y * 2))
        draw.ellipse([position[0], position[1], position[0] + radius_x * 2, position[1] + radius_y * 2], fill=255)
        shape_size = (radius_x * 2, radius_y * 2)
    elif shape == 'triangle':
        size = np.random.randint(40, 120)
        position = (np.random.randint(0, dimensions[0] - size), np.random.randint(0, dimensions[1] - size))
        vertices = [
            (np.random.randint(position[0], position[0] + size), np.random.randint(position[1], position[1] + size)) for
            _ in range(3)]
        draw.polygon(vertices, fill=255)
        shape_size = (size, size)
    else:
        raise ValueError("Unsupported shape! Supported shapes are 'ellipse', 'rectangle', and 'triangle'.")

    return mask, position, shape_size


def generate_simplex_noise(size, scale=10.0, octaves=1, persistence=0.5, lacunarity=2.0):
    """Generate a 2D Simplex noise texture."""
    width, height = size
    noise_array = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            noise_array[x, y] = snoise2(x / scale, y / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity)

    # Normalize to [0, 1]
    noise_array = (noise_array - np.min(noise_array)) / (np.max(noise_array) - np.min(noise_array))
    return noise_array


def generate_voronoi_noise(size, points_count=30):
    width, height = size
    mask = np.zeros((width, height))
    points = np.array([np.random.rand(2) * size for _ in range(points_count)])

    for x in range(width):
        for y in range(height):
            min_distance = float('inf')
            for px, py in points:
                distance = np.sqrt((px - x) ** 2 + (py - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
            mask[x, y] = min_distance

    max_val = np.max(mask)
    if max_val != 0:
        mask /= max_val
    return mask


def process_image_with_mask(image, idx, orgin_mask):
    # Check if the image is grayscale
    mask = orgin_mask
    if image.mode == 'RGB':
        mask = np.repeat(orgin_mask[:, :, np.newaxis], 3, axis=2)

    data1 = np.where(mask < 0.5, image, random.random() * 255)

    mask1 = np.where(orgin_mask < 0.5, 0, 255)

    return data1.astype(np.uint8), mask1


# Function to apply the mask and create the enlarged shape with dynamic scaling factor
def apply_variable_shape_mask(original_image, shape):
    mask, position, shape_size = create_variable_shape_mask(original_image.size, shape)
    # Choose a random scaling factor between 1.2 and 4
    scaling_factor = np.random.uniform(1.3, 3)
    enlarged_size = (int(shape_size[0] * scaling_factor), int(shape_size[1] * scaling_factor))

    # Crop the selected area based on the original shape size
    selected_area = original_image.crop(
        (position[0], position[1], position[0] + shape_size[0], position[1] + shape_size[1]))

    # Enlarge the selected area
    enlarged_area = selected_area.resize(enlarged_size, resample=Image.LANCZOS)

    # Create a new image to place the enlarged area
    final_image = Image.new('RGBA', original_image.size)
    final_image.paste(original_image, (0, 0))

    # Ensure there is enough space to paste the enlarged shape
    paste_x = np.random.randint(0, max(original_image.size[0] - enlarged_size[0], 1))
    paste_y = np.random.randint(0, max(original_image.size[1] - enlarged_size[1], 1))
    paste_position = (paste_x, paste_y)

    final_image.paste(enlarged_area, paste_position, mask.resize(enlarged_size))

    # Create a simple mask
    simple_mask = Image.new('L', original_image.size, 0)
    simple_mask_pos = Image.new('L', enlarged_area.size, 255)
    simple_mask.paste(simple_mask_pos, paste_position, mask.resize(enlarged_size))

    # Convert back to RGB to discard alpha channel before saving
    final_image_rgb = final_image.convert('RGB')
    return final_image_rgb, simple_mask


# Function to create a final image with variable shaped areas

def apply_variable_moise_mask(image_size, original_image, shapes, filename):
    new_size = (image_size[1], image_size[0])
    masks = []
    for s_width, g_width in zip(stripe_widths, gap_widths):
        horizontal_mask = generate_stripe_mask(new_size, 'horizontal', s_width, g_width)
        vertical_mask = generate_stripe_mask(new_size, 'vertical', s_width, g_width)
        masks.append(horizontal_mask)
        masks.append(vertical_mask)
    voronoi_mask = generate_voronoi_noise(size=new_size, points_count=random.choice([10, 15, 20, 25, 30, ]))
    simplex_mask = generate_simplex_noise(
        size=new_size,
        scale=random.choice([40, 50, 60]),
        octaves=random.choice([2, 3]),
        persistence=random.choice([1, 2]),
        lacunarity=random.choice([4, 6, 8]),
    )

    cur_mask = masks + [voronoi_mask, simplex_mask]
    final_image, simple_mask, file_list = [], [], []
    for idx, mask1 in enumerate(cur_mask):
        processed_image, mask1 = process_image_with_mask(original_image, idx, mask1)
        final_image.append(Image.fromarray(processed_image))
        simple_mask.append(Image.fromarray(mask1.astype(np.uint8)))
        file_list.append(f"{shapes[idx]}/{os.path.splitext(filename)[0]}.jpg")

    return final_image, simple_mask, file_list


def create_final_image_with_variable_shapes(image_size, image_path, filename, output_directory, shapes):
    original_image = Image.open(image_path)
    original_image = original_image.resize(image_size)

    origin_data = []
    labels_data = []
    files_list = []
    for shape in shapes[6:]:
        final_image, simple_mask = apply_variable_shape_mask(copy.deepcopy(original_image), shape)
        origin_data.append(final_image)
        labels_data.append(simple_mask)
        files_list.append(f"{shape}/{os.path.splitext(filename)[0]}.jpg")

    final_image, simple_mask, file_list = apply_variable_moise_mask(image_size, copy.deepcopy(original_image),
                                                                    shapes[:6], filename)
    origin_data.extend(final_image)
    labels_data.extend(simple_mask)
    files_list.extend(file_list)

    for idx, file in enumerate(files_list):
        output_path = os.path.join(output_directory, 'train',
                                   file)
        output_mask_path = os.path.join(output_directory, 'groundtruth',
                                        file)
        origin_data[idx].save(output_path)
        labels_data[idx].convert('RGB').save(output_mask_path)

    origin_data = np.array(origin_data)
    labels_data = np.array(labels_data)
    files_list = np.array(files_list)

    return origin_data, labels_data, files_list


def process_images_in_directory(cur_class,
                                train_number,
                                data_path,
                                output_directory,
                                image_size=(256, 256)
                                ):
    origin_data = []
    labels_data = []
    files_list = []

    fil_emu_list = os.listdir(data_path)

    bar_num = len(fil_emu_list)
    if bar_num > train_number:
        bar_num = train_number

    origin_data_all = np.array([])
    labels_data_all = np.array([])
    files_list_all = np.array([])
    masks_name = [
        'stride_h1',
        'stride_h3',
        'stride_v1',
        'stride_v3',
        'voronoi',
        'simplex',
        'ellipse',
        'rectangle',
        'triangle',
    ]
    for cur_name in masks_name:
        # utils.remove_directory(output_directory)
        os.makedirs(os.path.join(output_directory, 'train', cur_name), exist_ok=True)
        os.makedirs(os.path.join(output_directory, 'groundtruth', cur_name), exist_ok=True)

    pbar = tqdm(range(bar_num))
    for filename in fil_emu_list[0:bar_num]:
        pbar.update()
        if filename.endswith((".jpg", '.png')):  # check if it's an image
            image_path = os.path.join(data_path, filename)

            origins, labels, files = create_final_image_with_variable_shapes(
                image_size,
                image_path, filename,
                output_directory=output_directory,
                shapes=masks_name,
            )

            if len(origin_data_all) == 0:
                origin_data_all = origins
                labels_data_all = labels
                files_list_all = files
            else:
                origin_data_all = np.concatenate((origin_data_all, origins), axis=0)
                labels_data_all = np.concatenate((labels_data_all, labels), axis=0)
                files_list_all = np.concatenate((files_list_all, files), axis=0)
    np.save(rf'{output_directory}/origin.npy', origin_data_all)
    np.save(rf'{output_directory}/labels.npy', labels_data_all)
    np.save(rf'{output_directory}/files.npy', files_list_all)

    pbar.close()
    noise_data = labels_data  # + origin_data
    origin_data = origin_data  # + origin_data

    return origin_data, noise_data, files_list  # + files_list


def generate_stripe_mask(size, direction='horizontal', stripe_width=10, gap_width=10):
    mask = np.zeros(size)
    total_width = stripe_width + gap_width

    if direction == 'horizontal':
        for i in range(0, size[0], total_width):
            mask[i:i + stripe_width, :] = 1
    else:  # 'vertical'
        for j in range(0, size[1], total_width):
            mask[:, j:j + stripe_width] = 1
    return mask


import platform

if platform.node() == 'kaya3090':
    DATA_PATH = r'D:\pythonProject\AOI\dataset\Classification'
    OUT_PATH = r'F:\OUTPUT\SCM\Circle\0627'
    OUT_DATA = r'F:\OUTPUT\DATASET\Simulated'
    work_sample = 0
else:
    DATA_PATH = r'/home/u4542686/YanqinAD/Dataset'
    OUT_PATH = r'/work/u4542686/YanQinOutput/OUTPUT'
    OUT_DATA = r'/home/u4542686/YanqinAD/Dataset/Simulated'
    work_sample = 2

cate_list = [
    'YDFID_1',
    'mvtec',
]

datacategary = {
    'YDFID_1': [
        'CL1',
        'CL2',
        'CL3',
        'CL4',
        'CL10',
        'CL12',
        'SL1',  # 34 到這裏
        'SL8',
        'SL9',
        'SL10',
        'SL11',
        'SL13',
        'SL16',
        'SP3',
        'SP5',
        'SP19',
        'SP24',
    ],
    'mvtec': [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",  # 16
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper"
    ]
}

data_suffix = {
    'YDFID_1': 'TRAIN/defect-free',
    'mvtec': 'train/good'
}

for CUR_TYPE in cate_list:
    print(CUR_TYPE)

    data_path = f'{DATA_PATH}/{CUR_TYPE}'

    multi_classes = datacategary[CUR_TYPE]

    image_size = (256, 256)

    for cur_class in multi_classes:
        print(cur_class)
        process_images_in_directory(cur_class, 10000,
                                    f'{DATA_PATH}/{CUR_TYPE}/{cur_class}/{data_suffix[CUR_TYPE]}/',
                                    f'{OUT_DATA}/{CUR_TYPE}/{cur_class}',
                                    image_size=image_size)



