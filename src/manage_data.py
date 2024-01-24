import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random

def verify_image_vector(ax, image, ts, va, vb, title=None):
    ts = np.array(ts)[1], np.array(ts)[0]
    va = np.array(va)[1], np.array(va)[0]
    vb = np.array(vb)[1], np.array(vb)[0]

    ax.imshow(image)
    
    ax.set_ylabel('Y-axis')
    ax.set_xlabel('X-axis')
    
    array = np.array([[ts[0], ts[1], va[0], va[1]]])
    X, Y, U, V = zip(*array)
    ax.quiver(X, Y, U, V,color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.3)
    
    array = np.array([[ts[0], ts[1], vb[0], vb[1]]])
    X, Y, U, V = zip(*array)
    ax.quiver(X, Y, U, V,color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.3)

    ax.axis('off')
    ax.set_title(title)
    plt.draw()


def verify_image_in_hdf5_file(ds_path, n_list, group, viz=True):

    symmetry_dict = {'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8, 
                        'p4': 9, 'p4m': 10, 'p4g': 11, 'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16}

    symmetry_inv_dict = {v: k for k, v in symmetry_dict.items()}

    with h5py.File(ds_path, 'r') as h5:
        if viz:
            print('Total number of images in the dataset: ', len(h5[group]['data']))
        if isinstance(n_list, int):
           n_list = random.choices(range(len(h5[group]['data'])), k=n_list)
        n_list = np.sort(n_list)
        if viz:
            print('Randomly selected images: ', n_list)
        imgs = np.array(h5[group]['data'][n_list])
        labels = np.array(h5[group]['labels'][n_list])
        ss_list = np.array(h5[group]['source_start_point'][n_list])
        ts_list = np.array(h5[group]['translation_start_point'][n_list])
        va_list = np.array(h5[group]['primitive_uc_vector_a'][n_list])
        vb_list = np.array(h5[group]['primitive_uc_vector_b'][n_list])
        VA_list = np.array(h5[group]['translation_uc_vector_a'][n_list])
        VB_list = np.array(h5[group]['translation_uc_vector_b'][n_list])

    if viz:
        for i in range(len(n_list)):
            output_image = imgs[i]
            ss, ts, va, vb, VA, VB = ss_list[i], ts_list[i], va_list[i], vb_list[i], VA_list[i], VB_list[i]
            # visualize example
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            verify_image_vector(axes[0], output_image, ts, va, vb, title='Primitive unit cell')
            verify_image_vector(axes[1], output_image, ts, VA, VB, title='Translational unit cell')
            plt.axis('off')
            plt.show()

            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # verify_image_vector(axes[0], unit_cells[i], ts_list[i], va_list[i], vb_list[i])
            # verify_image_vector(axes[1], imgs[i], ts_list[i], va_list[i], vb_list[i])
            # verify_image_vector(axes[2], imgs[i], ts_list[i], VA_list[i], VB_list[i])
            # plt.title(symmetry_inv_dict[labels[i]])
            # plt.show()
    # return imgs, labels, ts_list, va_list, vb_list, VA_list, VB_list



def list_to_dict(lst):
    dictionary = {}
    for index, item in enumerate(lst):
        dictionary[index] = item
    return dictionary

def copy_dataset_without_chunking(source_file, destination_file, chunk_size=10000):
    # Open the source file in read mode
    with h5py.File(source_file, 'r') as f_src:
        with h5py.File(destination_file, 'w') as f_dst:
            for group in f_src.keys():
                f_dst_group = f_dst.create_group(group)
                for dataset in f_src[group].keys():
                    # Open the source dataset
                    dataset_src = f_src[group][dataset]

                    # Determine the shape and data type of the source dataset
                    shape = dataset_src.shape
                    dtype = dataset_src.dtype

                    # Create the destination dataset with the same shape and data type, without chunking
                    dataset_dst = f_dst_group.create_dataset(dataset, shape, dtype=dtype, chunks=None)

                    # Determine the number of chunks based on the chunk size
                    num_chunks = shape[0] // chunk_size + 1

                    # Iterate over chunks and copy data
                    for i in tqdm(range(num_chunks)):

                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, shape[0])

                        # Read a chunk of data from the source dataset
                        data = dataset_src[start:end]

                        # Write the chunk of data to the destination dataset
                        dataset_dst[start:end] = data
                        
def merge_h5(target_h5, source_h5_list, group, create_new_file, start_index=0, viz=False):

    total_images = 0
    for file in source_h5_list:
        with h5py.File(file, 'r') as h5_count:
            if group in h5_count.keys():
                total_images += h5_count[group]['data'].shape[0]
            else:
                total_images += h5_count['data'].shape[0]
    print('total images to merge:', total_images)
    output_size = (256, 256, 3)

    with h5py.File(target_h5, 'a') as h5_to:

        if create_new_file:
            h5_group = h5_to.create_group(group)
            create_data = h5_group.create_dataset('data', shape=(total_images, *output_size), dtype=np.uint8, chunks=None)
            create_unit_cell = h5_group.create_dataset('unit_cell', shape=(total_images, *output_size), dtype=np.uint8, chunks=None)  
            create_labels = h5_group.create_dataset('labels', shape=(total_images, ), dtype=np.uint8, chunks=None)    
            create_shapes = h5_group.create_dataset('shapes', shape=(total_images, ), dtype=np.uint8, chunks=None)   
            create_ss = h5_group.create_dataset('source_start_point', shape=(total_images, 2), dtype=np.int32, chunks=None)   
            create_ts = h5_group.create_dataset('translation_start_point', shape=(total_images, 2), dtype=np.int32, chunks=None)   
            create_va = h5_group.create_dataset('primitive_uc_vector_a', shape=(total_images, 2), dtype=np.int32, chunks=None)   
            create_vb = h5_group.create_dataset('primitive_uc_vector_b', shape=(total_images, 2), dtype=np.int32, chunks=None)   
            create_VA = h5_group.create_dataset('translation_uc_vector_a', shape=(total_images, 2), dtype=np.int32, chunks=None)   
            create_VB = h5_group.create_dataset('translation_uc_vector_b', shape=(total_images, 2), dtype=np.int32, chunks=None)
        else:
            if len(h5_to[group]['data']) < start_index + total_images:
                print('h5 file do not have right shape for data merging.')
                print(f"h5 file dataset size is {len(h5_to[group]['data'])}, while task want to fill in {start_index} to {start_index+total_images} index")
                return 
            else:
                h5_group = h5_to[group]
                create_data = h5_group['data']
                create_unit_cell = h5_group['unit_cell'] 
                create_labels = h5_group['labels']    
                create_shapes = h5_group['shapes']
                create_ss = h5_group['source_start_point'] 
                create_ts = h5_group['translation_start_point'] 
                create_va = h5_group['primitive_uc_vector_a']  
                create_vb = h5_group['primitive_uc_vector_b']   
                create_VA = h5_group['translation_uc_vector_a'] 
                create_VB = h5_group['translation_uc_vector_b']

        n_start = start_index
        for i, file in enumerate(source_h5_list):
            with h5py.File(file, 'r') as h5_from:
                n_images = h5_from[group]['data'].shape[0]
                n_end = n_start+n_images
                print(f'Loading {i}th file {file} with images from {n_start} to {n_end}')
                
                if group in h5_from.keys():
                    create_data[n_start:n_end] = np.array(h5_from[group]['data'])
                    create_unit_cell[n_start:n_end] = np.array(h5_from[group]['unit_cell'])
                    create_labels[n_start:n_end] = np.array(h5_from[group]['labels'])
                    create_shapes[n_start:n_end] = np.array(h5_from[group]['shapes'])
                    create_ss[n_start:n_end] = np.array(h5_from[group]['source_start_point'])
                    create_ts[n_start:n_end] = np.array(h5_from[group]['translation_start_point'])
                    create_va[n_start:n_end] = np.array(h5_from[group]['primitive_uc_vector_a'])
                    create_vb[n_start:n_end] = np.array(h5_from[group]['primitive_uc_vector_b'])
                    create_VA[n_start:n_end] = np.array(h5_from[group]['translation_uc_vector_a'])
                    create_VB[n_start:n_end] = np.array(h5_from[group]['translation_uc_vector_b'])
                    n_start += n_images
                
                else:
                    n_images = h5_from['data'].shape[0]
                    create_data[n_start:n_end] = np.array(h5_from['data'])
                    create_unit_cell[n_start:n_end] = np.array(h5_from['unit_cell'])
                    create_labels[n_start:n_end] = np.array(h5_from['labels'])
                    create_shapes[n_start:n_end] = np.array(h5_from['shapes'])
                    create_ss[n_start:n_end] = np.array(h5_from['source_start_point'])
                    create_ts[n_start:n_end] = np.array(h5_from['translation_start_point'])
                    create_va[n_start:n_end] = np.array(h5_from['primitive_uc_vector_a'])
                    create_vb[n_start:n_end] = np.array(h5_from['primitive_uc_vector_b'])
                    create_VA[n_start:n_end] = np.array(h5_from['translation_uc_vector_a'])
                    create_VB[n_start:n_end] = np.array(h5_from['translation_uc_vector_b'])
                    n_start += n_images
                    
        if viz:
            verify_image_in_hdf5_file(target_h5, 10, group)