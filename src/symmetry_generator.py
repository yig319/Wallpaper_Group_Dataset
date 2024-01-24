import os
import random
import h5py
import glob
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from manage_data import verify_image_vector 
from ImageNet_dataset_functions_v4 import transformation, debug_show, debug_config
from Atom_dataset_functions_v4 import wp_atoms


class symmetry_generator:
    '''
    Generate images with different symmetry classes and save to hdf5 file, with function to visualize examples.
    '''

    def __init__(self, ds_type, source_image_dir, output_size, rotation=False) -> None:

        '''
        Inputs:
            ds_type: 'imagenet' or 'atom';
            source_image_dir: path to the source image h5 file;
            output_size: output image size;
            rotation: whether to rotate the image;
        '''
        super().__init__()
        self.ds_type = ds_type
        self.source_image_dir = source_image_dir
        if os.path.isdir(self.source_image_dir):
            self.source_image_dir = self.source_image_dir + 'images/'
            self.backup_image_dir = self.source_image_dir + 'backup_images/'
        self.output_size = output_size
        self.rotation = rotation

    def make_input_and_list(self, symmetry_list, num_per_class, batch_size, order):
        '''
        Make input image list and symmetry list.
        Inputs:
            num_per_class: number of images per class;
            symmetry_list: list of symmetry classes;
            batch_size: batch size;
            order: order of symmetry classes;
        Returns: 
            inputs: numpy array of input images with shape number*height*width*channels;
            index_list_all: list of image index;
            symmetry_list_all: list of symmetry classes;
        '''
        if self.ds_type == 'imagenet':
            if os.path.isdir(self.source_image_dir):
                if self.source_image_dir[-1] != '/': self.source_image_dir += '/'
                all_files = glob.glob(self.source_image_dir+'*')
                if len(all_files) > batch_size:
                    file_list = random.choices(all_files, k=batch_size)
                    file_list = np.sort(file_list)
                else:
                    file_list = random.choices(all_files, k=batch_size-len(all_files))
                    file_list = np.sort(all_files)
                    file_list = np.concatenate((file_list, all_files))
                # print(file_list)
                inputs = []
                for file in file_list:
                    img = plt.imread(file)
                    while img.shape[0]<50 or img.shape[1]<50 or len(img.shape)<3:
                        # os.remove(file)
                        file = random.choices(all_files, k=1)[0]
                        img = plt.imread(file)
                    inputs.append(plt.imread(file)[:,:,:3])
                files = file_list
                
                self.backup_imgs = []
                for file in glob.glob(self.backup_image_dir+'*'):
                    img = plt.imread(file)
                    self.backup_imgs.append(plt.imread(file)[:,:,:3])

            elif self.source_image_dir[-3:] == '.h5':
                with h5py.File(self.source_image_dir, 'r') as f:
                    inputs = np.array(f['images'])
                    
                # # randomly select 10 images as backup
                # backup_list = random.choices(list(np.arange(len(inputs))), k=10)
                # backup_list = np.sort(backup_list)
                # self.backup_imgs = inputs[backup_list]
                
                # use inputs as backup_imgs to prevent  
                self.backup_imgs = inputs
                files = ['']*np.min((batch_size, len(symmetry_list)*num_per_class))
                # files = None                
        else:
            inputs = [None]*num_per_class
            self.backup_imgs = [None]*10
            files = ['']*np.min((batch_size, len(symmetry_list)*num_per_class))           
            # files = None
            
        # print('inputs are generated')

        if self.ds_type == 'imagenet':
            index_list = random.choices(list(np.arange(len(inputs))), k=num_per_class)
        else:
            index_list = list(np.arange(num_per_class))

        index_list_all = index_list * len(symmetry_list)
        symmetry_list_all = [[s]*num_per_class for s in (symmetry_list)]
        symmetry_list_all = [item for sublist in symmetry_list_all for item in sublist]

        if order == 'symmetry':
            random.shuffle(symmetry_list_all)

        # seperate the list into batches
        if len(index_list_all) > batch_size:
            index_list_all_list = [index_list_all[i:i+batch_size] for i in range(0, len(index_list_all), batch_size)]
            symmetry_list_all_list = [symmetry_list_all[i:i+batch_size] for i in range(0, len(symmetry_list_all), batch_size)]
            start_list = [i for i in range(0, len(index_list_all), batch_size)]
            end_list = [i for i in range(batch_size, len(index_list_all)+batch_size, batch_size)]
            end_list[-1] = len(index_list_all)
        else:
            index_list_all_list = [index_list_all]
            symmetry_list_all_list = [symmetry_list_all]
            start_list = [0]
            end_list = [len(index_list_all)]

        return inputs, files, index_list_all_list, symmetry_list_all_list, start_list, end_list

    def make_h5_dataset(self, ds_path, folder, symmetry_list, num_per_class=1000, cover_file=False, 
                        batch_size=2000, order='symmetry', num_workers=1):
        '''
        Make hdf5 dataset with different symmetry classes.
        Inputs:
            ds_path: string, path to the hdf5 file;
            folder: strings, target folder name;
            symmetry_list: list of strings, , list of symmetry classes;
            num_per_class: int, number of images per symmetry class;
            cover_file: boolean, whether to cover the existing file;
            batch_size: int, batch size;
            order: string, 'symmetry' or 'random', order for the class of images;
            num_workers: int, number of workers;
        '''
        # print('start1')
        if os.path.isfile(ds_path):
            print('h5 file exist.')
            if cover_file: 
                os.remove(ds_path)
                print('Replace with new file.')

        inputs, files, index_list_all_list, symmetry_list_all_list, start_list, end_list = self.make_input_and_list(symmetry_list, num_per_class, batch_size, order)
        print(f'Total images for {folder} group: {end_list[-1]}.')

        with h5py.File(ds_path, mode='a') as h5_file:
            h5_group = h5_file.create_group(folder)
            create_data = h5_group.create_dataset('data', shape=(end_list[-1], *self.output_size), 
                                                    dtype=np.uint8, chunks=True)
            
            create_labels = h5_group.create_dataset('labels', shape=(end_list[-1], ), dtype=np.uint8, chunks=True)    
            create_shapes = h5_group.create_dataset('shapes', shape=(end_list[-1], ), dtype=np.uint8, chunks=True)   
            create_ss = h5_group.create_dataset('source_start_point', shape=(end_list[-1], 2), dtype=np.int32, chunks=True)   
            create_ts = h5_group.create_dataset('translation_start_point', shape=(end_list[-1], 2), dtype=np.int32, chunks=True)   
            create_va = h5_group.create_dataset('primitive_uc_vector_a', shape=(end_list[-1], 2), dtype=np.int32, chunks=True)   
            create_vb = h5_group.create_dataset('primitive_uc_vector_b', shape=(end_list[-1], 2), dtype=np.int32, chunks=True)   
            create_VA = h5_group.create_dataset('translation_uc_vector_a', shape=(end_list[-1], 2), dtype=np.int32, chunks=True)   
            create_VB = h5_group.create_dataset('translation_uc_vector_b', shape=(end_list[-1], 2), dtype=np.int32, chunks=True)   

            # generate data with every batch_size images to avoid memory overflow
            start_time = time.time()
            for index_list_partial, symmetry_list_partial, start, end in zip(index_list_all_list, symmetry_list_all_list, start_list, end_list):
                print(f'Generating images from {start} to {end}...')

        
                results = self.generate_batch(inputs, files, index_list_partial, symmetry_list_partial, num_workers)
                img_partial = np.array([res[0] for res in results])
                metadata_partial = np.array([res[1] for res in results])

                create_data[start:end] = img_partial
                create_labels[start:end] = metadata_partial[:, 5, 0]
                create_shapes[start:end] = metadata_partial[:, 5, 1]
                create_ss[start:end] = metadata_partial[:, 0]
                create_ts[start:end] = metadata_partial[:, 1]
                create_va[start:end] = metadata_partial[:, 2]
                create_vb[start:end] = metadata_partial[:, 3]
                create_VA[start:end] = metadata_partial[:, 4]
                create_VB[start:end] = metadata_partial[:, 5]
                print(f'Batch generation for {folder} group - {start} to {end} images are finished in {(time.time()-start_time)/60:.0f} mins.')

    def generate(self, img, symmetry, file):

        # print(file)
        # if self.ds_type == 'imagenet':
        #     output, unit_cell, metadata = transformation(img, self.output_size, symmetry, rotation=self.rotation) 
        # if self.ds_type == 'atom':
        #     output, unit_cell, metadata = wp_atoms(self.output_size, symmetry, rotation=self.rotation)


        # run for up to 3 times, if not working, return blank image
        flag_blank, count_same_source_img, count_another_source_img = True, 0, 0
        while flag_blank and count_another_source_img <= 3:
            while flag_blank and count_same_source_img <= 3:
                try:
                    if self.ds_type == 'imagenet':
                        output, metadata, step_outputs = transformation(img, self.output_size, symmetry, rotation=self.rotation, file=file) 
                    if self.ds_type == 'noise':
                        img = (np.random.random((500,500,3))*255).astype(np.uint8)
                        output, metadata, step_outputs = transformation(img, self.output_size, symmetry, rotation=self.rotation, file=file) 
                    if self.ds_type == 'atom':
                        output, metadata, step_outputs = wp_atoms(self.output_size, symmetry, rotation=self.rotation)
                    flag_blank = False
                except:
                    output = np.zeros(self.output_size, dtype=np.uint8)
                    metadata = np.zeros((6, 2), dtype=np.float32)
                    flag_blank = True
                count_same_source_img += 1
            if flag_blank:
               img = self.backup_imgs[np.random.randint(0, len(self.backup_imgs))]
            count_another_source_img += 1
        return output, metadata, step_outputs
    
    def generate_batch(self, inputs, files, index_list_all, symmetry_list_all, num_workers):
        '''
        Generate images with different symmetry classes.
        Inputs:
            inputs: numpy array of input images with shape number*height*width*channels;
            index_list_all: list of image index;
            symmetry_list_all: list of symmetry classes;
            num_workers: number of workers;
        Returns:
            results: list of images with different symmetry classes;
        '''
        print(f'Number of workers: {num_workers}.')

        # print(len(index_list_all), len(symmetry_list_all), len(files))
        if num_workers > 1:
            tasks = [delayed(self.generate)(inputs[index], symmetry, file) for (index, symmetry, file) in zip(index_list_all, symmetry_list_all, files)]
            # print(tasks)
            results = Parallel(n_jobs=num_workers)(tasks)
        else:
            results = [self.generate(inputs[index], symmetry, file) for (index, symmetry, file) in zip(index_list_all, symmetry_list_all, files)]
        return results
    

    def visualize_example(self, file_index, symmetry, save_file=False):
        if self.ds_type == 'imagenet':
            with h5py.File(self.source_image_dir, 'r') as f:
                img = np.array(f['images'])[file_index]
        else:
            img = None
            
        img, metadata, step_outputs = self.generate(img, symmetry, file_index)
        ss, ts, va, vb, VA, VB, (symmetry_int, shape_int) = metadata
        source_image = step_outputs['source_image']
        primitive_unit_cell_full = step_outputs['primitive_unit_cell_full']
        translational_unit_cell_full = step_outputs['translational_unit_cell_full']
        output_image = step_outputs['output_image']

        # visualize example
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        verify_image_vector(axes[0], source_image, ss, va, vb, title='Source image')
        verify_image_vector(axes[1], primitive_unit_cell_full, ts, va, vb, title='Primitive unit cell')
        verify_image_vector(axes[2], translational_unit_cell_full, ts, VA, VB, title='Translational unit cell')
        verify_image_vector(axes[3], output_image, ts, VA, VB, title='Output image')
        plt.axis('off')
        if save_file:
            plt.savefig(f'{save_file}.png', bbox_inches='tight')
            plt.savefig(f'{save_file}.svg', bbox_inches='tight')
        plt.show()
