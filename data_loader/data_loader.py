import tensorflow as tf
from .data_augmentation import *
import os
import glob
import keras.backend as K

class DataLoader(object):
    def __init__(self, filenames, all_filenames, batch_size, dir_bm_eyes, 
                 resolution, num_cpus, sess, **da_config):
        self.filenames = filenames
        self.all_filenames = all_filenames
        self.batch_size = batch_size
        self.dir_bm_eyes = dir_bm_eyes
        self.resolution = resolution
        self.num_cpus = num_cpus
        self.sess = sess
        
        self.set_data_augm_config(
            da_config["prob_random_color_match"], 
            da_config["use_da_motion_blur"], 
            da_config["use_bm_eyes"])
        
        self.data_iter_next = self.create_tfdata_iter(
            self.filenames, 
            self.all_filenames,
            self.batch_size, 
            self.dir_bm_eyes,
            self.resolution,
            self.prob_random_color_match,
            self.use_da_motion_blur,
            self.use_bm_eyes,
        )
        
    def set_data_augm_config(self, prob_random_color_match=0.5, 
                             use_da_motion_blur=True, use_bm_eyes=True):
        self.prob_random_color_match = prob_random_color_match
        self.use_da_motion_blur = use_da_motion_blur
        self.use_bm_eyes = use_bm_eyes
        
    def create_tfdata_iter(self, filenames, fns_all_trn_data, batch_size, dir_bm_eyes, resolution, 
                           prob_random_color_match, use_da_motion_blur, use_bm_eyes):
        tf_fns = tf.constant(filenames, dtype=tf.string) # use tf_fns=filenames is also fine
        dataset = tf.data.Dataset.from_tensor_slices(tf_fns) 
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda filenames: tf.py_func(
                    func=read_image, 
                    inp=[filenames, 
                         fns_all_trn_data, 
                         dir_bm_eyes, 
                         resolution, 
                         prob_random_color_match, 
                         use_da_motion_blur, 
                         use_bm_eyes], 
                    Tout=[tf.float32, tf.float32, tf.float32]
                ), 
                batch_size=batch_size,
                num_parallel_batches=self.num_cpus, # cpu cores
                drop_remainder=True
            )
        )
        dataset = dataset.repeat()
        dataset = dataset.prefetch(32)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next() # this tensor can also be useed as Input(tensor=next_element)
        return next_element
        
    def get_next_batch(self):
        return self.sess.run(self.data_iter_next)

def test():
    # Number of CPU cores
    num_cpus = os.cpu_count()

    # Input/Output resolution
    RESOLUTION = 64  # 64x64, 128x128, 256x256
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

    # Batch size
    # batchSize = 8
    batchSize = 1

    # Use motion blurs (data augmentation)
    # set True if training data contains images extracted from videos
    use_da_motion_blur = False

    # Use eye-aware training
    # require images generated from prep_binary_masks.ipynb
    use_bm_eyes = True
    use_bm_eyes = False
    # Probability of random color matching (data augmentation)
    prob_random_color_match = 0.5

    da_config = {
        "prob_random_color_match": prob_random_color_match,
        "use_da_motion_blur": use_da_motion_blur,
        "use_bm_eyes": use_bm_eyes
    }

    img_dirA = './faces/faceA/aligned_faces'
    img_dirB = './faces/faceB/aligned_faces'
    img_dirA_bm_eyes = "./faces/faceA/binary_masks_eyes"
    img_dirB_bm_eyes = "./faces/faceB/binary_masks_eyes"

    # Get filenames
    train_A = glob.glob(img_dirA + "/*.*")
    train_B = glob.glob(img_dirB + "/*.*")

    train_AnB = train_A + train_B

    assert len(train_A), "No image found in " + str(img_dirA)
    assert len(train_B), "No image found in " + str(img_dirB)
    print("Number of images in folder A: " + str(len(train_A)))
    print("Number of images in folder B: " + str(len(train_B)))

    print('step2')

    # global train_batchA, train_batchB
    train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)
    train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)

    data_A = train_batchA.get_next_batch()
    warped_A, target_A, bm_eyes_A = data_A
    cv2.imshow('warp', warped_A)
    cv2.imshow('target', target_A)
    cv2.imshow('bm_eye', bm_eyes_A)
    cv2.waitKey(0)
    print('step3')

test()