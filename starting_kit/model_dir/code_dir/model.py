""" 
    started : 25 Aug 2021
    author  : Olivier F.L. Manette
    email   : olivier@flod.ai 
    url     : https://flod.ai 
"""
import os
import logging
import csv 

import tensorflow as tf

from metadl.api.api import MetaLearner, Learner, Predictor

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from meta_dataset.data import config
from meta_dataset.data import pipeline
from meta_dataset.data import learning_spec
from meta_dataset.data import dataset_spec as dataset_spec_lib

tf.get_logger().setLevel('INFO')

def plot_episode(support_images, support_class_ids, query_images,
                 query_class_ids, size_multiplier=1, max_imgs_per_col=10,
                 max_imgs_per_row=10):
    """Plots the content of an episode. Episodes are composed of a support set 
    (training set) and a query set (test set). The different numbers of examples in
    each set will be detailled in the starting kit.
    Args:
        - support_images : tuple, (Batch_size_support, Height, Width, Channels)
        - support_class_ids : tuple, (Batch_size_support, N_class)
        - query_images : tuple, (Batch_size_query, Height, Width, Channels)
        - size_multiplier : dilate or shrink the size of displayed images
        - max_imgs_per_col : Integer, Number of images in a column
        - max_imgs_per_row : Integer, Number of images in a row
    """
    
    for name, images, class_ids in zip(('Support', 'Query'),
                                     (support_images, query_images),
                                     (support_class_ids, query_class_ids)):
        n_samples_per_class = Counter(class_ids)
        n_samples_per_class = {k: min(v, max_imgs_per_col)
                               for k, v in n_samples_per_class.items()}
        id_plot_index_map = {k: i for i, k
                             in enumerate(n_samples_per_class.keys())}
        num_classes = min(max_imgs_per_row, len(n_samples_per_class.keys()))
        max_n_sample = max(n_samples_per_class.values())
        figwidth = max_n_sample
        figheight = num_classes
        if name == 'Support':
            print('#Classes: %d' % len(n_samples_per_class.keys()))
        figsize = (figheight * size_multiplier, figwidth * size_multiplier)
        fig, axarr = plt.subplots(
            figwidth, figheight, figsize=figsize)
        fig.suptitle('%s Set' % name, size='15')
        fig.tight_layout(pad=3, w_pad=0.1, h_pad=0.1)
        reverse_id_map = {v: k for k, v in id_plot_index_map.items()}
        for i, ax in enumerate(axarr.flat):
            ax.patch.set_alpha(0)
            # Print the class ids, this is needed since, we want to set the x axis
            # even there is no picture.
            ax.set(xlabel=reverse_id_map[i % figheight], xticks=[], yticks=[])
            ax.label_outer()
        for image, class_id in zip(images, class_ids):
            # First decrement by one to find last spot for the class id.
            n_samples_per_class[class_id] -= 1
            # If class column is filled or not represented: pass.
            if (n_samples_per_class[class_id] < 0 or
              id_plot_index_map[class_id] >= max_imgs_per_row):
                continue
            # If width or height is 1, then axarr is a vector.
            if axarr.ndim == 1:
                ax = axarr[n_samples_per_class[class_id]
                           if figheight == 1 else id_plot_index_map[class_id]]
            else:
                ax = axarr[n_samples_per_class[class_id], id_plot_index_map[class_id]]
            ax.imshow(image / 2 + 0.5)
        plt.show()

def iterate_dataset(dataset, n):
    """ Iterates over an episode generator represented by dataset.
    It yields n episodes. An episode is a tuple containing images from 
    the support (train set) and query set (test set). A full episode description
    is available in the starting kit.
    """
    if not tf.executing_eagerly():
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            for idx in range(n):
                yield idx, sess.run(next_element)
    else:
        for idx, episode in enumerate(dataset):
            if idx == n:
                break
            yield idx, episode

class MyMetaLearner(MetaLearner):

    def __init__(self):
        super().__init__()

    def meta_fit(self, meta_dataset_generator) -> Learner:
        """
        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        
        Returns:
            MyLearner object : a Learner that stores the meta-learner's 
                learning object. (e.g. a neural network trained on meta-train
                episodes)
        """
        print("hello from meta_fit! meta_dataset_generator:",type(meta_dataset_generator))
        # for att in dir(meta_dataset_generator):
        #     print (att)
        meta_train_generator = meta_dataset_generator.meta_train_pipeline

        N_EPISODES=1

        for idx, (episode, source_id) in iterate_dataset(meta_train_generator, N_EPISODES):
            episode = [a.numpy() for a in episode]
            print('Length of the tuple describing an episode : {} \n'.format(len(episode)))

        return MyLearner()


class MyLearner(Learner):

    def __init__(self):
        super().__init__()

    def fit(self, dataset_train) -> Predictor:
        """
        Args: 
            dataset_train : a tf.data.Dataset object. It is an iterator over 
                the support examples.
        Returns:
            ModelPredictor : a Predictor.
        """
        return MyPredictor()

    def save(self, model_dir):
        """ Saves the learning object associated to the Learner. It could be 
        a neural network for example. 

        Note : It is mandatory to write a file in model_dir. Otherwise, your 
        code won't be available in the scoring process (and thus it won't be 
        a valid submission).
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
        # Save a file for the code submission to work correctly.
        with open(os.path.join(model_dir,'dummy_sample.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Dummy example'])
            
    def load(self, model_dir):
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in save().
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
    
class MyPredictor(Predictor):

    def __init__(self):
        super().__init__()

    def predict(self, dataset_test):
        """ Predicts the label of the examples in the query set which is the 
        dataset_test in this case. The prototypes are already computed by
        the Learner.

        Args:
            dataset_test : a tf.data.Dataset object. An iterator over the 
                unlabelled query examples.
        Returns: 
            preds : tensors, shape (num_examples, N_ways). We are using the 
                Sparse Categorical Accuracy to evaluate the predictions. Valid 
                tensors can take 2 different forms described below.

        Case 1 : The i-th prediction row contains the i-th example logits.
        Case 2 : The i-th prediction row contains the i-th example 
                probabilities.

        Since in both cases the SparseCategoricalAccuracy behaves the same way,
        i.e. taking the argmax of the row inputs, both forms are valid.
        Note : In the challenge N_ways = 5 at meta-test time.
        """
        dummy_pred = tf.constant([[1.0, 0, 0, 0 ,0]], dtype=tf.float32)
        dummy_pred = tf.broadcast_to(dummy_pred, (95, 5))


        # create the iterator
        # for batch in dataset_test.take(1):
        #     print([arr.numpy() for arr in batch])


        return dummy_pred

