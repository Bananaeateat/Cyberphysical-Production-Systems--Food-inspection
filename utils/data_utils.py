"""
Data utilities for Food Quality Detection App.

This module provides functions for dataset statistics and sample image retrieval.
"""

import os
import random
import streamlit as st


@st.cache_data
def get_dataset_stats():
    """
    Calculate dataset statistics.

    This function counts images in the train and test directories
    and calculates various statistics about the dataset.

    Returns:
        dict: Dataset statistics including:
            - train_fresh: Number of fresh training images
            - train_stale: Number of stale training images
            - test_fresh: Number of fresh test images
            - test_stale: Number of stale test images
            - total_train: Total training images
            - total_test: Total test images
            - total: Total images in dataset
            - train_percentage: Percentage of training data
            - test_percentage: Percentage of test data
    """
    stats = {}

    # Count images in each directory
    train_fresh_dir = 'data/train/fresh'
    train_stale_dir = 'data/train/stale'
    test_fresh_dir = 'data/test/fresh'
    test_stale_dir = 'data/test/stale'

    # Count with error handling
    stats['train_fresh'] = _count_images(train_fresh_dir)
    stats['train_stale'] = _count_images(train_stale_dir)
    stats['test_fresh'] = _count_images(test_fresh_dir)
    stats['test_stale'] = _count_images(test_stale_dir)

    # Calculate totals
    stats['total_train'] = stats['train_fresh'] + stats['train_stale']
    stats['total_test'] = stats['test_fresh'] + stats['test_stale']
    stats['total'] = stats['total_train'] + stats['total_test']

    # Calculate percentages
    if stats['total'] > 0:
        stats['train_percentage'] = (stats['total_train'] / stats['total']) * 100
        stats['test_percentage'] = (stats['total_test'] / stats['total']) * 100

        if stats['total_train'] > 0:
            stats['train_fresh_percentage'] = (stats['train_fresh'] / stats['total_train']) * 100
            stats['train_stale_percentage'] = (stats['train_stale'] / stats['total_train']) * 100

        if stats['total_test'] > 0:
            stats['test_fresh_percentage'] = (stats['test_fresh'] / stats['total_test']) * 100
            stats['test_stale_percentage'] = (stats['test_stale'] / stats['total_test']) * 100
    else:
        stats['train_percentage'] = 0
        stats['test_percentage'] = 0
        stats['train_fresh_percentage'] = 0
        stats['train_stale_percentage'] = 0
        stats['test_fresh_percentage'] = 0
        stats['test_stale_percentage'] = 0

    return stats


def _count_images(directory):
    """
    Count image files in a directory.

    Args:
        directory: Path to directory

    Returns:
        int: Number of image files
    """
    if not os.path.exists(directory):
        return 0

    try:
        files = os.listdir(directory)
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [f for f in files if f.lower().endswith(image_extensions)]
        return len(image_files)
    except Exception:
        return 0


def get_sample_images(category, num_samples=5, seed=None):
    """
    Get random sample images from test set.

    Args:
        category: 'fresh', 'stale', or 'both'
        num_samples: Number of samples to retrieve (per category if 'both')
        seed: Random seed for reproducibility (optional)

    Returns:
        list: List of tuples (image_path, true_label)
    """
    if seed is not None:
        random.seed(seed)

    samples = []

    if category in ['fresh', 'both']:
        fresh_samples = _get_images_from_dir('data/test/fresh', num_samples)
        samples.extend([(path, 'Fresh') for path in fresh_samples])

    if category in ['stale', 'both']:
        stale_samples = _get_images_from_dir('data/test/stale', num_samples)
        samples.extend([(path, 'Stale') for path in stale_samples])

    return samples


def _get_images_from_dir(directory, num_samples):
    """
    Get random image paths from a directory.

    Args:
        directory: Path to directory
        num_samples: Number of samples to retrieve

    Returns:
        list: List of image file paths
    """
    if not os.path.exists(directory):
        return []

    try:
        files = os.listdir(directory)
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [f for f in files if f.lower().endswith(image_extensions)]

        # Random sampling
        num_to_sample = min(num_samples, len(image_files))
        selected = random.sample(image_files, num_to_sample)

        return [os.path.join(directory, img) for img in selected]
    except Exception:
        return []


def check_data_directories():
    """
    Check if data directories exist.

    Returns:
        dict: Status of each directory
    """
    directories = {
        'train_fresh': 'data/train/fresh',
        'train_stale': 'data/train/stale',
        'test_fresh': 'data/test/fresh',
        'test_stale': 'data/test/stale'
    }

    status = {}
    for key, path in directories.items():
        status[key] = os.path.exists(path)

    return status


def get_random_test_image(category=None):
    """
    Get a single random test image.

    Args:
        category: 'fresh' or 'stale', or None for random category

    Returns:
        tuple: (image_path, true_label) or (None, None) if no images found
    """
    if category is None:
        category = random.choice(['fresh', 'stale'])

    samples = get_sample_images(category, num_samples=1)

    if samples:
        return samples[0]
    else:
        return None, None
