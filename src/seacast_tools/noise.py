import numpy as np
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)

def flatten_to_2d(data, mask, shape=(300, 300)):
    """
    Converts a 1D data array into a 2D matrix using a mask.

    :param data: 1D data array.
    :param mask: Binary mask in 1D format.
    :param shape: Output dimensions of the 2D matrix.
    :return: 2D matrix with values from `datos` placed according to `mask`.
    """
    mask = mask.flatten()
    temperature_map = np.zeros_like(mask, dtype=float)

    j = 0
    for i in range(len(mask)):
        if mask[i] == 1:
            temperature_map[i] = data[j]
            j += 1

    return temperature_map.reshape(shape)


class Noise:
    def apply(self, X):
        raise NotImplementedError("This method should be overridden by subclasses")
    

class GaussianNoise(Noise):
    ### Configurations used:
    # mean = 0, std = 0.1
    # mean = 0, std = 0.05
    # mean = 0, std = 0.01
    def __init__(self, mean=0, std=0.1): 
        self.mean = mean
        self.std = std

    def apply(self, X):     
        noise = np.random.normal(self.mean, self.std, X.shape)
        data_with_noise = X + noise

        return data_with_noise


class PerlinNoise(Noise):
    ### Configurations used:
    # resolution = (2, 3, 3) tileable = (True, False, False)
    # resolution = (2, 12, 12) tileable = (True, False, False)
    def __init__(self, resolution=(2, 12, 12)):
        self.resolution = resolution
        self.mask = np.load(r"data\atlantic\static\sea_mask.npy")

    def apply(self, data):

        data_2d_list = []

        for entry in data:
            data_2d = flatten_to_2d(entry, self.mask) 
            data_2d_list.append(data_2d)

        data_2d_array = np.array(data_2d_list)

        noise = generate_perlin_noise_3d(
            shape=(data_2d_array.shape[0], data_2d_array.shape[1], data_2d_array.shape[2]), res=self.resolution, tileable=(True, False, False)
        )

        noisy_data_list = []             

        for i in range(len(data_2d_array)):

            noise_entry = noise[i]
            entry = data_2d_array[i]

            noisy_data = entry + noise_entry
            
            noisy_data_flat = noisy_data.flatten()
            noisy_data_flat = noisy_data_flat[noisy_data_flat >= 1]

            noisy_data_list.append(noisy_data_flat)

        return np.expand_dims(np.array(noisy_data_list), axis=-1)   



class PerlinFractalNoise(Noise):
    ### Configurations used:
    # resolution = (15, 15), tileable = (False, True), persistence = 0.5, octaves = 3, noise_scale = 0.2, lacunarity = 2.0
    # resolution = (15, 15), tileable = (False, False), persistence = 0.5, octaves = 3, noise_scale = 0.2, lacunarity = 2.0
    # resolution = (5, 5), tileable = (False, True), persistence = 0.5, octaves = 3, noise_scale = 0.2, lacunarity = 2.0
    # resolution = (15, 15), tileable = (False, True), persistence = 0.5, octaves = 3, noise_scale = 0.05, lacunarity = 2.0
    # resolution = (15, 15), tileable = (False, True), persistence = 0.5, octaves = 3, noise_scale = 0.4, lacunarity = 2.0
    def __init__(self, resolution=(15, 15), persistence=0.5, octaves=3, noise_scale=0.2, lacunarity=2.0):
        self.resolution = resolution
        self.persistence = persistence
        self.octaves = octaves
        self.noise_scale = noise_scale
        self.lacunarity = lacunarity
        self.mask = np.load(r"data\atlantic\static\sea_mask.npy")

    def apply(self, data):

        data_2d_list = []

        for entry in data:

            data_2d = flatten_to_2d(entry, self.mask)
            data_2d_list.append(data_2d)

        data_2d_array = np.array(data_2d_list)
        noisy_data_list = []            

        for i in range(len(data_2d_array)):

            entry = data_2d_array[i]

            noise = generate_fractal_noise_2d(
                shape=entry.shape, tileable=(False, True), res=self.resolution, persistence=self.persistence, octaves=self.octaves
            )

            noise *= self.noise_scale
            noisy_data = entry + noise
           
            noisy_data_flat = noisy_data.flatten()
            noisy_data_flat = noisy_data_flat[noisy_data_flat >= 1]

            noisy_data_list.append(noisy_data_flat)

        return np.expand_dims(np.array(noisy_data_list), axis=-1)