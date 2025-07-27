from .noise import GaussianNoise, PerlinNoise, PerlinFractalNoise 

class NoiseGenerator:
    def __init__(self, noise_type="gaussian"):
        self.noise = self._select_noise(noise_type)

    def _select_noise(self, noise_type):
        if noise_type == "gaussian":
            return GaussianNoise(mean=0, std=0.1)
        elif noise_type == "perlin":
            return PerlinNoise()
        elif noise_type == "perlin_fractal":
            return PerlinFractalNoise()
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

    def apply_noise(self, data):
        return self.noise.apply(data)