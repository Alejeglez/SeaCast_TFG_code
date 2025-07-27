import pytest
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from neural_lam.weather_dataset import WeatherDataset
from neural_lam import constants

dataset_name = "atlantic"
data_subset = "reanalysis"
forcing_prefix = "forcing"
split = "test"
pred_length = (constants.SAMPLE_LEN[split] // 1) - 2

@pytest.mark.parametrize("noise_type", ["gaussian", "perlin", "perlin_fractal"])
def test_noise_applied_only_to_initial_states(noise_type):
    ds_clean = WeatherDataset(
        dataset_name=dataset_name,
        split=split,
        noise=None,
        data_subset=data_subset,
        forcing_prefix=forcing_prefix,
        pred_length=pred_length,
    )
    ds_noisy = WeatherDataset(
        dataset_name=dataset_name,
        split=split,
        noise=noise_type,
        data_subset=data_subset,
        forcing_prefix=forcing_prefix,
        pred_length=pred_length,
    )

    idx = 0
    init_clean, target_clean, _ = ds_clean[idx]
    init_noisy, target_noisy, _ = ds_noisy[idx]

    diff_init = torch.abs(init_clean - init_noisy)
    diff_target = torch.abs(target_clean - target_noisy)

    assert torch.any(diff_init > 1e-4), f"[{noise_type}] No se aplicó ruido a los estados iniciales"
    assert torch.all(diff_target < 1e-6), f"[{noise_type}] Se aplicó ruido indebidamente a los estados objetivo"


@pytest.mark.parametrize("noise_type", ["gaussian", "perlin", "perlin_fractal"])
def test_noise_varies_each_call(noise_type):
    ds1 = WeatherDataset(
        dataset_name=dataset_name,
        split=split,
        noise=noise_type,
        data_subset=data_subset,
        forcing_prefix=forcing_prefix,
        pred_length=pred_length,
    )
    ds2 = WeatherDataset(
        dataset_name=dataset_name,
        split=split,
        noise=noise_type,
        data_subset=data_subset,
        forcing_prefix=forcing_prefix,
        pred_length=pred_length,
    )

    idx = 0
    init1, _, _ = ds1[idx]
    init2, _, _ = ds2[idx]

    assert not torch.allclose(init1, init2), f"[{noise_type}] El ruido aplicado es idéntico en ambas ejecuciones"

@pytest.mark.parametrize("noise_type", ["gaussian", "perlin", "perlin_fractal"])
def test_compare_noise_between_files(noise_type):
    dataset_name = "atlantic"

    ds_noisy = WeatherDataset(
        dataset_name=dataset_name,
        split=split,
        noise=noise_type,
        data_subset=data_subset,
        forcing_prefix=forcing_prefix,
        pred_length=pred_length,
    )

    idx1 = 0
    idx2 = 1

    init_noisy_1, _, _ = ds_noisy[idx1]
    init_noisy_2, _, _ = ds_noisy[idx2]

    diff_states = init_noisy_1[1] - init_noisy_2[0]

    assert not torch.allclose(diff_states, torch.zeros_like(diff_states)), (
        f"[{noise_type}] El ruido entre estados consecutivos de diferentes muestras es idéntico"
    )
