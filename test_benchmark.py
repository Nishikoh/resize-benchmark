from typing import TYPE_CHECKING

import cupy as cp
import cvcuda
import numpy as np
import pytest
import requests
import torch
from cykooz.resizer import (
    Algorithm,
    CpuExtensions,
    FilterType,
    ImageData,
    PixelType,
    ResizeAlg,
    ResizeOptions,
    Resizer,
)
from cykooz.resizer.alpha import set_image_mode
from PIL import Image
from PIL.Image import Resampling

from utils import BenchResults

if TYPE_CHECKING:
    from pytest_benchmark.stats import Metadata

CPU_EXTENSIONS = [
    CpuExtensions.none,
    CpuExtensions.sse4_1,
    CpuExtensions.avx2,
]


@pytest.fixture(
    name="cpu_extensions",
    params=CPU_EXTENSIONS,
    ids=[e.name for e in CPU_EXTENSIONS],
)
def cpu_extensions_fixture(request):
    return request.param


@pytest.fixture(
    name="resize_alg",
    params=[
        ResizeAlg.nearest(),
        ResizeAlg.convolution(FilterType.bilinear),
        ResizeAlg.convolution(FilterType.lanczos3),
    ],
    ids=[
        "nearest",
        "bilinear",
        "lanczos3",
    ],
)
def resize_alg_fixture(request):
    return request.param


@pytest.fixture(name="resizer")
def resizer_fixture(cpu_extensions):
    resizer = Resizer()
    resizer.cpu_extensions = cpu_extensions
    return resizer


@pytest.fixture(name="source_image", scope="session")
def source_image_fixture() -> Image.Image:
    return Image.open(
        requests.get(
            "https://github.com/Cykooz/cykooz.resizer/blob/main/tests/data/nasa-4928x3279.png?raw=true",
            stream=True,
        ).raw,
    )


@pytest.fixture(name="results", scope="session")
def results_fixture():
    results = BenchResults()
    yield results
    print()
    results.print_table()


DST_SIZE = (852, 567)

CVCVDA_PIL_FILTERS = {
    # cvcuda.Interp.NEAREST: "nearest", # error in cvcuda
    cvcuda.Interp.LINEAR: "linear",
    cvcuda.Interp.LANCZOS: "lanczos3",
}


@pytest.fixture(
    name="cvcuda_filter",
    params=list(CVCVDA_PIL_FILTERS.keys()),
    ids=list(CVCVDA_PIL_FILTERS.values()),
)
def cvcuda_pil_filter_fixture(request):
    return request.param


# Pillow

PIL_FILTERS = {
    Resampling.NEAREST: "nearest",
    Resampling.BILINEAR: "bilinear",
    Resampling.LANCZOS: "lanczos3",
}


@pytest.fixture(
    name="pil_filter",
    params=list(PIL_FILTERS.keys()),
    ids=list(PIL_FILTERS.values()),
)
def pil_filter_fixture(request):
    return request.param


def resize_pillow(src_image: Image.Image, pil_filter) -> None:
    src_image.resize(DST_SIZE, pil_filter)


@pytest.mark.skip("Only manual running")
def test_resize_pillow(
    benchmark, pil_filter, source_image, results: BenchResults
) -> None:
    if source_image.mode != "RGBA":
        source_image = source_image.convert("RGBA")

    def setup():
        src_image = source_image.copy()
        return (src_image, pil_filter), {}

    benchmark.pedantic(resize_pillow, setup=setup, rounds=10, warmup_rounds=3)

    alg = PIL_FILTERS[pil_filter]
    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add("Pillow", alg, f"{value:.2f}")


def resize_pillow_cuda(src_image, shape, cvcuda_filter, format) -> None:
    cvcuda.pillowresize(src=src_image, shape=shape, format=format, interp=cvcuda_filter)


def test_resize_pillow_cuda_from_tensor(
    benchmark,
    cvcuda_filter,
    source_image,
    results: BenchResults,
) -> None:
    if source_image.mode != "RGBA":
        source_image = source_image.convert("RGBA")

    def setup():
        src_image = source_image.copy()
        image_array = cp.array(src_image)
        image_tensor = cvcuda.as_tensor(
            torch.as_tensor(image_array).to(device="cuda", non_blocking=True),
            "HWC",
        )
        return (
            image_tensor,
            (DST_SIZE[0], DST_SIZE[1], 4),
            cvcuda_filter,
            cvcuda.Format.RGBA8,
        ), {}

    benchmark.pedantic(resize_pillow_cuda, setup=setup, rounds=10, warmup_rounds=3)

    alg = CVCVDA_PIL_FILTERS[cvcuda_filter]
    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add("cvcuda tensor", alg, f"{value:.2f}")


def resize_pillow_cuda_image(src_image, shape, cvcuda_filter, format) -> None:
    image_array = np.array(src_image)
    image_tensor = cvcuda.as_tensor(
        torch.as_tensor(image_array).to(device="cuda", non_blocking=True),
        "HWC",
    )
    cvcuda.pillowresize(
        src=image_tensor,
        shape=shape,
        format=format,
        interp=cvcuda_filter,
    )


def test_resize_pillow_cuda_from_pil_image(
    benchmark,
    cvcuda_filter,
    source_image,
    results: BenchResults,
) -> None:
    if source_image.mode != "RGBA":
        source_image = source_image.convert("RGBA")

    def setup():
        src_image = source_image.copy()
        return (
            src_image,
            (DST_SIZE[0], DST_SIZE[1], 4),
            cvcuda_filter,
            cvcuda.Format.RGBA8,
        ), {}

    benchmark.pedantic(
        resize_pillow_cuda_image,
        setup=setup,
        rounds=10,
        warmup_rounds=3,
    )

    alg = CVCVDA_PIL_FILTERS[cvcuda_filter]
    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add("cvcuda image", alg, f"{value:.2f}")


# cykooz.resizer - resize raw image


def resize_raw(
    resizer: Resizer,
    resize_options: ResizeOptions,
    src_image: ImageData,
    dst_image: ImageData,
) -> None:
    resizer.resize(src_image, dst_image, resize_options)


@pytest.mark.skip("Only manual running")
def test_resize_raw(benchmark, resizer, resize_alg, source_image) -> None:
    if source_image.mode != "RGBA":
        source_image = source_image.convert("RGBA")
    width, height = source_image.size
    dst_image = ImageData(DST_SIZE[0], DST_SIZE[1], PixelType.U8x4)
    resize_options = ResizeOptions(resize_alg)

    def setup():
        src_image = ImageData(width, height, PixelType.U8x4, source_image.tobytes())
        return (resizer, resize_options, src_image, dst_image), {}

    benchmark.pedantic(resize_raw, setup=setup, rounds=50, warmup_rounds=3)


# cykooz.resizer - resize PIL image


def resize_pil(
    resizer: Resizer,
    resize_options: ResizeOptions,
    src_image: Image.Image,
    dst_image: Image.Image,
) -> None:
    resizer.resize_pil(src_image, dst_image, resize_options)


def test_resize_pil(
    benchmark,
    resizer: Resizer,
    resize_alg,
    source_image,
    results: BenchResults,
) -> None:
    if source_image.mode != "RGBA":
        source_image = source_image.convert("RGBA")
    dst_image = Image.new("RGBA", DST_SIZE)
    resize_options = ResizeOptions(resize_alg)

    def setup():
        set_image_mode(dst_image, "RGBA")
        return (resizer, resize_options, source_image, dst_image), {}

    benchmark.pedantic(resize_pil, setup=setup, rounds=10, warmup_rounds=3)

    row_name = "cykooz.resizer"
    if resizer.cpu_extensions != CpuExtensions.none:
        row_name += f" - {resizer.cpu_extensions.name}"

    alg = resize_alg.algorithm
    alg = "nearest" if alg == Algorithm.nearest else resize_alg.filter_type.name

    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add(row_name, alg, f"{value:.2f}")


# Pillow - U8


def test_resize_pillow_u8(
    benchmark, pil_filter, source_image, results: BenchResults
) -> None:
    if source_image.mode != "L":
        source_image = source_image.convert("L")

    def setup():
        src_image = source_image.copy()
        return (src_image, pil_filter), {}

    benchmark.pedantic(resize_pillow, setup=setup, rounds=20, warmup_rounds=3)

    alg = PIL_FILTERS[pil_filter]
    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add("Pillow U8", alg, f"{value:.2f}")


def test_resize_pil_u8(
    benchmark,
    resizer: Resizer,
    resize_alg,
    source_image,
    results: BenchResults,
) -> None:
    if source_image.mode != "L":
        source_image = source_image.convert("L")
    dst_image = Image.new("L", DST_SIZE)
    resize_options = ResizeOptions(resize_alg)

    def setup():
        set_image_mode(dst_image, "L")
        return (resizer, resize_options, source_image, dst_image), {}

    benchmark.pedantic(resize_pil, setup=setup, rounds=10, warmup_rounds=3)

    row_name = "cykooz.resizer U8"
    if resizer.cpu_extensions != CpuExtensions.none:
        row_name += f" - {resizer.cpu_extensions.name}"

    alg = resize_alg.algorithm
    alg = "nearest" if alg == Algorithm.nearest else resize_alg.filter_type.name

    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add(row_name, alg, f"{value:.2f}")


def test_resize_pillow_cuda_from_tensor_u8(
    benchmark,
    cvcuda_filter,
    source_image,
    results: BenchResults,
) -> None:
    if source_image.mode != "L":
        source_image = source_image.convert("L")

    def setup():
        src_image = source_image.copy()
        image_array = np.array(src_image)
        image_array = np.reshape(image_array, (*image_array.shape, 1))
        image_tensor = cvcuda.as_tensor(
            torch.as_tensor(image_array).to(device="cuda", non_blocking=True),
            "HWC",
        )

        return (
            image_tensor,
            (DST_SIZE[0], DST_SIZE[1], 1),
            cvcuda_filter,
            cvcuda.Format.RGBA8,
        ), {}

    benchmark.pedantic(resize_pillow_cuda, setup=setup, rounds=10, warmup_rounds=3)

    alg = CVCVDA_PIL_FILTERS[cvcuda_filter]
    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add("cvcuda tensor u8", alg, f"{value:.2f}")


@pytest.mark.skip("Only manual running")
def test_resize_pillow_cuda_from_pil_image_u8(
    benchmark,
    cvcuda_filter,
    source_image,
    results: BenchResults,
) -> None:
    if source_image.mode != "L":
        source_image = source_image.convert("L")

    def setup():
        src_image = source_image.copy()
        return (
            src_image,
            (DST_SIZE[0], DST_SIZE[1], 1),
            cvcuda_filter,
            cvcuda.Format.RGBA8,
        ), {}

    benchmark.pedantic(
        resize_pillow_cuda_image,
        setup=setup,
        rounds=10,
        warmup_rounds=3,
    )

    alg = CVCVDA_PIL_FILTERS[cvcuda_filter]
    stats: Metadata = benchmark.stats
    value = stats.stats.mean * 1000
    results.add("cvcuda image u8", alg, f"{value:.2f}")
