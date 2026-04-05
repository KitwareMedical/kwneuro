from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import ants
import numpy as np
from ants.core import ANTsImage, ANTsTransform

from kwneuro.resource import InMemoryVolumeResource, VolumeResource


def average_volumes(
    volume_list: Sequence[VolumeResource], normalize: bool = True
) -> InMemoryVolumeResource:
    """
    Calculates the simple arithmetic average (mean) of a list of 3D scalar volumes.

    Input volumes are automatically resampled to the largest image space in the list.
    However, no registration is performed so all volumes should be in the same physical coordinate space to begin with.
    This is a simplified version of ANTs `average_images`. All volumes are treated with equal weight (1/N).

    Args:
        volume_list: A list of 3D scalar VolumeResource objects. Volumes do not need
                 to share the same shape/resolution but must be physically aligned.

    Returns:
        A VolumeResource object containing the element-wise arithmetic mean of all input volumes in the largest image space.
    """

    ref_volume = volume_list[0]
    for v in volume_list:
        max_size = -1
        data = v.get_array()
        if data.ndim > 3:
            error_message = (
                f"Input volume dimensions must be 2D or 3D. Found {data.ndim}D instead."
            )
            raise ValueError(error_message)
        if data.size > max_size:
            max_size = data.size
            ref_volume = v

    ants_avg = ref_volume.load().to_ants_image()
    ants_avg *= 0.0  # Initialize to zero image

    for vol in volume_list:
        ants_img = vol.load().to_ants_image()
        if normalize:
            ants_img = ants_img / ants_img.mean()

        # Resample to reference space
        temp = ants.resample_image_to_target(
            ants_img, ants_avg, interp_type="linear", imagetype=0
        )
        ants_avg = ants_avg + temp

    ants_avg = ants_avg / float(len(volume_list))

    return InMemoryVolumeResource.from_ants_image(ants_avg)


def _sharpen_template(template: ANTsImage, blending_weight: float = 0.75) -> ANTsImage:
    """
    Sharpens the input template using ANTs iMath Sharpen operation.
    Args:
        template: The ANTsImage template to be sharpened.
        blending_weight: The weight for blending the original and sharpened templates. Default is 0.75.
    Returns:
        The sharpened ANTsImage template.
    """
    sharpened_template = ants.iMath(template, "Sharpen")
    return template * blending_weight + sharpened_template * (1.0 - blending_weight)


def _update_template(
    template: ANTsImage | dict[str, ANTsImage],
    avg_warp: ANTsImage,
    avg_affine_transform: ANTsTransform,
) -> ANTsImage | dict[str, ANTsImage]:
    """
    Updates the current template by applying the average warp and affine transforms.
    Args:
        template: The current template (ANTsImage or dict of ANTsImages for multi-modality).
        avg_warp: The average warp field (ANTsImage).
        avg_affine_transform: The average affine transform (ANTsTransform).
    Returns:
        The updated template (ANTsImage or dict of ANTsImages for multi-modality).
    """
    # Save the transformations to file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Apply average transform to updated template
        aff_fn = str(Path(tmpdir) / "avgAffine.mat")
        ants.write_transform(avg_affine_transform, aff_fn)

        if isinstance(template, dict):
            ref_template = template[next(iter(template.keys()))]
        else:
            ref_template = template

        # Compute warp update in template space
        gradient_step = 0.2
        avg_warp = avg_warp * (-gradient_step)
        avg_warp_resliced = ants.apply_transforms(
            fixed=ref_template,
            moving=avg_warp,
            imagetype=1,
            transformlist=aff_fn,
            whichtoinvert=[1],
        )
        wavg_fn = str(Path(tmpdir) / "avgWarp.nii.gz")
        ants.image_write(avg_warp_resliced, wavg_fn)

        if isinstance(template, dict):
            updated_template = {}
            for modality, temp in template.items():
                updated_template[modality] = ants.apply_transforms(
                    fixed=temp,
                    moving=temp,
                    transformlist=[wavg_fn, aff_fn],
                    whichtoinvert=[0, 1],
                )

            return updated_template

        return ants.apply_transforms(
            fixed=template,
            moving=template,
            transformlist=[wavg_fn, aff_fn],
            whichtoinvert=[0, 1],
        )


def build_template(
    volume_list: Sequence[VolumeResource],
    initial_template: VolumeResource | None = None,
    iterations: int = 3,
) -> InMemoryVolumeResource:
    """
    Constructs an unbiased mean shape template from a list of 3D scalar volumes using
    an iterative group-wise registration approach based on ANTs.

    The process follows the standard iterative unbiased approach:
    1. Register all images to the current template (using SyN and Affine transforms).
    2. Average the warped images to create a new template estimate.
    3. Average the resulting transformations (warp and affine) to calculate the mean shift.
    4. Apply the inverse of the mean shift to the template to correct bias toward the true mean.
    5. Sharpen the template to enhance edge definition.

    Args:
        volume_list: A list of input 3D scalar volumes (VolumeResource objects).
                    If an initial template is not provided, the input volumes should all be rigidly aligned to
                    create a plausible initial average image.
        initial_template: An optional starting template volume. If None, the
                initial template is the simple average of all input volumes.
        iterations: The number of iterations for the template refinement process.
    Returns:
        A VolumeResource object representing the final group-wise mean template.
    """

    if initial_template is None:
        initial_template = average_volumes(volume_list)
    current_template = initial_template.load().to_ants_image()
    # Equal weighting of all images. ANTs has this as an input parameters.
    weights = np.repeat(1.0 / len(volume_list), len(volume_list))

    for _i in range(iterations):
        affine_list = []
        for idx, vol in enumerate(volume_list):
            moving_image = vol.load()
            if moving_image.get_array().ndim > 3:  # Check that the input volume is 3D
                error_message = f"Input volume dimensions must be 2D or 3D. Found {moving_image.get_array().ndim}D instead."
                raise ValueError(error_message)

            # This copies the data.
            # https://github.com/ANTsX/ANTsPy/blob/4888946a8b59a55cc784585796423c65268369ab/ants/core/ants_image_io.py#L137
            ants_moving_image = moving_image.to_ants_image()
            result = ants.registration(
                fixed=current_template,
                moving=ants_moving_image,
                type_of_transform="SyN",
            )

            # Syn result has L == 2 (warp, affine.mat)
            affine_list.append(result["fwdtransforms"][-1])  # Affine transform is last

            if idx == 0:
                avg_warp = ants.image_read(result["fwdtransforms"][0]) * weights[idx]
                new_template = result["warpedmovout"] * weights[idx]
            else:
                avg_warp = (
                    avg_warp
                    + ants.image_read(result["fwdtransforms"][0]) * weights[idx]
                )
                new_template = new_template + result["warpedmovout"] * weights[idx]

        # Average affine transforms and forward warp fields
        avg_affine_transform = ants.average_affine_transform(affine_list)

        current_template = _update_template(
            new_template, avg_warp, avg_affine_transform
        )

        # Sharpen template
        current_template = _sharpen_template(current_template, blending_weight=0.75)

    return InMemoryVolumeResource.from_ants_image(current_template)


# Helper function for multi-metric registration of a single subject
def _register_subject_multimodality(
    subject_volumes: dict[str, ANTsImage],
    current_template: dict[str, ANTsImage],
    weights: dict[str, float],
) -> tuple[str, str, dict[str, ANTsImage]]:
    """
    Performs affine and SyN-only registration for a single subject against the
    current template, returning the affine transform, transformation field and the warped images.
    Args:
        subject_volumes: A dictionary mapping modality names to ANTsImage objects for the subject.
        current_template: A dictionary mapping modality names to ANTsImage objects for the current template.
        weights: A dictionary mapping modality names to their corresponding weights for multivariate registration.
    Returns:
        A tuple containing:
            - The file path to the affine transform (str).
            - The file path to the warp field (str).
            - A dictionary mapping modality names to their corresponding warped ANTsImage objects.
    """

    modalities = list(subject_volumes.keys())
    primary_mod = modalities[0]

    # Affine registration using only the main modality
    affine_result = ants.registration(
        fixed=current_template[primary_mod],
        moving=subject_volumes[primary_mod],
        type_of_transform="Affine",
    )
    affine_transform = affine_result["fwdtransforms"][-1]

    # Setup multivariate metrics using remaining modalities
    multivariate_metrics = []
    for k in range(1, len(modalities)):
        modality_k = modalities[k]

        fixed_image = current_template[modality_k]
        moving_image = subject_volumes[modality_k]

        multivariate_metrics.append(
            [
                "mattes",
                fixed_image,
                moving_image,
                weights[modality_k],
                1,
            ]
        )

    # Deformable registration (SyNOnly, initialized with affine)
    deformable_result = ants.registration(
        fixed=current_template[primary_mod],
        moving=subject_volumes[primary_mod],
        multivariate_extras=multivariate_metrics,
        type_of_transform="SyNOnly",
        initial_transform=affine_transform,
    )

    warp_field = deformable_result["fwdtransforms"][0]

    # Warped image list (including the main one from the result)
    warped_images = {primary_mod: deformable_result["warpedmovout"]}

    # Apply transform to other modalities
    for k in range(1, len(modalities)):
        modality_k = modalities[k]
        warped_images[modality_k] = ants.apply_transforms(
            fixed=current_template[modality_k],
            moving=subject_volumes[modality_k],
            transformlist=[warp_field, affine_transform],
            whichtoinvert=[0, 0],
        )

    return affine_transform, warp_field, warped_images


def _reformat_subject_list(
    subject_list: Sequence[Mapping[str, VolumeResource]],
) -> dict[str, list[VolumeResource]]:
    """
    Reformats a list of subject dictionaries into a dictionary of lists per modality.
    Args:
        subject_list: A list of dictionaries mapping modality names to VolumeResource objects.
    Returns:
        A dictionary mapping modality names to lists of VolumeResource objects.
    """

    modalities = list(subject_list[0].keys())
    reformatted = defaultdict(list)
    for subject_dict in subject_list:
        # Check that all subjects have the same set of keys
        if Counter(list(subject_dict.keys())) != Counter(modalities):
            error_msg = "Inconsistent keys detected across subjects in subject_list"
            raise ValueError(error_msg)
        for modality, volume in subject_dict.items():
            reformatted[modality].append(volume)
    return dict(reformatted)


def build_multi_metric_template(
    subject_list: Sequence[Mapping[str, VolumeResource]],
    initial_template: Mapping[str, VolumeResource] | None = None,
    weights: Mapping[str, np.floating] | None = None,
    iterations: int = 3,
) -> Mapping[str, InMemoryVolumeResource]:
    """
    Constructs an unbiased mean shape template from a list of input volumes using
    an iterative group-wise registration approach based on ANTs, utilizing
    multiple image modalities simultaneously. See 'build_template' for more details.

    NOTE: The current implementation assumes input images are roughly pre-aligned.

    Args:
        subject_list: A list of dictionaries where each subject maps modality names to their corresponding 3D scalar volumes (VolumeResource objects).
                It is assumed that all volumes for a given subject are already co-registered.
        initial_template: An optional starting template volume. If None, the
                initial template is the simple average of all input volumes.
        weights: The weight given to each volume type (modality) during the multivariate registration step.
                The first entry is treated as the reference modality; all subsequent weights are normalized relative to
                this first value before being passed to ANTs. If None, equal weights are assumed.
                The dictionary length must match the number of modalities provided in subject_list.
        iterations: The number of iterations for the template refinement process.
    Returns:
        A VolumeResource object representing the final group-wise mean template (per modality)
    """

    # Get the list of modalities from the first subject
    modalities = list(subject_list[0].keys())
    primary_mod = modalities[0]

    # Check the the input variables all reference the same image modalities
    for d, name in zip(
        [weights, initial_template], ["weights", "initial_template"], strict=False
    ):
        if d and list(d.keys()) != modalities:
            error_msg = (
                f"Inconsistent keys detected in '{name}' compared to subject_list"
            )
            raise ValueError(error_msg)

    # Convert to a dictionary that contains a list of volumes for each modality type
    volumes_dict = _reformat_subject_list(subject_list)

    # Initialize the template - dictionary of volume resources
    if initial_template is None:
        initial_template = {}
        for m in modalities:
            initial_template[m] = average_volumes(volumes_dict[m])

    current_template: dict[str, ANTsImage] = {}
    for m in modalities:
        current_template[m] = initial_template[m].load().to_ants_image()

    # weights for multivariate extras are relative to the primary modality
    if weights is None:
        weights = dict.fromkeys(modalities, 1.0)  # Equal weighting

    weights = {m: weights[m] / weights[primary_mod] for m in modalities}

    n_subj = len(subject_list)
    for _ in range(iterations):
        affine_list: list[str] = []  # list of paths
        new_template: dict[str, ANTsImage] = {}
        for idx in range(n_subj):
            # Convert volumes to ANTs images
            ants_image_list: dict[str, ANTsImage] = {}
            for m in modalities:
                moving_image = volumes_dict[m][idx]
                moving_image = moving_image.load()
                ants_image_list[m] = moving_image.to_ants_image()

            affine_transform, warp_field, warped_images = (
                _register_subject_multimodality(
                    subject_volumes=ants_image_list,
                    current_template=current_template,
                    weights=weights,
                )
            )

            # Accumulate affine transforms, warped images and warp fields for averaging
            affine_list.append(affine_transform)
            warp_field_ants = ants.image_read(warp_field)
            if idx == 0:
                avg_warp = warp_field_ants * (1 / n_subj)
                # Initialize template average for each modality
                for m in modalities:
                    new_template[m] = warped_images[m] * (1 / n_subj)
            else:
                avg_warp = avg_warp + warp_field_ants * (1 / n_subj)
                # Accumulate template average for each modality
                for m in modalities:
                    new_template[m] = new_template[m] + warped_images[m] * (1 / n_subj)

        # Average affine transforms and forward warp fields
        avg_affine_transform = ants.average_affine_transform(affine_list)

        current_template = _update_template(
            new_template, avg_warp, avg_affine_transform
        )

        # Sharpen template
        for modality, template in current_template.items():
            current_template[modality] = _sharpen_template(
                template, blending_weight=0.75
            )

    # Return a dictionary of volume resource templates
    result = {}
    for m in modalities:
        result[m] = InMemoryVolumeResource.from_ants_image(current_template[m])
    return result
