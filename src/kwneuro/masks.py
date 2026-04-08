from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from kwneuro.io import NiftiVolumeResource
from kwneuro.util import PathLike, normalize_path

if TYPE_CHECKING:
    from kwneuro.dwi import Dwi


def _run_hd_bet(
    hd_bet_input: list[str],
    hd_bet_output: list[str],
    device: str | None = None,
    use_tta: bool | None = None,
) -> None:  # TODO UPDATE THE DOCSTRING!
    """
    Run HD-BET inference on the given input files.
    The expensive HD-BET import and run call are isolated in this function.

    :param hd_bet_input: List of input filepaths. We have only observed nii.gz files to work.
    :param hd_bet_output: List of output filepaths. We have only observed nii.gz files to work.
    :param device: A string indicating the device on which to use pytorch. Defaults to 'cuda' if available, otherwise 'cpu'.
    :param use_tta: Whether to tell HD-BET to do test time augmentation. Takes a bit longer but is a bit more accurate.
        Default is to use it if the device is GPU and omit it if the device is CPU; this is HD-BET's recommendation.
    """

    try:
        import torch
    except ImportError:
        msg = (
            "HD-BET requires PyTorch but it is not installed. "
            "Install it with: pip install kwneuro[hdbet]"
        )
        raise ImportError(msg) from None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_tta is None:
        use_tta = torch.device(device).type == "cuda"

    # try to catch certain issues before going into HD-BET
    if any(
        not Path(path).name.endswith(".nii.gz") for path in hd_bet_input + hd_bet_output
    ):
        logging.warning(
            "kwneuro HD-BET runner has only been tested with *.nii.gz files. Masking might not work."
        )
    if any(Path(path).exists() for path in hd_bet_output):
        logging.warning(
            "HD-BET output already exists and will be overwritten for the following files: %s",
            [path for path in hd_bet_output if Path(path).exists()],
        )
    if any(not Path(path).exists() for path in hd_bet_input):
        msg = f"Some input files to HD-BET do not exist: {[path for path in hd_bet_input if not Path(path).exists()]}"
        raise FileNotFoundError(msg)

    logging.debug("Loading HD_BET")
    try:
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor
    except ImportError:
        msg = (
            "HD-BET is required for brain extraction but is not installed. "
            "Install it with: pip install kwneuro[hdbet]"
        )
        raise ImportError(msg) from None

    maybe_download_parameters()
    predictor = get_hdbet_predictor(
        use_tta=use_tta,
        device=torch.device(device),
    )

    logging.debug("Generate %s masks", len(hd_bet_input))
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[i] for i in hd_bet_input],
        output_folder_or_list_of_truncated_output_files=hd_bet_output,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=8,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def brain_extract_batch(cases: list[tuple[Dwi, Path]]) -> list[NiftiVolumeResource]:
    """Run brain extraction on a batch of cases.
    HD-BET does not run in parallel, but it does have some initialization time so it helps to run cases in batches.

    :param cases: A list of pairs each consisting of an input Dwi and desired output path for the mask.

    Returns a list of computed brain masks that is in correspondence with the list of input `cases`.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        hd_bet_input = []
        hd_bet_output = []
        for i, (dwi, output_path) in enumerate(cases):
            b0_path = (
                tmpdir_path / f"b0_mean_{i}.nii.gz"
            )  # (for some reason HD-BET does not work with .nii)
            b0_resource = NiftiVolumeResource.save(dwi.compute_mean_b0(), b0_path)
            hd_bet_input.append(str(b0_resource.path))
            hd_bet_output.append(str(output_path))

        _run_hd_bet(hd_bet_input, hd_bet_output)

    for _, output_path in cases:
        if not output_path.exists():
            logging.error(
                "After running brain masking, expected output does not seem to exist: %s.",
                output_path,
            )

    return [NiftiVolumeResource(output_path) for _, output_path in cases]


def brain_extract_single(dwi: Dwi, output_path: PathLike) -> NiftiVolumeResource:
    """Run brain extraction on a single case.

    HD-BET has significant initialization time, so it is not advised to run this function in a loop;
    see `brain_extract_batch`.

    :param dwi: Input DWI
    :param output_path: Output path for brain mask

    Returns the computed brain mask.
    """

    return brain_extract_batch([(dwi, normalize_path(output_path))])[0]
