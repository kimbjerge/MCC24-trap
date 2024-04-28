import datetime
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import torchvision

#KBE??? from trapdata import logger

#KBE??? 
class Logger():
    
    def info(self, text):
        print("Info:", text)
    
    def debug(self, text):
        print("Debug:", text)
        
logger = Logger()
#KBE??? 

def get_device(device_str=None) -> torch.device:
    """
    Select CUDA if available.

    @TODO add macOS Metal?
    @TODO check Kivy settings to see if user forced use of CPU
    """
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device '{device}' for inference")
    return device


def get_or_download_file(
    path, destination_dir=None, prefix=None, suffix=None
) -> pathlib.Path:
    """
    >>> filename, headers = get_weights("https://drive.google.com/file/d/1KdQc56WtnMWX9PUapy6cS0CdjC8VSdVe/view?usp=sharing")

    """
    if not path:
        raise Exception("Specify a URL or path to fetch file from.")

    # If path is a local path instead of a URL then urlretrieve will just return that path
    destination_dir = destination_dir or os.environ.get("LOCAL_WEIGHTS_PATH")
    fname = path.rsplit("/", 1)[-1]
    if destination_dir:
        destination_dir = pathlib.Path(destination_dir)
        if prefix:
            destination_dir = destination_dir / prefix
        if not destination_dir.exists():
            logger.info(f"Creating local directory {str(destination_dir)}")
            destination_dir.mkdir(parents=True, exist_ok=True)
        local_filepath = pathlib.Path(destination_dir) / fname
        if suffix:
            local_filepath = local_filepath.with_suffix(suffix)
    else:
        raise Exception(
            "No destination directory specified by LOCAL_WEIGHTS_PATH or app settings."
        )

    if local_filepath and local_filepath.exists():
        logger.info(f"Using existing {local_filepath}")
        return local_filepath

    else:
        logger.info(f"Downloading {path} to {local_filepath}")
        resulting_filepath, headers = urllib.request.urlretrieve(
            url=path, filename=local_filepath
        )
        resulting_filepath = pathlib.Path(resulting_filepath)
        logger.info(f"Downloaded to {resulting_filepath}")
        return resulting_filepath


def synchronize_clocks():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        pass


def bbox_relative(bbox_absolute, img_width, img_height):
    """
    Convert bounding box from absolute coordinates (x1, y1, x2, y2)
    like those used by pytorch, to coordinates that are relative
    percentages of the original image size like those used by
    the COCO cameratraps format.
    https://github.com/Microsoft/CameraTraps/blob/main/data_management/README.md#coco-cameratraps-format
    """

    box_numpy = bbox_absolute.detach().cpu().numpy()
    bbox_percent = [
        round(box_numpy[0] / img_width, 4),
        round(box_numpy[1] / img_height, 4),
        round(box_numpy[2] / img_width, 4),
        round(box_numpy[3] / img_height, 4),
    ]
    return bbox_percent


def crop_bbox(image, bbox):
    """
    Create cropped image from region specified in a bounding box.

    Bounding boxes are assumed to be in the format:
    [(top-left-coordinate-pair), (bottom-right-coordinate-pair)]
    or: [x1, y1, x2, y2]

    The image is assumed to be a numpy array that can be indexed using the
    coordinate pairs.
    """

    x1, y1, x2, y2 = bbox

    cropped_image = image[
        :,
        int(y1) : int(y2),
        int(x1) : int(x2),
    ]
    transform_to_PIL = torchvision.transforms.ToPILImage()
    cropped_image = transform_to_PIL(cropped_image)
    yield cropped_image


def get_user_data_dir() -> pathlib.Path:
    """
    Return the path to the user data directory if possible.
    Otherwise return the system temp directory.
    """
    try:
        from trapdata.settings import read_settings

        settings = read_settings()
        return settings.user_data_path
    except Exception:
        import tempfile

        return pathlib.Path(tempfile.gettempdir())


@dataclass
class Taxon:
    gbif_id: int
    name: Optional[str]
    genus: Optional[str]
    family: Optional[str]
    source: Optional[str]


def fetch_gbif_species(gbif_id: int) -> Optional[Taxon]:
    """
    Look up taxon name from GBIF API. Cache results in user_data_path.
    """

    base_url = "https://api.gbif.org/v1/species/{gbif_id}"
    url = base_url.format(gbif_id=gbif_id)

    try:
        taxon_data = get_or_download_file(
            url, destination_dir=get_user_data_dir(), prefix="taxa/gbif", suffix=".json"
        )
        data: dict = json.load(taxon_data.open())
    except urllib.error.HTTPError:
        logger.warn(f"Could not find species with gbif_id {gbif_id} in {url}")
        return None
    except json.decoder.JSONDecodeError:
        logger.warn(f"Could not parse JSON response from {url}")
        return None

    taxon = Taxon(
        gbif_id=gbif_id,
        name=data["canonicalName"],
        genus=data["genus"],
        family=data["family"],
        source="gbif",
    )
    return taxon


def lookup_gbif_species(species_list_path: str, gbif_id: int) -> Taxon:
    """
    Look up taxa names from a Darwin Core Archive file (DwC-A).

    Example:
    https://docs.google.com/spreadsheets/d/1E3-GAB0PSKrnproAC44whigMvnAkbkwUmwXUHMKMOII/edit#gid=1916842176

    @TODO Optionally look up species name from GBIF API
    Example https://api.gbif.org/v1/species/5231190
    """
    local_path = get_or_download_file(
        species_list_path, destination_dir=get_user_data_dir(), prefix="taxa"
    )
    df = pd.read_csv(local_path)
    taxon = None
    # look up single row by gbif_id
    try:
        row = df.loc[df["taxon_key_gbif_id"] == gbif_id].iloc[0]
    except IndexError:
        logger.warn(
            f"Could not find species with gbif_id {gbif_id} in {species_list_path}"
        )
    else:
        taxon = Taxon(
            gbif_id=gbif_id,
            name=row["search_species_name"],
            genus=row["genus_name"],
            family=row["family_name"],
            source=row["source"],
        )

    if not taxon:
        taxon = fetch_gbif_species(gbif_id)

    if not taxon:
        return Taxon(
            gbif_id=gbif_id, name=str(gbif_id), genus=None, family=None, source=None
        )

    return taxon


def replace_gbif_id_with_name(name) -> str:
    """
    If the name appears to be a GBIF ID, then look up the species name from GBIF.
    """
    try:
        gbif_id = int(name)
    except ValueError:
        return name
    else:
        taxon = fetch_gbif_species(gbif_id)
        if taxon and taxon.name:
            return taxon.name
        else:
            return name


class StopWatch:
    """
    Measure inference time with GPU support.

    >>> with stopwatch() as t:
    >>>     sleep(5)
    >>> int(t.duration)
    >>> 5
    """

    def __enter__(self):
        synchronize_clocks()
        # self.start = time.perf_counter()
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        synchronize_clocks()
        # self.end = time.perf_counter()
        self.end = time.time()
        self.duration = self.end - self.start

    def __repr__(self):
        start = datetime.datetime.fromtimestamp(self.start).strftime("%H:%M:%S")
        end = datetime.datetime.fromtimestamp(self.end).strftime("%H:%M:%S")
        seconds = int(round(self.duration, 1))
        return f"Started: {start}, Ended: {end}, Duration: {seconds} seconds"
