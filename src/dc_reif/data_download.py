from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import requests

from dc_reif.config import ProjectConfig
from dc_reif.utils import ensure_directory, get_logger, sha256_file

LOGGER = get_logger(__name__)


class DownloadError(RuntimeError):
    """Raised when dataset download fails."""


def is_aria2_available() -> bool:
    return shutil.which("aria2c") is not None


def _is_kaggle_url(url: str) -> bool:
    return url.startswith("kaggle://") or "kaggle.com" in url


def _download_with_aria2(url: str, target_dir: Path, filename: str) -> None:
    command = [
        "aria2c",
        "-x",
        "8",
        "-s",
        "8",
        "-k",
        "1M",
        "-d",
        str(target_dir),
        "-o",
        filename,
        url,
    ]
    LOGGER.info("Download method: aria2c")
    subprocess.run(command, check=True)


def _download_with_requests(url: str, destination: Path) -> None:
    LOGGER.info("Download method: requests")
    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code in {401, 403}:
            raise DownloadError(
                f"Authenticated download required for {url}. "
                "Provide an anonymous direct file URL or configure the required credentials."
            )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type and _is_kaggle_url(url):
            raise DownloadError(
                "The configured URL appears to require interactive authentication. "
                "For Kaggle, configure the Kaggle API credentials and use a kaggle:// dataset reference."
            )

        with destination.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)


def _download_with_wget(url: str, destination: Path) -> bool:
    wget = shutil.which("wget")
    if not wget:
        return False
    LOGGER.info("Download method: wget")
    subprocess.run([wget, "-O", str(destination), url], check=True)
    return True


def _download_with_urllib(url: str, destination: Path) -> None:
    LOGGER.info("Download method: urllib")
    try:
        urllib.request.urlretrieve(url, destination)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise DownloadError(
                f"Authenticated download required for {url}. "
                "Provide a direct downloadable link or configure the relevant credentials."
            ) from exc
        raise


def _download_from_kaggle(url: str, target_dir: Path) -> Path:
    if not shutil.which("kaggle"):
        raise DownloadError(
            "Kaggle download requested but the Kaggle CLI is not installed. "
            "Install the Kaggle package, configure credentials, and retry."
        )

    if not url.startswith("kaggle://"):
        raise DownloadError(
            "Only kaggle://owner/dataset/file.csv references are supported for authenticated Kaggle downloads."
        )
    parts = url.removeprefix("kaggle://").split("/")
    if len(parts) < 3:
        raise DownloadError("Kaggle references must look like kaggle://owner/dataset/file.csv")
    dataset = f"{parts[0]}/{parts[1]}"
    target_name = "/".join(parts[2:])

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-f",
        target_name,
        "-p",
        str(target_dir),
    ]
    LOGGER.info("Download method: kaggle CLI")
    subprocess.run(command, check=True)
    zip_candidates = list(target_dir.glob("*.zip"))
    for zip_path in zip_candidates:
        with ZipFile(zip_path) as zipped:
            zipped.extractall(target_dir)
        zip_path.unlink()
    return target_dir / target_name


def _validate_existing_file(path: Path, checksum: str | None, force_download: bool) -> bool:
    if not path.exists():
        return False

    if not checksum:
        if force_download:
            path.unlink()
            return False
        LOGGER.info("Existing file found with no checksum configured. Reusing %s", path)
        return True

    observed = sha256_file(path)
    if observed == checksum:
        LOGGER.info("Existing file checksum matched. Reusing %s", path)
        return True
    if not force_download:
        raise DownloadError(
            f"Checksum mismatch for {path.name}. Expected {checksum}, observed {observed}. "
            "Set FORCE_DOWNLOAD=true to overwrite the file."
        )
    LOGGER.warning("Checksum mismatch detected, removing %s due to force-download.", path)
    path.unlink()
    return False


def download_dataset(config: ProjectConfig) -> Path:
    data_dir = ensure_directory(config.data_dir)
    destination = data_dir / config.data_filename

    LOGGER.info("Resolved data directory: %s", data_dir)
    LOGGER.info("Resolved dataset path: %s", destination)
    LOGGER.info("Configured DATA_URL: %s", config.data_url)

    if _validate_existing_file(destination, config.data_checksum, config.force_download):
        return destination

    if config.data_url.startswith("kaggle://"):
        destination = _download_from_kaggle(config.data_url, data_dir)
    elif "kaggle.com" in config.data_url:
        raise DownloadError(
            "The configured URL looks like a Kaggle-hosted source. "
            "Use a direct downloadable URL for the default workflow, or configure a kaggle://owner/dataset/file.csv reference."
        )
    else:
        if config.use_aria2:
            if is_aria2_available():
                _download_with_aria2(config.data_url, data_dir, config.data_filename)
            else:
                LOGGER.warning("aria2c not found in PATH. Falling back to Python-based download methods.")
                try:
                    _download_with_requests(config.data_url, destination)
                except Exception as exc:
                    LOGGER.warning("Requests download failed: %s", exc)
                    if not _download_with_wget(config.data_url, destination):
                        _download_with_urllib(config.data_url, destination)
        else:
            try:
                _download_with_requests(config.data_url, destination)
            except Exception as exc:
                LOGGER.warning("Requests download failed: %s", exc)
                if not _download_with_wget(config.data_url, destination):
                    _download_with_urllib(config.data_url, destination)

    if not destination.exists():
        raise DownloadError(f"Download finished without creating {destination}.")

    if config.data_checksum:
        observed_checksum = sha256_file(destination)
        if observed_checksum != config.data_checksum:
            if not config.force_download:
                raise DownloadError(
                    f"Downloaded file checksum mismatch. Expected {config.data_checksum}, observed {observed_checksum}."
                )
            LOGGER.warning("Downloaded file checksum mismatch after forced download.")

    LOGGER.info("Dataset available at %s", destination)
    return destination
