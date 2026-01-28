from pathlib import Path
import subprocess


ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = (ROOT_DIR / "models")
DATA_DIR = (ROOT_DIR / "data")

def wget_download(url, output_path, retries=3, show_progress=True):
    cmd = ['wget', url]
    cmd.extend(['-O', str(output_path)])
    cmd.extend(['--tries', str(retries)])
    cmd.extend(['--timeout', str(60)])
    if not show_progress:
        cmd.append('--quiet')

    try:
        subprocess.run(cmd, check=True)
        return {"success": True, "message": "Download completed successfully."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": f"Download failed: {e}"}

def extract_tar_gz(archive_path, extract_to_dir):
    cmd = ['tar', '-xzf', str(archive_path), '-C', str(extract_to_dir)]
    try:
        subprocess.run(cmd, check=True)
        return {"success": True, "message": "Extraction completed successfully."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": f"Extraction failed: {e}"}