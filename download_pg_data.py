"""
Retrieve and decompress input files to run PowerGenome and set PowerGenome
parameters to access them.

Will download all files and folders named in pg_data.yml by default, or only the
ones that contain any word specified on the command line, if any, e.g.,

python download_pg_data.py misc_tables
"""

import sys, zipfile, platform, urllib
from pathlib import Path
import gdown, yaml


def unzip_if_needed(filename):
    filepath = Path(filename)
    if filepath.suffix == ".zip":
        print(f"unzipping {filepath}")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            # identify the files we want to extract (ignore metadata)
            names = [n for n in zip_ref.namelist() if not n.startswith("__MACOSX")]
            top_level_files = set(n.split("/")[0] for n in names)
            if len(top_level_files) == 1:
                # contains one file or one directory; expand in place
                dest = filepath.parent
            else:
                # use zip file name as outer subdir
                dest = filepath.with_suffix(' ')
                if dest.exists():
                    dest.unlink()
            zip_ref.extractall(dest, members=names)
        filepath.unlink()

def make_parent(dest):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)

def main(filter=[]):
    with open("pg_data.yml", "r") as f:
        settings = yaml.safe_load(f)

    # Warn about Windows limit on filenames if needed
    maximum_path_length = settings.get('maximum_path_length', 0)
    if platform.system == "Windows" and maximum_path_length:
        test_file = Path("x" * maximum_path_length).resolve()
        if len(str(test_file)) > 260:
            try:
                f = open(test_file, "w")
            except OSError:
                raise RuntimeError(
                    "This script needs to create files with names longer than 260 "
                    "characters, which are not supported by your current Windows "
                    "configuration. Please see "
                    "https://answers.microsoft.com/en-us/windows/forum/all/how-to-extend-file-path-characters-maximum-limit/691ea463-2e15-452b-8426-8d372003504f"
                    " for options to fix this. Alternatively, try running the script "
                    f"in a directory with a path shorter than {260 - maximum_path_length} "
                    "characters, including separators."
                )
            else:
                f.close()
                test_file.unlink()

    setting_items = lambda k: (settings.get(k) or {}).items()

    for dest, url in setting_items("download_gdrive_folders"):
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            make_parent(dest)
            files = gdown.download_folder(url, output=dest)
            for filename in files:
                unzip_if_needed(filename)

    for dest, url in setting_items("download_gdrive_files"):
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            make_parent(dest)
            filename = gdown.download(url, fuzzy=True, output=dest)
            unzip_if_needed(filename)

    for dest, url in setting_items("download_files"):
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            make_parent(dest)
            urllib.request.urlretrieve(url, dest)
            unzip_if_needed(dest)

    # create model_dir/env.yml
    for model_dir, model_settings in setting_items("env.yml"):
        yml_file = Path(model_dir) / "env.yml"
        print(f"\ncreating {yml_file}")
        with open(yml_file, "w") as f:
            for var, dest in model_settings.items():
                # convert from relative to absolute path for PowerGenome
                abs_dest = Path(dest).resolve()
                f.write(f"{var}: '{abs_dest}'\n")

    print(f"\n{sys.argv[0]} finished.")


if __name__ == "__main__":
    main(filter=sys.argv[1:])
