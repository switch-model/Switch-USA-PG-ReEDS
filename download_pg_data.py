"""
Retrieve and decompress input files to run PowerGenome and set PowerGenome
parameters to access them.

Will download all files and folders named in pg_data.yml by default, or only the
ones that contain any word specified on the command line, if any, e.g.,

python download_pg_data.py misc_tables
"""

import os, sys, zipfile, platform, urllib
import gdown, yaml


def unzip_if_needed(filename):
    if filename.endswith(".zip"):
        print(f"unzipping {filename}")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            # identify the files we want to extract (ignore metadata)
            names = [n for n in zip_ref.namelist() if not n.startswith("__MACOSX")]
            top_level_files = set(n.split("/")[0] for n in names)
            if len(top_level_files) == 1:
                # contains one file or one directory; expand in place
                dest = os.path.dirname(filename)
            else:
                # use zip file name as outer subdir
                dest = os.path.splitext(filename)[0]
            if os.path.exists(dest):
                os.remove(dest)
            zip_ref.extractall(dest, members=names)
        os.remove(filename)
    else:
        pass


def main(filter=[]):
    with open("pg_data.yml", "r") as f:
        settings = yaml.safe_load(f)

    os.makedirs("pg_data", exist_ok=True)

    # Warn about Windows limit on filenames if needed
    # 130 char file name, equal to the longest path created by this script
    test_file = os.path.abspath(os.path.join("pg_data", "x" * 130))
    if platform.system == "Windows" and len(test_file) > 260:
        try:
            f = open(test_file, "w")
        except OSError:
            raise RuntimeError(
                "This script needs to create files with names longer than 260 "
                "characters, which are not supported by your current Windows "
                "configuration. Please see "
                "https://answers.microsoft.com/en-us/windows/forum/all/how-to-extend-file-path-characters-maximum-limit/691ea463-2e15-452b-8426-8d372003504f"
                " for options to fix this."
            )
        else:
            f.close()
            os.remove(test_file)

    for dest, url in settings["download_gdrive_folders"].items():
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            files = gdown.download_folder(url, output=dest)
            for filename in files:
                unzip_if_needed(filename)

    for dest, url in settings["download_gdrive_files"].items():
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            filename = gdown.download(url, fuzzy=True, output=dest)
            unzip_if_needed(filename)

    for dest, url in settings["download_files"].items():
        if not filter or any(f in dest for f in filter):
            print(f"\nretrieving {dest}")
            urllib.request.urlretrieve(url, dest)
            unzip_if_needed(dest)

    # create model_dir/env.yml
    for model_dir, model_settings in settings["env.yml"].items():
        yml_file = os.path.join(model_dir, "env.yml")
        print(f"\ncreating {yml_file}")
        with open(yml_file, "w") as f:
            for var, dest in model_settings.items():
                abs_dest = os.path.abspath(dest)
                f.write(f"{var}: '{abs_dest}'\n")

    print(f"\n{sys.argv[0]} finished.")


if __name__ == "__main__":
    main(filter=sys.argv[1:])
