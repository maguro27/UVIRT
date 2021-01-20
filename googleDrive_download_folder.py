import os
import argparse
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


gauth = GoogleAuth()
gauth.CommandLineAuth()
drive = GoogleDrive(gauth)


def download_file_recursively(parent_id, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    file_list = drive.ListFile(
        {"q": '"{}" in parents and trashed = false'.format(parent_id)}
    ).GetList()

    for f in file_list:
        if f["mimeType"] == "application/vnd.google-apps.folder":
            download_file_recursively(f["id"], os.path.join(dst_dir, f["title"]))
        else:
            dst_path = os.path.join(dst_dir, f["title"])
            f.GetContentFile(dst_path)
            print("Download {} to {}".format(f["title"], dst_path))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--parent_id",
    "-p",
    type=str,
    help="Download folder id, e.g. in https://drive.google.com/drive/folders/hogehoge, id is hogehoge.",
)
parser.add_argument("--save_dir", "-s", type=str, help="Save directory")
args = parser.parse_args()

download_file_recursively(args.parent_id, args.save_dir)
