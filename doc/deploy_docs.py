#!/usr/tce/packages/python/python-3.10.8/bin/python3
"""Scripts for deploying user documentation."""
import os
import subprocess
from shutil import rmtree, copytree

def remove_file_or_directory(file_path: str) -> None:
    """Remove a file or directory if it exists.

    Args:
        file_path (str): the path to the file or directories that should be removed
    """
    try:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                rmtree(file_path)
            else:
                os.remove(file_path)
    except Exception as my_exception:
        print(f"Issue removing file {file_path}")
        print(f"The current directory is {os.getcwd()}")
        raise my_exception

def install_docs(deploy_path, source_folder):
    """Install the documents to a location.

    Args:
        install_path (str): location for the merge request to be installed to
        source_folder (str): location where the files should be copied from
    """
    deploy_location: str = os.path.join(deploy_path)
    os.makedirs(deploy_location, exist_ok=True)
    copytree(source_folder, deploy_location, dirs_exist_ok=True)

def set_group_permissions_to_user_recursive(path: str, set_group: str = "") -> None:
    """Sets the permissions in a directory to same as the users permissions.

    This is recursive through the directory and sub-directories.

    Args:
        path (str): the path of the directory that should have permissions updated.
        set_group (str, default=""): If there is a string then set the group to the group name in
            addition to changing the permissions.
    """
    if set_group:
        subprocess.run(['chgrp', '-R', set_group, path], check=True)
    subprocess.run(['chmod', 'g=u', '-R', path], check=True)

def deploy_docs():
    """Deploy documentation to mili-python docs website."""
    docs_path: str = "/usr/global/web-pages/lc/www/mili-python/"

    # Remove user_docs directory
    print(f"Removing {os.path.join(docs_path, '*')}")
    remove_file_or_directory(os.path.join(docs_path, '*'))

    # Install new version of docs
    print(f"Installing new documentation to {os.path.join(docs_path)}")
    install_docs(docs_path, "./build/html")

    # set group permissions on directory
    print(f"Updating permissions")
    set_group_permissions_to_user_recursive(os.path.join(docs_path), "mdgdev")


if __name__ == "__main__":
    deploy_docs()