"""
Make it easy to set up and manage project artifacts, data, and metadata
"""
import re
import sys

import git

from pathlib import Path


def repo_root(fpath: str) -> str:
    """
    Absolute path for nearest ancestor parent Git repository.

    Args:
        fpath (str): Path to file or directory.

    Returns:
        str: Absolute path to nearest ancestor parent Git repository.
    """
    git_repo = git.Repo(fpath, search_parent_directories=True)

    return git_repo.git.rev_parse("--show-toplevel")


class ProjectPath:
    """
    Common project path attributes and methods to manage paths and files.

    This class maps absolute paths to project directories and files as
    attributes that have the same name as the directory or file and the path to
    the project root is mapped to the `root` attribute.

    Additionally, for the data directory, this class maps absolute paths to each
    of the data subdirectories as attributes that have the same name as the
    subdirectory so that data files in the subdirectories can be easily accessed.
    """

    def __init__(self):
        # Get absolute path to project root
        fpath = Path(__file__).resolve()
        git_repo = git.Repo(fpath, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        self.root = Path(git_root)

        # First level subdirectories
        root_subdirs = [x for x in self.root.iterdir() if x.is_dir()]
        root_subdirs = list(
            filter(lambda x: not bool(re.match(r"\..*", x.name)), root_subdirs)
        )

        for x in root_subdirs:
            setattr(self, x.name, self.root.joinpath(x))

        # Map data subdirectories to attributes if any exist
        if hasattr(self, "data"):
            data_subdirs = [x for x in self.data.iterdir() if x.is_dir()]
            for x in data_subdirs:
                setattr(self, x.name, self.root.joinpath(x))


if __name__ == "__main__":
    sys.path.append(repo_root())
