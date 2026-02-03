# project file management for EONS experiments

__version__ = "0.0.1"

from functools import cached_property
from ast import literal_eval
import pathlib as pl
import tempfile
import zipfile
import shutil
import time
import csv
import sys
import os
from swarmsim import yaml
from . import jsontools as jst

# typing
from typing import override


# required for training
DEFAULT_PROJECT_BASEPATH = pl.Path("results")
LOGFILE_NAME = "training.log"
BESTNET_NAME = "best.json"
POPULATION_FITNESS_NAME = "population_fitnesses.log"
# used in project mode
RUNINFO_NAME = "runinfo.yaml"
BACKUPNET_NAME = "previous.json"
NETWORKS_DIR_NAME = "networks"
ARTIFACTS_DIR_NAME = "artifacts"


def _NONE1(x):
    pass


def is_project_dir(path):
    return path.is_dir() and (path / RUNINFO_NAME).is_file()


def contains_single_dir(path: pl.Path | None):
    if path is None or not path.is_dir():
        return False
    it = path.iterdir()
    try:
        d = next(it)
    except StopIteration:
        return False
    if not d.is_dir():
        return False
    try:
        next(it)
    except StopIteration:
        return d if d.is_dir() else False


def find_lastmodified_dir(basepath):
    basepath = pl.Path(basepath)
    projectdirs = [(child, os.path.getmtime(child)) for child in basepath.iterdir() if is_project_dir(child)]
    return max(projectdirs, key=lambda x: x[1])[0]


def check_if_writable(path):
    if not os.access(path, os.W_OK):
        msg = f"{path} could not be accessed. Check that you have permissions to write to it."
        raise PermissionError(msg)


def inquire_project(root=None):
    if root is None:
        root = DEFAULT_PROJECT_BASEPATH
    from InquirerPy import inquirer
    from InquirerPy.base import Choice
    projects = list(root.iterdir())
    if not list:
        return None
    projects = sorted(projects)
    newest = max(projects, key=lambda f: f.stat().st_mtime)
    projects.insert(0, Choice(newest, name=f"Suggested (newest): {newest.name}"))
    return inquirer.fuzzy(choices=projects, message="Select a project").execute()


def inquire_size(file: pl.Path, limit=300e6):  # 300 MB
    size = file.stat().st_size
    if size < limit:
        return
    mb = round(size / 1e6, 3)
    message = f"The file {file} is {mb} MB. Do you want to continue?"
    try:
        from InquirerPy import inquirer
        return not inquirer.confirm(message=message, default=True).execute()
    except ImportError:
        print(message)
        input("Press enter to continue, ctrl-c to cancel.")


def read_tsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return list(reader)


def yaml_safe_load(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_dict(obj):
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "asdict"):
        return obj.asdict()
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {i: get_dict(val) for i, val in enumerate(obj)}
    if hasattr(obj, "__dict__"):
        return vars(obj)

    return obj


def get_config_dict(obj):
    if hasattr(obj, "as_config_dict"):
        return obj.as_config_dict()
    return get_dict(obj)


cache = {}
def ensure_dir_exists(path: os.PathLike, parents=True, exist_ok=True, **kwargs):  # noqa: E302
    global cache
    path = pl.Path(path)
    fullpath = path.resolve()
    key = str(fullpath)
    if not cache.get(key, False):
        path.mkdir(parents=parents, exist_ok=exist_ok, **kwargs)
        cache[key] = path.is_dir()


class File:
    def __init__(self, path: os.PathLike):
        self.path = pl.Path(path)

    def write(self, s):
        with open(self.path, 'w') as f:
            f.write(s)

    def append(self, s):
        with open(self.path, 'a') as f:
            f.write(s)

    def __add__(self, s):
        self.append(s)
        return self

    def __str__(self):
        return str(self.path)

    def as_config_dict(self):
        return self.as_dict()

    def as_dict(self):
        return {'path': str(self)}

    def exists(self):
        return self.path.exists()


class Logger(File):
    def __init__(self, path, firstcall=None):
        super().__init__(path)
        self._initialized = False
        self.firstcall = _NONE1 if firstcall is None else firstcall

    @override
    def append(self, s):
        if not self._initialized:
            self.firstcall(self)
            self._initialized = True
        super().append(s)

    def as_dict(self):
        d = super().as_dict()
        d.update({'firstcall': repr(self.firstcall)})
        return d

    def read_text(self):
        return self.path.read_text()

    def read_lines(self):
        return self.path.read_text().splitlines()


class FolderlessProject:
    isproj = False

    def __init__(self, network_path_or_jsonstr, logfile_path, name=None):
        self.name = name
        self.logfile = Logger(logfile_path)
        self._network_path_or_jsonstr = network_path_or_jsonstr
        self.bestnet_file: None | File = None
        self.bestnet = dict()

    def load_bestnet(self,
        path_or_jsonstr: None | str | os.PathLike | File | dict = None,
        update_path=True,
    ) -> None:
        if path_or_jsonstr is None:
            path_or_jsonstr = self._network_path_or_jsonstr
        if isinstance(path_or_jsonstr, File):
            path_or_jsonstr = path_or_jsonstr.path

        # HACK: on most code paths, the json file will be loaded twice (json_detect, smartload)
        detected_type = jst.json_detect(path_or_jsonstr)
        if detected_type is jst.ValidJSONFilePath:
            self.bestnet = jst.smartload(path_or_jsonstr)
            self.bestnet_file = File(path_or_jsonstr)
        elif detected_type is jst.ValidJSONStr:
            self.bestnet = jst.smartload(path_or_jsonstr)
            self.bestnet_file = None
        elif detected_type is dict:
            self.bestnet = path_or_jsonstr
            self.bestnet_file = None
        else:
            raise ValueError("Unsupported type for argument 'path_or_jsonstr'")

    def check_bestnet_writable(self):
        # check that the best network path is writable
        path = pl.Path(self.bestnet_file.path)
        if path.is_file():
            check_if_writable(path)
            print(f"WARNING: The output network file\n    {path}\nexists and will be overwritten!")
        elif path.is_dir():
            raise OSError(1, f"{path} is a directory. Please specify a valid path for the output network file.")
        else:  # parent dir probably doesn't exist.
            if path.parent.is_dir():
                try:
                    f = open(path, 'ab')
                except BaseException as err:
                    raise err
                finally:
                    f.close()  # type: ignore
            else:
                raise OSError(2, f"One or more parent directories are missing. Cannot write to {path}.")

    def backup_bestnet(self, new_path=None):
        if self.bestnet_file.path.is_file():
            if new_path is None:
                new_path = self.bestnet_file.path.with_name(BACKUPNET_NAME)
            self.bestnet_file.path.rename(new_path)

    @property
    def is_bestnet_loaded(self):
        return bool(self.bestnet)


class Project(FolderlessProject):
    """Project class for managing project data.

    Contains Paths to stuff in the project folder.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the project folder. Does not need to exist.
    name : str, optional
        Name of the project. If not specified, the name will be the same as the folder name.
    overwrite : bool, default=False
        Whether to overwrite existing files in the project folder. Defaults to False.
    """

    isproj = True

    def __init__(self, name=None, path=None, overwrite=False):
        self.name = name
        self.root = pl.Path(path) if path is not None else None
        if name is None and isinstance(self.root, pl.Path):
            # determine name from path
            self.name = self.root.name
        elif name is not None and path is None:
            # if name is given, use as project directory name
            self.root = pl.Path(DEFAULT_PROJECT_BASEPATH / name)
        self.allow_overwrite = overwrite
        self.bestnet_file = File(self.root / BESTNET_NAME)
        self.networks = Networks(self, NETWORKS_DIR_NAME)
        self.logfile = Logger(self.root / LOGFILE_NAME)
        self.popfit_file = Logger(self.root / POPULATION_FITNESS_NAME)
        self.popfit_file.firstcall = self._default_firstcall
        self.artifacts = self.root / 'artifacts'
        self._opened = True

    @property
    def original_path(self):
        return self.root

    def possibly_valid(self):
        checks = [
            self.logfile.exists(),
            self.networks.exists(),
            self.bestnet_file.exists(),
            self.popfit_file.exists(),
        ]
        return self.root.exists() and any(checks)

    @staticmethod
    def _check_relpath(relpath):
        path = pl.PurePath(relpath)
        if path.is_absolute():
            msg = f"Expected a relative path, got {relpath}"
            raise ValueError(msg)

    def pathrel(self, relpath: str | bytes | os.PathLike):
        """Get a path relative to the project root.

        Parameters
        ----------
        relpath : str | bytes | os.PathLike
            The path relative to the project root.

        Returns
        -------
        pathlib.Path
            The joined path, relative to the project root.
        """
        self._check_relpath(relpath)
        return self.root / relpath

    def pathabs(self, relpath: str | bytes | os.PathLike):
        """Get a path relative to the project root.

        Parameters
        ----------
        relpath : str | bytes | os.PathLike
            The path relative to the project root.

        Returns
        -------
        pathlib.Path
            The joined path, relative to the absolute path to the project root.
        """
        self._check_relpath(relpath)
        return self.root.resolve() / relpath

    __div__ = pathrel

    def __truediv__(self, relpath):
        return self.pathrel(relpath)

    __getitem__ = pathrel

    def __fspath__(self):
        if self.root is None:
            raise RuntimeError("Project has no path.")
        return str(self.root)

    def load_bestnet(self,
        path_or_jsonstr: None | str | os.PathLike | File | dict = None,
        update_path=True,
    ) -> None:
        if not path_or_jsonstr:
            path_or_jsonstr = self.bestnet_file
        super().load_bestnet(path_or_jsonstr, update_path)

    def _default_firstcall(self, f):
        f.write(f"{time.time()}\t{0}\t[]\n")

    def make_root_interactive(self):
        create_parents = False
        if self.root.is_dir():
            if self.allow_overwrite:
                s = 'rm'
            else:
                s = input(f"Project folder already exists:\n\t{str(self.root)}\n'y' to continue, 'rm' to delete the contents of the folder, anything else to exit. ")  # noqa: E501
            if s.lower() not in ('y', 'yes', 'rm'):
                print("Exiting. Your filesystem has not been modified.")
                sys.exit(1)
            if s.lower() == 'rm':
                shutil.rmtree(self.root)   # type: ignore[reportArgumentType]
                print(f"Deleted {self.root}.")
        elif not self.root.parent.is_dir():
            if 'MAKE_PARENTS' in os.environ and os.environ['MAKE_PARENTS'].lower() in ('true', '1'):
                env_ask_parents = False
            else:
                env_ask_parents = True
            if env_ask_parents:
                print("WARNING: You're trying to put the project in")
                print(str(self.root.parent))
                print("but some part of it does not exist! Would you like to create it?")
                s = input("Type 'y' to create it, anything else to exit. ")
                if s.lower() not in ('y', 'yes'):
                    print("Exiting. Your filesystem has not been modified.")
                    sys.exit(1)
            create_parents = True
        print(f"Creating project folder at {self.root}")
        ensure_dir_exists(self.root, parents=create_parents)
        self._opened = True

    @property
    @override
    def logfile_path(self):
        return self.root / LOGFILE_NAME

    @property
    def runinfo_path(self):
        return self.root / RUNINFO_NAME

    @property
    def popfit_path(self):
        return self.root / POPULATION_FITNESS_NAME

    def log_popfit(self, info):
        self.popfit_file += f"{time.time()}\t{info.i}\t{repr(info.fitnesses)}\n"

    def read_popfit(self, error=True):
        from swarmsim.yaml.mathexpr import safe_eval
        if not self._opened:
            raise RuntimeError("Project is not open.")
        try:
            data = read_tsv(self.popfit_path)
        except FileNotFoundError as err:
            if not error:
                return []
            msg = f"Could not read population fitness of project {self.name} "
            msg += f"because it has no recorded {POPULATION_FITNESS_NAME} file."
            raise FileNotFoundError(msg) from err
        return list(zip(*([safe_eval(x) for x in line] for line in data)))

    def ensure_dir(self, relpath, parents=True, exist_ok=True, **kwargs):
        path = self.root / relpath
        ensure_dir_exists(path, parents=parents, exist_ok=exist_ok, **kwargs)
        return path

    def ensure_file_parents(self, relpath, parents=True, exist_ok=True, **kwargs):
        path = self.root / relpath
        ensure_dir_exists(path.parent, parents=parents, exist_ok=exist_ok, **kwargs)
        return path

    def save_yaml_artifact(self, name, obj):
        artifacts = pl.Path(ARTIFACTS_DIR_NAME)
        with open(self.ensure_file_parents(artifacts / name), "w") as f:
            yaml.dump(get_config_dict(obj), f)

    def load_yaml(self, name):
        with open(self.root / name, "r") as f:
            return yaml.load(f)

    def safe_load_yaml(self, name):
        with open(self.root / name, "r") as f:
            return yaml.load(f)

    def explore(self):
        from showinfm import show_in_file_manager
        assert self.root
        show_in_file_manager(str(self.root))

    @cached_property
    def env(self):
        return yaml_safe_load(self.artifacts / 'env.yaml')

    @cached_property
    def experiment(self):
        return yaml_safe_load(self.artifacts / 'experiment.yaml')

    @cached_property
    def evolver(self):
        return yaml_safe_load(self.artifacts / 'evolver.yaml')


class Networks:
    def __init__(self, project, relpath):
        self.path = project.root / relpath

    def get_file(self, epoch, population_id):
        ensure_dir_exists(self.path, parents=False)
        return File(self.path / f"e{epoch}-{population_id}.json")

    __call__ = get_file

    def exists(self):
        return self.path.exists()

    def any(self):
        return any(self.path.iterdir())


class UnzippedProject(Project):
    """Wrapper around Project which allows for usage of zipped project data.

    The primary use case assumes that it gets initialized with a path to a .zip file.
    In that case, it will unzip the file to a tmp directory.
    However, if the path is an existing folder with project data, data won't be moved.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the project folder or zip file.
    name : str, optional
        Name of the project. If not specified, the name will be the same as the folder name.
    overwrite : bool, default=False
        Whether to overwrite existing files in the project folder. Defaults to False.
    temp_path : str | pathlib.Path, optional
        Path to the temporary directory. If not specified, a random prefix will be used.
    """
    def __init__(self, path, name=None, overwrite=False, temp_path=None):
        self._original_path = pl.Path(path)
        self._tempdir = None
        self._opened = False
        self.temp_path = temp_path
        if self._original_path.is_dir() and not temp_path:
            super().__init__(name=name, path=path, overwrite=overwrite)
        else:
            self.name = self._original_path.stem
            self.allow_overwrite = overwrite

    def downgrade(self):
        super().__init__(name=self.original_path.stem,
                         path=self.original_path,
                         overwrite=self.allow_overwrite)
        self._opened = True

    @property
    def original_path(self):
        return self._original_path

    def check_valid_zip(self):
        if not self._original_path.is_file():
            return False
        return zipfile.is_zipfile(self._original_path)

    def unzip(self):
        if self._opened:
            return self
        if self._tempdir:
            raise RuntimeError("Project is already unzipped.")
        self._tempdir = tempfile.TemporaryDirectory(self.name, dir=self.temp_path)
        if self._original_path.is_file():
            with zipfile.ZipFile(self._original_path, "r") as d:
                d.extractall(self._tempdir.name)
        elif self._original_path.is_dir():
            shutil.copytree(self._original_path, self._tempdir.name)
        super().__init__(path=self._tempdir.name, name=self.name, overwrite=self.allow_overwrite)
        if not self.possibly_valid() and (d := contains_single_dir(self.root)):
            if UnzippedProject(path=d, name=d.name).possibly_valid():
                super().__init__(path=d, name=d.name, overwrite=self.allow_overwrite)
        return self

    def __enter__(self):
        self.unzip()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self):
        if self._tempdir:
            self._tempdir.cleanup()
            self._opened = False
        self._tempdir = None


if __name__ == "__main__":
    # if a user runs this file as a script, unzip the project, then print some info
    # and
    import argparse
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", help="path to project directories or zips", nargs="+")
    args = parser.parse_args()
    for filename in args.filenames:
        with UnzippedProject(filename) as proj:
            print(f"Name: {proj.name}" '' if proj.root.exists() else " (missing)")
            if hasattr(proj, "original_path"):
                print(f" -Original root: {proj.original_path.parent}")
            print(f"  logfile: {proj.logfile_path}" + ('' if proj.logfile_path.exists() else " (missing)"))
            print(f"  runinfo: {proj.popfit_path}" + ('' if proj.popfit_path.exists() else " (missing)"))
            print(f"  bestnet: {proj.bestnet_file.path}" + ('' if proj.bestnet_file.path.exists() else " (missing)"))
            print(f"  networks: {proj.networks.path}" + ('' if proj.networks.path.exists() else " (missing)"))
            print(f"  artifacts: {proj.root / ARTIFACTS_DIR_NAME}" + ('' if (proj.root / ARTIFACTS_DIR_NAME).exists() else " (missing)"))
            if proj.bestnet_file.path.exists():
                dest = (proj.original_path.parent / proj.name).with_suffix(".json")
                shutil.copy(proj.bestnet_file.path, dest)
