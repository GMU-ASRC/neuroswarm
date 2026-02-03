# Basic tools for getting environment info.
# does not require pygit2 or gitpython

__version__ = "0.0.1"

import json
import string
import subprocess as sp
from urllib.parse import urlparse
from importlib.metadata import Distribution, version
import pathlib as pl


class GitRepositoryNotFoundError(Exception):
    pass


def is_root(path):
    # https://stackoverflow.com/questions/9823143/check-if-a-directory-is-a-file-system-root
    # resolves symlinks and ../ https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
    path = pl.Path(path).resolve()
    return path == pl.Path(path.anchor)


def module_editable_path(module):
    # https://stackoverflow.com/questions/43348746/how-to-detect-if-module-is-installed-in-editable-mode
    is_module = not isinstance(module, str)
    name = module.__name__ if is_module else module
    direct_url = Distribution.from_name(name).read_text("direct_url.json")
    if direct_url is None:
        return
    info = json.loads(direct_url)
    is_editable = info.get("dir_info", {}).get("editable", False)
    if is_editable:
        if is_module:
            if (file := pl.Path(module.__file__)).exists():
                return file.parent
        if url := info.get("url", None):
            parsed = urlparse(url)
            assert parsed.scheme == "file", f"Received a non-file url for editable module {name}: {url}"
            return pl.Path(urlparse(url).path)


def get_module_version(module):
    if isinstance(module, str):
        return version(module)
    else:
        return version(module.__version__)


def search_git_root(path, max_recursions=10):
    path = pl.Path(path)
    return _search_git_root(path, 0, max_recursions=max_recursions)


def _search_git_root(path: pl.Path, recursions, max_recursions) -> pl.Path | None:
    if (path / ".git").is_dir():
        return path
    elif recursions > max_recursions:
        msg = f"Reached max directory search depth. Could not find .git folder in parents after {max_recursions} levels up"
        raise RecursionError(msg)
    elif not is_root(path):
        return _search_git_root(path.parent, recursions + 1, max_recursions)
    raise GitRepositoryNotFoundError


def get_branch_ref(path):
    repo_root = search_git_root(path)
    if repo_root is None:
        raise RuntimeError("Could not find git root")
    return _get_branch_ref(repo_root)


def _get_branch_ref(repo_root) -> str:
    gitdir = repo_root / '.git'
    return (gitdir / 'HEAD').read_text().split(':')[1].strip()


def get_branch_name(path) -> str:
    ref_path = pl.PurePosixPath(get_branch_ref(path))
    if ref_path.is_absolute():
        msg = f"References to absolute paths are not supported. ref_path must be relative. Got {ref_path}"
        raise ValueError(msg)
    return str(ref_path.relative_to('refs/heads')).strip(string.whitespace + '/\\')


def git_porcelain(path='.'):
    ret = sp.run(['git', 'status', '--porcelain'], stdout=sp.PIPE, stderr=sp.PIPE, cwd=path)
    return ret.stdout.decode('utf-8').strip()


def git_hash(path='.'):
    ret = sp.run(['git', 'rev-parse', 'HEAD'], stdout=sp.PIPE, stderr=sp.PIPE, cwd=path)
    return ret.stdout.decode('utf-8').strip()


def framework_root():
    try:
        import neuro
    except ImportError:
        return
    framework = pl.Path(neuro.__file__).parents[4]
    if framework.name == 'framework':
        return framework
    else:
        raise FileNotFoundError("Could not find framework folder.")
