"""Adapted from
https://github.com/ad12/meddlr/blob/main/meddlr/utils/env.py."""
import importlib
import re
from importlib import util

from packaging import version

_SUPPORTED_PACKAGES = {}


def package_available(name: str) -> bool:
    """Returns if package is available.

    Args:
        name (str): Name of the package.

    Returns:
        bool: Whether module exists in environment.
    """
    global _SUPPORTED_PACKAGES
    if name not in _SUPPORTED_PACKAGES:
        _SUPPORTED_PACKAGES[name] = importlib.util.find_spec(name) is not None
    return _SUPPORTED_PACKAGES[name]


def get_package_version(package_or_name) -> str:
    """Returns package version.

    Args:
        package_or_name (``module`` or ``str``): Module or name of module.
            This package must have the version accessible through
            ``<module>.__version__``.

    Returns:
        str: The package version.

    Examples:
        >>> get_version("numpy")
        "1.20.0"
    """
    if isinstance(package_or_name, str):
        if not package_available(package_or_name):
            raise ValueError(f"Package {package_or_name} not available")
        spec = util.find_spec(package_or_name)
        package_or_name = util.module_from_spec(spec)
        spec.loader.exec_module(package_or_name)
    version = package_or_name.__version__
    return version


def is_package_installed(pkg_str) -> bool:
    """Verify that a package dependency is installed and in the expected
    version range.

    This is useful for optional third-party dependencies where implementation
    changes are not backwards-compatible.

    Args:
        pkg_str (str): The pip formatted dependency string.
            E.g. "numpy", "numpy>=1.0.0", "numpy>=1.0.0,<=1.10.0", "numpy==1.10.0"

    Returns:
        bool: Whether dependency is satisfied.

    Note:
        This cannot resolve packages where the pip name does not match the python
        package name. ``'-'`` characters are automatically changed to ``'_'``.
    """
    ops = {
        "==": lambda x, y: x == y,
        "<=": lambda x, y: x <= y,
        ">=": lambda x, y: x >= y,
        "<": lambda x, y: x < y,
        ">": lambda x, y: x > y,
    }
    comparison_patterns = "(==|<=|>=|>|<)"

    pkg_str = pkg_str.strip()
    pkg_str = pkg_str.replace("-", "_")
    dependency = list(re.finditer(comparison_patterns, pkg_str))

    if len(dependency) == 0:
        return package_available(pkg_str)

    pkg_name = pkg_str[: dependency[0].start()]
    if not package_available(pkg_name):
        return False

    pkg_version = version.Version(get_package_version(pkg_name))
    version_limits = pkg_str[dependency[0].start() :].split(",")

    for vlimit in version_limits:
        comp_loc = list(re.finditer(comparison_patterns, vlimit))
        if len(comp_loc) != 1:
            raise ValueError(f"Invalid version string: {pkg_str}")
        comp_op = vlimit[comp_loc[0].start() : comp_loc[0].end()]
        comp_version = version.Version(vlimit[comp_loc[0].end() :])
        if not ops[comp_op](pkg_version, comp_version):
            return False
    return True


def is_torch_available() -> bool:
    """Returns if torch is available.

    Returns:
        bool: Whether torch is available.
    """
    return package_available("torch")
