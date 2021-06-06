# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy loader class."""

from __future__ import absolute_import, division, print_function

import importlib
import logging
import types

logger = logging.getLogger(__name__)


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    `contrib`, and `ffmpeg` are examples of modules that are large and
    not always needed, and this allows them to only be loaded when they
    are used.
    """

    # The lint error here is incorrect.
    def __init__(
        self,
        local_name,
        parent_module_globals=None,
        name=None,
        warning=None,
        error=None,
    ):  # pylint: disable=super-on-old-class
        if parent_module_globals is None:
            parent_module_globals = globals()

        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning
        self._error = error

        if name is None:
            name = local_name

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
        except ImportError as e:
            raise ImportError(self._error) if self._error else e
        self._parent_module_globals[self._local_name] = module

        # Emit a warning if one was specified
        if self._warning:
            logger.warning(self._warning)
            # Make sure to only warn once.
            self._warning = None

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
