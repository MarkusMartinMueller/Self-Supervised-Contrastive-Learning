#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  File: config.py
  Author: Tristan Kreuziger (tristan.kreuziger@tu-berlin.de)
  Created: 2020-07-29 13:28
  Copyright (c) 2020 Tristan Kreuziger under MIT license
"""

import os
import json
import yaml


def parse_config(filename):
    """
    Loads the specified configuration file and returns it as a dictionary.

    Args:
        filename (str): the fully qualified name and path of the configuration file.

    Returns:
        dict[str -> obj]: the parsed configuration
    """

    if os.path.exists(filename):
        try:
            if filename.endswith('.yaml'):
                with open(filename, 'r') as file_handle:
                    return yaml.load(file_handle, Loader=yaml.FullLoader)
            elif filename.endswith('.json'):
                with open(filename, 'r') as file_handle:
                    return json.load(file_handle)
            else:
                raise ValueError(
                    f'The type of the configuration file (YAML or JSON) could not be determined from the extension ("{filename}").')

        except IOError as error:
            raise ValueError(
                f'The config file could not be loaded. Original exception is: {error}')

    else:
        raise ValueError(
            f'The specified config file "{filename}" does not exist.')
