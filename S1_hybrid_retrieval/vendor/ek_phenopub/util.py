#  Copyright (c) 2018-2023 Beijing Ekitech Co., Ltd.
#  All rights reserved.

import logging
import logging.config
import os
from collections.abc import Sequence
from datetime import datetime

import errno
import pandas as pd
from six import string_types


def ensure_dir(path):
    """os.path.makedirs without EEXIST."""
    try:
        os.makedirs(os.path.abspath(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def is_str(arg):
    return isinstance(arg, string_types)


def to_list(arg):
    if is_str(arg):
        return [arg]
    elif isinstance(arg, Sequence):
        return arg
    return [arg]


def to_numeric_date_str(dt):
    """ Returns 8-digit numeric date string e.g. 20170522 """
    return dt.strftime("%Y%m%d")


def to_py_date(timestamp_obj):
    return to_timestamp(timestamp_obj).date()


def to_timestamp(timestamp_obj):
    """
    Creates a pandas pd.Timestamp using its constructor if not already
    a pd.Timestamp object, otherwise just returns it
    """
    if isinstance(timestamp_obj, pd.Timestamp):
        return timestamp_obj
    else:
        return pd.Timestamp(timestamp_obj)


def iter_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def magic_open(file_path, mode):
    if file_path.endswith('.gz'):
        import gzip
        return gzip.open(file_path, mode)
    elif file_path.endswith('.zst'):
        import zstandard
        return zstandard.open(file_path, mode)
    elif file_path.endswith('.xz'):
        import lzma
        return lzma.open(file_path, mode)
    elif file_path.endswith('.lz4'):
        import lz4.frame
        return lz4.frame.open(file_path, mode)
    else:
        return open(file_path, mode)


def dump_file(data_object, file_path, serializer='dill', allow_external=False):
    """ Deserialize object from file using dill """
    file_path = str(file_path)
    ensure_dir(os.path.dirname(file_path))

    if allow_external and file_path.endswith('xz'):
        external_xz_compress(data_object, file_path)
        return file_path

    if serializer == 'dill':
        import dill
        dump_func = dill.dump
    elif serializer == 'pickle':
        import pickle
        dump_func = pickle.dump
    else:
        raise RuntimeError('Must be (dill, pickle)')

    with magic_open(file_path, 'wb') as f:
        dump_func(data_object, f)

    return file_path


def load_file(file_path):
    """ Serialize object to file using dill """
    file_path = str(file_path)

    with magic_open(file_path, 'rb') as f:
        import dill
        return dill.load(f)


def external_xz_compress(obj, file_path):
    """ Dump object using dill with external xz utilizing all available threads """
    with open(file_path, 'wb') as fw:
        from subprocess import Popen, PIPE
        process = Popen(['xz', '-z', '-c', '-T0'], stdin=PIPE, stdout=fw, bufsize=1)
        with process.stdin:
            import dill
            dill.dump(obj, process.stdin)
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f'xz returned with error code: {returncode}')


def split_params(text, delim=',', dedupe=False):
    if text is None:
        return []

    text = text.strip()
    if not text:
        return []

    splits = (p.strip() for p in text.split(delim))
    return list(_dedupe(splits) if dedupe else splits)


def _dedupe(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)


def get_usable_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing
        return multiprocessing.cpu_count()


def configure_logging(log_level=None, stream='stdout'):
    # use a variable in the function object to determine if it has run before
    if getattr(configure_logging, 'has_run', False):
        return

    log_level = log_level or 'INFO'

    logging.config.dictConfig(_make_logging_config(log_level, stream=stream))

    logging.captureWarnings(True)

    configure_logging.has_run = True


def _make_logging_config(log_level, stream='stdout'):
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(name)s [%(threadName)s] %(levelname)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': f'ext://sys.{stream}',
            },
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'propagate': True,
                'level': log_level,
            },
        }
    }


class StopWatch(object):
    def __init__(self, init_timestamp=None):
        self.init_timestamp = None
        self.last_timestamp = None
        self.reset(init_timestamp)

    def record(self):
        current_timestamp = datetime.datetime.utcnow()
        total_seconds = (current_timestamp - self.last_timestamp).total_seconds()
        self.last_timestamp = current_timestamp
        return total_seconds

    def stop(self):
        return (datetime.datetime.utcnow() - self.init_timestamp).total_seconds()

    def reset(self, init_timestamp=None):
        self.init_timestamp = init_timestamp or datetime.datetime.utcnow()
        self.last_timestamp = self.init_timestamp


dedupe = _dedupe
stop_watch = StopWatch
