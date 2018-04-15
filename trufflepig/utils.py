import datetime
import logging
import os
import time

import numpy as np
from steembase.exceptions import RPCError, PostDoesNotExist

logger = logging.getLogger(__name__)


class _Progressbar(object):
    """Implements a progress bar.

    This class is supposed to be a singleton. Do not
    import the class itself but use the `progressbar` function from this module.

    Borrowed from pypet (https://github.com/SmokinCaterpillar/pypet).

    """
    def __init__(self):
        self._start_time = None   # Time of start/reset
        self._start_index = None  # Index of start/reset
        self._current_index = np.inf  # Current index
        self._percentage_step = None  # Percentage step for bar update
        self._total = None  # Total steps of the bas (float) not to be mistaken for length
        self._total_minus_one = None  # (int) the above minus 1
        self._length = None  # Length of the percentage bar in `=` signs
        self._norm_factor = None  # Normalization factor
        self._current_interval = None  # The current interval,
        # to check if bar needs to be updated

    def _reset(self, index, total, percentage_step, length):
        """Resets to the progressbar to start a new one"""
        self._start_time = datetime.datetime.now()
        self._start_index = index
        self._current_index = index
        self._percentage_step = percentage_step
        self._total = float(total)
        self._total_minus_one = total - 1
        self._length = length
        self._norm_factor = max(total * percentage_step / 100.0, 1)
        self._current_interval = int((index + 1.0) / self._norm_factor)

    def _get_remaining(self, index):
        """Calculates remaining time as a string"""
        try:
            current_time = datetime.datetime.now()
            time_delta = current_time - self._start_time
            try:
                total_seconds = time_delta.total_seconds()
            except AttributeError:
                # for backwards-compatibility
                # Python 2.6 does not support `total_seconds`
                total_seconds = ((time_delta.microseconds +
                                    (time_delta.seconds +
                                     time_delta.days * 24 * 3600) * 10 ** 6) / 10.0 ** 6)
            remaining_seconds = int((self._total - self._start_index - 1.0) *
                                    total_seconds / float(index - self._start_index) -
                                    total_seconds)
            remaining_delta = datetime.timedelta(seconds=remaining_seconds)
            remaining_str = ', remaining: ' + str(remaining_delta)
        except ZeroDivisionError:
            remaining_str = ''
        return remaining_str

    def __call__(self, index, total, percentage_step=5, logger='print', log_level=logging.INFO,
                 reprint=False, time=True, length=20, fmt_string=None,  reset=False):
        """Plots a progress bar to the given `logger` for large for loops.

        To be used inside a for-loop at the end of the loop.

        :param index: Current index of for-loop
        :param total: Total size of for-loop
        :param percentage_step: Percentage step with which the bar should be updated
        :param logger:

            Logger to write to, if string 'print' is given, the print statement is
            used. Use None if you don't want to print or log the progressbar statement.

        :param log_level: Log level with which to log.
        :param reprint:

            If no new line should be plotted but carriage return (works only for printing)

        :param time: If the remaining time should be calculated and displayed
        :param length: Length of the bar in `=` signs.
        :param fmt_string:

            A string which contains exactly one `%s` in order to incorporate the progressbar.
            If such a string is given, ``fmt_string % progressbar`` is printed/logged.

        :param reset:

            If the progressbar should be restarted. If progressbar is called with a lower
            index than the one before, the progressbar is automatically restarted.

        :return:

            The progressbar string or None if the string has not been updated.


        """
        reset = (reset or
                 index <= self._current_index or
                 total != self._total)
        if reset:
            self._reset(index, total, percentage_step, length)

        statement = None
        indexp1 = index + 1.0
        next_interval = int(indexp1 / self._norm_factor)
        ending = index >= self._total_minus_one

        if next_interval > self._current_interval or ending or reset:
            if time:
                remaining_str = self._get_remaining(index)
            else:
                remaining_str = ''

            if ending:
                statement = '[' + '=' * self._length +']100.0%'
            else:
                bars = int((indexp1 / self._total) * self._length)
                spaces = self._length - bars
                percentage = indexp1 / self._total * 100.0
                if reset:
                    statement = ('[' + '=' * bars +
                                 ' ' * spaces + ']' + ' %4.1f' % percentage + '%')
                else:
                    statement = ('[' + '=' * bars +
                                 ' ' * spaces + ']' + ' %4.1f' % percentage + '%' +
                                 remaining_str)

            if fmt_string:
                statement = fmt_string % statement
            if logger == 'print':
                if reprint:
                    print('\r' + statement, end='', flush=True)
                else:
                    print(statement)
            elif logger is not None:
                if isinstance(logger, str):
                    logger = logging.getLogger(logger)
                logger.log(msg=statement, level=log_level)

        self._current_interval = next_interval
        self._current_index = index

        return statement


_progressbar = _Progressbar()


def progressbar(index, total, percentage_step=10, logger='print', log_level=logging.INFO,
                 reprint=True, time=True, length=20, fmt_string=None, reset=False):
    """Plots a progress bar to the given `logger` for large for loops.

    To be used inside a for-loop at the end of the loop:

    .. code-block:: python

        for irun in range(42):
            my_costly_job() # Your expensive function
            progressbar(index=irun, total=42, reprint=True) # shows a growing progressbar


    There is no initialisation of the progressbar necessary before the for-loop.
    The progressbar will be reset automatically if used in another for-loop.

    :param index: Current index of for-loop
    :param total: Total size of for-loop
    :param percentage_step: Steps with which the bar should be plotted
    :param logger:

        Logger to write to - with level INFO. If string 'print' is given, the print statement is
        used. Use ``None`` if you don't want to print or log the progressbar statement.

    :param log_level: Log level with which to log.
    :param reprint:

        If no new line should be plotted but carriage return (works only for printing)

    :param time: If the remaining time should be estimated and displayed
    :param length: Length of the bar in `=` signs.
    :param fmt_string:

        A string which contains exactly one `%s` in order to incorporate the progressbar.
        If such a string is given, ``fmt_string % progressbar`` is printed/logged.

    :param reset:

        If the progressbar should be restarted. If progressbar is called with a lower
        index than the one before, the progressbar is automatically restarted.

    :return:

        The progressbar string or `None` if the string has not been updated.

    """
    return _progressbar(index=index, total=total, percentage_step=percentage_step,
                        logger=logger, log_level=log_level, reprint=reprint,
                        time=time, length=length, fmt_string=fmt_string, reset=reset)


def clean_up_directory(directory, keep_last=25):
    """ Removes files in `directory`

    Sorts files lexicographically and removes all except
    the `keep_last` ones.

    """
    filenames = os.listdir(directory)
    filenames = [os.path.join(directory, x) for x in filenames]
    filenames = sorted([x for x in filenames if os.path.isfile(x)])
    nfiles = len(filenames)
    if nfiles > keep_last:
        until = nfiles - keep_last
        logger.info('Founc {} files, will delete {} and keep '
                    '{}'.format(nfiles, until, keep_last))
        filenames = filenames[:until]
        for kdx, filename in enumerate(filenames):
            logger.info('Removing file {} ({}/{})'.format(filename,
                                                          kdx + 1,
                                                          keep_last))
            os.remove(filename)
    else:
        logger.info('Found only {} files in `{}`, less than the number to keep '
                    '({})'.format(nfiles, directory, keep_last))


def configure_logging(directory, current_datetime, bot_account='trufflepig'):
    """ Configures logging to stdout and file

    Parameters
    ----------
    directory: str
    current_datetime: datetime
    bot_account: str

    """
    if not os.path.isdir(directory):
        os.makedirs(directory)

    filename = '{bot_account}_{time}.txt'.format(bot_account=bot_account,
                                                 time=current_datetime.strftime('%Y-%m-%d'))
    filename = os.path.join(directory, filename)

    format=('%(asctime)s %(processName)s:%(name)s:'
                  '%(funcName)s:%(lineno)s:%(levelname)s: %(message)s')
    handlers = [logging.StreamHandler(), logging.FileHandler(filename)]
    logging.basicConfig(level=logging.INFO, format=format,
                        handlers=handlers)


def error_retry(f, retries=3, sleep_time=11, errors=(RPCError,),
                not_log_errors=(PostDoesNotExist,)):
    """Explicit decorator for Error retries"""
    def wrapped(*args, **kwargs):
        for retry in range(retries + 1):
            try:
                result =  f(*args, **kwargs)
                if retry > 0:
                    logger.warning('Needed retry {} out of {} for '
                                 '{}!'.format(retry, retries, f))
                return result
            except errors as exc:
                if retry + 1 >= retries:
                    if not isinstance(exc, not_log_errors):
                        logger.exception('Failed all {} retries for '
                                         '{}!'.format(retries, f))
                    raise
                time.sleep(sleep_time)
    return wrapped


def none_retry(f, retries=16, sleep_time=2):
    """Explicit decorator for not None retries"""
    def wrapped(*args, **kwargs):
        for retry in range(retries + 1):
            result =  f(*args, **kwargs)

            if result is not None:
                if retry > 0:
                    logger.warning('Needed retry {} out of {} for '
                                 '{}!'.format(retry, retries, f))
                return result

            if retry + 1 >= retries:
                logger.error('Failed all {} retries for '
                                 '{}! Return None!'.format(retries, f))
            time.sleep(sleep_time)
        return None
    return wrapped


def none_error_retry(f, retries=7, sleep_time=11,
                     errors=(RPCError,),
                     not_log_errors=(PostDoesNotExist,)):
    """Combines Error and None retry"""
    return none_retry(error_retry(f,
                                  retries=retries,
                                  sleep_time=sleep_time,
                                  errors=errors,
                                  not_log_errors=not_log_errors),
                      retries=retries,
                      sleep_time=sleep_time)
