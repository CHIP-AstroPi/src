"""
CHIP: CHIP Human Impact Prevision

Useful links:
- GitHub: https://github.com/CHIP-AstroPi

By the CHIP Team:
- Agbonson Fabrizio
- Bruno Luca
- Ferrando Filippo
- Jomini Pietro
- Nardi Simone

TODO project description
"""


# IMPORTS
# for autopep8 disable rule E402
# ----------------------------------------
from pathlib import Path
import logzero
import logging
# ----------------------------------------


class Config():
    """Config namespace"""

    # fs
    here = Path(__file__).parent.resolve()
    logfile_data = 'data.csv'
    logfile_log = 'runtime.log'

    # loggers
    log_format_date = '%Y-%m-%d %H:%M:%S'
    log_data_format = '%(asctime)s, %(message)s'
    log_log_format = '[%(levelname)s] %(asctime)-15s - %(message)s'


# LOGGERS SETUP
# ----------------------------------------

# runtime logger
# print to `stderr` and `Config.logfile_log`
logger: logging.Logger = logzero.setup_logger(
    name='info_logger',
    logfile=Config.here / Config.logfile_log,
    formatter=logging.Formatter(
        Config.log_log_format,
        Config.log_format_date
    )
)


# data logger
# print to `Config.logfile_data`
# shouldn't be used directly, but through the `log_data` function
_data_logger: logging.Logger = logzero.setup_logger(
    name='data_logger',
    logfile=Config.here / Config.logfile_data,
    formatter=logging.Formatter(
        Config.log_data_format,
        Config.log_format_date
    ),
    disableStderrLogger=True
)


def log_data(*args: any) -> None:
    """Data logging utility

    Utilize the `_data_logger` logger to log data to csv.
    Each `arg` becomes a csv cell.
    """

    _data_logger.info(', '.join(map(str, args)))


# ----------------------------------------
