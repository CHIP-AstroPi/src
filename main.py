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
TODO check english (!)
"""


# IMPORTS
# for autopep8 disable rule E402
# ----------------------------------------
from pathlib import Path
import cv2 as cv
import picamera
import logzero
import logging
import time
# ----------------------------------------


class Config():
    """Config namespace"""

    # runtime schedule
    # see the `runtime_schedule` function for more info
    rs_step = 1  # [seconds]
    rs_tot = 10  # [seconds]

    # fs
    fs_here = Path(__file__).parent.resolve()
    logfile_data = 'data.csv'
    logfile_log = 'runtime.log'

    # camera
    cam_resolution = (2592, 1944)
    cam_framerate = 15
    cam_format = "bgr"

    # loggers
    log_format_date = '%Y-%m-%d %H:%M:%S'
    log_data_format = '%(asctime)s, %(message)s'
    log_log_format = '(%(asctime)s.%(msecs)03d)  [%(levelname)s] %(message)s'
    log_stderr_level = logging.DEBUG  # TODO in production, switch to `logging.INFO`
    log_file_level = logging.INFO


# ----------------------------------------
# CAMERA SETUP

camera = picamera.PICamera()
camera_raw = picamera.PiRGBArray(camera, size=Config.cam_resolution)

camera.resolution = Config.cam_resolution
camera.framerate = Config.cam_framerate


def camera_capture():
    """camera raw capture utility"""
    camera.capture(camera_raw, format=Config.cam_format)
    return camera_raw.array

# /CAMERA SETUP
# ----------------------------------------


# ----------------------------------------
# LOGGERS SETUP

# runtime logger
# print to `stderr` and `Config.logfile_log`
logger: logging.Logger = logzero.setup_logger(
    name='info_logger',
    logfile=Config.fs_here / Config.logfile_log,
    formatter=logging.Formatter(
        Config.log_log_format,
        Config.log_format_date
    ),
    level=Config.log_stderr_level,
    fileLoglevel=Config.log_file_level
)

# data logger
# print to `Config.logfile_data`
# shouldn't be used directly, but through the `log_data` function
_data_logger: logging.Logger = logzero.setup_logger(
    name='data_logger',
    logfile=Config.fs_here / Config.logfile_data,
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

# /LOGGER SETUP
# ----------------------------------------


def runtime_scheduler(task: callable) -> None:
    """Handle runtime task scheduling

    Given the total runtime `Config.rs_tot` and the minimum step length `Config.rs_step`,
    it will try to fit the `task` in each step.

    If the `task` execution takes less than `Config.rs_step` it will wait until the step is complete.

    If the `task` execution exceeds `Config.rs_step` it will merge the current and the next step.
    """

    logger.info('RS:start')
    start = time.time()             # starting timestamp
    end = start + Config.rs_tot     # expected end timestamp

    # we keeps time of the time the task takes to run,
    # so that we can avoid to run the last task if it would exceed `Config.rs_tot`
    max_task_exec_time = 0

    # main loop, in which we advance between steps
    # loops until we reach the moment in which the last step
    # or task would exceed `Config.rs_tot`
    while time.time() < end - max_task_exec_time:

        # task starting timestamp
        task_start = time.time()
        logger.debug(f'RS:task:start @={round(time.time() - start, 4)}')

        # task execution
        task()

        # execution time calculation
        task_exec_time = time.time() - task_start
        logger.debug(f'RS:task:end Δ={round(task_exec_time, 4)}')

        # maximum task execution time update
        if max_task_exec_time < task_exec_time:
            max_task_exec_time = task_exec_time

        # dynamic calculation of the padding time
        # to match the next step starting point.
        padding_time = Config.rs_step - task_exec_time

        # "merge" steps if they overlap
        while padding_time < 0:
            padding_time += Config.rs_step
        logger.debug(f'RS:padding={round(padding_time, 4)}')

        # waiting for the next step
        time.sleep(padding_time)

    logger.info(f'RS:end elapsed={round(time.time() - start, 3)}')


# ----------------------------------------
# MAIN


imid = 0


def main():
    global imid
    image = camera_capture()
    log_data(imid)
    cv.imwrite(f'{Config.fs_here}/imgs/{imid}.jpg', image)
    imid += 1


# /MAIN
# ----------------------------------------


# ----------------------------------------
# RUNTIME ENTRY POINT

runtime_scheduler(main)

# /RUNTIME ENTRY POINT
# ----------------------------------------
