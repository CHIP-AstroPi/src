"""
CHIP: CHIP Human Impact Prevision

Test main without camera input

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
import numpy as np
import cv2 as cv
import logzero
import logging
import ephem
import time
import math
# ----------------------------------------


class Config():
    """Config namespace"""

    # runtime schedule
    # see the `runtime_schedule` function for more info
    rs_step = 1  # [seconds]
    rs_tot = 5  # [seconds]

    # fs
    fs_here = Path(__file__).parent.resolve()
    logfile_data = 'data.csv'
    logfile_log = 'runtime.log'

    # loggers
    log_format_date = '%Y-%m-%d %H:%M:%S'
    log_data_format = '%(asctime)s, %(message)s'
    log_log_format = '(%(asctime)s.%(msecs)03d)  [%(levelname)s] %(message)s'
    log_stderr_level = logging.DEBUG  # TODO in production, switch to `logging.INFO`
    log_file_level = logging.INFO

    # iss position
    iss_name = 'ISS (ZARYA)'
    iss_l1 = '1 25544U 98067A   20016.35580316  .00000752  00000-0  21465-4 0  9996'
    iss_l2 = '2 25544  51.6452  24.6741 0004961 136.6310 355.9024 15.49566400208322'


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


# ----------------------------------------
# ISS SETUP

iss = ephem.readtle(Config.iss_name, Config.iss_l1, Config.iss_l2)


def get_iss_data():
    """Get data about the ISS

    - DMS
    - latitude [8th decimal]
    - longitude [8th decimal]
    - elevation
    - eclipsed
    """

    # compute the last position
    iss.compute()

    # round the latitude at eighth decimal
    lat_decimal = round(math.degrees(iss.sublat), 8)

    # split the latitude in degrees, minutes and seconds
    lat = str(iss.sublat)
    lat_d, lat_m, lat_s = map(float, lat.split(':'))

    # evaluate the correct equator
    equator = 'N' if lat_d > 0 else 'S'
    lat_d = abs(lat_d)

    # round the longitude at eighth decimal
    long_decimal = round(math.degrees(iss.sublong), 8)

    # split the longitude in degrees, minutes and seconds
    long = str(iss.sublong)
    long_d, long_m, long_s = map(float, long.split(':'))

    # evaluate the correct quadrant
    quadrant = 'E' if long_d > 0 else 'W'
    long_d = abs(long_d)

    # format DMS
    dms = f"{lat_d}° {lat_m}' {lat_s}'' {equator} {long_d}° {long_m}' {long_s}'' {quadrant}"

    return dms, lat_decimal, long_decimal, iss.elevation, iss.eclipsed

# /ISS SETUP
# ----------------------------------------


# ----------------------------------------
# FUNCTIONS

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


def cut_image(img: np.ndarray, target_height_perc=65, target_width_perc=65) -> np.ndarray:
    """Cut an image in a rectangle of size defined by parameters, to remove black borders"""

    # retrieve image shape
    old_width, old_height, _ = img.shape
    half_width = old_width // 2
    half_height = old_height // 2

    # evaluate the needed padding on the x an y axis
    padding_x = (half_width * target_height_perc) // 100
    padding_y = (half_height * target_height_perc) // 100

    # cut the image
    return img[
        half_width - padding_x:half_width + padding_x,
        half_height - padding_y:half_height+padding_y
    ]


def is_day(img: np.ndarray, center_size_perc=30, threshold=70) -> bool:
    """Check if the photo represent a daytime scenary

    If the average brightness of the center of the image is higher than a `threshold`, 
    it is assumed to be taken at daytime.
    """

    # retrieve image shape
    height, width, _ = img.shape

    # calculate center coordinate
    center_x = width // 2
    center_y = height // 2

    # evaluates the dislocation of the more external sample points along the x and y axis
    dislocation_x = (width * center_size_perc) // 100
    dislocation_y = (height * center_size_perc) // 100

    # finds the position of the more external sample points
    border_left = center_x - dislocation_x
    border_right = center_x + dislocation_x
    border_bottom = center_y - dislocation_y
    border_top = center_y + dislocation_y

    # add all the sample points to a list
    sample_points = []
    for x in range(border_left, border_right):
        for y in range(border_bottom, border_top):
            sample_points.append(img[y, x])

    # convert the sample points list in a numpy array
    numpy_sample_points = np.array(sample_points)

    # calculate the average value of the sample points
    average_bgr = np.average(numpy_sample_points, axis=0)

    # convert the bgr values to python ints
    average_bgr = average_bgr.astype(int)

    # convert the average value to opencv compliant format
    average_bgr = np.uint8([[average_bgr]])

    # convert the average color from bgr to grayscale
    average_grayscale = cv.cvtColor(average_bgr, cv.COLOR_BGR2GRAY)

    # retrieve the grayscale value
    average_grayscale = np.squeeze(average_grayscale)

    # compare the average grayscale value with the threshold
    return average_grayscale >= threshold


def cloud_percent(img: np.ndarray) -> float:
    """Calculate the percentage of clouds in an image"""

    # convert the image to hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # extract the saturation of the image
    saturation = hsv[:, :, 1]

    # apply otsu threshold to the saturation
    threshold_type = cv.THRESH_BINARY+cv.THRESH_OTSU
    _, threshold = cv.threshold(saturation, 0, 255, threshold_type)

    # find the contours of the threshold
    contours = cv.findContours(
        threshold,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE
    )[-2]

    # create an `img` clone
    img_clone = np.copy(img)

    # draw the contours on the img clone
    cv.drawContours(img_clone, contours, -1, (0, 255, 0), 2)

    # TODO what does it do?
    x, y, w, h = cv.boundingRect(contours[0])
    threshold[y:y+h, x:x+w] = 255 - threshold[y:y+h, x:x+w]
    contours = cv.findContours(
        threshold,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE
    )[-2]

    # fill contours
    for contour in contours:
        if cv.contourArea(contour) > 4:
            cv.drawContours(
                img_clone,
                [contour],
                -1,
                (255, 0, 0),
                thickness=cv.FILLED
            )

    # calculate the percentage of clouds
    pixels_count = img_clone.shape[0] * img_clone.shape[1]
    red_pixels = np.all(img_clone == [255, 0, 0], axis=2)
    red_pixels_count = np.count_nonzero(red_pixels)
    return 100 * red_pixels_count / pixels_count


# /FUNCTIONS
# ----------------------------------------


# ----------------------------------------
# MAIN


def main():
    """
    - take image            x
    - [ cut image ]         x
    > is day                x
    > cloud
    - [ save image ]
    - iss -> data
    - rivers -> data
    - forest -> data
    - coasts -> data
    """

    # dummy image reading
    raw_image = cv.imread('./imgs/Image4.jpg', flags=cv.IMREAD_COLOR)

    cutted_image = cut_image(raw_image)
    if not is_day(cutted_image):
        print('not day')
        return


# /MAIN
# ----------------------------------------
# ----------------------------------------
# RUNTIME ENTRY POINT
# runtime_scheduler(main)
main()

# /RUNTIME ENTRY POINT
# ----------------------------------------