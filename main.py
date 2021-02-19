"""
CHIP: CHIP Human Impact Prevision

Useful links:
- GitHub: https://github.com/CHIP-AstroPi/src

By the CHIP Team:
- Agbonson Fabrizio
- Bruno Luca
- Ferrando Filippo
- Jomini Pietro
- Nardi Simone

This project aims to compare modern and old data and, by analyzing them, try to predict near-future conditions.
In particular, we analyze the way coastlines changes over time and, from this, extrapolate information about the environment.

This script, in particular, performs the data collection tasks required.
It runs for a little less than 3 hours and repeatedly collects images from the live NoIr camera, with the blue filter,
and collect data about the position of the ISS at the moment of the capture.
It than analyze the image applying the following algorithms:
- it checks if the image is captured at daytime, and stops if it's not
- it checks the clouds coverage in the image
- it detects coastlines
- it detects the fractal dimension of the detected coastlines

Finally, it dumps the collected data into a log file and passes to the next image.
"""


# ----------------------------------------
# IMPORTS

from typing import Tuple, List, Union
from pathlib import Path
import numpy as np
import cv2 as cv
import logzero
import logging
import ephem
import time
import math
import json

# while developing we don't always works directly on the respberry.
# hence, we try to import the PiCamera module, and if it fails to
# we simulate a dummy camera that uses images on disk as frames
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    NOCAM = False
except ImportError:
    NOCAM = True

# /IMPORTS
# ----------------------------------------


# ----------------------------------------
# ISS TLE data

name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   21039.89161926 -.00000043  00000-0  73771-5 0  9995"
line2 = "2 25544  51.6440 257.2891 0002486 351.4127  16.5936 15.48938694268710"

# /ISS TLE data
# ----------------------------------------


# ----------------------------------------
# CONFIG

class Config():
    """Config namespace

    All configuration data should be represented here.
    """

    # runtime schedule
    # see the `runtime_schedule` function for more info
    rs_step = 20  # [seconds]
    rs_tot = 175 * 60  # [seconds]

    # fs
    fs_here = Path(__file__).parent.resolve()
    logfile_data = 'data.csv'
    logfile_log = 'runtime.log'

    # camera
    cam_resolution = (2592, 1952)
    cam_framerate = 15
    cam_format = "bgr"

    # dummy camera
    cam_dummy_folder = 'dummy_images'
    cam_dummy_formats = ['.jpg']

    # loggers
    log_format_date = '%Y-%m-%d %H:%M:%S'
    log_data_format = '%(asctime)s, %(message)s'
    log_log_format = '(%(asctime)s.%(msecs)03d)  [%(levelname)s] %(message)s'
    log_stderr_level = logging.INFO
    log_file_level = logging.INFO

    # iss position
    iss_name = name
    iss_l1 = line1
    iss_l2 = line2

    # cut image
    cut_image_height = 65
    cut_image_width = 65

    # is day
    is_day_center_size = 30
    is_day_threshold = 70

    # cloud detection
    cloud_lower = [195] * 3
    cloud_upper = [255] * 3
    cloud_saturation_scale = 1.5

    # ghost islands detection
    ghost_island_thresh_thresh = 134
    ghost_island_thresh_maxval = 255
    ghost_island_thresh_type = 1
    ghost_island_max_white_percentage = 60

    # coasts detection
    coast_threshold_max_value = 100
    coast_threshold_blocksize = 1101
    coast_threshold_c = -6
    coast_contours_min_length = 100
    coast_contours_ratio_lowerbound = 0
    coast_contours_ratio_upperbound = 1

# /CONFIG
# ----------------------------------------


# ----------------------------------------
# CAMERA SETUP

class Camera:
    """Camera wrapper

    If the PiCamera module is loaded, works as a wrapper. If it's
    not loaded read images from the disk and exposes them as frames.
    """

    def __init__(self, image_first_id=0, is_picam_loaded=not NOCAM, dummy_images_dir=Config.cam_dummy_folder, dummy_images_formats=Config.cam_dummy_formats):

        # image incremental id
        self.image_id = image_first_id

        # checks if the picamera module is loaded
        self._is_picam_loaded = is_picam_loaded

        # if the picamera module is loaded, init it's camera
        if self._is_picam_loaded:

            # picamera creation
            self._picam = PiCamera()
            self._picam_raw = PiRGBArray(
                self._picam, size=Config.cam_resolution)

            # picamera configuration
            self._picam.resolution = Config.cam_resolution
            self._picam.framerate = Config.cam_framerate

        # if the picamera module is not loaded, we looks for images
        # to feed as fake frames
        else:

            # finds dummy images
            dummy_dir = Config.fs_here / dummy_images_dir
            dummy_images = filter(
                lambda file: file.is_file() and file.suffix in dummy_images_formats,
                dummy_dir.iterdir()
            )

            self._dummy_images = sorted(dummy_images)

            # check if there is no images, and throws if it's the case
            if not len(self._dummy_images) > 0:
                raise ValueError('No images found')

    def capture(self) -> np.ndarray:
        """Capture raw frames from camera or disk"""

        # if the picamera module is loaded
        if self._is_picam_loaded:
            self._picam.capture(self._picam_raw, format=Config.cam_format)
            return self._picam_raw.array

        # otherwise loads image from disk
        # cicle over images if index (id) exceeds the number of images
        image_index = self.image_id % len(self._dummy_images)
        image_path = self._dummy_images[image_index]
        return cv.imread(str(image_path))

    def build_image_path(self, image_format='.jpg') -> str:
        """Builds image path based on `Cofig.fs_here`"""
        return f'{Config.fs_here}/img_{self.image_id}{image_format}'

    def camera_update(self) -> None:
        """Update camera id and reset picamera raw input, if loaded"""

        # increment the image id
        self.image_id += 1

        # truncate picamera raw stream
        if self._is_picam_loaded:
            self._picam_raw.truncate(0)

    def __enter__(self) -> Union[np.ndarray, bool]:
        """Syntactic sugar for `Camera.capture`"""

        try:
            # return a capture
            return self.capture()
        except _:
            # if the capture fails, return False
            return False

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Syntactic sugar for `Camera.camera_update`"""
        self.camera_update()


# create a global camera object
camera = Camera()

# /CAMERA SETUP
# ----------------------------------------


# ----------------------------------------
# LOGGERS SETUP

# runtime logger
# print to `stderr` and `Config.logfile_log`
logger = logzero.setup_logger(
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
_data_logger = logzero.setup_logger(
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


def json_dump(image_id: int, name: str, data: dict, indent: Union[None, int] = None) -> None:
    """Json data logging utility

    Dumps the `data` object to a json file named as `image_id`_`name`.json
    """

    # build the filename
    filename = Config.fs_here / f'{image_id}_{name}.json'

    # write `data` to the json file
    with filename.open('w') as out:
        dump = json.dumps(data, indent=indent)
        out.write(dump)

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

def cut_image(img: np.ndarray, target_height_perc=Config.cut_image_height, target_width_perc=Config.cut_image_width) -> np.ndarray:
    """Cut an image in a rectangle of size `target_height_perc`% x `target_width_perc`%"""

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


def is_day(img: np.ndarray, center_size_perc=Config.is_day_center_size, threshold=Config.is_day_threshold) -> bool:
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


def white_percentage(img: np.ndarray) -> float:
    """Calculate the percentage of white pixels in an image"""

    # extract the shape of the image
    height, width = img.shape[:2]

    # count the white pixels
    white_pixels_count = np.sum(img == 255)

    # calculate the percentage
    return 100 * white_pixels_count / (height * width)


def scale_saturation(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale saturation in an image"""

    # convert the image to float32 hsv
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype('float32')

    # scale the saturation by `scale`
    (h, s, v) = cv.split(hsv)
    s = np.clip(s * scale, 0, 255)
    hsv = cv.merge([h, s, v])

    # convert the image back to uint8 bgr
    return cv.cvtColor(hsv.astype('uint8'), cv.COLOR_HSV2BGR)


def detect_cloud(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect clouds in an image"""

    # scale the saturation by `Config.cloud_saturation_scale`
    img = scale_saturation(img, Config.cloud_saturation_scale)

    # mask the image in the range defined by `Config.cloud_lower` and `Config.cloud_upper`
    lower_white = np.array(Config.cloud_lower, dtype=np.uint8)
    upper_white = np.array(Config.cloud_upper, dtype=np.uint8)
    mask = cv.inRange(img, lower_white, upper_white)

    return mask, white_percentage(mask)


def is_island_ghost(img: np.ndarray, cont: np.ndarray, max_white_percentage=Config.ghost_island_max_white_percentage) -> bool:
    """Check if a contour represent a ghost island"""

    # extract the bounding rectangle around the contour
    x, y, w, h = cv.boundingRect(cont)
    n = img[y:y+h, x:x+w]

    # convert the rectangle to grayscale
    n = cv.cvtColor(n, cv.COLOR_BGR2GRAY)

    # apply a threshold to the rectangle
    _, n = cv.threshold(
        n,
        Config.ghost_island_thresh_thresh,
        Config.ghost_island_thresh_maxval,
        Config.ghost_island_thresh_type
    )

    # calculate the percentage of white pixles in the rectangle
    return white_percentage(n) >= max_white_percentage


def find_coasts(img_gray: np.ndarray, img_color: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Finds coastlines"""

    # apply an adaptive threshold to the image
    th = cv.adaptiveThreshold(
        img_gray,
        Config.coast_threshold_max_value,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        Config.coast_threshold_blocksize,
        Config.coast_threshold_c
    )

    # find the contours in the thresholded image
    contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # filter the contours by their length
    contours = [
        contour for contour in contours
        if cv.arcLength(contour, False) > Config.coast_contours_min_length
    ]

    # find contours that should be ignored
    contours_to_ignore = []
    for index, contour in enumerate(contours):

        # calculate the ration between the length and the area of the contour
        length = cv.arcLength(contour, True)
        area = cv.contourArea(contour)
        ratio = abs(length / area - 1) if area > 0 else None

        # check if ration is in defined range
        is_in_range = ratio and Config.coast_contours_ratio_lowerbound <= ratio <= Config.coast_contours_ratio_upperbound

        if not is_in_range or is_island_ghost(img_color, contour):
            contours_to_ignore.append(index)

    return contours, contours_to_ignore


def fractal_dimension(contours: List[np.ndarray], shape: Tuple[int]) -> float:
    """Compute fractal dimension of coastlines

    Given an image of a coastline and a list of contours,
    this function comput the fractal value of the coastline
    using a boxcounting algorithm.

    """

    w, h = shape[:2]  # get width and height from the image
    # create a black background image with same dimensions
    img = np.zeros((w, h, 1), np.uint8)

    # draw contours on the image
    cv.drawContours(img, contours, -1, (255, 255, 255), 1)

    while w % 5 > 0:  # adjust dimensions so it get perfect 5x5 pixel squares
        w = w-1
    while h % 5 > 0:
        h = h-1

    higher_scale = 0
    for r in range(int(h/5)):
        for c in range(int(w/5)):
            if cv.countNonZero(img[r*5:r*5+5, c*5:c*5+5]) > 0:
                higher_scale = higher_scale + 1

    # INCREASE RESOLUTION OF THE GRID
    while w % 2 > 0:  # adjust dimensions so it get perfect 2x2 pixel squares
        w = w-1
    while h % 2 > 0:
        h = h-1

    lower_scale = 0

    for r in range(int(h/2)):
        for c in range(int(w/2)):
            if cv.countNonZero(img[r*2:r*2+2, c*2:c*2+2]) > 0:
                lower_scale = lower_scale + 1

    try:
        fractal = math.log(lower_scale/higher_scale, 10) / \
            math.log(2, 10)  # fractal dimension
    except ZeroDivisionError:
        fractal = -1

    return fractal

# /FUNCTIONS
# ----------------------------------------


# ----------------------------------------
# MAIN

def runtime_scheduler(task: callable) -> None:
    """Handle runtime task scheduling.

    Given the total runtime `Config.rs_tot` and the minimum step length `Config.rs_step`,
    it will try to fit the `task` in each step.

    If the `task` execution takes less than `Config.rs_step` it will wait until the step is complete.

    If the `task` execution exceeds `Config.rs_step` it will merge the current and the next step waiting time.
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
        logger.info(f'RS:task:start at={round(time.time() - start, 4)}')

        # task execution
        task()

        # execution time calculation
        task_exec_time = time.time() - task_start

        # maximum task execution time update
        if max_task_exec_time < task_exec_time:
            max_task_exec_time = task_exec_time

        # dynamic calculation of the padding time
        # to match the next step starting point.
        padding_time = Config.rs_step - task_exec_time

        # "merge" steps if they overlap
        while padding_time < 0:
            padding_time += Config.rs_step

        # log execution data
        log_str = f'RS:task:end duration={round(task_exec_time, 4)} padding={round(padding_time, 4)}'
        logger.info(log_str)

        # waiting for the next step
        time.sleep(padding_time)

    logger.info(f'RS:end elapsed={round(time.time() - start, 3)}')


def main():
    with camera as raw_image:

        # check if the capture is successfull
        if raw_image is False:
            logger.error('Camera error')
            return

        # get iss data
        iss_data = get_iss_data()

        # image basic modding
        image = cut_image(raw_image)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # check if the image is taken at daytime
        if not is_day(image):
            logger.info('Not daytime, closing')
            return

        # detect clouds
        _, clouds = detect_cloud(image)

        # detect coastlines
        coasts, to_ignore = find_coasts(gray_image, image)
        
        # compute the fractal dimension of the detected coastlines
        fractal_ratio = fractal_dimension(coasts, image.shape)

        # dumps data to json
        json_dump(
            camera.image_id,
            'coasts',
            {
                'coasts': list(map(lambda c: c.tolist(), coasts)),
                'ignorable': to_ignore
            },
            indent=4
        )

        image_path = camera.build_image_path()
        cv.imwrite(image_path, image)
        log_data(camera.image_id, image_path, clouds, fractal_ratio, *iss_data)

# /MAIN
# ----------------------------------------


# ----------------------------------------
# RUNTIME ENTRY POINT

# run the main function thorugh the `runtime_scheduler` function
runtime_scheduler(main)

# /RUNTIME ENTRY POINT
# ----------------------------------------
