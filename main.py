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
import numpy as np
import cv2 as cv
import logzero
import logging
import ephem
import time
import math

# for test purpose, we don't always works directly on the respberry.
# hence, we try to import the PiCamera module, and if it fails to
# we simulate a dummy camera that uses images on disk as frames
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    NOCAM = False
except ImportError:
    NOCAM = True
# ----------------------------------------


class Config():
    """Config namespace"""

    # runtime schedule
    # see the `runtime_schedule` function for more info
    rs_step = 0.1 * 60  # [seconds]
    rs_tot = 0.5 * 60  # [seconds]

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
    log_stderr_level = logging.DEBUG  # TODO in production, switch to `logging.INFO`
    log_file_level = logging.INFO

    # iss position
    iss_name = 'ISS (ZARYA)'
    iss_l1 = '1 25544U 98067A   21026.29455175  .00001781  00000-0  40614-4 0  9995'
    iss_l2 = '2 25544  51.6464 324.5349 0002308 292.1434 166.6739 15.48892138266603'


# ----------------------------------------
# CAMERA SETUP


class Camera:
    """Camera wrapper

    If the PiCamera module is loaded, works as a wrapper. If it's 
    not loaded read images from the disk and exposes them as frames.
    """

    def __init__(self, image_first_id=0,  is_picam_loaded=not NOCAM, dummy_images_dir=Config.cam_dummy_folder, dummy_images_formats=Config.cam_dummy_formats):

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
            assert len(self._dummy_images) > 0, Exception('No images found')

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

    def __enter__(self) -> np.ndarray:
        """Syntactic sugar for `Camera.capture`"""
        return self.capture()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Syntactic sugar for `Camera.camera_update`"""
        self.camera_update()


camera = Camera()

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
    # img_id, raw_image = camera_capture()
    # img_path = image_path()
    # iss_data = get_iss_data()

    # cutted_image = cut_image(raw_image)
    # if not is_day(cutted_image):
    #     logger.info('Not daytime, closing')

    # cv.imwrite(img_path, raw_image)
    # log_data(img_id, img_path, cloud_percent(cutted_image), *iss_data)

    # camera_reset()

    with camera as raw_image:

        # remove borders from image
        cutted_image = cut_image(raw_image)

        cv.imwrite(camera.build_image_path(), cutted_image)


# /MAIN
# ----------------------------------------


# ----------------------------------------
# RUNTIME ENTRY POINT

runtime_scheduler(main)

# /RUNTIME ENTRY POINT
# ----------------------------------------
