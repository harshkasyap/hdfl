import inspect
import json
import logging
import os.path

logger = logging.getLogger(__name__)


def init(arg='info', filename=None):
    format = "%(asctime)s - %(message)s [%(processName)s : %(threadName)s (%(levelname)s)]"
    if filename is not None:
        logdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../out/logs'
        filename = logdir + '/' + filename + ".log"
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        if not os.path.isfile(filename):
            open(filename, "w")

        if arg == 'info' or arg == 'INFO':
            logging.basicConfig(filename=filename, format=format, level=logging.INFO)
        elif arg == 'debug' or arg == 'DEBUG':
            logging.basicConfig(filename=filename, format=format, level=logging.DEBUG)
    else:
        if arg == 'info' or arg == 'INFO':
            logging.basicConfig(format=format, level=logging.INFO)
        elif arg == 'debug' or arg == 'DEBUG':
            logging.basicConfig(format=format, level=logging.DEBUG)


def logmodule(trace):
    filename, method, line = str(trace[1][1]), str(trace[1][3]), str(trace[1][2])
    return filename + "::" + method + "(l:" + line + ") : "


def info(_log):
    logger.info(logmodule(inspect.stack()) + _log)


def debug(_log):
    logger.debug(logmodule(inspect.stack()) + _log)


def warning(_log):
    logger.warning(logmodule(inspect.stack()) + _log)


def error(_log):
    logger.error(logmodule(inspect.stack()) + _log)


def critical(_log):
    logger.critical(logmodule(inspect.stack()) + _log)


def modeldebug(nn_model, info):
    logger.debug(logmodule(inspect.stack()) + info + " ")
    for params in nn_model.parameters():
        logger.debug(str(params))


def jsoninfo(_json, info):
    logger.info(logmodule(inspect.stack()) + info + " " + json.dumps(_json, indent=4, sort_keys=True))


def jsondebug(_json, info):
    logger.debug(logmodule(inspect.stack()) + info + " " + json.dumps(_json, indent=4, sort_keys=True))
