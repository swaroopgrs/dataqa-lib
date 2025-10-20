import logging
import sys


class CustomFormatterLevel(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ColorFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        "asctime": "\033[95m",  # Purple
        "name": "\033[94m",  # Blue
        "levelname": {
            "DEBUG": "\033[90m",  # Grey
            "INFO": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "CRITICAL": "\033[91m",  # Red (same as ERROR)
        },
        "message": "\033[93m",  # Yellow
        "reset": "\033[0m",  # Reset to default
    }

    def format(self, record):
        # Format the message with colors
        formatted_message = super().format(record)

        # Apply colors to specific parts of the formatted message
        formatted_message = formatted_message.replace(
            record.asctime,
            f"{self.COLORS['asctime']}{record.asctime}{self.COLORS['reset']}",
        )
        formatted_message = formatted_message.replace(
            record.name,
            f"{self.COLORS['name']}{record.name}{self.COLORS['reset']}",
        )
        formatted_message = formatted_message.replace(
            record.levelname,
            f"{self.COLORS['levelname'][record.levelname]}{record.levelname}{self.COLORS['reset']}",
        )
        formatted_message = formatted_message.replace(
            record.getMessage(),
            f"{self.COLORS['message']}{record.getMessage()}{self.COLORS['reset']}",
        )

        return formatted_message


def get_logger(
    name: str, file_path: str, level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a stream handler to output logs to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")

    # Set the custom formatter for the handler
    formatter = ColorFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger_level(
    name: str, file_path: str, level: int = logging.INFO
) -> logging.Logger:
    """
    Creates and returns a logger that outputs to both stdout and a local file.

    :param name: The name of the logger.
    :param file_path: The path to the log file.
    :param level: The logging level (default is DEBUG).
    :return: Configured logger object.
    """
    # Create a logger with the specified name
    logger = logging.getLogger(name)

    # Set the logging level
    logger.setLevel(level)

    # Create a stream handler to output logs to stdout
    stream_handler = logging.StreamHandler(sys.stdout)

    # Create a file handler to output logs to a file
    file_handler = logging.FileHandler(file_path)

    # Set the format for the handlers
    formatter = CustomFormatterLevel()
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
