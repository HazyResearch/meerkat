import os


class stderr_suppress(object):
    """A context manager for doing a "deep suppression" of stdout and stderr in
    Python.

    This is necessary when reading in a corrupted video, or else stderr
    will emit 10000s of errors via ffmpeg. Great for decoding IRL, not
    great for loading 100s of corrupted videos.
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        # Save stderr (2) file descriptor.
        self.save_fd = os.dup(2)

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fd, 2)
        # Close all file descriptors
        os.close(self.null_fd)
        os.close(self.save_fd)
