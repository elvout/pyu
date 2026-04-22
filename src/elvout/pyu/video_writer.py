import os
import subprocess
import threading
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image as PIL_Image


class VideoWriter:
    """
    Writes a video to disk from image frames.

    This class gives more fine-grained control over specific video encoding
    settings compared to iio/pyav.

    This class makes the following decisions for the user:
        - Backend: ffmpeg
        - Video encoding: x264, crf 18, preset medium
        - Video extension: .mp4
        - Chroma subsampling: yuv444p (doesn't require even dimensions)

    Videos are created by streaming images to an ffmpeg subprocess. The
    dimensions of each frame are extracted from the first streamed image and is
    enforced thereafter.

    Example:
    ```python
    writer = VideoWriter("video.mp4")
    for image in images:
        writer.add_frame(image)
    writer.close()
    ```
    """

    def __init__(self, filename: str | Path, fps: int | float) -> None:
        """Constructs a VideoWriter.

        Args:
            filename (str | Path): The output file path.
            fps (int): The output video frame rate.

        Raises:
            FileNotFoundError: ffmpeg cannot be found, probably because it is
                either missing or not in $PATH
            BrokenPipeError: The ffmpeg subprocess' stdin handle could not be
                opened, thus preventing images from being streamed.
        """
        filename = Path(filename)
        if filename.suffix != ".mp4":
            filename = filename.parent / f"{filename.stem}.mp4"
        os.makedirs(filename.parent, exist_ok=True)
        self._filename = filename

        # fmt: off
        ffmpeg_args = [
            "ffmpeg",
            "-loglevel", "warning",
            "-y",
            "-f", "image2pipe",
            "-probesize", "32M",
            "-r", str(fps),  # input fps needs to be specified or frames may be dropped
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-vf", "format=yuv444p",
            "-r", str(fps),
            str(filename)
        ]
        # fmt: on

        self._shape: tuple[int, ...] | None = None
        """The expected shape of image frames. Note that numpy-style dimension
        ordering is used (e.g., (1080, 1920)) instead of the width-height
        convention used for images (e.g., (1920, 1080))."""

        self._ffmpeg = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)
        if self._ffmpeg.stdin is None:
            raise BrokenPipeError("Subprocess' stdin could not be opened.")

        self._io_thread: threading.Thread | None = None
        """Thread to continue pipe I/O operations in the background."""
        self._io_exc: Exception | None = None

    def _write_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        try:
            assert self._ffmpeg.stdin is not None
            # PIL is slightly faster than OpenCV due to not needing to convert RGB to BGR
            PIL_Image.fromarray(frame).save(self._ffmpeg.stdin, "BMP")
        except Exception as exc:
            self._io_exc = exc

    def _join_io_thread(self) -> None:
        if self._io_thread is not None:
            self._io_thread.join()
            self._io_thread = None
        if self._io_exc is not None:
            exc = self._io_exc
            self._io_exc = None
            raise ChildProcessError("Failed writing frame to ffmpeg stdin") from exc

    def add_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Add a frame to the video.

        Args:
            frame (npt.NDArray[np.uint8]): The unencoded frame as an array.

        Raises:
            ValueError: The shape of `frame` is invalid or does not match a
                previously added image.
            ChildProcessError: The ffmpeg process cannot be communicated with or
                has terminated already.
        """
        if frame.ndim < 2 or frame.ndim > 3:
            raise ValueError(f"Expected 2 or 3 dimensions, got {frame.ndim}.")

        if self._shape is None:
            self._shape = frame.shape
        elif frame.shape != self._shape:
            raise ValueError(f"Expected array shape {self._shape}, got {frame.shape}.")

        if self._ffmpeg.poll() is not None:
            raise ChildProcessError("The ffmpeg process has terminated.")

        # Wait for the thread to finish as images must be sent sequentially.
        self._join_io_thread()
        self._io_thread = threading.Thread(
            target=self._write_frame,
            args=(frame.copy(),),
        )
        self._io_thread.start()

    def close(self) -> None:
        """Closes the ffmpeg subprocess."""
        if self._ffmpeg.returncode is None:
            self._join_io_thread()

            assert self._ffmpeg.stdin
            self._ffmpeg.stdin.close()
            self._ffmpeg.wait()

            if self._ffmpeg.returncode != 0:
                warnings.warn(
                    f"ffmpeg process closed with nonzero exit code ({self._ffmpeg.returncode}).",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # def __del__(self) -> None:
    #     """Closes the ffmpeg subprocess as a fallback."""
    #     if self._ffmpeg.returncode is None:
    #         warnings.warn(
    #             "Closing ffmpeg subprocess at object destruction."
    #             " This should not be relied upon: use close() instead.",
    #             RuntimeWarning,
    #             stacklevel=2,
    #         )
    #         self.close()
