from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


class VideoIterator:
    """
    Iterator over video frames.

    Args:
        video_file: Path to the input video.
        cvtcolor: If True, convert frames from OpenCV's native BGR to RGB.
            If False, frames are returned as BGR.
        reuse_buffer: If True, reuse the same mutable frame buffer across
            iterations when OpenCV supports it. Callers must copy returned frames
            if they need to keep them after advancing the iterator.

    Example:
    ```python
    with VideoIterator("/path") as it:
        for frame_num, frame in it:
            pass
    ```
    """

    def __init__(
        self,
        video_file: str | Path,
        cvtcolor: bool = True,
        reuse_buffer: bool = False,
    ) -> None:
        self.cap: cv2.VideoCapture | None = None
        self.frame_height: int
        self.frame_width: int
        self.expected_frame_count: int
        self.fps: float
        self.cvtcolor = cvtcolor
        self.reuse_buffer = reuse_buffer
        self._buffer: npt.NDArray[np.uint8] | None = None

        self.cap = cv2.VideoCapture(str(video_file))
        if not self.cap.isOpened():
            raise RuntimeError("cv2 VideoCapture could not be opened")
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.expected_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __len__(self) -> int:
        return self.expected_frame_count

    def reset(self, pos: int = 0) -> "VideoIterator":
        if pos < 0 or self.expected_frame_count <= pos:
            raise ValueError(
                f"Reset pos ({pos}) exceeds bounds [0, {self.expected_frame_count})"
            )

        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) != pos:
                raise RuntimeError(f"Unable to perform exact seek to pos {pos}")
        self._buffer = None
        return self

    def __iter__(self) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        return self

    def __next__(self) -> tuple[int, npt.NDArray[np.uint8]]:
        if self.cap is None:
            raise StopIteration

        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read(self._buffer)
        if not ret:
            raise StopIteration

        if frame is None:
            raise ValueError("Frame is None")
        if frame.dtype != np.uint8:
            raise ValueError("Frame is not np.uint8")

        if self.reuse_buffer:
            self._buffer = frame

        if self.cvtcolor:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._buffer)

        return frame_number, frame  # type: ignore

    # Seeking is very slow, so this API is discouraged. Similar functionality
    # can be achieved using reset() and next() if absolutely necessary. For
    # strided access (with relatively small stride), it is usually faster to
    # manually skip frames.
    # def __getitem__(self, idx: int) -> npt.NDArray[np.uint8]:
    #     assert self.cap is not None

    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    #     assert int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == idx
    #     ret, frame = self.cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     assert ret
    #     assert frame.dtype == np.uint8
    #     return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._buffer = None

    def __enter__(self) -> "VideoIterator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
