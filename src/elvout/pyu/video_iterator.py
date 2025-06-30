from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


class VideoIterator:
    """Wrapper class around cv2.VideoCapture."""

    def __init__(self, video_file: str | Path) -> None:
        self.cap: cv2.VideoCapture | None = None
        self.frame_height: int
        self.frame_width: int
        self.expected_frame_count: int
        self.fps: float

        self.cap = cv2.VideoCapture(str(video_file))
        assert self.cap.isOpened()
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.expected_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __len__(self) -> int:
        return self.expected_frame_count

    def reset(self, pos: int = 0) -> "VideoIterator":
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            assert int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) == pos
        return self

    def __iter__(self) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        return self  # type: ignore

    def __next__(self) -> tuple[int, npt.NDArray[np.uint8]]:
        if self.cap is None:
            raise StopIteration

        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        assert frame.dtype == np.uint8
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
    #     return frame  # type: ignore

    def __del__(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
