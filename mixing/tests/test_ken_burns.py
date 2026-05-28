"""mixing re-exports the Ken Burns renderers from the ``burns`` package.

The implementation (and its full render/parsing test suite) now lives in
``burns``. mixing keeps ``ken_burns_video`` / ``ken_burns_film`` in its public
video toolkit by re-exporting them, so this test only guards that contract —
that ``mixing``'s names are the same objects as ``burns``'.
"""

import burns
import mixing
from mixing.video.video_ops import ken_burns_video, ken_burns_film


def test_mixing_reexports_burns_renderers():
    assert mixing.ken_burns_video is burns.ken_burns_video
    assert mixing.ken_burns_film is burns.ken_burns_film
    # And the module-level binding in video_ops points at the same objects.
    assert ken_burns_video is burns.ken_burns_video
    assert ken_burns_film is burns.ken_burns_film
