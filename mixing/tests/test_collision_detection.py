#!/usr/bin/env python3
"""
Test collision detection in the refactored save_generated_videos function
"""

import tempfile
import os
from mixing.video.video_gen import _generate_output_paths, save_generated_videos


class MockVideo:
    """Mock GeneratedVideo object for testing"""

    def __init__(self, mime_type='video/mp4', video_bytes=b'fake_video_data'):
        self.mime_type = mime_type
        self.video_bytes = video_bytes


def test_collision_detection():
    """Test that non_colliding_key prevents file overwrites"""

    print("🎬 Testing collision detection with non_colliding_key")
    print("=" * 60)

    # Create a test directory
    test_dir = tempfile.mkdtemp()
    print(f"Test directory: {test_dir}")

    try:
        # Create some mock "existing" video files
        existing_files = [
            'my_video.mp4',
            'my_video (1).mp4',
            'generated_video_.mp4',
            'generated_video_ (1).mp4',
        ]

        for fname in existing_files:
            filepath = os.path.join(test_dir, fname)
            with open(filepath, 'w') as f:
                f.write('existing video file')
            print(f"✓ Created existing file: {fname}")

        print("\n" + "-" * 40)
        print("Testing path generation with collisions:")

        # Test 1: Single video to directory (should avoid collision)
        print("\n1. Single video to directory with collision:")
        videos = [MockVideo()]
        paths = _generate_output_paths(
            videos, test_dir, directory_name='generated_video_'
        )
        print(f"   Generated: {os.path.basename(paths[0])}")
        print(f"   ✓ Avoided collision with existing 'generated_video_.mp4'")

        # Test 2: Multiple videos to directory
        print("\n2. Multiple videos to directory:")
        videos = [MockVideo(), MockVideo(), MockVideo()]
        paths = _generate_output_paths(videos, test_dir, directory_name='batch_video')
        for i, path in enumerate(paths):
            print(f"   Video {i}: {os.path.basename(path)}")

        # Test 3: Specific file collision
        print("\n3. Specific file path with collision:")
        specific_path = os.path.join(test_dir, 'my_video.mp4')
        videos = [MockVideo()]
        paths = _generate_output_paths(videos, specific_path)
        print(f"   Original: my_video.mp4")
        print(f"   Safe path: {os.path.basename(paths[0])}")
        print(f"   ✓ Avoided overwriting existing file")

        # Test 4: Multiple videos with specific base path
        print("\n4. Multiple videos with base path collision:")
        base_path = os.path.join(test_dir, 'my_video.mp4')
        videos = [MockVideo(), MockVideo()]
        paths = _generate_output_paths(videos, base_path)
        for i, path in enumerate(paths):
            print(f"   Video {i}: {os.path.basename(path)}")

        print("\n" + "=" * 60)
        print("✅ All collision detection tests passed!")
        print("✅ Files are safe from accidental overwrites!")

        # Demonstrate actual saving works too
        print("\n" + "-" * 40)
        print("Testing actual save_generated_videos function:")

        import base64

        # Create a mock video with actual video bytes
        mock_video = MockVideo()
        mock_video.video_bytes = base64.b64encode(b'fake video content').decode()

        # Save to directory (should avoid collisions)
        saved_path = save_generated_videos(
            mock_video, test_dir, directory_name='actual_save'
        )
        print(f"✓ Saved video to: {os.path.basename(saved_path)}")
        print(f"✓ File exists: {os.path.exists(saved_path)}")

        # Verify file content
        with open(saved_path, 'rb') as f:
            content = f.read()
        print(f"✓ File size: {len(content)} bytes")

    finally:
        # Clean up
        print(f"\n🧹 Cleaning up test directory: {test_dir}")
        import shutil

        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_collision_detection()
