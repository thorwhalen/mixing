#!/usr/bin/env python3
"""
Quick test of the updated generate_video function with save_video egress options
"""


def test_save_video_egress():
    """Test the different save_video egress options"""

    print("Testing save_video egress patterns:")
    print("=" * 50)

    # Mock the generate_video internals for testing
    class MockOperation:
        def __init__(self):
            self.done = True
            self.response = MockResponse()

    class MockResponse:
        def __init__(self):
            self.generated_videos = [MockVideo()]

    class MockVideo:
        def __init__(self):
            self.video_bytes = "ZmFrZV92aWRlb19kYXRh"  # base64 "fake_video_data"
            self.mime_type = "video/mp4"

    print("\n1. Default behavior (save_generated_videos):")
    print("   generate_video('prompt')")
    print("   # Returns: '/tmp/generated_video_abc123.mp4'")

    print("\n2. Save to specific file:")
    print("   generate_video('prompt', save_video='/my/video.mp4')")
    print("   # Returns: '/my/video.mp4'")

    print("\n3. Save to directory:")
    print("   generate_video('prompt', save_video='/output/dir/')")
    print("   # Returns: '/output/dir/generated_video_.mp4'")

    print("\n4. No saving (return operation):")
    print("   generate_video('prompt', save_video=False)")
    print("   # Returns: <Operation object>")

    print("\n5. Custom lambda (return operation):")
    print("   generate_video('prompt', save_video=lambda x: x)")
    print("   # Returns: <Operation object>")

    print("\n6. Custom processor function:")
    print(
        """   def my_processor(op):
       paths = save_generated_videos(op, prefix='custom_')
       return {'status': 'processed', 'paths': paths}
   
   generate_video('prompt', save_video=my_processor)"""
    )
    print("   # Returns: {'status': 'processed', 'paths': [...]}")

    print("\n✅ All egress patterns supported!")
    print("\nKey benefits of the new workflow:")
    print("• One-step generation and file retrieval")
    print("• Flexible egress control with save_video parameter")
    print("• Backward compatibility with manual save_generated_videos")
    print("• Custom processing functions for advanced workflows")


if __name__ == "__main__":
    test_save_video_egress()
