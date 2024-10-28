from utils import read_video, save_video
from trackers import Tracker
import gc
import cv2

def main():
    # Read Video
    video_frames = read_video('input_videos/FootVid.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Draw output video
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_videos/FootVid_output.avi')

    gc.collect()

if __name__ == '__main__':
    main()
