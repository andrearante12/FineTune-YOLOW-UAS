import argparse
import os
import cv2


def extract_frames(video_path, output_dir, num_frames):
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{video_name}: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")

    # Evenly space frame indices across the video
    if num_frames >= total_frames:
        indices = range(total_frames)
    else:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"{video_name}_frame_{idx:06d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved += 1

    cap.release()
    print(f"  Saved {saved} frames")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract evenly-spaced frames from videos for Roboflow labeling"
    )
    parser.add_argument("--videos", nargs="+",
                        default=["VID_01.MP4", "VID_02.MP4", "VID_03.MP4"],
                        help="Video files to extract from")
    parser.add_argument("--output", default="datasets/OpenVocab/images",
                        help="Output directory for extracted frames")
    parser.add_argument("--frames-per-video", type=int, default=75,
                        help="Number of frames to extract per video (default: 75, total ~225)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    total_saved = 0
    for video in args.videos:
        if not os.path.isfile(video):
            print(f"Warning: video not found, skipping: {video}")
            continue
        total_saved += extract_frames(video, args.output, args.frames_per_video)

    print(f"\nDone. {total_saved} total frames saved to {args.output}")
    print("Upload this directory to Roboflow for labeling.")


if __name__ == "__main__":
    main()
