import os
import sys
from pathlib import Path
import argparse
#TODO: This is bad practice
p = Path(os.path.abspath(__file__))
sys.path.append(str(p.parent.parent))
sys.path.append(str(p.parent))

from  data.dataprep_mediapipe import VideoProcessor

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_raw_frames", default=0, type=int)
    parser.add_argument("--process_raw_frames", default=0, type=int)

    FILTER = 'face_smile'
    IMG_SIZE = 224

    input_directory = 'data/raw'
    output_directory = 'data/processed'
    video_processor = VideoProcessor(input_directory,
                                    output_directory)

    
    #patiend_ids = set([int(i.split('_')[0]) for i in video_processor.video_filter(input_directory,video_type=FILTER)])
    #processed_ids = set([int(i.split('_')[0]) for i in video_processor.video_filter(output_directory,extension='pkl',video_type=FILTER)])
    #remaining_ids = patiend_ids-processed_ids

    #print(remaining_ids)

    args = parser.parse_args()

    video_names = video_processor.video_filter(input_directory,video_type=FILTER)

    if args.extract_raw_frames==1:
        video_processor.extract_frames(video_names)
        
    if args.process_raw_frames==1:
        video_processor.pre_process()

