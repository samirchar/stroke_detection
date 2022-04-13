import os
import sys
#TODO: This is bad practice
sys.path.append("/Users/samirchar/Google_Drive/Masters/DSI Scholars/stroke_detection/src/")
sys.path.append("/Users/samirchar/Google_Drive/Masters/DSI Scholars/stroke_detection/src/data")
from  data.dataprep_mediapipe import VideoProcessor

if __name__ == "__main__":    

    FILTER = 'face_smile'
    IMG_SIZE = 224

    input_directory = 'data/raw'
    output_directory = 'data/processed'
    video_processor = VideoProcessor(input_directory,
                                    output_directory)

    
    patiend_ids = set([int(i.split('_')[0]) for i in video_processor.video_filter(input_directory,video_type=FILTER)])
    processed_ids = set([int(i.split('_')[0]) for i in video_processor.video_filter(output_directory,extension='pkl',video_type=FILTER)])
    remaining_ids = patiend_ids-processed_ids

    print(remaining_ids)

    for p_id in remaining_ids:
        video_names = video_processor.video_filter(input_directory,str(p_id),video_type=FILTER)
        video_processor.read(video_names)
        video_processor.pre_process(img_size = IMG_SIZE)

