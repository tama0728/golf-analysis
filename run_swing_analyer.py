import os
import shutil
import cv2
import matplotlib.pyplot as plt
from process_swing import DataProcessor, Evaluator, VideoProcessor
from MediaPipe_class import MediaPipe_PoseEstimation

# 프로그램 시작과 종료 시 처리된 파일 정리
def cleanup_files(FOLDER):
    for filename in os.listdir(FOLDER):
        file_path = os.path.join(FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def run_swing_analyzer(file_id):
    input_file = "uploads/" + f'/{file_id}_input.mp4'
    output_video_path = f"processed/{file_id}/{file_id}_output.mp4"
    csv_file_name = f'{file_id}.csv'

    # Create 'processed/{file_id}' folder
    processed_folder = f"processed/{file_id}"
    os.makedirs(processed_folder, exist_ok=True)
    csv_file_path = os.path.join(processed_folder, os.path.basename(csv_file_name))

    video_processor = MediaPipe_PoseEstimation(input_file, csv_file_path, output_video_path)
    video_processor.process_video()

    data_processor = DataProcessor(csv_file_path)
    data_processor.load_data()
    data_processor.preprocess_data()

    evaluator = Evaluator(data_processor)
    evaluator.evaluate_all_swing_parts()

    video_processor = VideoProcessor(processed_folder, data_processor, evaluator)
    video_processor.save_frame()


    image_files = [f for f in os.listdir(processed_folder) if f.split('_')[0] == file_id and f.endswith(('.jpg', '.jpeg', '.png'))]

    # Define the keywords for sorting
    keywords = ['address', 'top', 'contact']
    def extract_keyword_index(filename):
        filename_lower = filename.lower()
        for idx, keyword in enumerate(keywords):
            if keyword in filename_lower:
                return idx
        return len(keywords)  # Assign a higher index if keyword not found

    sorted_image_files = sorted(image_files, key=lambda x: extract_keyword_index(x))
    images = []

    # Read each image and append it to the list
    for image_file in sorted_image_files:
        image_path = os.path.join(processed_folder, image_file)
        image = cv2.imread(image_path)
        # Convert BGR to RGB format for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    # Create a figure to display the images
    # num_images = len(images)
    # fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    #
    # # Display each image in its respective subplot
    # for i in range(num_images):
    #     axes[i].imshow(images[i])
    #     axes[i].set_title(f'{sorted_image_files[i].split("_")[-1].split(".")[0]}')
    #     axes[i].axis('off')
    # plt.tight_layout()
    # plt.show()
    return video_processor.print_swing_analysis()