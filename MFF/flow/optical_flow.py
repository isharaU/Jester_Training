import cv2
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

class CUDAStyleFlowGenerator:
    @staticmethod
    def _calculate_flow(prev_frame, curr_frame):
        """Calculate optical flow between two frames"""
        flow_calculator = cv2.optflow.DualTVL1OpticalFlow_create()
        return flow_calculator.calc(prev_frame, curr_frame, None)

    @staticmethod
    def _normalize_flow(flowx, flowy):
        """Normalize flow using absolute max value approach"""
        max_val = max(abs(flowx).max(), abs(flowy).max())
        flowx_n = np.clip(((flowx + max_val) * (255.0 / (2 * max_val))), 0, 255).astype(np.uint8)
        flowy_n = np.clip(((flowy + max_val) * (255.0 / (2 * max_val))), 0, 255).astype(np.uint8)
        return flowx_n, flowy_n

    @staticmethod
    def process_video_folder(folder_info):
        """Process all frames in a video folder"""
        input_dir, output_dir_u, output_dir_v = folder_info
        os.makedirs(output_dir_u, exist_ok=True)
        os.makedirs(output_dir_v, exist_ok=True)
        frames = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

        if not frames:
            print(f"No jpg files found in {input_dir}")
            return

        for i in range(len(frames) - 1):
            frame1_path = os.path.join(input_dir, frames[i])
            frame2_path = os.path.join(input_dir, frames[i + 1])

            frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

            if frame1 is None or frame2 is None:
                print(f"Error reading frames: {frame1_path} or {frame2_path}")
                continue

            flow = CUDAStyleFlowGenerator._calculate_flow(frame1, frame2)
            flowx_n, flowy_n = CUDAStyleFlowGenerator._normalize_flow(flow[..., 0], flow[..., 1])

            output_name = frames[i]
            cv2.imwrite(os.path.join(output_dir_u, output_name), flowx_n, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(output_dir_v, output_name), flowy_n, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def process_from_text_file(self, text_file_path, root_dir, output_root):
        """Process videos listed in a text file in parallel"""
        output_u = os.path.join(output_root, 'flow_abs_max', 'u')
        output_v = os.path.join(output_root, 'flow_abs_max', 'v')
        os.makedirs(output_u, exist_ok=True)
        os.makedirs(output_v, exist_ok=True)

        with open(text_file_path, 'r') as f:
            folders = [line.strip() for line in f if line.strip()]

        folder_info = [(os.path.join(root_dir, folder), os.path.join(output_u, folder), os.path.join(output_v, folder)) for folder in folders]

        print(f"Processing {len(folder_info)} folders using {cpu_count()} processes...")

        with Pool(cpu_count()) as pool:
            list(tqdm(pool.imap(CUDAStyleFlowGenerator.process_video_folder, folder_info), total=len(folder_info), desc="Processing videos"))

def main():
    generator = CUDAStyleFlowGenerator()
    text_file_path = "/content/video_names.txt"
    root_dir = "/content/drive/MyDrive/V2E/test/jester/rgb"
    output_root = "/content/drive/MyDrive/V2E/test/jester/flow"
    generator.process_from_text_file(text_file_path, root_dir, output_root)

if __name__ == "__main__":
    main()
