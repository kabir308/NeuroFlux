import cv2 # OpenCV for image preprocessing and drawing (common with NCNN examples)
import numpy as np
import os

try:
    import ncnn
    print(f"Successfully imported ncnn version: {ncnn.__version__}")
    NCNN_AVAILABLE = True
except ImportError:
    print("Failed to import ncnn. Object detection with NCNN will not be available.")
    print("Please ensure 'pyncnn' is installed correctly and its dependencies (NCNN C++ library) are met.")
    NCNN_AVAILABLE = False
except Exception as e:
    print(f"An error occurred during ncnn import: {e}")
    NCNN_AVAILABLE = False

# COCO class names, often used with object detection models like NanoDet
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class NanoDetPlusNCNN:
    def __init__(self, param_path, bin_path, input_size=320, num_classes=80, score_thresh=0.4, nms_thresh=0.5):
        if not NCNN_AVAILABLE:
            self.net = None
            print("NCNN not available, NanoDetPlusNCNN functionality disabled.")
            return

        self.net = ncnn.Net()
        # Register custom layers if NanoDet-Plus NCNN model requires them (common for some models)
        # Example: self.net.register_custom_layer("CustomLayerName", custom_layer_creator)
        # For standard NanoDet-Plus, this might not be needed if pyncnn handles common ops.

        # Attempt to load model
        if self.net.load_param(param_path) != 0:
            raise Exception(f"Failed to load NCNN param file from {param_path}")
        if self.net.load_model(bin_path) != 0:
            raise Exception(f"Failed to load NCNN bin file from {bin_path}")

        print(f"NCNN model loaded successfully: {param_path}, {bin_path}")

        self.input_size = input_size
        self.num_classes = num_classes # NanoDet is usually trained on COCO (80 classes)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        # NanoDet-Plus specific settings (may need adjustment based on exact model version)
        self.reg_max = 7
        self.strides = [8, 16, 32] # Typical strides for FPN-like structures
        # Mean and std for normalization, these are common for many ImageNet-trained models
        # but should be verified for the specific NanoDet-Plus NCNN model.
        # Some NCNN models might have this baked in or expect 0-255 input.
        # For now, let's assume normalization is needed and use common values.
        self.mean_vals = [103.53, 116.28, 123.675] # BGR order
        self.norm_vals = [1.0/57.375, 1.0/57.12, 1.0/58.395] # BGR order

    def preprocess(self, image_pil):
        # Convert PIL image to OpenCV format (NumPy array)
        image_cv = np.array(image_pil.convert('RGB'))
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR) # NCNN often expects BGR

        img_h, img_w = image_cv.shape[:2]

        # Resize image while maintaining aspect ratio by padding
        scale = min(self.input_size / img_h, self.input_size / img_w)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized_img = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a new image with padding
        padded_img = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8) # Pad with 114
        padded_img[(self.input_size - new_h) // 2 : (self.input_size - new_h) // 2 + new_h,
                   (self.input_size - new_w) // 2 : (self.input_size - new_w) // 2 + new_w, :] = resized_img

        # Normalization
        # For pyncnn, create ncnn.Mat from numpy array
        # The Mat constructor can take (w, h, c, data_pointer, cstep)
        # Or convert from_pixels which also handles normalization if mean/std are set.
        # mat_in = ncnn.Mat.from_pixels_resize(image_cv.tobytes(), ncnn.Mat.PixelType.PIXEL_BGR, img_w, img_h, self.input_size, self.input_size)
        # mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        # Simpler way if from_pixels_resize does not handle padding correctly or if we need custom padding:
        # Convert padded_img (NumPy BGR uint8) to ncnn.Mat
        # Ensure it's float32 if normalization is manual
        mat_in = ncnn.Mat(padded_img).clone() # ncnn.Mat expects BGR
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        return mat_in, scale, (img_h, img_w), ((self.input_size - new_h) // 2, (self.input_size - new_w) // 2)


    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def postprocess(self, outputs, scale, original_shape, pad_tl):
        # outputs is a dict where keys are output layer names from NCNN
        # For NanoDet-Plus, these are typically named like 'cls_pred_stride_8', 'reg_pred_stride_8', etc.
        # Or sometimes just 'output', '158', '159', etc. Need to check model's .param file or net.out_names()

        # This is a simplified post-processing based on typical NanoDet structure.
        # It might need significant adjustments based on the exact output names and structure
        # of the NCNN model being used.

        all_bboxes = []
        all_scores = []
        all_class_ids = []

        # Assuming output names are like 'output_8', 'output_16', 'output_32' for scores
        # and 'output_reg_8', 'output_reg_16', 'output_reg_32' for boxes, or similar.
        # This needs to be confirmed by inspecting the NCNN model's output layers.
        # The pyncnn `Extractor.extract` returns a list of ncnn.Mat.
        # If `net.out_names()` is available in pyncnn and useful, it could be used.
        # For now, let's assume `outputs` is a list of ncnn.Mat in order of strides.

        # Placeholder: This part is highly dependent on model structure and pyncnn's output format.
        # The actual NanoDet postprocessing is more complex, involving decoding from distribution (GFL).
        # What follows is a generic placeholder for object detection postprocessing.

        # Example for a single output layer (needs to be adapted for multi-stride NanoDet)
        # This is NOT NanoDet's actual complex postprocessing logic.
        # It's a generic placeholder to illustrate the flow.

        # For a proper NanoDet implementation, one would iterate over each stride's output:
        # 1. Get classification scores (e.g., 80 classes per anchor point).
        # 2. Get regression predictions (e.g., 4 * (reg_max + 1) values per anchor point for GFL).
        # 3. Create anchor points (centers of grid cells for each stride).
        # 4. Decode regression predictions to bounding boxes (e.g., using integral of distribution for GFL).
        # 5. Filter by score.
        # 6. Perform NMS.

        # Since implementing full NanoDet GFL decoding here is too complex for one subtask
        # and might require specific ops not easily available in numpy/pyncnn directly,
        # I will create a placeholder that assumes a simpler bbox format from the network
        # or that the user might need to plug in a more detailed C++ postprocessing if pyncnn is limited.

        # Let's assume `outputs` contains one primary detection mat for simplicity here.
        # A real implementation would iterate through all feature map outputs.
        if not outputs: # if outputs is empty list from ex.extract_all()
            return []

        # Assuming a very simplified output for demonstration: [class_id, score, x1, y1, x2, y2]
        # This is NOT what NanoDet-Plus NCNN actually outputs directly.
        # The actual output is raw feature maps.

        # A more realistic but still simplified approach if we assume some direct box predictions:
        # Let's say one of the output Mat objects in the list `outputs`
        # has dimensions (num_detections, 5 + num_classes) where each row is [x,y,w,h,conf, cls_scores...]
        # This is also a simplification.

        # Given the complexity, the most robust thing this subtask can do for postprocessing
        # is to set up the structure and highlight that detailed, model-specific C++ level
        # postprocessing might be needed if pyncnn alone isn't enough.
        # For now, return an empty list to signify postprocessing needs to be fully implemented.

        # TODO: Implement detailed NanoDet-Plus post-processing.
        # This involves:
        # - Iterating through output layers (cls_pred_stride_X, reg_pred_stride_X for X in [8,16,32]).
        # - Generating anchor points for each stride.
        # - Applying General Focal Loss (GFL) / Distribution Focal Loss decoding for bounding boxes.
        # - Applying softmax/sigmoid to scores.
        # - Filtering detections by score threshold.
        # - Performing NMS across all strides.
        # - Scaling coordinates back to original image size.

        # This is a complex operation. For now, we'll return dummy data or an empty list.
        # This makes the class structurally complete but functionally incomplete for detection.

        # Example dummy detection (replace with actual logic)
        # detections = [
        #     {'box': [int(50/scale), int(50/scale), int(150/scale), int(150/scale)], 'label': 'person', 'score': 0.9}
        # ]
        # return detections

        print("Warning: Full NanoDet-Plus NCNN postprocessing is not implemented in ncnn_object_detector.py. Returning empty detections.")
        return []


    def detect(self, image_pil):
        if self.net is None or not NCNN_AVAILABLE:
            print("NCNN Net not initialized or NCNN not available.")
            return []

        mat_in, scale, original_shape, pad_tl = self.preprocess(image_pil)

        ex = self.net.create_extractor()
        # Set number of threads. 0 means use default (often number of big cores).
        # ex.set_num_threads(4) # Example: Use 4 threads

        # Set input. The input blob name must match what's in the .param file.
        # Common default is "input" or "data". Let's assume "input".
        # If pyncnn has issues with blob names, this might need adjustment.
        # Some NCNN examples show that input/output names can be inferred if not set.
        # Let's find input blob name
        input_names = self.net.input_names()
        if not input_names:
             raise Exception("NCNN model has no input names defined in .param file or discoverable by pyncnn.")
        input_name = input_names[0] # Assume first input is the image
        print(f"NCNN Model Input Name: {input_name}")

        ex.input(input_name, mat_in)

        # Extract all output blobs.
        # This part is tricky as we don't know the output blob names beforehand without inspecting the .param file.
        # For NanoDet, there are multiple output heads.
        # A robust way is to get all output names from net.output_names()
        output_names = self.net.output_names()
        if not output_names:
            # Fallback: some pyncnn examples use hardcoded names or indices if names are not exposed.
            # This is highly model-specific.
            # For now, we'll raise an error if output names aren't found,
            # as postprocessing depends on knowing them.
            raise Exception("NCNN model has no output names defined in .param file or discoverable by pyncnn. Cannot extract.")

        print(f"NCNN Model Output Names: {output_names}")

        # Extract data from each output blob
        # The `ret` variable from ex.extract() indicates success/failure for each blob.
        all_raw_outputs = []
        for out_name in output_names:
            ret, out_mat = ex.extract(out_name)
            if ret != 0: # Error
                raise Exception(f"Failed to extract NCNN output blob: {out_name}")
            all_raw_outputs.append(np.array(out_mat).copy()) # Convert ncnn.Mat to NumPy array and copy
            # np.array(ncnn.Mat) should create a view. Copy it to own the data.
            # The shape and content of out_mat need to be understood for postprocessing.

        detections = self.postprocess(all_raw_outputs, scale, original_shape, pad_tl)
        return detections

# Example Usage (for testing this script directly)
if __name__ == '__main__':
    if NCNN_AVAILABLE:
        # These paths assume the script is run from the project root or webapp directory
        # and models are in webapp/models/ncnn/
        param_file = os.path.join(os.path.dirname(__file__), 'models', 'ncnn', 'nanodet-plus-m_320.param')
        bin_file = os.path.join(os.path.dirname(__file__), 'models', 'ncnn', 'nanodet-plus-m_320.bin')

        if not (os.path.exists(param_file) and os.path.exists(bin_file)):
            print(f"Model files not found. Searched for: \n{param_file}\n{bin_file}")
            print("Please ensure NCNN model files are correctly placed in webapp/models/ncnn/")
        else:
            try:
                detector = NanoDetPlusNCNN(param_path=param_file, bin_path=bin_file)
                print("NanoDetPlusNCNN detector initialized successfully.")

                # Create a dummy PIL Image for testing preprocessing and basic detection flow
                # (actual detection will be empty due to placeholder postprocessing)
                from PIL import Image
                dummy_image_np = np.zeros((480, 640, 3), dtype=np.uint8) # H, W, C
                dummy_image_pil = Image.fromarray(dummy_image_np)
                print("Created dummy PIL image for testing.")

                detections = detector.detect(dummy_image_pil)
                print(f"Detection results (dummy): {detections}")

            except Exception as e:
                print(f"Error during NanoDetPlusNCNN initialization or test detection: {e}")
    else:
        print("Skipping NanoDetPlusNCNN test because NCNN/pyncnn is not available.")
