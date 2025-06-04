# Deploying NeuroFlux Models on Android

This guide provides suggestions and high-level steps for developers looking to use models from the NeuroFlux project in their Android applications. While NeuroFlux focuses on creating highly optimized models for edge devices, it does not directly output pre-built Android applications (APKs). Instead, it provides models in formats like TFLite, which are well-suited for Android deployment.

## Using TFLite Models on Android

The NeuroFlux project, particularly through the scripts in the `huggingface/` directory (e.g., `neuroflux.py` using `NeuroFluxDataset`), can generate `.tflite` models. These models are optimized for mobile and edge devices.

Here's a general workflow to integrate these TFLite models into your Android application:

1.  **Generate or Obtain a TFLite Model:**
    *   Use the `huggingface/neuroflux.py` script as described in the [Quickstart Guide](quickstart.md#accéder-aux-modèles-neuroflux-pré-optimisés) to produce `.tflite` model files.
    *   Ensure you have the specific `.tflite` file (e.g., `mobilenet_optimized.tflite`) you intend to deploy.

2.  **Set up Your Android Project:**
    *   If you don't have one, create a new Android project in Android Studio.
    *   Ensure your project's `minSdkVersion` is compatible with the TensorFlow Lite library requirements.

3.  **Add TensorFlow Lite Dependency:**
    *   Open your app module's `build.gradle` file (usually `app/build.gradle`).
    *   Add the TensorFlow Lite Android library dependency. The latest version can be found on the official TensorFlow website. It will look something like this:
        ```gradle
        dependencies {
            // ... other dependencies
            implementation 'org.tensorflow:tensorflow-lite:2.9.0' // Example version, use the latest
            // For GPU delegation, you might need:
            // implementation 'org.tensorflow:tensorflow-lite-gpu:2.9.0' // Example version
            // For specific task libraries (e.g., vision, nlp):
            // implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0' // Example version
        }
        ```
    *   Sync your project with the Gradle files.

4.  **Add TFLite Model to Assets:**
    *   In Android Studio, switch to the "Project" view in the Project Explorer.
    *   Navigate to `app/src/main/`.
    *   Right-click on the `main` directory and select `New > Directory`. Name it `assets`.
    *   Copy your `.tflite` model file into this `app/src/main/assets/` directory.

5.  **Load and Run the Model in Your Code (Java/Kotlin):**

    *   **Load the model:**
        ```java
        // Java Example
        import org.tensorflow.lite.Interpreter;
        import java.io.FileInputStream;
        import java.io.IOException;
        import java.nio.MappedByteBuffer;
        import java.nio.channels.FileChannel;
        import android.content.res.AssetFileDescriptor;

        private MappedByteBuffer loadModelFile(String modelName) throws IOException {
            AssetFileDescriptor fileDescriptor = this.getAssets().openFd(modelName);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }

        try {
            Interpreter tflite = new Interpreter(loadModelFile("your_model.tflite"));
            // For GPU delegation:
            // Interpreter.Options options = new Interpreter.Options();
            // GpuDelegate delegate = new GpuDelegate();
            // options.addDelegate(delegate);
            // Interpreter tflite = new Interpreter(loadModelFile("your_model.tflite"), options);
        } catch (IOException e) {
            Log.e("TFLite-Error", "Error loading model file: " + e.getMessage());
        }
        ```

    *   **Prepare Input Data:**
        The input data needs to be converted into a `ByteBuffer` with the correct shape and data type that your model expects. This often involves preprocessing images, text, or other data. Refer to the model's documentation or use the TensorFlow Lite Task Library for easier input handling.

    *   **Run Inference:**
        ```java
        // Assuming 'inputBuffer' is your prepared ByteBuffer and 'outputBuffer' is allocated
        // tflite.run(inputBuffer, outputBuffer);
        ```

    *   **Process Output:**
        Interpret the `outputBuffer` according to your model's output specifications.

6.  **Permissions (if needed):**
    *   If your app uses features like the camera for live inference, ensure you request the necessary permissions in your `AndroidManifest.xml`.

### Official TensorFlow Lite Resources:

*   **TensorFlow Lite Android Quickstart:** [https://www.tensorflow.org/lite/android/quickstart](https://www.tensorflow.org/lite/android/quickstart)
*   **TensorFlow Lite Android Support Library:** [https://www.tensorflow.org/lite/android/support_library](https://www.tensorflow.org/lite/android/support_library) (Provides convenient APIs for common tasks)
*   **TensorFlow Lite Model Maker:** [https://www.tensorflow.org/lite/models/model_maker](https://www.tensorflow.org/lite/models/model_maker) (Can help in converting and optimizing models for mobile)

This provides a basic outline. Actual implementation details will vary based on the specific model and application requirements.

## Other Mobile Inference Engines

While TensorFlow Lite is a common choice for Android, the NeuroFlux project's models (especially if available in ONNX format or convertible to it) can also be deployed using other inference engines that support mobile platforms. As mentioned in the [Quickstart Guide](quickstart.md#autres-solutions-dinférence), these include:

*   **ONNX Runtime Mobile:**
    *   If your model is in ONNX format or can be converted to it, ONNX Runtime provides cross-platform acceleration for ML models. It has specific builds optimized for mobile.
    *   You would typically include the ONNX Runtime Android AAR in your project.
    *   Official Documentation: [https://onnxruntime.ai/docs/execution-providers/mobile-options.html](https://onnxruntime.ai/docs/execution-providers/mobile-options.html)

*   **NCNN:**
    *   NCNN is a high-performance neural network inference framework optimized for mobile platforms, particularly ARM CPUs. It's developed by Tencent.
    *   Models often need to be converted to NCNN's specific format. The `webapp/` directory in this project demonstrates NCNN usage for the NanoDet-Plus model.
    *   Official GitHub: [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn)

*   **MNN:**
    *   MNN is a lightweight deep learning inference engine from Alibaba. It supports various model formats and offers good performance on mobile devices.
    *   Official GitHub: [https://github.com/alibaba/MNN](https://github.com/alibaba/MNN)

Choosing an inference engine depends on factors like the original model format, performance requirements, hardware acceleration needs (CPU, GPU, DSP), and the complexity of integrating the engine into your Android application.
