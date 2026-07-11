#import tf2onnx
import os
import subprocess
import sys

def export_model(save_keras=False, save_ONNX=False, model_name="model") -> None:
    """
    Экспортирует модель в формате TensorFlow SavedModel.
    :param model: Keras-модель.
    :param model_name: Имя модели (будет использовано для создания папки).
    """
    from main import model

    if save_keras:
        model.save(f"exports/{model_name}.keras")
    if save_ONNX:
        # Шаг 1 Экспорт в формате SavedModel
        model.save(f"exports/{model_name}_SavedModel")

        # Шаг 2 Переключается на использование Python-реализации протокольных буферов, 
        # чтобы избежать ошибок при конвертации в ONNX. 
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
        # Шаг 3 Конвертация SavedModel в ONNX с использованием tf2onnx
        # !python -m tf2onnx.convert --saved-model exports/saved_model --output exports/model.onnx --opset 17
        # sys.executable - указывает на текущий интерпретатор Python, который используется для запуска скрипта.
        
        subprocess.run([
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", f"exports/{model_name}_SavedModel",
            "--output", f"exports/{model_name}.onnx",
            "--opset", "17"
            ], check=True)
