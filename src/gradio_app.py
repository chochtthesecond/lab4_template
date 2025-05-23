import gradio as gr
import joblib
import numpy as np
import pandas as pd

from config import app_config

model = joblib.load(app_config.path_to_modelfile)
FEATURE_NAMES = app_config.feature_names
CLASS_NAMES_MAP = {
    1: 'Mammal',
    2: 'Bird',
    3: 'Reptile',
    4: 'Fish',
    5: 'Amphibian',
    6: 'Bug',
    7: 'Invertebrate'
}


# Функция для инференса
def predict_species(legs,*checks):
    processed_checks = [int(c) for c in checks]
    input_values = [legs] + processed_checks

    # Создание Pandas DataFrame с именами признаков
    input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

    # Предсказание
    prediction = model.predict(input_df)

    numeric_pred = 0 # Default or error value
    numeric_pred = int(prediction[0])

    class_name = CLASS_NAMES_MAP.get(numeric_pred, f"Unknown class ({numeric_pred})")
    
    return class_name


# Создание интерфейса с gr.Blocks
with gr.Blocks(title="Инференс модели") as demo:
    gr.Markdown("## 🐾 Прогнозирование вида животного")

    with gr.Row():
        # Входные слайдеры для признаков
        # Можно определить текстовые поля для ввода
        labels=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
        checks=[]
        for x in labels:
            if x=='legs': break
            checks.append(gr.Checkbox(label=x))
        legs = gr.Slider(
            minimum=0, maximum=8, step=1, label="legs"
        )
        checks.append(legs)
        for x in labels[13:]:
            checks.append(gr.Checkbox(label=x))
            
    # Кнопка для запуска инференса
    predict_btn = gr.Button("Предсказать вид")

    # Вывод результата
    result = gr.Textbox(label="Результат")

    # Связь кнопки и функции
    predict_btn.click(
        fn=predict_species,
        inputs=[*checks],
        outputs=result,
    )

# Запуск приложения
if __name__ == "__main__":
    demo.launch()
