from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import traceback
app = Flask(__name__)

# Cargar modelos y preprocesadores
encoder = joblib.load('models/laptop_encoder.pkl')
scaler = joblib.load('models/laptop_scaler.pkl')
model = joblib.load('models/laptop_price_model.pkl')

# # Columnas esperadas por el modelo (deben coincidir con tu entrenamiento)
categorical_cols = ['Company', 'Product', 'Touchscreen', 'IPSpanel', 'TypeName',
                    'OS', 'Screen', 'RetinaDisplay', 'CPU_company', 'CPU_model',
                    'PrimaryStorageType', 'SecondaryStorageType', 'GPU_company', 
                    'GPU_model']


numerical_cols = ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 
                 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict-laptop-price', methods=['POST'])
def predict():
    try:
        print("\n=== Iniciando predicción ===")
        input_data = request.json
        print("Datos recibidos:", input_data)

        if not input_data:
            return jsonify({'error': 'No se recibieron datos', 'status': 'error'}), 400

        # 1. Obtener el orden EXACTO de features del modelo
        required_features = list(model.feature_names_in_)
        print("\nOrden requerido por el modelo:", required_features)

        # 2. Crear DataFrame en el ORDEN correcto
        laptop_data = pd.DataFrame({feature: [input_data.get(feature)] for feature in required_features})
        print("\nDataFrame inicial:")
        print(laptop_data)

        # 3. Verificar valores faltantes
        missing_values = laptop_data.isnull().any()
        if missing_values.any():
            error_msg = f"Valores faltantes: {list(missing_values[missing_values].index)}"
            print(error_msg)
            return jsonify({'error': error_msg, 'status': 'error'}), 400

        # 4. Reconfirmar orden
        laptop_data = laptop_data[required_features]
        print("\nDataFrame con orden corregido:")
        print(laptop_data)

        # 5. ✅ PREPROCESAMIENTO CORREGIDO
        try:
            print("\nAplicando preprocesamiento...")

            # Orden correcto de columnas esperadas por el encoder
            encoder_cols = list(encoder.feature_names_in_)
            print("→ Columnas usadas por encoder:", encoder_cols)
            laptop_data[encoder_cols] = encoder.transform(laptop_data[encoder_cols])

            # Orden correcto de columnas esperadas por el scaler
            if hasattr(scaler, "feature_names_in_"):
                scaler_cols = list(scaler.feature_names_in_)
            else:
                scaler_cols = numerical_cols  # fallback si no tiene atributo

            print("→ Columnas usadas por scaler:", scaler_cols)
            laptop_data[scaler_cols] = scaler.transform(laptop_data[scaler_cols])

            print("\n✅ DataFrame después de preprocesamiento:")
            print(laptop_data)

        except ValueError as e:
            error_msg = f"Error en preprocesamiento: {str(e)}"
            print(error_msg)
            print("Columnas disponibles:", laptop_data.columns.tolist())
            print("Columnas esperadas por el encoder:", getattr(encoder, "feature_names_in_", []))
            print("Columnas esperadas por el scaler:", getattr(scaler, "feature_names_in_", []))
            return jsonify({'error': 'Error procesando los datos', 'status': 'error'}), 400

        # 6. Asegurar el orden final antes de predecir
        laptop_data = laptop_data[required_features]

        # 7. Predicción
        try:
            prediction = model.predict(laptop_data)
            predicted_price = float(prediction[0])
            print(f"\n✅ Predicción exitosa: €{predicted_price:.2f}")

            return jsonify({
                'predicted_price': predicted_price,
                'exchange_rate': 20.50,
                'status': 'success'
            })

        except Exception as e:
            error_msg = f"Error en predicción: {str(e)}"
            print(error_msg)
            print("Traceback:", traceback.format_exc())
            return jsonify({'error': 'Error generando predicción', 'status': 'error'}), 500

    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        print(error_msg)
        print("Traceback completo:", traceback.format_exc())
        return jsonify({'error': 'Error interno del servidor', 'status': 'error'}), 500

def get_exchange_rate():
    """Obtiene el tipo de cambio actual de EUR a MXN"""
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/EUR', timeout=3)
        return response.json()['rates']['MXN']
    except:
        # Puedes implementar un fallback a otra API o base de datos aquí
        return 20.50  # Valor por defecto
if __name__ == '__main__':
    app.run(debug=True)