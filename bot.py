# ... (imports anteriores se mantienen igual)
import os  # Importación del módulo os
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from collections import deque
import json
from telegram import Update
from telegram.ext import CallbackContext, Updater, CommandHandler, MessageHandler
from telegram.ext.filters import Filters
from datetime import datetime, time
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import logging

# Configuración del logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Reemplaza 'TU_TOKEN' y 'TU_CANAL_ID' con tus valores reales
TOKEN = "7928518493:AAE4tvmK8MBiZtaSwRbSKj1Io95WPiuyADI"
CANAL_ID = "-1002371093140"

# ----------------- FUNCIONES AUXILIARES (Simuladas) -----------------
def get_klines(symbol, limit=500):
    # Simulación de datos de precios
    return [[0, 100, 110, 95, 105, 1000], [1, 105, 115, 100, 112, 1100], [2, 112, 120, 108, 118, 1200]] * (limit // 3)

def get_fundamental_news(limit=100):
    # Simulación de noticias
    return [{'title': 'Noticia positiva sobre BTC'}, {'title': 'Noticia neutral sobre ETH'}] * (limit // 2)

def get_price(symbol):
    # Simulación del precio actual
    return 115.0

# ----------------- CLASE BASE (Simulada) -----------------
class CryptoAI:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def predict_price(self, symbol):
        return [118.0, 120.0, 122.0]

    def prepare_data(self, closes):
        # Simulación de preparación de datos
        return np.random.rand(10, 10), np.random.rand(10, 1), joblib.load('scaler_example.pkl')

    def analyze_news(self, text):
        return {'positive': 1, 'negative': 0}

# ----------------- CONFIGURACIÓN AVANZADA -----------------
MODEL_SAVE_PATH = "ai_models/saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ----------------- IA CON AUTOAPRENDIZAJE -----------------
class SelfLearningCryptoAI(CryptoAI):
    def __init__(self):
        super().__init__()
        self.news_model = None
        self.feedback_db = deque(maxlen=1000)  # Base de datos de feedback
        self.load_models()

    def load_models(self):
        """Carga modelos guardados"""
        try:
            # Modelo de noticias
            if os.path.exists(f"{MODEL_SAVE_PATH}/news_model.model"):
                self.news_model = Word2Vec.load(f"{MODEL_SAVE_PATH}/news_model.model")

            # Modelos de precios
            for symbol in ["BTC", "ETH", "SOL"]:
                if os.path.exists(f"{MODEL_SAVE_PATH}/{symbol}_model.h5"):
                    self.models[symbol] = load_model(f"{MODEL_SAVE_PATH}/{symbol}_model.h5")
                    self.scalers[symbol] = joblib.load(f"{MODEL_SAVE_PATH}/{symbol}_scaler.pkl")

        except Exception as e:
            logger.error(f"Error cargando modelos: {str(e)}")

    def save_models(self):
        """Guarda los modelos periódicamente"""
        try:
            # Modelo de noticias
            if self.news_model:
                self.news_model.save(f"{MODEL_SAVE_PATH}/news_model.model")

            # Modelos de precios
            for symbol, model in self.models.items():
                model.save(f"{MODEL_SAVE_PATH}/{symbol}_model.h5")
                joblib.dump(self.scalers[symbol], f"{MODEL_SAVE_PATH}/{symbol}_scaler.pkl")

        except Exception as e:
            logger.error(f"Error guardando modelos: {str(e)}")

    def update_with_feedback(self, symbol, actual_price, predicted_price):
        """Ajusta el modelo basado en feedback"""
        error = actual_price - predicted_price
        self.feedback_db.append({
            'symbol': symbol,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

        # Reentrenamiento periódico
        if len(self.feedback_db) % 100 == 0:
            self.retrain_models()

    def retrain_models(self):
        """Reentrena modelos con nuevos datos"""
        try:
            logger.info("Iniciando reentrenamiento de modelos...")

            # 1. Reentrenar modelos de precios
            for symbol in self.models.keys():
                new_data = get_klines(symbol, limit=500)  # Datos recientes
                closes = np.array([float(k[4]) for k in new_data]).reshape(-1, 1)
                X, y, scaler = self.prepare_data(closes)
                self.models[symbol].fit(X, y, epochs=20, batch_size=16, verbose=0)
                self.scalers[symbol] = scaler

            # 2. Actualizar modelo de noticias
            articles = get_fundamental_news(limit=100)
            processed_news = [self.preprocess_text(a['title']) for a in articles]
            if not self.news_model:
                self.news_model = Word2Vec(processed_news, vector_size=100, window=5, min_count=1, workers=4)
            else:
                self.news_model.build_vocab(processed_news, update=True)
                self.news_model.train(processed_news, total_examples=len(processed_news), epochs=10)

            self.save_models()
            logger.info("Reentrenamiento completado con éxito")

        except Exception as e:
            logger.error(f"Error en reentrenamiento: {str(e)}")

    def analyze_news_advanced(self, text):
        """Análisis semántico avanzado de noticias"""
        try:
            # Análisis de sentimiento básico
            sentiment = super().analyze_news(text)

            # Análisis semántico
            processed = self.preprocess_text(text)
            if self.news_model:
                vectors = [self.news_model.wv[word] for word in processed if word in self.news_model.wv]
                if vectors:
                    cluster_model = KMeans(n_clusters=3)
                    clusters = cluster_model.fit_predict(vectors)
                    sentiment['clusters'] = len(set(clusters))
                    sentiment['topics'] = self.extract_topics(processed)

            return sentiment
        except Exception as e:
            logger.error(f"Error en análisis avanzado: {str(e)}")
            return None

    def preprocess_text(self, text):
        """Preprocesamiento de texto para NLP"""
        # Implementar limpieza de texto (stopwords, stemming, etc.)
        return text.lower().split()

    def extract_topics(self, tokens):
        """Extrae temas clave usando el modelo de noticias"""
        if not self.news_model:
            return []

        # Implementar extracción de temas importantes
        return list(set(token for token in tokens if token in self.news_model.wv))

# ----------------- NUEVOS COMANDOS -----------------
def market_analysis(update: Update, context):
    """Análisis completo del mercado"""
    try:
        symbol = context.args[0].upper() if context.args else 'BTC'

        # 1. Predicción de IA
        predictions = crypto_ai.predict_price(symbol)

        # 2. Análisis Técnico
        ta_report = generate_ta_report(symbol)

        # 3. Análisis Fundamental
        news = get_fundamental_news()
        sentiment = crypto_ai.analyze_news_advanced(" ".join([n['title'] for n in news]))

        # 4. Recomendación integrada
        recommendation = generate_recommendation(predictions, ta_report, sentiment)

        message = (
            f"🔍 *Análisis Completo - {symbol}*\n\n"
            f"📈 *Predicción IA (3 días)*:\n"
            f"{format_predictions(predictions)}\n\n"
            f"📊 *Análisis Técnico*:\n"
            f"{ta_report}\n\n"
            f"📰 *Análisis Fundamental*:\n"
            f"• Sentimiento: {sentiment['positive']}👍 / {sentiment['negative']}👎\n"
            f"• Temas clave: {', '.join(sentiment.get('topics', []))}\n\n"
            f"💡 *Recomendación*:\n"
            f"{recommendation}"
        )

        update.message.reply_text(message, parse_mode='Markdown')

    except Exception as e:
        update.message.reply_text(f"❌ Error: {str(e)}")

def learn_command(update: Update, context):
    """Permite enseñar nuevos comandos al bot"""
    try:
        command = context.args[0].lower()
        response = " ".join(context.args[1:])

        # Guardar en base de conocimiento
        with open("knowledge_base.json", "a") as f:
            json.dump({"command": command, "response": response}, f)
            f.write("\n")

        update.message.reply_text(f"✅ Aprendido nuevo comando: /{command}")

    except Exception as e:
        update.message.reply_text("Formato: /aprender <comando> <respuesta>")

def execute_custom_command(update: Update, context):
    """Ejecuta comandos personalizados"""
    try:
        command = update.message.text[1:].lower()

        with open("knowledge_base.json", "r") as f:
            for line in f:
                data = json.loads(line)
                if data["command"] == command:
                    update.message.reply_text(data["response"])
                    return

        update.message.reply_text("❌ Comando desconocido. Use /aprender para enseñarme")

    except Exception as e:
        update.message.reply_text(f"Error: {str(e)}")

# ----------------- FUNCIONES AUXILIARES -----------------
def generate_ta_report(symbol):
    """Genera reporte técnico detallado"""
    # Implementar análisis técnico completo
    return "• Tendencia: Alcista\n• Momentum: Fuerte\n• Volumen: Creciente"

def generate_recommendation(predictions, ta, sentiment):
    """Genera recomendación integrada"""
    # Lógica compleja de recomendación
    if sentiment['positive'] > sentiment['negative'] * 1.5:
        return "✅ Fuerte recomendación de COMPRA (Fundamentales positivos)"
    else:
        return "⚠️ Neutral (Esperar confirmación técnica)"

def format_predictions(preds):
    """Formatea predicciones para visualización"""
    return "\n".join(f"Día {i+1}: ${p:.2f}" for i, p in enumerate(preds))

# ----------------- CONFIGURACIÓN MEJORADA -----------------
crypto_ai = SelfLearningCryptoAI()

def check_alerts(context: CallbackContext):
    """Verifica condiciones de mercado para alertas"""
    try:
        for symbol in ["BTC", "ETH"]:
            preds = crypto_ai.predict_price(symbol)
            current = get_price(symbol)

            # Aprendizaje automático
            crypto_ai.update_with_feedback(symbol, current, preds[-1])

            # Alertas inteligentes
            if abs(preds[-1] - current) > current * 0.05:
                context.bot.send_message(
                    chat_id=CANAL_ID,
                    text=f"⚠️ Alerta {symbol}: Gran discrepancia IA/mercado\n"
                         f"Predicho: ${preds[-1]:.2f} vs Actual: ${current:.2f}"
                )

        # Reentrenamiento periódico
        if datetime.now().hour == 3:  # 3 AM
            crypto_ai.retrain_models()

    except Exception as e:
        logger.error(f"Error en check_alerts: {str(e)}")

def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher

    # Comandos principales
    dp.add_handler(CommandHandler("prediccion", market_analysis))
    dp.add_handler(CommandHandler("analisis", market_analysis))
    dp.add_handler(CommandHandler("mercado", market_analysis))

    # Nuevos comandos autoaprendizaje
    dp.add_handler(CommandHandler("aprender", learn_command))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, execute_custom_command))

    # Configuración avanzada
    job_queue = updater.job_queue
    job_queue.run_repeating(check_alerts, interval=3600, first=0)
    job_queue.run_daily(
        lambda ctx: crypto_ai.retrain_models(),
        time=time(hour=3)  # 3 AM diario
    )

    updater.start_polling()
    logger.info("Bot con autoaprendizaje iniciado")
    updater.idle()

if __name__ == '__main__':
    main()



