import pandas as pd
import numpy as np
import ccxt
import time
import ta
import logging
from datetime import datetime
import telegram
from telegram.ext import Updater, CommandHandler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os

# Configuración del sistema de registro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crypto_trading_bot")

# Cargar variables de entorno
load_dotenv()

# Configuración de Telegram
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Configuración del exchange
EXCHANGE_ID = 'binance'  # Puedes cambiar a 'ftx', 'bybit', etc.
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Configuración del trading
SYMBOLS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAME = '1h'  # Timeframe para análisis
FAST_MA = 9  # Media móvil rápida
SLOW_MA = 21  # Media móvil lenta
RSI_PERIOD = 14  # Período RSI
RSI_OVERBOUGHT = 70  # Nivel de sobrecompra RSI
RSI_OVERSOLD = 30  # Nivel de sobreventa RSI
MACD_FAST = 12  # MACD rápido
MACD_SLOW = 26  # MACD lento
MACD_SIGNAL = 9  # Señal MACD

# Inicializar exchange
def init_exchange():
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Para usar futuros en lugar de spot
        }
    })
    return exchange

# Inicializar bot de Telegram
def init_telegram():
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    return bot

# Función para obtener datos históricos
def get_historical_data(exchange, symbol, timeframe, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error al obtener datos históricos para {symbol}: {e}")
        return None

# Aplicar indicadores técnicos al dataframe
def apply_technical_indicators(df):
    # Media móvil simple
    df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=FAST_MA)
    df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=SLOW_MA)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)
    
    # MACD
    macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()
    df['bollinger_mid'] = bollinger.bollinger_mavg()
    
    # Soporte y resistencia basados en máximos y mínimos recientes
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    
    return df

# Analizar señales de trading
def analyze_signals(df):
    signals = []
    
    # Obtener la fila más reciente para análisis
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # Señal por cruce de medias móviles
    if (prev_row['sma_fast'] <= prev_row['sma_slow']) and (last_row['sma_fast'] > last_row['sma_slow']):
        signals.append("LONG: Cruce alcista de medias móviles")
    elif (prev_row['sma_fast'] >= prev_row['sma_slow']) and (last_row['sma_fast'] < last_row['sma_slow']):
        signals.append("SHORT: Cruce bajista de medias móviles")
    
    # Señal por RSI
    if prev_row['rsi'] < RSI_OVERSOLD and last_row['rsi'] > RSI_OVERSOLD:
        signals.append(f"LONG: RSI saliendo de sobreventa ({last_row['rsi']:.2f})")
    elif prev_row['rsi'] > RSI_OVERBOUGHT and last_row['rsi'] < RSI_OVERBOUGHT:
        signals.append(f"SHORT: RSI saliendo de sobrecompra ({last_row['rsi']:.2f})")
    
    # Señal por MACD
    if (prev_row['macd'] <= prev_row['macd_signal']) and (last_row['macd'] > last_row['macd_signal']):
        signals.append("LONG: Cruce alcista MACD")
    elif (prev_row['macd'] >= prev_row['macd_signal']) and (last_row['macd'] < last_row['macd_signal']):
        signals.append("SHORT: Cruce bajista MACD")
    
    # Señal por Bollinger Bands
    if last_row['close'] < last_row['bollinger_low']:
        signals.append("LONG: Precio por debajo de banda inferior de Bollinger")
    elif last_row['close'] > last_row['bollinger_high']:
        signals.append("SHORT: Precio por encima de banda superior de Bollinger")
    
    # Señal por soporte/resistencia
    if (prev_row['close'] > last_row['support']) and (last_row['close'] < last_row['support']):
        signals.append("SHORT: Ruptura de soporte")
    elif (prev_row['close'] < last_row['resistance']) and (last_row['close'] > last_row['resistance']):
        signals.append("LONG: Ruptura de resistencia")
    
    return signals

# Generar gráfico para análisis visual
def generate_chart(df, symbol, signals=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Configuración de estilo
    plt.style.use('seaborn-darkgrid')
    
    # Gráfico de precios y medias móviles
    axes[0].plot(df.index, df['close'], label='Precio', color='blue', linewidth=1)
    axes[0].plot(df.index, df['sma_fast'], label=f'SMA {FAST_MA}', color='red', linewidth=0.7)
    axes[0].plot(df.index, df['sma_slow'], label=f'SMA {SLOW_MA}', color='green', linewidth=0.7)
    axes[0].plot(df.index, df['bollinger_high'], label='BB Superior', color='gray', linestyle='--', linewidth=0.5)
    axes[0].plot(df.index, df['bollinger_mid'], label='BB Medio', color='gray', linestyle='-', linewidth=0.5)
    axes[0].plot(df.index, df['bollinger_low'], label='BB Inferior', color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_title(f'Análisis Técnico de {symbol}')
    axes[0].set_ylabel('Precio')
    axes[0].legend(loc='upper left')
    
    # RSI
    axes[1].plot(df.index, df['rsi'], color='purple', linewidth=1)
    axes[1].axhline(y=RSI_OVERBOUGHT, color='red', linestyle='--', linewidth=0.5)
    axes[1].axhline(y=RSI_OVERSOLD, color='green', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim(0, 100)
    
    # MACD
    axes[2].plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1)
    axes[2].plot(df.index, df['macd_signal'], label='Señal', color='red', linewidth=1)
    axes[2].bar(df.index, df['macd_histogram'], label='Histograma', color='green', width=0.02)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('MACD')
    axes[2].legend(loc='upper left')
    
    # Formato de fecha
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Añadir señales al gráfico si hay
    if signals:
        for signal in signals:
            direction = "LONG" if "LONG" in signal else "SHORT"
            color = "green" if direction == "LONG" else "red"
            axes[0].axvline(x=df.index[-1], color=color, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    chart_path = f"chart_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# Enviar notificación por Telegram
def send_telegram_notification(bot, symbol, signals, chart_path=None):
    if not signals:
        return
    
    message = f"🚨 *Alerta de Trading para {symbol}* 🚨\n\n"
    message += f"*Fecha:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    message += "*Señales Detectadas:*\n"
    
    for signal in signals:
        if "LONG" in signal:
            message += f"🟢 {signal}\n"
        else:
            message += f"🔴 {signal}\n"
    
    message += "\n*Nota:* Esta es una señal automática. Realiza tu propio análisis antes de operar."
    
    try:
        if chart_path:
            with open(chart_path, 'rb') as chart:
                bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=chart, caption=message, parse_mode=telegram.ParseMode.MARKDOWN)
        else:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=telegram.ParseMode.MARKDOWN)
        logger.info(f"Notificación enviada para {symbol}")
    except Exception as e:
        logger.error(f"Error al enviar notificación: {e}")

# Función principal
def main():
    logger.info("Iniciando bot de trading de criptomonedas...")
    
    exchange = init_exchange()
    #telegram_bot = init_telegram()
    
    try:
        # Enviar mensaje de inicio
        # telegram_bot.send_message(
        #     chat_id=TELEGRAM_CHAT_ID,
        #     text="🤖 *Bot de Trading de Criptomonedas Iniciado* 🤖\n\nVigilando: " + ", ".join(SYMBOLS),
        #     parse_mode=telegram.ParseMode.MARKDOWN
        # )
        
        while True:
            for symbol in SYMBOLS:
                logger.info(f"Analizando {symbol}...")
                
                # Obtener datos
                df = get_historical_data(exchange, symbol, TIMEFRAME)
                if df is None:
                    continue
                
                # Aplicar indicadores técnicos
                df = apply_technical_indicators(df)
                
                # Analizar señales
                signals = analyze_signals(df)
                
                if signals:
                    logger.info(f"Señales detectadas para {symbol}: {signals}")
                    
                    # Generar gráfico
                    chart_path = generate_chart(df, symbol, signals)
                    
                    # Enviar notificación
                    print("-----------------------------")
                    print(symbol, signals, chart_path)
                    print("-----------------------------")
                    #send_telegram_notification(telegram_bot, symbol, signals, chart_path)
                else:
                    logger.info(f"No hay señales para {symbol} en este momento")
            
            # Esperar antes del próximo análisis
            logger.info(f"Esperando {60} segundos antes del próximo análisis...")
            time.sleep(60)  # Ajustar según necesidad
            
    except KeyboardInterrupt:
        logger.info("Bot detenido manualmente")
    except Exception as e:
        logger.error(f"Error en el bot: {e}")
        # telegram_bot.send_message(
        #     chat_id=TELEGRAM_CHAT_ID,
        #     text=f"❌ *Error en el Bot* ❌\n\n{str(e)}",
        #     parse_mode=telegram.ParseMode.MARKDOWN
        # )

if __name__ == "__main__":
    main()