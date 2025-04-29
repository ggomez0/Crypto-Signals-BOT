import pandas as pd
import numpy as np
import ccxt
import time
import ta
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os
import colorama
from colorama import Fore, Back, Style
import requests  # For Telegram integration

# Inicializar colorama para colores en terminal
colorama.init(autoreset=True)

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

# Configuración del exchange
EXCHANGE_ID = 'binance'
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Configuración de Telegram
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_ENABLED = TELEGRAM_TOKEN is not None and TELEGRAM_CHAT_ID is not None

# Configuración del trading
SYMBOLS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAME = '15m'
FAST_MA = 9
SLOW_MA = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # Umbral para considerar una tendencia fuerte
VOLUME_THRESHOLD = 1.5  # Multiplicador para considerar aumento de volumen significativo

# Variables para backtesting
BACKTEST_MODE = True  # Activar/desactivar modo backtesting
BACKTEST_START_DATE = '2025-03-01'
BACKTEST_END_DATE = '2025-04-27'
BACKTEST_RESULTS = []  # Para almacenar resultados del backtesting

# Inicializar exchange
def init_exchange():
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    return exchange

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
    
    # ADX - Average Directional Index para medir fuerza de tendencia
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=ADX_PERIOD)
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()  # Indicador Direccional Positivo
    df['di_minus'] = adx.adx_neg()  # Indicador Direccional Negativo
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_high'] = bollinger.bollinger_hband()
    df['bollinger_low'] = bollinger.bollinger_lband()
    df['bollinger_mid'] = bollinger.bollinger_mavg()
    
    # NUEVO: Análisis de Volumen
    df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
    # Detectar tendencia de volumen decreciente (3 periodos consecutivos)
    df['volume_decreasing_trend'] = (
        (df['volume'] < df['volume'].shift(1)) & 
        (df['volume'].shift(1) < df['volume'].shift(2)) & 
        (df['volume'].shift(2) < df['volume'].shift(3))
    )
    
    # NUEVO: Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(
        high=df['high'],
        low=df['low'],
        window1=9,   # Tenkan-sen (Conversion Line)
        window2=26,  # Kijun-sen (Base Line)
        window3=52   # Senkou Span B (Leading Span B)
    )
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    df['ichimoku_a'] = ichimoku.ichimoku_a()  # Senkou Span A (Leading Span A)
    df['ichimoku_b'] = ichimoku.ichimoku_b()  # Senkou Span B (Leading Span B)
    # Calcular si el precio está por encima o por debajo de la nube
    df['above_cloud'] = (df['close'] > df['ichimoku_a'].shift(26)) & (df['close'] > df['ichimoku_b'].shift(26))
    df['below_cloud'] = (df['close'] < df['ichimoku_a'].shift(26)) & (df['close'] < df['ichimoku_b'].shift(26))

    df['in_cloud'] = ~(df['above_cloud'] | df['below_cloud'])
    # Calcular el estado del Kumo futuro (nube adelantada)
    df['future_cloud_bullish'] = df['ichimoku_a'] > df['ichimoku_b']
    df['future_cloud_bearish'] = df['ichimoku_a'] < df['ichimoku_b']
    
    # Soportes y resistencias más avanzados
    # Método 1: Basado en mínimos y máximos locales
    window = 20  # Ventana para identificar picos locales
    
    # Picos para resistencias (máximos locales)
    df['is_resistance'] = df['high'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] == max(x), raw=True
    ).fillna(0).astype(bool)
    
    # Valles para soportes (mínimos locales)
    df['is_support'] = df['low'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] == min(x), raw=True
    ).fillna(0).astype(bool)
    
    # Método 2: Niveles de soporte y resistencia basados en cuartiles de precio
    price_range = df['high'].max() - df['low'].min()
    q1 = df['low'].min() + price_range * 0.25
    q3 = df['low'].min() + price_range * 0.75
    df['support_level'] = q1
    df['resistance_level'] = q3
    
    # NUEVO: Soportes y resistencias dinámicas
    # Detectar última resistencia superada
    df['broke_resistance'] = (df['close'] > df['close'].shift(1)) & df['is_resistance'].shift(1) & (df['close'] > df['high'].shift(1))
    # Detectar último soporte roto
    df['broke_support'] = (df['close'] < df['close'].shift(1)) & df['is_support'].shift(1) & (df['close'] < df['low'].shift(1))
    
    # Marcar últimos puntos donde ocurrieron estos eventos
    df['last_broke_resistance'] = None
    df['last_broke_support'] = None
    
    # Encontrar y marcar la última resistencia superada
    resistance_indices = df.index[df['broke_resistance']]
    if len(resistance_indices) > 0:
        last_resistance_idx = resistance_indices[-1]
        df.loc[last_resistance_idx:, 'last_broke_resistance'] = df.loc[last_resistance_idx, 'high']
    
    # Encontrar y marcar el último soporte roto
    support_indices = df.index[df['broke_support']]
    if len(support_indices) > 0:
        last_support_idx = support_indices[-1]
        df.loc[last_support_idx:, 'last_broke_support'] = df.loc[last_support_idx, 'low']
    
    # Tendencia
    df['trend'] = np.where(df['close'] > df['sma_slow'], 'BULLISH',
                         np.where(df['close'] < df['sma_slow'], 'BEARISH', 'NEUTRAL'))
    
    return df

# Identificar zonas de soporte y resistencia importantes
def identify_key_levels(df):
    # Encontrar soportes recientes
    recent_df = df.iloc[-100:]  # Analizar últimas 100 velas
    supports = []
    resistances = []
    
    # Identificar niveles de soporte recientes
    for i in range(5, len(recent_df)-5):
        if recent_df['is_support'].iloc[i]:
            support_price = recent_df['low'].iloc[i]
            # Verificar si este nivel ha sido testeado múltiples veces
            touches = sum(abs(recent_df['low'] - support_price) < support_price * 0.005)
            if touches >= 2:  # Al menos 2 toques para considerarlo importante
                supports.append((support_price, touches))
    
    # Identificar niveles de resistencia recientes
    for i in range(5, len(recent_df)-5):
        if recent_df['is_resistance'].iloc[i]:
            resistance_price = recent_df['high'].iloc[i]
            # Verificar si este nivel ha sido testeado múltiples veces
            touches = sum(abs(recent_df['high'] - resistance_price) < resistance_price * 0.005)
            if touches >= 2:  # Al menos 2 toques para considerarlo importante
                resistances.append((resistance_price, touches))
    
    # NUEVO: Añadir soporte y resistencia dinámicos
    last_row = recent_df.iloc[-1]
    
    # Añadir última resistencia superada como nuevo soporte
    if last_row['last_broke_resistance'] is not None and not np.isnan(last_row['last_broke_resistance']):
        resistance_price = last_row['last_broke_resistance']
        # Verificar que no esté ya en la lista
        if not any(abs(s[0] - resistance_price) < resistance_price * 0.005 for s in supports):
            supports.append((resistance_price, 1))
    
    # Añadir último soporte roto como nueva resistencia
    if last_row['last_broke_support'] is not None and not np.isnan(last_row['last_broke_support']):
        support_price = last_row['last_broke_support']
        # Verificar que no esté ya en la lista
        if not any(abs(r[0] - support_price) < support_price * 0.005 for r in resistances):
            resistances.append((support_price, 1))
    
    # Ordenar y filtrar para obtener los más importantes
    if supports:
        supports.sort(key=lambda x: x[1], reverse=True)
        supports = supports[:3]  # Top 3 soportes más importantes
    
    if resistances:
        resistances.sort(key=lambda x: x[1], reverse=True)
        resistances = resistances[:3]  # Top 3 resistencias más importantes
    
    return supports, resistances

# Analizar señales de trading con explicación detallada
def analyze_signals(df, key_levels=None):
    signals = []
    explanations = []
    current_price = df['close'].iloc[-1]
    
    # Obtener la fila más reciente para análisis
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # Señal por cruce de medias móviles
    if (prev_row['sma_fast'] <= prev_row['sma_slow']) and (last_row['sma_fast'] > last_row['sma_slow']):
        signals.append("LONG")
        explanations.append(f"Cruce alcista de medias móviles (SMA{FAST_MA} cruza por encima de SMA{SLOW_MA})")
    elif (prev_row['sma_fast'] >= prev_row['sma_slow']) and (last_row['sma_fast'] < last_row['sma_slow']):
        signals.append("SHORT")
        explanations.append(f"Cruce bajista de medias móviles (SMA{FAST_MA} cruza por debajo de SMA{SLOW_MA})")
    
    # Señal por RSI
    if prev_row['rsi'] < RSI_OVERSOLD and last_row['rsi'] > RSI_OVERSOLD:
        signals.append("LONG")
        explanations.append(f"RSI saliendo de zona de sobreventa ({last_row['rsi']:.2f})")
    elif prev_row['rsi'] > RSI_OVERBOUGHT and last_row['rsi'] < RSI_OVERBOUGHT:
        signals.append("SHORT")
        explanations.append(f"RSI saliendo de zona de sobrecompra ({last_row['rsi']:.2f})")
    
    # Señal por MACD
    if (prev_row['macd'] <= prev_row['macd_signal']) and (last_row['macd'] > last_row['macd_signal']):
        signals.append("LONG")
        explanations.append("Cruce alcista MACD (MACD cruza por encima de línea de señal)")
    elif (prev_row['macd'] >= prev_row['macd_signal']) and (last_row['macd'] < last_row['macd_signal']):
        signals.append("SHORT") 
        explanations.append("Cruce bajista MACD (MACD cruza por debajo de línea de señal)")
    
    # Señal por ADX y Direccional
    if last_row['adx'] > ADX_THRESHOLD:
        if last_row['di_plus'] > last_row['di_minus']:
            signals.append("LONG")
            explanations.append(f"ADX fuerte ({last_row['adx']:.2f}) con tendencia alcista (DI+ > DI-)")
        elif last_row['di_minus'] > last_row['di_plus']:
            signals.append("SHORT")
            explanations.append(f"ADX fuerte ({last_row['adx']:.2f}) con tendencia bajista (DI- > DI+)")
    
    # Señal por Bollinger Bands
    if last_row['close'] < last_row['bollinger_low']:
        signals.append("LONG")
        explanations.append("Precio por debajo de banda inferior de Bollinger (posible sobreventa)")
    elif last_row['close'] > last_row['bollinger_high']:
        signals.append("SHORT")
        explanations.append("Precio por encima de banda superior de Bollinger (posible sobrecompra)")
    
    # NUEVO: Señal por volumen
    if last_row['volume_ratio'] > VOLUME_THRESHOLD:
        # Confirmar señales si hay alto volumen
        if "LONG" in signals:
            explanations.append(f"Alto volumen ({last_row['volume_ratio']:.2f}x promedio) confirma señal de compra")
        elif "SHORT" in signals:
            explanations.append(f"Alto volumen ({last_row['volume_ratio']:.2f}x promedio) confirma señal de venta")
        # O generar nueva señal basada en ruptura de precio con alto volumen
        elif last_row['close'] > last_row['close'].shift(1) * 1.01:  # Subida de más del 1%
            signals.append("LONG")
            explanations.append(f"Ruptura alcista con volumen alto ({last_row['volume_ratio']:.2f}x promedio)")
        elif last_row['close'] < last_row['close'].shift(1) * 0.99:  # Bajada de más del 1%
            signals.append("SHORT")
            explanations.append(f"Ruptura bajista con volumen alto ({last_row['volume_ratio']:.2f}x promedio)")
    
    # NUEVO: Alerta de agotamiento por volumen decreciente
    if last_row['volume_decreasing_trend']:
        if last_row['trend'] == 'BULLISH':
            signals.append("SHORT")
            explanations.append("Volumen decreciente en tendencia alcista (posible agotamiento)")
        elif last_row['trend'] == 'BEARISH':
            signals.append("LONG")
            explanations.append("Volumen decreciente en tendencia bajista (posible agotamiento)")
    
    # NUEVO: Señales basadas en Ichimoku Cloud
    if last_row['above_cloud']:
        if last_row['ichimoku_conversion_line'] > last_row['ichimoku_base_line']:
            signals.append("LONG")
            explanations.append("Precio por encima de la nube Ichimoku con TK Cross alcista")
    elif last_row['below_cloud']:
        if last_row['ichimoku_conversion_line'] < last_row['ichimoku_base_line']:
            signals.append("SHORT")
            explanations.append("Precio por debajo de la nube Ichimoku con TK Cross bajista")
    
    # NUEVO: Señal basada en Kumo futuro
    if last_row['future_cloud_bullish'] and not prev_row['future_cloud_bullish']:
        signals.append("LONG")
        explanations.append("Kumo futuro se torna alcista (Senkou Span A cruza por encima de Senkou Span B)")
    elif last_row['future_cloud_bearish'] and not prev_row['future_cloud_bearish']:
        signals.append("SHORT")
        explanations.append("Kumo futuro se torna bajista (Senkou Span A cruza por debajo de Senkou Span B)")
    
    # Señales basadas en soportes y resistencias
    if key_levels:
        supports, resistances = key_levels
        
        # Comprobar si el precio está cerca de soporte
        for support_price, touches in supports:
            if 0.99 * support_price <= current_price <= 1.01 * support_price:
                if last_row['trend'] == 'BULLISH' or last_row['macd_histogram'] > 0:
                    signals.append("LONG")
                    explanations.append(f"Precio en SOPORTE fuerte (testado {touches} veces): {support_price:.2f}")
        
        # Comprobar si el precio está cerca de resistencia
        for resistance_price, touches in resistances:
            if 0.99 * resistance_price <= current_price <= 1.01 * resistance_price:
                if last_row['trend'] == 'BEARISH' or last_row['macd_histogram'] < 0:
                    signals.append("SHORT")
                    explanations.append(f"Precio en RESISTENCIA fuerte (testada {touches} veces): {resistance_price:.2f}")
    
    return signals, explanations

# NUEVO: Función para enviar alertas a Telegram
def send_telegram_alert(symbol, signal_type, price, explanations):
    if not TELEGRAM_ENABLED:
        return False
    
    try:
        # Determinar emoji según tipo de señal
        if signal_type == "LONG":
            emoji = "🚀"
            color = "🟢"
        elif signal_type == "SHORT":
            emoji = "📉"
            color = "🔴"
        else:
            emoji = "⚠️"
            color = "🟡"
        
        # Construir mensaje
        message = f"{emoji} *ALERTA DE TRADING* {emoji}\n\n"
        message += f"{color} *{signal_type}* en *{symbol}* a *{price:.2f} USD*\n\n"
        message += f"*Señales detectadas:*\n"
        
        for explanation in explanations:
            message += f"• {explanation}\n"
        
        message += f"\n*Fecha y hora:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Enviar mensaje
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            logger.info(f"Alerta enviada a Telegram: {symbol} {signal_type}")
            return True
        else:
            logger.error(f"Error al enviar alerta a Telegram: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error en envío de alerta a Telegram: {e}")
        return False

# Generar análisis detallado para terminal
def generate_terminal_analysis(symbol, df, signals, explanations, key_levels=None):
    current_price = df['close'].iloc[-1]
    last_row = df.iloc[-1]
    
    # Encabezado con información general
    print("\n" + "="*80)
    if "LONG" in signals:
        print(f"{Fore.GREEN}{Style.BRIGHT}🚨 ALERTA: {symbol} - SEÑALES DE COMPRA (LONG) DETECTADAS 🚨{Style.RESET_ALL}")
        # NUEVO: Enviar alerta a Telegram
        if TELEGRAM_ENABLED:
            send_telegram_alert(symbol, "LONG", current_price, explanations)
    elif "SHORT" in signals:
        print(f"{Fore.RED}{Style.BRIGHT}🚨 ALERTA: {symbol} - SEÑALES DE VENTA (SHORT) DETECTADAS 🚨{Style.RESET_ALL}")
        # NUEVO: Enviar alerta a Telegram
        if TELEGRAM_ENABLED:
            send_telegram_alert(symbol, "SHORT", current_price, explanations)
    else:
        print(f"{Fore.YELLOW}{Style.BRIGHT}ℹ️ ANÁLISIS: {symbol} - SIN SEÑALES CLARAS{Style.RESET_ALL}")
    
    print("="*80)
    print(f"📊 Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💵 Precio actual: {Fore.CYAN}{current_price:.2f} USD{Style.RESET_ALL}")
    print(f"⏱️ Timeframe: {TIMEFRAME}")
    print("-"*80)
    
    # Análisis técnico
    print(f"{Fore.YELLOW}{Style.BRIGHT}📈 ANÁLISIS TÉCNICO:{Style.RESET_ALL}")
    print(f"• Tendencia: {Fore.GREEN if last_row['trend'] == 'BULLISH' else Fore.RED if last_row['trend'] == 'BEARISH' else Fore.YELLOW}{last_row['trend']}{Style.RESET_ALL}")
    print(f"• RSI ({RSI_PERIOD}): {Fore.GREEN if last_row['rsi'] < 50 else Fore.RED}{last_row['rsi']:.2f}{Style.RESET_ALL} {'(Sobreventa)' if last_row['rsi'] < 30 else '(Sobrecompra)' if last_row['rsi'] > 70 else ''}")
    print(f"• MACD: {Fore.GREEN if last_row['macd'] > last_row['macd_signal'] else Fore.RED}{last_row['macd']:.6f}{Style.RESET_ALL} (Señal: {last_row['macd_signal']:.6f})")
    print(f"• Histograma MACD: {Fore.GREEN if last_row['macd_histogram'] > 0 else Fore.RED}{last_row['macd_histogram']:.6f}{Style.RESET_ALL}")
    print(f"• ADX ({ADX_PERIOD}): {Fore.GREEN if last_row['adx'] > ADX_THRESHOLD else Fore.YELLOW}{last_row['adx']:.2f}{Style.RESET_ALL} {'(Tendencia fuerte)' if last_row['adx'] > ADX_THRESHOLD else '(Tendencia débil)'}")
    print(f"• DI+ / DI-: {Fore.GREEN}{last_row['di_plus']:.2f}{Style.RESET_ALL} / {Fore.RED}{last_row['di_minus']:.2f}{Style.RESET_ALL}")
    print(f"• SMA{FAST_MA}: {last_row['sma_fast']:.2f}")
    print(f"• SMA{SLOW_MA}: {last_row['sma_slow']:.2f}")
    print(f"• Bandas de Bollinger: {last_row['bollinger_low']:.2f} - {last_row['bollinger_mid']:.2f} - {last_row['bollinger_high']:.2f}")
    
    # NUEVO: Análisis de volumen
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}📊 ANÁLISIS DE VOLUMEN:{Style.RESET_ALL}")
    print(f"• Volumen: {last_row['volume']:.2f}")
    print(f"• Media 20 periodos: {last_row['volume_sma']:.2f}")
    print(f"• Ratio Volumen/Media: {Fore.GREEN if last_row['volume_ratio'] > 1 else Fore.RED}{last_row['volume_ratio']:.2f}x{Style.RESET_ALL}")
    if last_row['volume_ratio'] > VOLUME_THRESHOLD:
        print(f"{Fore.GREEN}• Volumen significativamente alto (>{VOLUME_THRESHOLD}x promedio){Style.RESET_ALL}")
    if last_row['volume_decreasing_trend']:
        print(f"{Fore.RED}• Alerta: Volumen decreciente en últimos 3 periodos (posible agotamiento){Style.RESET_ALL}")
    
    # NUEVO: Análisis Ichimoku
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}☁️ ICHIMOKU CLOUD:{Style.RESET_ALL}")
    cloud_status = "Por ENCIMA" if last_row['above_cloud'] else "Por DEBAJO" if last_row['below_cloud'] else "DENTRO"
    print(f"• Posición del precio: {Fore.GREEN if last_row['above_cloud'] else Fore.RED if last_row['below_cloud'] else Fore.YELLOW}{cloud_status} de la nube{Style.RESET_ALL}")
    print(f"• Tenkan-sen (9): {last_row['ichimoku_conversion_line']:.2f}")
    print(f"• Kijun-sen (26): {last_row['ichimoku_base_line']:.2f}")
    print(f"• Senkou Span A: {last_row['ichimoku_a']:.2f}")
    print(f"• Senkou Span B: {last_row['ichimoku_b']:.2f}")
    print(f"• Kumo futuro: {Fore.GREEN if last_row['future_cloud_bullish'] else Fore.RED}{'ALCISTA (A>B)' if last_row['future_cloud_bullish'] else 'BAJISTA (A<B)'}{Style.RESET_ALL}")
    
    # Soportes y resistencias
    if key_levels:
        supports, resistances = key_levels
        print("-"*80)
        print(f"{Fore.YELLOW}{Style.BRIGHT}🧱 SOPORTES Y RESISTENCIAS:{Style.RESET_ALL}")
        
        if supports:
            print(f"{Fore.GREEN}Soportes importantes:")
            for i, (price, touches) in enumerate(supports, 1):
                print(f"  {i}. {price:.2f} USD (testado {touches} veces)")
        else:
            print(f"{Fore.GREEN}No se identificaron soportes importantes")
            
        if resistances:
            print(f"{Fore.RED}Resistencias importantes:")
            for i, (price, touches) in enumerate(resistances, 1):
                print(f"  {i}. {price:.2f} USD (testado {touches} veces)")
        else:
            print(f"{Fore.RED}No se identificaron resistencias importantes")
        
        # NUEVO: Mostrar soportes y resistencias dinámicos
        print(f"\n{Fore.YELLOW}Soportes y resistencias dinámicos:{Style.RESET_ALL}")
        if last_row['last_broke_resistance'] is not None and not np.isnan(last_row['last_broke_resistance']):
            print(f"{Fore.GREEN}• Última resistencia superada (nuevo soporte): {last_row['last_broke_resistance']:.2f}{Style.RESET_ALL}")
        if last_row['last_broke_support'] is not None and not np.isnan(last_row['last_broke_resistance']):
            print(f"{Fore.GREEN}• Última resistencia superada (nuevo soporte): {last_row['last_broke_resistance']:.2f}{Style.RESET_ALL}")
        if last_row['last_broke_support'] is not None and not np.isnan(last_row['last_broke_support']):
            print(f"{Fore.RED}• Último soporte roto (nueva resistencia): {last_row['last_broke_support']:.2f}{Style.RESET_ALL}")
    
    # Señales y explicaciones
    if explanations:
        print("-"*80)
        print(f"{Fore.YELLOW}{Style.BRIGHT}🔍 ANÁLISIS DE SEÑALES:{Style.RESET_ALL}")
        for i, explanation in enumerate(explanations, 1):
            if "LONG" in signals and any(keyword in explanation.upper() for keyword in ["ALCISTA", "LONG", "SOPORTE", "SOBREVENTA"]):
                print(f"{Fore.GREEN}✅ {explanation}")
            elif "SHORT" in signals and any(keyword in explanation.upper() for keyword in ["BAJISTA", "SHORT", "RESISTENCIA", "SOBRECOMPRA"]):
                print(f"{Fore.RED}❌ {explanation}")
            else:
                print(f"ℹ️ {explanation}")
    
    # Recomendación final
    print("-"*80)
    print(f"{Fore.YELLOW}{Style.BRIGHT}📝 RECOMENDACIÓN:{Style.RESET_ALL}")
    
    long_count = sum(1 for s in signals if s == "LONG")
    short_count = sum(1 for s in signals if s == "SHORT")
    
    if long_count > short_count and long_count >= 2:
        print(f"{Fore.GREEN}{Style.BRIGHT}LONG (COMPRA) ⬆️ - {long_count} señales alcistas detectadas{Style.RESET_ALL}")
    elif short_count > long_count and short_count >= 2:
        print(f"{Fore.RED}{Style.BRIGHT}SHORT (VENTA) ⬇️ - {short_count} señales bajistas detectadas{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}NEUTRAL - Esperar mejor configuración{Style.RESET_ALL}")
    
    print("="*80)
    
    # NUEVO: Guardar señal para backtesting
    if BACKTEST_MODE and (long_count >= 2 or short_count >= 2):
        signal_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'price': current_price,
            'signal': 'LONG' if long_count > short_count else 'SHORT',
            'signal_count': max(long_count, short_count),
            'indicators': explanations
        }
        BACKTEST_RESULTS.append(signal_record)

# NUEVO: Función para realizar backtesting
def run_backtest(exchange, symbol, timeframe, start_date, end_date):
    print(f"{Fore.CYAN}{Style.BRIGHT}Ejecutando backtesting para {symbol} desde {start_date} hasta {end_date}...{Style.RESET_ALL}")
    
    try:
        # Convertir fechas a timestamps
        start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000)
        
        # Obtener datos históricos para el período completo
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp, limit=1000)
        all_data = []
        
        # Si necesitamos más datos, hacemos múltiples solicitudes
        while ohlcv and ohlcv[-1][0] < end_timestamp:
            all_data.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=last_timestamp + 1, limit=1000)
        
        # Filtrar sólo los datos dentro del rango
        filtered_data = [candle for candle in all_data if start_timestamp <= candle[0] <= end_timestamp]
        
        # Convertir a DataFrame
        df = pd.DataFrame(filtered_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        results = []
        signals_count = {'LONG': 0, 'SHORT': 0}
        successful_signals = {'LONG': 0, 'SHORT': 0}
        
        # Procesar cada punto de tiempo como si fuera "ahora"
        for i in range(100, len(df) - 20):  # Empezamos después de suficientes datos para indicadores y dejamos margen para evaluar
            current_df = df.iloc[:i+1].copy()
            
            # Calcular indicadores
            current_df = apply_technical_indicators(current_df)
            
            # Identificar soportes/resistencias
            key_levels = identify_key_levels(current_df)
            
            # Obtener señales
            signals, explanations = analyze_signals(current_df, key_levels)
            
            # Contar señales
            long_count = sum(1 for s in signals if s == "LONG")
            short_count = sum(1 for s in signals if s == "SHORT")
            
            # Si tenemos una señal clara
            if long_count >= 2 and long_count > short_count:
                signal = "LONG"
                signals_count['LONG'] += 1
                entry_price = current_df['close'].iloc[-1]
                
                # Evaluar resultado (miramos 10 velas adelante)
                if i + 10 < len(df):
                    future_prices = df['close'].iloc[i+1:i+11]
                    max_price = future_prices.max()
                    # Si el precio subió al menos un 1%
                    if max_price >= entry_price * 1.01:
                        successful_signals['LONG'] += 1
                        success = True
                    else:
                        success = False
                    
                    results.append({
                        'date': current_df.index[-1],
                        'signal': signal,
                        'entry_price': entry_price,
                        'max_future_price': max_price,
                        'profit_potential': ((max_price - entry_price) / entry_price) * 100,
                        'success': success
                    })
            
            elif short_count >= 2 and short_count > long_count:
                signal = "SHORT"
                signals_count['SHORT'] += 1
                entry_price = current_df['close'].iloc[-1]
                
                # Evaluar resultado (miramos 10 velas adelante)
                if i + 10 < len(df):
                    future_prices = df['close'].iloc[i+1:i+11]
                    min_price = future_prices.min()
                    # Si el precio bajó al menos un 1%
                    if min_price <= entry_price * 0.99:
                        successful_signals['SHORT'] += 1
                        success = True
                    else:
                        success = False
                    
                    results.append({
                        'date': current_df.index[-1],
                        'signal': signal,
                        'entry_price': entry_price,
                        'min_future_price': min_price,
                        'profit_potential': ((entry_price - min_price) / entry_price) * 100,
                        'success': success
                    })
        
        # Convertir resultados a DataFrame
        results_df = pd.DataFrame(results)
        
        # Calcular estadísticas
        if not results_df.empty:
            total_signals = signals_count['LONG'] + signals_count['SHORT']
            total_successful = successful_signals['LONG'] + successful_signals['SHORT']
            
            success_rate = (total_successful / total_signals) * 100 if total_signals > 0 else 0
            
            long_success_rate = (successful_signals['LONG'] / signals_count['LONG']) * 100 if signals_count['LONG'] > 0 else 0
            short_success_rate = (successful_signals['SHORT'] / signals_count['SHORT']) * 100 if signals_count['SHORT'] > 0 else 0
            
            avg_profit = results_df['profit_potential'].mean() if 'profit_potential' in results_df.columns else 0
            
            # Mostrar resultados
            print("\n" + "="*80)
            print(f"{Fore.CYAN}{Style.BRIGHT}RESULTADOS DEL BACKTESTING PARA {symbol}{Style.RESET_ALL}")
            print("="*80)
            print(f"Período: {start_date} a {end_date}")
            print(f"Timeframe: {timeframe}")
            print(f"Total señales generadas: {total_signals}")
            print(f"  - Señales LONG: {signals_count['LONG']}")
            print(f"  - Señales SHORT: {signals_count['SHORT']}")
            print("-"*80)
            print(f"{Fore.GREEN}Tasa de éxito global: {success_rate:.2f}%{Style.RESET_ALL}")
            print(f"  - Tasa éxito LONG: {long_success_rate:.2f}%")
            print(f"  - Tasa éxito SHORT: {short_success_rate:.2f}%")
            print(f"Potencial de beneficio promedio: {avg_profit:.2f}%")
            print("="*80)
            
            return results_df
        else:
            print(f"{Fore.YELLOW}No se generaron señales durante el período de backtesting{Style.RESET_ALL}")
            return None
            
    except Exception as e:
        print(f"{Fore.RED}Error durante el backtesting: {e}{Style.RESET_ALL}")
        return None

# NUEVO: Función para graficar resultados
def plot_results(symbol, df, signals=None):
    plt.figure(figsize=(14, 10))
    
    # Configurar la figura con subplots
    gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # Subplot para precio y medias móviles
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f'Análisis Técnico de {symbol}', fontsize=14)
    
    # Graficar precio
    ax1.plot(df.index, df['close'], label='Precio', color='black', linewidth=1.5)
    
    # Graficar medias móviles
    ax1.plot(df.index, df['sma_fast'], label=f'SMA {FAST_MA}', color='blue', linewidth=1)
    ax1.plot(df.index, df['sma_slow'], label=f'SMA {SLOW_MA}', color='red', linewidth=1)
    
    # Graficar Bandas de Bollinger
    ax1.plot(df.index, df['bollinger_high'], 'g--', linewidth=0.8, alpha=0.7)
    ax1.plot(df.index, df['bollinger_mid'], 'g-', linewidth=0.8, alpha=0.7)
    ax1.plot(df.index, df['bollinger_low'], 'g--', linewidth=0.8, alpha=0.7)
    ax1.fill_between(df.index, df['bollinger_high'], df['bollinger_low'], color='gray', alpha=0.1, label='Bollinger Bands')
    
    # Graficar Ichimoku Cloud
    ax1.plot(df.index, df['ichimoku_conversion_line'], color='blue', linewidth=0.8, label='Tenkan-sen')
    ax1.plot(df.index, df['ichimoku_base_line'], color='red', linewidth=0.8, label='Kijun-sen')
    
    # Graficar la nube
    cloud_df = df[['ichimoku_a', 'ichimoku_b']].shift(26)  # Nube actual (26 períodos adelante)
    ax1.fill_between(df.index, cloud_df['ichimoku_a'], cloud_df['ichimoku_b'],
                    where=cloud_df['ichimoku_a'] >= cloud_df['ichimoku_b'],
                    color='green', alpha=0.1)
    ax1.fill_between(df.index, cloud_df['ichimoku_a'], cloud_df['ichimoku_b'],
                    where=cloud_df['ichimoku_a'] < cloud_df['ichimoku_b'],
                    color='red', alpha=0.1)
    
    # Graficar señales si se proporcionan
    if signals is not None:
        for signal in signals:
            if signal['signal'] == 'LONG':
                ax1.scatter(signal['date'], signal['entry_price'], marker='^', color='green', s=100, label='Señal LONG' if 'Long' not in ax1.get_legend_handles_labels()[1] else "")
            elif signal['signal'] == 'SHORT':
                ax1.scatter(signal['date'], signal['entry_price'], marker='v', color='red', s=100, label='Señal SHORT' if 'Short' not in ax1.get_legend_handles_labels()[1] else "")
    
    # Configurar leyenda y grid
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Precio', fontsize=12)
    
    # Subplot para RSI
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=1)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, df['rsi'], 70, where=(df['rsi']>=70), color='r', alpha=0.3)
    ax2.fill_between(df.index, df['rsi'], 30, where=(df['rsi']<=30), color='g', alpha=0.3)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 100)
    
    # Subplot para MACD
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1)
    ax3.plot(df.index, df['macd_signal'], label='Señal', color='red', linewidth=1)
    ax3.bar(df.index, df['macd_histogram'], label='Histograma', color=np.where(df['macd_histogram'] >= 0, 'green', 'red'), alpha=0.5)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('MACD', fontsize=12)
    ax3.legend(loc='upper left')
    
    # Subplot para Volumen
    ax4 = plt.subplot(gs[3], sharex=ax1)
    volumen_colors = np.where(df['volume'] > df['volume'].shift(1), 'green', 'red')
    ax4.bar(df.index, df['volume'], color=volumen_colors, label='Volumen', alpha=0.7)
    ax4.plot(df.index, df['volume_sma'], label='Media Volumen', color='blue', linewidth=1)
    ax4.set_ylabel('Volumen', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    # Formato de fecha en el eje x
    plt.gcf().autofmt_xdate()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    return plt

# Función principal
def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}Iniciando bot de trading de criptomonedas - Análisis Avanzado (15min)...{Style.RESET_ALL}")
    
    exchange = init_exchange()
    
    # Verificar configuración de Telegram
    if TELEGRAM_ENABLED:
        print(f"{Fore.GREEN}Alertas Telegram activadas - Envío a chat ID: {TELEGRAM_CHAT_ID}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Alertas Telegram desactivadas - Configure TELEGRAM_TOKEN y TELEGRAM_CHAT_ID en .env para activar{Style.RESET_ALL}")
    
    # NUEVO: Verificar si estamos en modo backtesting
    if BACKTEST_MODE:
        print(f"{Fore.CYAN}Modo backtesting activado - Período: {BACKTEST_START_DATE} hasta {BACKTEST_END_DATE}{Style.RESET_ALL}")
        for symbol in SYMBOLS:
            results_df = run_backtest(exchange, symbol, TIMEFRAME, BACKTEST_START_DATE, BACKTEST_END_DATE)
            if results_df is not None and not results_df.empty:
                # Obtener datos para graficar
                df = get_historical_data(exchange, symbol, TIMEFRAME, limit=1000)
                if df is not None:
                    df = apply_technical_indicators(df)
                    plt = plot_results(symbol, df, results_df.to_dict('records'))
                    plt.savefig(f"backtest_results_{symbol.replace('/', '_')}.png")
                    plt.close()
                    print(f"{Fore.GREEN}Gráfico de resultados guardado como backtest_results_{symbol.replace('/', '_')}.png{Style.RESET_ALL}")
    else:
        try:            
            # Tiempo de espera entre análisis - ajustado para 15min
            WAIT_TIME = 30  # 30 segundos de espera entre análisis
            
            while True:
                for symbol in SYMBOLS:
                    print(f"\n{Fore.CYAN}Analizando {symbol} (15min)...{Style.RESET_ALL}")
                    
                    # Obtener datos
                    df = get_historical_data(exchange, symbol, TIMEFRAME)
                    if df is None:
                        continue
                    
                    # Aplicar indicadores técnicos
                    df = apply_technical_indicators(df)
                    
                    # Identificar niveles clave de soporte y resistencia
                    key_levels = identify_key_levels(df)
                    
                    # Analizar señales
                    signals, explanations = analyze_signals(df, key_levels)
                    
                    if signals:
                        # Generar análisis para terminal
                        generate_terminal_analysis(symbol, df, signals, explanations, key_levels)
                        
                        # NUEVO: Guardar gráfico con la situación actual
                        plt = plot_results(symbol, df)
                        plt.savefig(f"signal_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                        plt.close()
                    else:
                        print(f"{Fore.YELLOW}No hay señales claras para {symbol} en este momento{Style.RESET_ALL}")
                
                # Mostrar tiempo de espera
                wait_message = f"Esperando {WAIT_TIME} segundos antes del próximo análisis..."
                print(f"\n{Fore.BLUE}{wait_message}{Style.RESET_ALL}")
                
                # Temporizador visual
                for i in range(WAIT_TIME, 0, -1):
                    print(f"\r{Fore.BLUE}Próximo análisis en: {i} segundos...{Style.RESET_ALL}", end="")
                    time.sleep(1)
                print("\r" + " " * len("Próximo análisis en: 30 segundos..."), end="\r")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Bot detenido manualmente{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error en el bot: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()