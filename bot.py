import pandas as pd
import numpy as np
import ccxt
import time
import ta
from datetime import datetime
import colorama
from colorama import Fore, Back, Style
from env import API_KEY, API_SECRET
from api_telegram import send_telegram_alert
from logger_config import logger
from plt_graph import generate_plt
from initial_config import SYMBOLS, TIMEFRAME, FAST_MA, SLOW_MA, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, MACD_FAST, MACD_SIGNAL, MACD_SLOW, ADX_PERIOD, ADX_THRESHOLD, VOLUME_THRESHOLD

# Inicializar colorama para colores en terminal
colorama.init(autoreset=True)


# Inicializar exchange
def init_exchange():
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
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
        print(f"Datos históricos obtenidos para {symbol} ({len(df)} velas)")
        #print (df)
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
    
    # Análisis de Volumen
    if isinstance(df['volume'], pd.Series):
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
        # Detectar tendencia de volumen decreciente (3 periodos consecutivos)
        df['volume_decreasing_trend'] = (
            (df['volume'] < df['volume'].shift(1)) & 
            (df['volume'].shift(1) < df['volume'].shift(2)) & 
            (df['volume'].shift(2) < df['volume'].shift(3))
        )
    else:
        # If volume is a scalar, set default values
        df['volume_sma'] = np.nan
        df['volume_ratio'] = np.nan
        df['volume_increasing'] = False
        df['volume_decreasing_trend'] = False
    
    # Ichimoku Cloud
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
    window = 20
    
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
    
    # Soportes y resistencias dinámicas
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
    print (f"Analizando soportes y resistencias en las últimas {len(recent_df)} velas")
    print (recent_df)
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
    
    # Añadir soporte y resistencia dinámicos
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
    
    # Señal por volumen
    if isinstance(last_row['volume_ratio'], (float, np.float64, np.float32, int, np.int64, np.int32)):
        volume_ratio_check = last_row['volume_ratio'] > VOLUME_THRESHOLD
    else:
        volume_ratio_check = last_row['volume_ratio'] > VOLUME_THRESHOLD if not pd.isna(last_row['volume_ratio']) else False
    
    if volume_ratio_check:
        # Confirmar señales si hay alto volumen
        if "LONG" in signals:
            explanations.append(f"Alto volumen ({last_row['volume_ratio']:.2f}x promedio) confirma señal de compra")
        elif "SHORT" in signals:
            explanations.append(f"Alto volumen ({last_row['volume_ratio']:.2f}x promedio) confirma señal de venta")
        # O generar nueva señal basada en ruptura de precio con alto volumen
        elif isinstance(last_row['close'], pd.Series) and last_row['close'] > last_row['close'].shift(1) * 1.01:  # Subida de más del 1%
            signals.append("LONG")
            explanations.append(f"Ruptura alcista con volumen alto ({last_row['volume_ratio']:.2f}x promedio)")
        elif isinstance(last_row['close'], pd.Series) and last_row['close'] < last_row['close'].shift(1) * 0.99:  # Bajada de más del 1%
            signals.append("SHORT")
            explanations.append(f"Ruptura bajista con volumen alto ({last_row['volume_ratio']:.2f}x promedio)")
        # Si close no es una Series, usamos una comparación con el valor anterior (que debería estar disponible)
        elif not isinstance(last_row['close'], pd.Series) and 'close' in prev_row and last_row['close'] > prev_row['close'] * 1.01:
            signals.append("LONG")
            explanations.append(f"Ruptura alcista con volumen alto ({last_row['volume_ratio']:.2f}x promedio)")
        elif not isinstance(last_row['close'], pd.Series) and 'close' in prev_row and last_row['close'] < prev_row['close'] * 0.99:
            signals.append("SHORT")
            explanations.append(f"Ruptura bajista con volumen alto ({last_row['volume_ratio']:.2f}x promedio)")
    
    #  Alerta de agotamiento por volumen decreciente
    if isinstance(last_row['volume_decreasing_trend'], bool) and last_row['volume_decreasing_trend']:
        if last_row['trend'] == 'BULLISH':
            signals.append("SHORT")
            explanations.append("Volumen decreciente en tendencia alcista (posible agotamiento)")
        elif last_row['trend'] == 'BEARISH':
            signals.append("LONG")
            explanations.append("Volumen decreciente en tendencia bajista (posible agotamiento)")
    
    #  Señales basadas en Ichimoku Cloud
    if last_row['above_cloud']:
        if last_row['ichimoku_conversion_line'] > last_row['ichimoku_base_line']:
            signals.append("LONG")
            explanations.append("Precio por encima de la nube Ichimoku con TK Cross alcista")
    elif last_row['below_cloud']:
        if last_row['ichimoku_conversion_line'] < last_row['ichimoku_base_line']:
            signals.append("SHORT")
            explanations.append("Precio por debajo de la nube Ichimoku con TK Cross bajista")
    
    #  Señal basada en Kumo futuro
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

# Generar análisis detallado para terminal
def generate_terminal_analysis(symbol, df, signals, explanations, key_levels=None):
    current_price = df['close'].iloc[-1]
    last_row = df.iloc[-1]
    
    print("\n" + "="*80)
    if "LONG" in signals:
        print(f"{Fore.GREEN}{Style.BRIGHT}🚨 ALERTA: {symbol} - SEÑALES DE COMPRA (LONG) DETECTADAS 🚨{Style.RESET_ALL}")
        #  Enviar alerta a Telegram
        send_telegram_alert(symbol, "LONG", current_price, explanations)
    elif "SHORT" in signals:
        print(f"{Fore.RED}{Style.BRIGHT}🚨 ALERTA: {symbol} - SEÑALES DE VENTA (SHORT) DETECTADAS 🚨{Style.RESET_ALL}")
        #  Enviar alerta a Telegram
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
    
    # Análisis de volumen
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}📊 ANÁLISIS DE VOLUMEN:{Style.RESET_ALL}")
    print(f"• Volumen: {last_row['volume']:.2f}")
    print(f"• Media 20 periodos: {last_row['volume_sma']:.2f}")
    print(f"• Ratio Volumen/Media: {Fore.GREEN if last_row['volume_ratio'] > 1 else Fore.RED}{last_row['volume_ratio']:.2f}x{Style.RESET_ALL}")
    if last_row['volume_ratio'] > VOLUME_THRESHOLD:
        print(f"{Fore.GREEN}• Volumen significativamente alto (>{VOLUME_THRESHOLD}x promedio){Style.RESET_ALL}")
    if last_row['volume_decreasing_trend']:
        print(f"{Fore.RED}• Alerta: Volumen decreciente en últimos 3 periodos (posible agotamiento){Style.RESET_ALL}")
    
    # Ichimoku
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
        #  Mostrar soportes y resistencias dinámicas
        print(f"\n{Fore.YELLOW}Soportes y resistencias dinámicas:{Style.RESET_ALL}")
        if last_row['last_broke_resistance'] is not None and not np.isnan(last_row['last_broke_resistance']):
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
    
    # Guardar señal para backtesting
    # if BACKTEST_MODE and (long_count >= 2 or short_count >= 2):
    #     signal_record = {
    #         'timestamp': datetime.now(),
    #         'symbol': symbol,
    #         'price': current_price,
    #         'signal': 'LONG' if long_count > short_count else 'SHORT',
    #         'signal_count': max(long_count, short_count),
    #         'indicators': explanations
    #     }
    #     BACKTEST_RESULTS.append(signal_record)

def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}Iniciando bot de trading de criptomonedas - Análisis Avanzado (15min)...{Style.RESET_ALL}")
    
    exchange = init_exchange()    
    
    try:  
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

                #Uncomment to get Graph
                #generate_plt(symbol,df)
            else:
                print(f"{Fore.YELLOW}No hay señales claras para {symbol} en este momento{Style.RESET_ALL}")
        
        
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Bot detenido manualmente{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error en el bot: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
        WAIT_TIME = 30
        while True:
            main()
            wait_message = f"Esperando {WAIT_TIME} segundos antes del próximo análisis..."
            print(f"\n{Fore.BLUE}{wait_message}{Style.RESET_ALL}")
            for i in range(WAIT_TIME, 0, -1):
                print(f"\r{Fore.BLUE}Próximo análisis en: {i} segundos...{Style.RESET_ALL}", end="")
                time.sleep(1)
            print("\r" + " " * len("Próximo análisis en: 30 segundos..."), end="\r")