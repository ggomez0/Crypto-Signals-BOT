from colorama import Fore, Style
import pandas as pd
from bot import SYMBOLS, TIMEFRAME, apply_technical_indicators, get_historical_data, identify_key_levels, analyze_signals, init_exchange, plot_results

BACKTEST_MODE = False
BACKTEST_START_DATE = '2025-03-01'
BACKTEST_END_DATE = '2025-04-27'
BACKTEST_RESULTS = []

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
    
if BACKTEST_MODE:
    exchange = init_exchange()
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