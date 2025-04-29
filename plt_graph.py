from datetime import datetime
import numpy as np
from initial_config import FAST_MA, SLOW_MA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_plt(symbol, df, signals=None):
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
    if 'ichimoku_a' in df.columns and 'ichimoku_b' in df.columns:
    # Crear una copia para evitar warning de SettingWithCopyWarning
        cloud_df = df[['ichimoku_a', 'ichimoku_b']].copy()
        # Aplicar shift para obtener la nube actual
        if len(cloud_df) > 26:
            cloud_df = cloud_df.shift(26)
            # Llenar valores NaN
            cloud_df = cloud_df.ffill(method='ffill')
            
            # Determinar áreas para colorear
            above_areas = cloud_df['ichimoku_a'] >= cloud_df['ichimoku_b']
            below_areas = cloud_df['ichimoku_a'] < cloud_df['ichimoku_b']
            
            # Dibujar áreas
            if above_areas.any():
                ax1.fill_between(df.index, cloud_df['ichimoku_a'], cloud_df['ichimoku_b'],
                                where=above_areas,
                                color='green', alpha=0.1)
            if below_areas.any():
                ax1.fill_between(df.index, cloud_df['ichimoku_a'], cloud_df['ichimoku_b'],
                                where=below_areas,
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
    plt.savefig(f"signal_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    return plt

