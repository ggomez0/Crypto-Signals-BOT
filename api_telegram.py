from datetime import datetime
from colorama import Fore, Style
import requests
from env import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN
from logger_config import logger

TELEGRAM_ENABLED = TELEGRAM_TOKEN is not None and TELEGRAM_CHAT_ID is not None
    
def send_telegram_alert(symbol, signal_type, price, explanations, telegram_chatid = TELEGRAM_CHAT_ID):
    if TELEGRAM_ENABLED:
        print(f"{Fore.GREEN}Alertas Telegram activadas - Envío a chat ID: {TELEGRAM_CHAT_ID}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Alertas Telegram desactivadas - Configure TELEGRAM_TOKEN y TELEGRAM_CHAT_ID en .env para activar{Style.RESET_ALL}")
        return False
    
    try:
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

        send_msg(telegram_chatid, message)
        return True
    except Exception as e:
        logger.error(f"Error al enviar alerta a Telegram: {e}")
        return False
    
def send_msg(telegram_chatid, message):
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": telegram_chatid,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            logger.info(f"Alerta enviada a Telegram")
            return True
        else:
            logger.error(f"Error al enviar alerta a Telegram: {response.status_code} - {response.text}")
            return False