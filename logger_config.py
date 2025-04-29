import logging

# Configuración del sistema de registro
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_trading.log"),
        logging.StreamHandler()
    ]
)
# Definir logger como variable global
logger = logging.getLogger("crypto_trading_bot") 