# Crypto Trading/Signals Bot

A cryptocurrency trading bot that analyzes BTC/USDT and ETH/USDT markets on Binance Futures, using multiple technical indicators to generate trading signals.

## Features

- **Real-time analysis** of cryptocurrency markets
- **Multiple technical indicators**:
  - Simple Moving Averages (SMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - ADX (Average Directional Index)
  - Bollinger Bands
  - Ichimoku Cloud
  - Volume analysis
  - Dynamic support and resistance levels

- **Trading signals** for LONG and SHORT positions
- **Telegram alerts** to notify important signals
- **Graphical visualization** of technical analysis
- **REST API** to access bot functionalities

## Requirements

- Python 3.8+
- Binance account with API Key and Secret (Its Free)
- (Optional) Telegram bot with Token and Chat ID for alerts or u can get the results on terminal

## Installation

1. Clone the repository:
```
git clone https://github.com/ggomez0/crypto-signal-bot.git
cd crypto-trading-bot
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure environment variables:
   - Create a `.env` file with the following variables:
   ```
   API_KEY=your_binance_api_key
   API_SECRET=your_binance_api_secret
   TELEGRAM_TOKEN=your_telegram_token (optional)
   TELEGRAM_CHAT_ID=your_telegram_chat_id (optional)
   ```

## Usage

### Run the bot directly:
```
python bot.py
```

### Run the web server:
```
python server.py
```

The server will be available at `http://localhost:5000`

## API Endpoints

- `/` - Initiates bot analysis and returns a confirmation message
- `/api/test` - Test endpoint that returns a status message

## Project Structure

- `bot.py` - Main trading bot logic
- `server.py` - Flask web server to access bot functionalities
- `api_telegram.py` - Module for sending Telegram alerts
- `logger_config.py` - Logging system configuration
- `env.py` - Environment variables and configuration

## Warning

**IMPORTANT**: This bot is for educational and research purposes only. Cryptocurrency trading carries significant risks and may result in financial losses. We are not responsible for any losses that may occur when using this software.
