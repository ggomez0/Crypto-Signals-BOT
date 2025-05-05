from flask import Flask, jsonify, request
from bot import main
from logger_config import logger

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def home():
    try:
        return jsonify({"message": "Hello, World!"})
    except Exception as e:
        logger.error(f"Error in bot execution: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    print (f"Datos recibidos: {data}")

    message_text = data['message']['text']
    user_id = data['message']['from']['id']
    username = data['message']['from'].get('username', '')
    first_name = data['message']['from'].get('first_name', '')

    print (f"ID del usuario: {user_id}")
    print(f"Mensaje: {message_text}")
    print(f"Usuario: {username} (ID: {user_id}, Nombre: {first_name})")
    main()

    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
