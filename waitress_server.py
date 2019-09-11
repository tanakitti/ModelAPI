from waitress import serve
import predict
serve(predict.app, host='0.0.0.0', port=8080)