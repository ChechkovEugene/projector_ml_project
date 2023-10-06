import logging

import tornado.ioloop
import tornado.web
from tornado.httpclient import AsyncHTTPClient

from src.deploy.api import Healthcheck, WinePriceRegressor
from src.predictor import WinePricePredictor

logging.getLogger("transformers").setLevel(logging.ERROR)


MODEL_PATH = "run_scripts/model/"
PORT = 1492


def make_app(predictor: WinePricePredictor, version: int) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/wine_prices_regressor/healthcheck", Healthcheck, dict(version=version)),
            (r"/wine_prices_regressor/", WinePriceRegressor, dict(predictor=predictor, version=version)),
        ]
    )


if __name__ == "__main__":
    AsyncHTTPClient.configure("tornado.curl_httpclient.CurlAsyncHTTPClient", max_clients=100)
    http_client = AsyncHTTPClient()
    model_version = int(open("model.version").readline().rstrip())
    wine_prices_predictor = WinePricePredictor.load(MODEL_PATH)
    app = make_app(predictor=wine_prices_predictor, version=model_version)
    app.listen(PORT)
    tornado.ioloop.IOLoop.current().start()
