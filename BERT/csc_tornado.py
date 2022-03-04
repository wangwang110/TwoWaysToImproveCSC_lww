# coding: utf-8

import json
import tornado.web
from tornado.options import define, options
from csc_main import CSC

port = 18000
num_process = 1
define("port", default=port, help="run on the given port", type=int)
bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
model_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_all/sighan13/model.pkl"
check_object = CSC(bert_path, model_path)


class IndexHandler(tornado.web.RequestHandler):

    def get(self):
        greeting = self.get_argument('testing', "testing")
        self.write(greeting)

    def post(self):
        # result = {}
        data = json.loads(self.request.body.decode('utf-8'))
        all_error_list = check_object.correct(data)
        # result["res"] = all_error_list
        # self.write(json.dumps(all_error_list, ensure_ascii=False))
        self.write(json.dumps(all_error_list, indent=6, separators=(', ', ': '), ensure_ascii=False))


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    # http_server.listen(options.port)
    http_server.bind(options.port, address='0.0.0.0')
    http_server.start(num_processes=1)
    print('http://127.0.0.1:{}'.format(port))
    tornado.ioloop.IOLoop.instance().start()
