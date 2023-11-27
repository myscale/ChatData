import json
import datetime

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return datetime.datetime.isoformat(obj)
        return json.JSONEncoder.default(self, obj)

class CustomJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, source):
        for k, v in source.items():
            if isinstance(v, str):
                try:
                    source[k] = datetime.datetime.fromisoformat(str(v))
                except:
                    pass
        return source