import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        try:
            event = json.loads(message)
        except (ValueError, TypeError):
            event = message
        return json.dumps({"level": record.levelname, "logger": record.name, "event": event})
