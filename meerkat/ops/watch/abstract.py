class WatchLogger:
    def log_errand(self, **kwargs):
        raise NotImplementedError

    def log_errand_start(self, **kwargs):
        raise NotImplementedError

    def log_errand_end(self, **kwargs):
        raise NotImplementedError

    def log_engine_run(self, **kwargs):
        raise NotImplementedError