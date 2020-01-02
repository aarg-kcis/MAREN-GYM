class Preprocessor:
    def __init__(self, preprocess_fns):
        self.preprocess_fns = preprocess_fns

    def __call__(self, x):
        obs = {}
        for k, v in x.items():
            if k in self.preprocess_fns.keys() and self.preprocess_fns[k] is not None:
                obs[k] = self.preprocess_fns[k](v)
            else:
                obs[k] = v
        return obs

