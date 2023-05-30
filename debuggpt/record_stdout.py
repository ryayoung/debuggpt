import sys

recorder = None

class StdoutRecorder:
    def __init__(self):
        self.original_stdout = sys.stdout
        self.recorded_text = []
        sys.stdout = self

    def __getattr__(self, attr):
        return getattr(self.original_stdout, attr)

    def write(self, text):
        self.original_stdout.write(text)
        self.recorded_text.append(text)

    def finalize_recorded_text(self) -> str:
        sys.stdout = self.original_stdout
        return "".join(self.recorded_text)
