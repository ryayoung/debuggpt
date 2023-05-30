from debuggpt.gpt_debug import gpt_debug
from debuggpt.record_stdout import StdoutRecorder, recorder
from debuggpt import options

TESTING = False

import os
api_key = os.environ.get("OPENAI_API_KEY")


