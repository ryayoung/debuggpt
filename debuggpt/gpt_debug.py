import inspect
import re
from functools import wraps
from traceback import (
    extract_tb,
    format_list,
)
from types import FrameType
from typing import *
import traceback

import os
import site

import tiktoken
import openai
import debuggpt
from debuggpt import record_stdout, options

tokenizer = tiktoken.encoding_for_model("gpt-4")


def gpt_debug(
    __func: Optional[Callable] = None,
    context_line_limit: Optional[int] = None,
    stdout_token_limit: Optional[int] = None,
    variable_value_token_limit: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
    show_token_report: Optional[bool] = None,
    extend_external_tracebacks: Optional[bool] = None,
    ask_for_confirmation: Optional[bool] = None,
    prompt_include_line_numbers: Optional[bool] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    show_prompt_only: Optional[bool] = None,
    **params,
):
    api_key = api_key or debuggpt.api_key
    assert api_key, "No API key provided."

    context_line_limit = context_line_limit or options.context_line_limit
    stdout_token_limit = stdout_token_limit or options.stdout_token_limit
    variable_value_token_limit = (
        variable_value_token_limit or options.variable_value_token_limit
    )
    max_prompt_tokens = max_prompt_tokens or options.max_prompt_tokens
    show_token_report = show_token_report or options.show_token_report
    extend_external_tracebacks = (
        extend_external_tracebacks or options.extend_external_tracebacks
    )
    ask_for_confirmation = ask_for_confirmation or options.ask_for_confirmation
    prompt_include_line_numbers = (
        prompt_include_line_numbers or options.prompt_include_line_numbers
    )
    model = model or options.model
    temperature = temperature or options.temperature
    show_prompt_only = show_prompt_only or options.show_prompt_only

    params["model"] = model
    params["temperature"] = temperature

    if record_stdout.recorder is None:
        record_stdout.recorder = record_stdout.StdoutRecorder()

    def decorator(ai_ignore_this_func):
        func_name = ai_ignore_this_func.__name__

        @wraps(ai_ignore_this_func)
        def ai_ignore_this_wrapper(*args, **kwargs):
            try:
                return ai_ignore_this_func(*args, **kwargs)
            except Exception as e:
                curr_frame = inspect.currentframe()
                assert curr_frame is not None
                handle_exception(
                    context_line_limit=context_line_limit,
                    stdout_token_limit=stdout_token_limit,
                    variable_value_token_limit=variable_value_token_limit,
                    max_prompt_tokens=max_prompt_tokens,
                    show_token_report=show_token_report,
                    show_prompt_only=show_prompt_only,
                    extend_external_tracebacks=extend_external_tracebacks,
                    ask_for_confirmation=ask_for_confirmation,
                    prompt_include_line_numbers=prompt_include_line_numbers,
                    api_key=api_key,
                    params=params,
                    func_name=func_name,
                    e=e,
                    curr_frame=curr_frame,
                )

        return ai_ignore_this_wrapper

    if callable(__func):
        return decorator(__func)
    return decorator


def handle_exception(
    context_line_limit: int,
    stdout_token_limit: int,
    variable_value_token_limit: int,
    max_prompt_tokens: int,
    show_token_report: bool,
    show_prompt_only: bool,
    extend_external_tracebacks: bool,
    ask_for_confirmation: bool,
    prompt_include_line_numbers: bool,
    api_key: str,
    params: dict,
    func_name: str,
    e: Exception,
    curr_frame: FrameType,
) -> NoReturn:
    token_report = dict(
        prompt_total=0,
        stdout=0,
        context_code=0,
        variable_values=0,
    )
    err_text = f"{type(e).__name__}: {e}"

    def get_main_error_detail():
        idx = -1
        summary = traceback.extract_tb(e.__traceback__)[idx]
        while is_external_library(summary.filename):
            idx -= 1
            summary = traceback.extract_tb(e.__traceback__)[idx]
        return (summary.filename, summary.lineno, summary.line)

    err_detail = get_main_error_detail()

    def get_stdout_and_traceback():
        assert record_stdout.recorder is not None, "No stdout recorder found."
        stdout = record_stdout.recorder.finalize_recorded_text().strip()
        if count_tokens(stdout) > stdout_token_limit:
            stdout = "..." + detokenize(tokenize(stdout)[-stdout_token_limit:])
        token_report["stdout"] += len(tokenize(stdout))

        tback = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        msg = "STDOUT FROM USER'S PROGRAM"
        return f"{msg}:\n```\n{stdout.strip()}\n{tback.strip()}\n```"

    def get_context_leading_to_error(fr):
        assert fr
        lineno = fr.f_lineno
        filename = truncate_path(fr.f_code.co_filename)
        context = get_context_lines(fr, lineno, context_line_limit, err_detail)
        token_report["context_code"] += len(tokenize(context))
        msg = f"CONTEXT SURROUNDING THE LAST CALL TO `{func_name}` BEFORE THE ERROR WAS RAISED:"
        return format_traceback_item(filename, context, lineno, msg)

    def get_context_primary_debug_frame(fr):
        lineno = fr.f_lineno
        filename = truncate_path(fr.f_code.co_filename)
        context = get_context_lines(fr, lineno, 100, err_detail)
        token_report["context_code"] += len(tokenize(context))
        msg = f"SOURCE CODE FOR `{func_name}`, THE FUNCTION YOU'RE DEBUGGING:"
        return format_traceback_item(filename, context, lineno, msg)

    def get_frame_locals(fr):
        msg = f"LOCAL NAMES, TYPES, AND VALUES, IN THE ABOVE CONTEXT ({frame_name(fr)}) WHEN THE ERROR OCCURRED:\n"
        locals = {
            k: v
            for k, v in fr.f_locals.items()
            if not k.startswith("__") and not k.endswith("__")
        }
        for k, v in locals.items():
            details = repr_variable(k, v, variable_value_token_limit)
            token_report["variable_values"] += len(tokenize(details))
            msg += details
        return msg

    def get_extended_tracebacks(frames) -> list:
        items = []
        any_used = False
        for fr in frames:
            lineno = fr.f_lineno
            filename = fr.f_code.co_filename
            if is_external_library(filename):
                if not extend_external_tracebacks:
                    continue
                n_lines = 3
                msg = f"NOTE: This call is from an external library:"
            else:
                n_lines = context_line_limit
                msg = None
            any_used = True
            filename = truncate_path(filename)
            context = get_context_lines(fr, lineno, n_lines, err_detail)
            token_report["context_code"] += len(tokenize(context))
            formatted = format_traceback_item(filename, context, lineno, msg)
            items.append(formatted)

        if not any_used:
            return []
        return ["EXTENDED TRACEBACK"] + items

    def format_token_report():
        msg = format_header("TOKEN REPORT") + "\n"
        items = {k: f"{v} tokens" for k, v in token_report.items()}
        msg += format_bullets(items, indent=2)
        return msg

    tb_walked = list(traceback.walk_tb(e.__traceback__))
    frames = [tb[0] for tb in tb_walked[1:]]
    first_frame = frames.pop(0)

    stdout_msg = get_stdout_and_traceback()
    messages = [
        get_context_leading_to_error(curr_frame.f_back),
        get_frame_locals(curr_frame.f_back),
        get_context_primary_debug_frame(first_frame),
        get_frame_locals(first_frame),
        *get_extended_tracebacks(frames),
    ]

    messages_str = "\n\n".join(messages)
    prompt = "\n\n".join(
        [
            stdout_msg,
            options.opening_prompt,
            f"You are debugging a call to `{func_name}`.",
            messages_str,
            "END OF DEBUG CODE REPORT",
            f'As a reminder, the original error was: "{err_text}"',
            options.closing_prompt,
        ]
    )
    if prompt_include_line_numbers is True:
        prompt += "\n" + options.line_number_prompt

    prompt_tokens = len(tokenize(prompt))
    token_report["prompt_total"] = prompt_tokens

    if show_prompt_only:
        raise GPTDebugException(prompt + "\n\n" + format_token_report())

    if prompt_tokens > max_prompt_tokens:
        print(stdout_msg + "\n\n" + messages_str + "\n\n")
        raise GPTDebugException(
            f"`max_prompt_tokens` limit exceeded: Prompt had {prompt_tokens} tokens. No API request made. "
            + TOKEN_LIMIT_EXCEEDED_ERROR_MSG
            + "\n\n"
            + format_token_report()
        )

    if debuggpt.TESTING:
        response = "<RESPONSE>"
    else:
        if ask_for_confirmation is True:
            print("\n\n".join([stdout_msg, messages_str, format_token_report()]))
            proceed = input("Proceed with API request? [y/N] ")
            if proceed.lower() != "y":
                raise GPTDebugException(
                    "No API request made."
                )
        response = request_to_gpt(prompt, api_key, params)

    if show_token_report:
        response += "\n\n" + format_token_report()
    raise GPTDebugException(prompt + "\n\n" + response)


def tokenize(text: str) -> List:
    return tokenizer.encode(text)


def detokenize(tokens: List) -> str:
    return tokenizer.decode(tokens)


def is_external_library(filename):
    site_packages = site.getsitepackages()
    for sp in site_packages:
        if filename.startswith(sp):
            return True
    return False


def format_numbered_lines(lines, start_line_number):
    return [f"{start_line_number + i}| {line}" for i, line in enumerate(lines)]


def get_context_lines(frame, line_number, n_context_lines, err_detail):
    source_lines, start_line = inspect.getsourcelines(frame)
    # read documentation for inspect.getsourcelines. They're a bunch of idiots for this. Start line is zero based if frame is a module.
    start_line = max(start_line, 1)
    last_line = start_line + len(source_lines) - 1
    idx_of_line_number = line_number - start_line
    if n_context_lines == 1:
        return source_lines[idx_of_line_number].strip()

    source_lines = source_lines[: idx_of_line_number + 3]
    idx_of_original_err_line = err_detail[1] - start_line
    in_same_file_as_err = frame.f_code.co_filename == err_detail[0]
    err_visible = (start_line <= err_detail[1] <= last_line) and in_same_file_as_err

    if err_visible:
        curr = source_lines[idx_of_original_err_line]
        new = re.sub(
            r"\n$",
            f"  # <-- LINE {err_detail[1]}, THE ORIGINAL LINE THAT CAUSED THE ERROR\n",
            curr,
        )
        source_lines[idx_of_original_err_line] = new

    if (idx_of_line_number != idx_of_original_err_line) or not in_same_file_as_err:
        source_lines[idx_of_line_number] = re.sub(
            r"\n$",
            f"  # <-- LINE {line_number} CALLED\n",
            source_lines[idx_of_line_number],
        )

    start_index = max(0, line_number - start_line - n_context_lines)
    end_index = min(len(source_lines), line_number - start_line + n_context_lines)
    if options.prompt_include_line_numbers is True:
        source_lines = format_numbered_lines(source_lines, start_line)
    return "".join(source_lines[start_index:end_index])


def format_traceback_item(filename, context_lines, lineno=None, message=None):
    message = (message + "\n") if message else ""
    line = f", line {lineno}" if lineno else ""
    file = f'File "{filename}"{line}'
    code = re.sub(r"^\n+", "", context_lines)
    code = re.sub(r"\n+$", "", code)
    return f"{message}{file}\n```python\n{code}\n```"


class GPTDebugException(Exception):
    def __init__(self, *args, **kwargs):
        import sys

        sys.excepthook = lambda _, value, __: print(value)
        super().__init__(*args, **kwargs)


def count_tokens(text) -> int:
    return len(tokenize(text))


model_map = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4",
}


def request_to_gpt(prompt, api_key, options) -> str:

    openai.api_key = api_key
    options["model"] = model_map.get(options["model"], options["model"])
    response = openai.ChatCompletion.create(
        **options, messages=[{"role": "user", "content": prompt}]
    )
    res_msg = response["choices"][0]["message"]["content"]
    return res_msg


def repr_variable(name, value, limit):
    header = f"{name}: {type(value).__qualname__}"
    if (
        inspect.ismodule(value)
        or inspect.isclass(value)
        or inspect.isfunction(value)
        or inspect.ismethod(value)
    ):
        return f"{header}\n"

    header += " ="

    if isinstance(value, str):
        value = f"'{value}'"

    val = str(value)
    if count_tokens(val) > limit:
        val = detokenize(tokenize(val)[:limit]) + "... (truncated)"

    return f"{header}\n{val}\n\n"


def truncate_path(path):
    import sys

    sys_path_heads = [os.path.split(p)[0] for p in sys.path]
    for head in sys_path_heads:
        if path.startswith(head):
            return path[len(head) :]
    return path


def format_header(header):
    return header + "\n" + ("-" * len(header))


def format_bullet_item(key, value, tab: int):
    key = f"{key}: " + (" " * (tab - len(key)))
    return f"{key}{value}"


def format_bullets(items: dict, tab: Optional[int] = None, indent: int = 0) -> str:
    tab = tab or (1 + max([len(k) for k in items.keys()]))
    indent_str = " " * indent
    return indent_str + f"\n{indent_str}".join(
        [format_bullet_item(k, v, tab) for k, v in items.items()]
    )


def frame_name(fr: FrameType):
    co_name = fr.f_code.co_qualname
    if co_name in ["<module>", "__main__"]:
        fname = truncate_path(fr.f_code.co_filename)
        return f"module, '{fname}'"
    return f"function, `{co_name}`"


TOKEN_LIMIT_EXCEEDED_ERROR_MSG = (
    "No prompt was sent to the API. You can debug manually using the information above, or increase the "
    "`max_prompt_tokens` option, or decrease the `context_lines` or `stdout_token_limit` parameters. "
    "Alternatively, you can customize the `debuggpt.opening_prompt` and `debuggpt.closing_prompt` strings to "
    "a shorter length."
)
