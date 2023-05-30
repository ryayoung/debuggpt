# debuggpt

> The realest AI debugger in the game.

Like many other solutions, `debuggpt` takes advantage of high-context language models like GPT-4 to let the model make informed decisions.

_**The difference is in the prompt.**_

### `debuggpt` will provide a GPT model with a comprehensive report on the state of your program at the moment an error is thrown.

## Usage

**Place a `@debug_gpt` above a function you want to debug, and run your program. That's it.**

If your function fails, the OpenAI model will see (in addition to intro/closing prompts)
1. The recorded history of all printed outputs (stdout text) for the life of your program
2. The full, original traceback that would have been shown if you weren't debugging.
3. The state of your code **leading up to the final call to your target function**
   - All surrounding lines of code
     - Annotated with actual line numbers before each line.
     - Annotated with a comment pointing to the line which called your function
   - The names, types, and **values** of all in-scope variables at the time of error
4. The state of your code **within your target function**
   - All of the code in your target function, annotated the same way as above, pointing to the line that caused the error.
   - The names, types, and **values** of all in-scope variables at the time of error
5. For every remaining call in the traceback stack, if it *isn't* to an external libary, the annotated code context will be provided.
   - In other words, if your target function makes a call to another function which errors, the annotated source code for *that* function, and those that follow, are provided.

# Example

`/test/test_debug.py`:

```python
from debuggpt import gpt_debug
import pickle

@gpt_debug
def func():
    with open('test.pkl', 'rb') as f:
        df = pickle.load(f)
    df['c'] = df.a + df.b

func()
```

Here's what happens when the above code runs:

`test.pkl` stores a pandas DataFrame:

| a | b |
|---|---|
| 1 | 4 |
| 2 | foo |
| 3 | 6 |

Since column `b` contains a string, 'foo', we get an error when adding it to `a`:

> *TypeError: unsupported operand type(s) for +: 'int' and 'str'*

This is tricky to debug because:
1. Our code NEVER indicates that `df` is a Pandas DataFrame. In fact, we don't even import pandas!
2. Our code NEVER mentions the data types, or contents, of `df`.
3. We *can't* know which column ('a' or 'b') has mixed types, not even by looking at the error or traceback.

But GPT-4 knew *all* of this. **Including #3.**

## GPT-4's Response:

---

\<START\>

The error "TypeError: unsupported operand type(s) for +: 'int' and 'str'" is occurring because you are trying to add an integer and a string together in line 9 of `test/test_debug.py`:

```python
df['c'] = df.a + df.b
```

The DataFrame `df` has mixed types in column 'b' - it contains both integers and the string 'foo'. You cannot add an integer and a string together directly, which is causing the error.

To fix this issue, you need to decide how you want to handle the mixed types in column 'b'. One option is to convert the non-numeric values to a default value, like 0, before performing the addition. You can do this using the `pd.to_numeric` function with the `errors='coerce'` argument, which will replace non-numeric values with NaN, and then use the `fillna` method to replace NaN with 0.

Here's the updated code for `func`:

```python
def func():
    with open('test.pkl', 'rb') as f:
        df = pickle.load(f)

    df['b'] = pd.to_numeric(df['b'], errors='coerce').fillna(0)
    df['c'] = df.a + df.b
```

This code will first convert any non-numeric values in column 'b' to NaN, then replace NaN with 0, and finally perform the addition. If you have a different way you'd like to handle the mixed types in column 'b', you can modify the code accordingly.

\<END\>

---

This is an excellent response.

The model could see *everything*. 
- It knew the type and exact contents of `df`, the value returned by `pickle.load`.
- It could see all relevant code, *not* just the traceback (more on this later)
- It could see everything that was printed during life of our program (including the original error traceback).


Here's what the original traceback looked like:
```
Traceback (most recent call last):
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/ops/array_ops.py", line 171, in _na_arithmetic_op
    result = func(left, right)
             ^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/computation/expressions.py", line 239, in evaluate
    return _evaluate(op, op_str, a, b)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/computation/expressions.py", line 70, in _evaluate_standard
    return op(a, b)
           ^^^^^^^^
TypeError: unsupported operand type(s) for +: 'int' and 'str'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/Ryan.Young3/main/code/debuggpt/test/test_debug.py", line 10, in <module>
    func()
  File "/Users/Ryan.Young3/main/code/debuggpt/test/test_debug.py", line 8, in func
    df['c'] = df.a + df.b
              ~~~~~^~~~~~
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/ops/common.py", line 81, in new_method
    return method(self, other)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/arraylike.py", line 186, in __add__
    return self._arith_method(other, operator.add)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/series.py", line 6108, in _arith_method
    return base.IndexOpsMixin._arith_method(self, other, op)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/base.py", line 1348, in _arith_method
    result = ops.arithmetic_op(lvalues, rvalues, op)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/ops/array_ops.py", line 232, in arithmetic_op
    res_values = _na_arithmetic_op(left, right, op)  # type: ignore[arg-type]
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/ops/array_ops.py", line 178, in _na_arithmetic_op
    result = _masked_arith_op(left, right, op)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Ryan.Young3/.pyenv/versions/debuggpt/lib/python3.11/site-packages/pandas/core/ops/array_ops.py", line 116, in _masked_arith_op
    result[mask] = op(xrav[mask], yrav[mask])
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

While we do pass this traceback to the model, the magic comes from what we provide next: the code report.

Here is the exact code report we gave the model (in addition, we also provided a prompt, and the printed stdout history)

````

You are debugging a call to `func`.

CONTEXT SURROUNDING THE LAST CALL TO `func` BEFORE THE ERROR WAS RAISED:
File "/test/test_debug.py", line 10
```python
1| from debuggpt import gpt_debug
2| import pickle
3| 
4| @gpt_debug
5| def func():
6|     with open('test.pkl', 'rb') as f:
7|         df = pickle.load(f)
8|     df['c'] = df.a + df.b  # <-- LINE 8, THE ORIGINAL LINE THAT CAUSED THE ERROR
9| 
10| func()  # <-- LINE 10 CALLED
```

LOCAL NAMES, TYPES, AND VALUES, IN THE ABOVE CONTEXT (module, '/test/test_debug.py') WHEN THE ERROR OCCURRED:
gpt_debug: function
pickle: module
func: function


SOURCE CODE FOR `func`, THE FUNCTION YOU'RE DEBUGGING:
File "/test/test_debug.py", line 8
```python
4| @gpt_debug
5| def func():
6|     with open('test.pkl', 'rb') as f:
7|         df = pickle.load(f)
8|     df['c'] = df.a + df.b  # <-- LINE 8, THE ORIGINAL LINE THAT CAUSED THE ERROR
```

LOCAL NAMES, TYPES, AND VALUES, IN THE ABOVE CONTEXT (function, `func`) WHEN THE ERROR OCCURRED:
f: BufferedReader =
<_io.BufferedReader name='test.pkl'>

df: DataFrame =
   a     b
0  1     4
1  2  blah
2  3     6
````

The report stops here because the remaining calls in the traceback are from pandas code. But normally,
if there are more calls from our code, we would provide an extended traceback, showing the context
for each call in the stack.
