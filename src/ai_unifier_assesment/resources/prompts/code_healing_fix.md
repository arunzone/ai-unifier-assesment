# Self-Healing Code Fix Prompt

The previous code failed testing. Analyze the errors carefully and generate corrected code.

## Previous Code

{previous_code}

## Test Output (Errors)

{test_output}

## Your Task

1. **Analyze the errors** line-by-line
2. **Identify root causes** (logic errors, syntax errors, missing edge cases, etc.)
3. **Generate corrected code** following the same format rules

## Critical Rules for Fixes

1. **Output ONLY corrected complete files** - no explanations
2. **Use the exact same FILE format** as before
3. **Fix ALL identified issues**, not just the first one
4. **Maintain the same file structure** (don't add/remove files unless necessary)
5. **Make minimal changes** - only fix what's broken
6. **Ensure tests cover the failing scenarios**

## Common Error Patterns

- **Rust Compilation Errors**: Missing imports, type mismatches, borrow checker issues
- **Python Errors**: Import errors, attribute errors, type errors
- **Test Failures**: Logic bugs, off-by-one errors, missing edge cases
- **Edge Cases**: Empty inputs, single elements, already sorted, reverse sorted

Fix the code now using the same output format.
