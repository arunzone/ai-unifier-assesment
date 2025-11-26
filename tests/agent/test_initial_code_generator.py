import pytest
from assertpy import assert_that
from pytest_httpx import HTTPXMock
from httpx import Request, Response

from ai_unifier_assesment.agent.initial_code_generator import InitialCodeGenerator
from ai_unifier_assesment.agent.language import Language
from ai_unifier_assesment.agent.state import CodeHealingState
from ai_unifier_assesment.config import Settings
from ai_unifier_assesment.large_language_model.model import Model
from ai_unifier_assesment.resources.prompts.prompt_loader import PromptLoader
from llm_helper import ai_response_for, extract_user_content

PYTHON_CODE_BLOCK = """
FILE: quicksort.py
```python
def quicksort(arr):
    # Sort an array using the quicksort algorithm.
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

if __name__ == "__main__":
    print(quicksort([3, 6, 8, 10, 1, 2, 1]))
```

FILE: test_quicksort.py
```python
import pytest
from quicksort import quicksort

def test_quicksort():
    assert quicksort([]) == []
    assert quicksort([1]) == [1]
    assert quicksort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]
    assert quicksort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert quicksort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert quicksort([2, 2, 2, 2]) == [2, 2, 2, 2]

def test_quicksort_with_duplicates():
    assert quicksort([1, 1, 1, 2, 2, 1]) == [1, 1, 1, 2, 2]

def test_quicksort_with_negative_numbers():
    assert quicksort([3, -1, 2, -5, 0]) == [-5, -1, 0, 2, 3]
```
"""
PYTHON_QUICKSORT_RESPONSE = ai_response_for(PYTHON_CODE_BLOCK)

RUST_CODE_BLOCK = """FILE: lib.rs
```rust
pub fn quicksort<T: Ord>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    let pivot_index = arr.len() / 2;
    arr.swap(pivot_index, arr.len() - 1);
    let pivot_pos = partition(arr);
    quicksort(&mut arr[..pivot_pos]);
    quicksort(&mut arr[pivot_pos + 1..]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let pivot = &arr[arr.len() - 1];
    let mut i = 0;
    for j in 0..arr.len() - 1 {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, arr.len() - 1);
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quicksort() {
        let mut arr = vec![3, 6, 8, 10, 1, 2, 1];
        quicksort(&mut arr);
        assert_eq!(arr, vec![1, 1, 2, 3, 6, 8, 10]);
    }

    #[test]
    fn test_quicksort_empty() {
        let mut arr: Vec<i32> = vec![];
        quicksort(&mut arr);
        assert_eq!(arr, vec![]);
    }

    #[test]
    fn test_quicksort_single_element() {
        let mut arr = vec![42];
        quicksort(&mut arr);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_quicksort_sorted() {
        let mut arr = vec![1, 2, 3, 4, 5];
        quicksort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_quicksort_reverse_sorted() {
        let mut arr = vec![5, 4, 3, 2, 1];
        quicksort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }
}
```"""

RUST_QUICKSORT_RESPONSE = ai_response_for(RUST_CODE_BLOCK)


def llm_response(request: Request):
    request_content = request.read().decode("utf-8")
    user_content = extract_user_content(request_content)

    if "quicksort" in user_content.lower() and "python" in user_content.lower():
        return Response(status_code=200, json=PYTHON_QUICKSORT_RESPONSE)
    elif "quicksort" in user_content.lower() and "rust" in user_content.lower():
        return Response(status_code=200, json=RUST_QUICKSORT_RESPONSE)
    return None


@pytest.mark.asyncio
async def test_should_have_main_file_name_in_python_code(httpx_mock: HTTPXMock):
    httpx_mock.add_callback(llm_response)
    state = CodeHealingState(
        task_description="Write a Python function to sort an array using quicksort", language=Language.PYTHON
    )
    settings = Settings()
    generator = InitialCodeGenerator(model=Model(settings), prompt_loader=PromptLoader(), settings=settings)

    result = await generator.generate_initial_code(state)

    assert_that(result.current_code).contains(PYTHON_CODE_BLOCK)


@pytest.mark.asyncio
async def test_should_contain_rust_code_block(httpx_mock: HTTPXMock):
    httpx_mock.add_callback(llm_response)
    state = CodeHealingState(
        task_description="Write a Rust function to sort an array using quicksort", language=Language.RUST
    )
    settings = Settings()
    generator = InitialCodeGenerator(model=Model(settings), prompt_loader=PromptLoader(), settings=settings)

    result = await generator.generate_initial_code(state)

    assert_that(result.current_code).contains(RUST_CODE_BLOCK)


@pytest.mark.asyncio
async def test_should_handle_empty_response(httpx_mock: HTTPXMock):
    httpx_mock.add_callback(lambda request: Response(status_code=200, json=ai_response_for("")))
    state = CodeHealingState(task_description="Write a simple function", language=Language.PYTHON)
    settings = Settings()
    generator = InitialCodeGenerator(model=Model(settings), prompt_loader=PromptLoader(), settings=settings)

    result = await generator.generate_initial_code(state)

    assert_that(result.current_code).is_empty()
