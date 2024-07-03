import asyncio
import concurrent.futures
from openai import OpenAI, AsyncOpenAI
from typing import List, Tuple


class OpenAIMetaPrompter(object):

    def __init__(
        self,
        api_key: str,
        model_name: str,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)


class OpenAISourceSummarizer(OpenAIMetaPrompter):

    def __init__(
        self, 
        api_key: str, 
        model_name: str
    ):
        super().__init__(api_key, model_name)
        self.meta_system_prompt = \
        "You are an experienced C/C++ software developer."

        self.user_prompt = \
"""You are provided with the following function:
```C/C++
{}
```
First generate a brief step-by-step description of its functionality in the format:
**Description**:
...

Then generate a high-level summary of its functionality in the format:
**Summary**:
The function ...

After that, generate a brief description of its general purpose in the format:
**Purpose**:
The purpose of the function is to ...

"""

    def pack_src(
        self,
        source_function,
    ):
        message = [
            {"role": "system", "content": self.meta_system_prompt},
            {
                "role": "user",
                "content": self.user_prompt.format(source_function),
            },
        ]
        return message
    
    def generate(
        self,
        inputs,
        **kwargs
    ):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.pack_src(inputs),
            **kwargs
        )
        return response

    def batch_generate(
        self,
        batch_inputs,
        **kwargs
    ):
        responses = []
        # iterative implementation
        for inputs in batch_inputs:
            responses.append(
                self.generate(inputs, **kwargs)
            )
        return responses


class OpenAIDecompSummarizer(OpenAIMetaPrompter):

    def __init__(
        self, 
        api_key: str, 
        model_name: str
    ):
        super().__init__(api_key, model_name)

        self.meta_system_prompt = \
        "You are an experienced binary reverse engineer to understand "\
        "decompiled C code that lacks symbol information. "

        self.default_user_prompt = \
"""You are provided with the following decompiled function that is hardly human readable:
```C
{}
```
First generate a brief step-by-step description of its functionality in the format:
**Description**:
...

Then try to generate a summary of it that can help human understand / inspect its original high-level source code functionality in the format:
**Summary**:
The function ...

After that, inspect and generate a brief description of its general purpose in the format:
**Purpose**:
The purpose of the function seems to ...
"""

        self.augmented_user_prompt = \
"""You are provided with the following decompiled function that is not human readable:
```C
{}
```

First generate a brief step-by-step description of the functionality of the decompiled code in the format:
**Description**:
...

Then try to generate a summary of it that can help human understand / inspect its original high-level source code functionality in the format:
**Summary**:
The function ...

After that, consider the following source functions (if any) that are potentially relevant to the this decompiled function.
{}

Analyze whether they are relevant to the decompiled function in the format:
**Analysis**:
...

Finally, given relevant source code information, try to inspect and generate the general purpose of the decompiled function in the format:
**Purpose**:
The purpose of the function seems to ...
"""

    def pack_dec(
        self,
        decompiled_function,
    ):
        message = [
            {"role": "system", "content": self.meta_system_prompt},
            {
                "role": "user", 
                "content": self.default_user_prompt.format(decompiled_function)
            },
        ]
        return message
    
    def pack_dec_w_aug(
        self,
        decompiled_function: str,
        source_candidates: List[str],
    ):
        message = [
            {"role": "system", "content": self.meta_system_prompt},
            {
                "role": "user", 
                "content": self.augmented_user_prompt.format(
                    decompiled_function,
                    '\n'.join(
                    [f'Potential source function {i+1}:\n```{s.rstrip()}\n```' \
                        for i, s in enumerate(source_candidates)])
                )
            },
        ]
        return message
    
    def generate(
        self,
        inputs,
        **kwargs
    ):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.pack_dec(inputs),
            **kwargs
        )

    def batch_generate(
        self,
        batch_inputs,
        **kwargs
    ):
        responses = []
        for inputs in batch_inputs:
            responses.append(self.generate(inputs, **kwargs))
        return responses
    
    def generate_with_augmentation(
        self,
        inputs,
        **kwargs
    ):
        assert len(inputs) == 2
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.pack_dec_w_aug(inputs[0], inputs[1]),
            **kwargs
        )

    def batch_generate_with_augmentation(
        self,
        batch_inputs: List[Tuple[str, List[str]]],
        **kwargs
    ):
        responses = []
        for inputs in batch_inputs:
            responses.append(self.generate_with_augmentation(inputs, **kwargs))
        return responses


class OpenAIDecompFuncNamer(OpenAIMetaPrompter):

    def __init__(
        self, 
        api_key: str, 
        model_name: str
    ):
        super().__init__(api_key, model_name)
        self.system_prompt = \
        "You are an experienced binary reverse engineer to understand "\
        "decompiled C code that lacks symbol information. "

        self.default_user_prompt = \
"""You have decompiled a function from an executable, which currently has a generic name like `sub_xxx`. The decompiled function code is as follows:
```C
{}
```
Generate a more human-understandable function name for the decompiled code to replace the original `sub_xxx` in the format:
**Function Name**: `function_name_goes_here`
"""
        self.augmented_user_prompt = \
"""You have decompiled a function from an executable, which currently has a generic name like `sub_xxx`. The decompiled function code is as follows:
```C
{}
```

Consider the following source functions (if any) that are potentially relevant to the this decompiled function.
{}

Analyze whether these source functions are relevant to the decompiled function in the format:
**Analysis**:
...

Then, based on the analysis, generate a more human-understandable function name for the decompiled code to replace the original `sub_xxx` in the format:
**Function Name**:  `function_name_goes_here`
"""
# Then, properly leverage the naming information in the relevant source functions based on the analysis, and generate a more human-understandable function name for the decompiled code to replace the original `sub_xxx` in the format:

    def pack_dec(
        self,
        decompiled_function,
    ):
        message = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user", 
                "content": self.default_user_prompt.format(decompiled_function)
            },
        ]
        return message
    
    def pack_dec_w_aug(
        self,
        decompiled_function: str,
        source_candidates: List[str],
    ):
        message = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user", 
                "content": self.augmented_user_prompt.format(
                    decompiled_function,
                    '\n'.join(
                    [f'Potential source function {i+1}:\n```{s.rstrip()}\n```' \
                        for i, s in enumerate(source_candidates)])
                )
            },
        ]
        return message

    def generate(
        self,
        inputs,
        **kwargs
    ):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.pack_dec(inputs),
            **kwargs
        )

    def generate_with_augmentation(
        self,
        inputs,
        **kwargs
    ):
        assert len(inputs) == 2
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.pack_dec_w_aug(inputs[0], inputs[1]),
            **kwargs
        )